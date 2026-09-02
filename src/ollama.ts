const DEFAULT_OLLAMA_PORT = 11434;
const MAX_METADATA_BYTES = 2 * 1024 * 1024;
const MAX_MODELS = 1_000;
const MAX_MODEL_ID_LENGTH = 512;
const METADATA_TIMEOUT_MS = 10_000;

type OllamaRoute = "/api/version" | "/api/tags" | "/api/show";
type ModelCapability = "completion" | "embedding";
type Fetcher = (
  input: RequestInfo | URL,
  init?: RequestInit,
) => Promise<Response>;

export type OllamaErrorCode =
  | "ollama_unavailable"
  | "ollama_incompatible"
  | "timeout"
  | "canceled"
  | "no_models"
  | "remote_model_disallowed"
  | "chat_model_unavailable"
  | "embedding_model_unavailable"
  | "invalid_request"
  | "rate_limited"
  | "ollama_server_error"
  | "invalid_response"
  | "http_error";

export interface LLMvaultSettings {
  ollamaPort: number;
  chatModel: string | null;
  embeddingModel: string | null;
}

export interface OllamaDiscovery {
  version: string;
  installedModelCount: number;
  chatModels: string[];
  embeddingModels: string[];
  remoteModels: string[];
}

export type ModelValidation = "compatible" | "incompatible" | "remote";

export class OllamaError extends Error {
  readonly code: OllamaErrorCode;
  readonly route?: OllamaRoute;
  readonly status?: number;

  constructor(
    code: OllamaErrorCode,
    route?: OllamaRoute,
    status?: number,
  ) {
    super(code);
    this.name = "OllamaError";
    this.code = code;
    this.route = route;
    this.status = status;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isModelId(value: unknown): value is string {
  return (
    typeof value === "string" &&
    value.length > 0 &&
    value.length <= MAX_MODEL_ID_LENGTH
  );
}

function isRemote(value: Record<string, unknown>): boolean {
  return [value.remote_host, value.remote_model].some(
    (field) =>
      field !== undefined &&
      field !== null &&
      (typeof field !== "string" || field.length !== 0),
  );
}

function errorForStatus(route: OllamaRoute, status: number): OllamaError {
  if (status === 429) return new OllamaError("rate_limited", route, status);
  if (status === 500 || status === 502) {
    return new OllamaError("ollama_server_error", route, status);
  }
  if (route === "/api/version" || route === "/api/tags") {
    return new OllamaError("ollama_incompatible", route, status);
  }
  if (status === 400) return new OllamaError("invalid_request", route, status);
  return new OllamaError("http_error", route, status);
}

async function readBoundedJson(
  response: Response,
  route: OllamaRoute,
): Promise<unknown> {
  const declaredLength = Number(response.headers.get("content-length"));
  if (Number.isFinite(declaredLength) && declaredLength > MAX_METADATA_BYTES) {
    throw new OllamaError("invalid_response", route, response.status);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new OllamaError("invalid_response", route, response.status);

  const decoder = new TextDecoder("utf-8", { fatal: true });
  let bytes = 0;
  let finished = false;
  let text = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        finished = true;
        break;
      }
      bytes += value.byteLength;
      if (bytes > MAX_METADATA_BYTES) {
        throw new OllamaError("invalid_response", route, response.status);
      }
      try {
        text += decoder.decode(value, { stream: true });
      } catch {
        throw new OllamaError("invalid_response", route, response.status);
      }
    }
    try {
      text += decoder.decode();
      return JSON.parse(text);
    } catch {
      throw new OllamaError("invalid_response", route, response.status);
    }
  } finally {
    if (!finished) await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

export function normalizeSettings(value: unknown): LLMvaultSettings {
  const source = isRecord(value) ? value : {};
  const ollamaPort =
    typeof source.ollamaPort === "number" &&
    Number.isInteger(source.ollamaPort) &&
    source.ollamaPort >= 1 &&
    source.ollamaPort <= 65535
      ? source.ollamaPort
      : DEFAULT_OLLAMA_PORT;
  return {
    ollamaPort,
    chatModel: isModelId(source.chatModel) ? source.chatModel : null,
    embeddingModel: isModelId(source.embeddingModel)
      ? source.embeddingModel
      : null,
  };
}

export function ollamaOrigin(port: number): string {
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new TypeError("Port must be an integer from 1 to 65535");
  }
  return `http://127.0.0.1:${port}`;
}

export function revalidateSelections(
  settings: LLMvaultSettings,
  discovery: OllamaDiscovery,
): { settings: LLMvaultSettings; recoveryCodes: OllamaErrorCode[] } {
  const next = { ...settings };
  const recoveryCodes: OllamaErrorCode[] = [];
  const remote = new Set(discovery.remoteModels);

  if (next.chatModel && !discovery.chatModels.includes(next.chatModel)) {
    recoveryCodes.push(
      remote.has(next.chatModel)
        ? "remote_model_disallowed"
        : "chat_model_unavailable",
    );
    next.chatModel = null;
  }
  if (
    next.embeddingModel &&
    !discovery.embeddingModels.includes(next.embeddingModel)
  ) {
    const code = remote.has(next.embeddingModel)
      ? "remote_model_disallowed"
      : "embedding_model_unavailable";
    if (!recoveryCodes.includes(code)) recoveryCodes.push(code);
    next.embeddingModel = null;
  }

  return { settings: next, recoveryCodes };
}

export class OllamaClient {
  private readonly controllers = new Set<AbortController>();
  private readonly fetcher: Fetcher;

  constructor(fetcher: Fetcher = fetch) {
    this.fetcher = fetcher;
  }

  abortAll(): void {
    for (const controller of this.controllers) controller.abort();
    this.controllers.clear();
  }

  async discover(port: number): Promise<OllamaDiscovery> {
    const versionValue = await this.request(port, "/api/version", "GET");
    if (
      !isRecord(versionValue) ||
      typeof versionValue.version !== "string" ||
      versionValue.version.trim().length === 0 ||
      versionValue.version.length > 128 ||
      [...versionValue.version].some((character) => {
        const code = character.charCodeAt(0);
        return code <= 31 || code === 127;
      })
    ) {
      throw new OllamaError("ollama_incompatible", "/api/version");
    }

    const tagsValue = await this.request(port, "/api/tags", "GET");
    if (
      !isRecord(tagsValue) ||
      !Array.isArray(tagsValue.models) ||
      tagsValue.models.length > MAX_MODELS ||
      tagsValue.models.some(
        (model) => !isRecord(model) || !isModelId(model.model),
      )
    ) {
      throw new OllamaError("ollama_incompatible", "/api/tags");
    }

    const catalog = new Map<string, boolean>();
    for (const model of tagsValue.models) {
      if (!isRecord(model) || !isModelId(model.model)) continue;
      catalog.set(model.model, (catalog.get(model.model) ?? false) || isRemote(model));
    }

    const chatModels: string[] = [];
    const embeddingModels: string[] = [];
    const remoteModels: string[] = [];
    for (const [model, remote] of catalog) {
      if (remote) {
        remoteModels.push(model);
        continue;
      }
      const details = await this.inspectModel(port, model);
      if (details === "remote") {
        remoteModels.push(model);
      } else if (details) {
        if (details.has("completion")) chatModels.push(model);
        if (details.has("embedding")) embeddingModels.push(model);
      }
    }

    return {
      version: versionValue.version,
      installedModelCount: catalog.size,
      chatModels: chatModels.sort(),
      embeddingModels: embeddingModels.sort(),
      remoteModels: remoteModels.sort(),
    };
  }

  async validateModel(
    port: number,
    model: string,
    capability: ModelCapability,
  ): Promise<ModelValidation> {
    if (!isModelId(model)) return "incompatible";
    const details = await this.inspectModel(port, model);
    if (details === "remote") return "remote";
    return details?.has(capability) ? "compatible" : "incompatible";
  }

  private async inspectModel(
    port: number,
    model: string,
  ): Promise<Set<string> | "remote" | null> {
    let value: unknown;
    try {
      value = await this.request(port, "/api/show", "POST", { model });
    } catch (error) {
      if (
        error instanceof OllamaError &&
        (error.status === 404 || error.code === "invalid_response")
      ) {
        return null;
      }
      throw error;
    }
    if (isRecord(value) && isRemote(value)) return "remote";
    if (
      !isRecord(value) ||
      !Array.isArray(value.capabilities) ||
      value.capabilities.some((capability) => typeof capability !== "string")
    ) {
      return null;
    }
    return new Set(value.capabilities);
  }

  private async request(
    port: number,
    route: OllamaRoute,
    method: "GET" | "POST",
    body?: Record<string, string>,
  ): Promise<unknown> {
    const url = `${ollamaOrigin(port)}${route}`;
    const controller = new AbortController();
    this.controllers.add(controller);
    let timedOut = false;
    const timeout = globalThis.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, METADATA_TIMEOUT_MS);

    try {
      const response = await this.fetcher(url, {
        method,
        ...(body
          ? {
              body: JSON.stringify(body),
              headers: { "Content-Type": "application/json" },
            }
          : {}),
        cache: "no-store",
        credentials: "omit",
        redirect: "error",
        signal: controller.signal,
      });
      if (!response.ok) {
        await response.body?.cancel().catch(() => undefined);
        throw errorForStatus(route, response.status);
      }
      return await readBoundedJson(response, route);
    } catch (error) {
      if (error instanceof OllamaError) throw error;
      if (timedOut) throw new OllamaError("timeout", route);
      if (controller.signal.aborted) throw new OllamaError("canceled", route);
      throw new OllamaError("ollama_unavailable", route);
    } finally {
      globalThis.clearTimeout(timeout);
      this.controllers.delete(controller);
    }
  }
}

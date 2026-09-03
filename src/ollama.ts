const DEFAULT_OLLAMA_PORT = 11434;
const CHAT_INACTIVITY_TIMEOUT_MS = 5 * 60_000;
const CHAT_DIAGNOSTIC_FIELDS = [
  "done_reason",
  "total_duration",
  "load_duration",
  "prompt_eval_count",
  "prompt_eval_duration",
  "eval_count",
  "eval_duration",
] as const;
const CHAT_DONE_REASONS = new Set(["length", "stop"]);
const MAX_CHAT_RESPONSE_BYTES = 16 * 1024 * 1024;
const EMBEDDING_TIMEOUT_MS = 5 * 60_000;
const MAX_EMBEDDING_RESPONSE_BYTES = 16 * 1024 * 1024;
const MAX_METADATA_BYTES = 2 * 1024 * 1024;
const MAX_MODELS = 1_000;
const MAX_MODEL_ID_LENGTH = 512;
const METADATA_TIMEOUT_MS = 10_000;

type OllamaRoute = "/api/version" | "/api/tags" | "/api/show" | "/api/embed" | "/api/chat";
const OLLAMA_METHODS: Record<OllamaRoute, "GET" | "POST"> = {
  "/api/chat": "POST",
  "/api/embed": "POST",
  "/api/show": "POST",
  "/api/tags": "GET",
  "/api/version": "GET",
};
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
  | "stream_interrupted"
  | "http_error";

export interface OllamaMessage {
  content: string;
  role: "assistant" | "system" | "user";
}

export interface OllamaChatResult {
  content: string;
  diagnostics: Record<string, number | string>;
}

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
  modelDigests: Record<string, string>;
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

function isDigest(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= 512;
}

function isRemote(value: Record<string, unknown>): boolean {
  return [value.remote_host, value.remote_model].some(
    (field) =>
      field !== undefined &&
      field !== null &&
      (typeof field !== "string" || field.length !== 0),
  );
}

function parseModelCatalog(
  value: unknown,
): Map<string, { digest?: string; remote: boolean }> | null {
  if (
    !isRecord(value) ||
    !Array.isArray(value.models) ||
    value.models.length > MAX_MODELS ||
    value.models.some((model) => !isRecord(model) || !isModelId(model.model))
  ) {
    return null;
  }
  const catalog = new Map<string, { digest?: string; remote: boolean }>();
  for (const model of value.models) {
    if (!isRecord(model) || !isModelId(model.model)) continue;
    const previous = catalog.get(model.model);
    catalog.set(model.model, {
      ...(isDigest(model.digest)
        ? { digest: model.digest }
        : previous?.digest
          ? { digest: previous.digest }
          : {}),
      remote: (previous?.remote ?? false) || isRemote(model),
    });
  }
  return catalog;
}

function errorForStatus(route: OllamaRoute, status: number): OllamaError {
  if (status === 429) return new OllamaError("rate_limited", route, status);
  if (status === 500 || status === 502) {
    return new OllamaError("ollama_server_error", route, status);
  }
  if (route === "/api/version" || route === "/api/tags") {
    return new OllamaError("ollama_incompatible", route, status);
  }
  if (status === 404 && route === "/api/chat") {
    return new OllamaError("chat_model_unavailable", route, status);
  }
  if (status === 404 && route === "/api/embed") {
    return new OllamaError("embedding_model_unavailable", route, status);
  }
  if (status === 400) return new OllamaError("invalid_request", route, status);
  return new OllamaError("http_error", route, status);
}

async function rejectResponse(response: Response, route: OllamaRoute): Promise<never> {
  await response.body?.cancel().catch(() => undefined);
  throw errorForStatus(route, response.status);
}

async function readBoundedJson(
  response: Response,
  route: OllamaRoute,
  maximumBytes = MAX_METADATA_BYTES,
): Promise<unknown> {
  assertResponseLength(response, route, maximumBytes);

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
      if (bytes > maximumBytes) {
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

function assertResponseLength(
  response: Response,
  route: OllamaRoute,
  maximumBytes: number,
): void {
  const declaredLength = Number(response.headers.get("content-length"));
  if (Number.isFinite(declaredLength) && declaredLength > maximumBytes) {
    throw new OllamaError("invalid_response", route, response.status);
  }
}

function parseChatResponse(
  value: unknown,
  status: number,
): { content: string; diagnostics: Record<string, number | string>; done: boolean } {
  const route = "/api/chat";
  if (!isRecord(value)) throw new OllamaError("invalid_response", route, status);
  if (typeof value.error === "string" && value.error.length > 0) {
    throw new OllamaError("stream_interrupted", route, status);
  }
  if (
    typeof value.done !== "boolean" ||
    !isRecord(value.message) ||
    typeof value.message.content !== "string"
  ) {
    throw new OllamaError("invalid_response", route, status);
  }
  const diagnostics: Record<string, number | string> = {};
  for (const field of CHAT_DIAGNOSTIC_FIELDS) {
    const diagnostic = value[field];
    if (
      (field === "done_reason" && typeof diagnostic === "string" && CHAT_DONE_REASONS.has(diagnostic)) ||
      (field !== "done_reason" && typeof diagnostic === "number" && Number.isFinite(diagnostic))
    ) {
      diagnostics[field] = diagnostic;
    }
  }
  return { content: value.message.content, diagnostics, done: value.done };
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

  constructor(fetcher: Fetcher = globalThis.fetch.bind(globalThis)) {
    this.fetcher = fetcher;
  }

  abortAll(): void {
    for (const controller of this.controllers) controller.abort();
    this.controllers.clear();
  }

  async discover(port: number): Promise<OllamaDiscovery> {
    const versionValue = await this.request(port, "/api/version");
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

    const tagsValue = await this.request(port, "/api/tags");
    const catalog = parseModelCatalog(tagsValue);
    if (!catalog) {
      throw new OllamaError("ollama_incompatible", "/api/tags");
    }

    const chatModels: string[] = [];
    const embeddingModels: string[] = [];
    const modelDigests: Record<string, string> = {};
    const remoteModels: string[] = [];
    for (const [model, metadata] of catalog) {
      if (metadata.remote) {
        remoteModels.push(model);
        continue;
      }
      const details = await this.inspectModel(port, model);
      if (details === "remote") {
        remoteModels.push(model);
      } else if (details) {
        if (metadata.digest) modelDigests[model] = metadata.digest;
        if (details.has("completion")) chatModels.push(model);
        if (details.has("embedding")) embeddingModels.push(model);
      }
    }

    return {
      version: versionValue.version,
      installedModelCount: catalog.size,
      chatModels: chatModels.sort(),
      embeddingModels: embeddingModels.sort(),
      modelDigests,
      remoteModels: remoteModels.sort(),
    };
  }

  async validatePinnedModel(
    port: number,
    model: string,
    digest: string,
    capability: ModelCapability,
  ): Promise<ModelValidation> {
    if (!isModelId(model) || !isDigest(digest)) return "incompatible";
    const catalog = parseModelCatalog(await this.request(port, "/api/tags"));
    if (!catalog) throw new OllamaError("ollama_incompatible", "/api/tags");
    const metadata = catalog.get(model);
    if (!metadata) return "incompatible";
    if (metadata.remote) return "remote";
    if (metadata.digest !== digest) return "incompatible";
    const details = await this.inspectModel(port, model);
    if (details === "remote") return "remote";
    return details?.has(capability) ? "compatible" : "incompatible";
  }

  async embed(port: number, model: string, inputs: string[]): Promise<number[][]> {
    if (!isModelId(model) || inputs.length === 0 || inputs.some((input) => typeof input !== "string" || input.length === 0)) {
      throw new OllamaError("invalid_request", "/api/embed");
    }
    const value = await this.request(
      port,
      "/api/embed",
      { input: inputs, model, truncate: false },
      EMBEDDING_TIMEOUT_MS,
      MAX_EMBEDDING_RESPONSE_BYTES,
    );
    if (!isRecord(value) || !Array.isArray(value.embeddings) || value.embeddings.length !== inputs.length) {
      throw new OllamaError("invalid_response", "/api/embed");
    }
    let dimension = 0;
    const vectors: number[][] = [];
    for (const vector of value.embeddings) {
      if (!Array.isArray(vector) || vector.length === 0 || vector.some((item) => typeof item !== "number" || !Number.isFinite(item))) {
        throw new OllamaError("invalid_response", "/api/embed");
      }
      dimension ||= vector.length;
      if (vector.length !== dimension) throw new OllamaError("invalid_response", "/api/embed");
      vectors.push(vector as number[]);
    }
    return vectors;
  }

  async chat(
    port: number,
    model: string,
    messages: OllamaMessage[],
    onContent: (content: string) => void,
    stream = true,
    evaluationOptions?: { seed: 0; temperature: 0 },
  ): Promise<OllamaChatResult> {
    const route = "/api/chat";
    ollamaOrigin(port);
    if (
      !isModelId(model) ||
      messages.length === 0 ||
      messages.some(
        (message) =>
          !isRecord(message) ||
          !["assistant", "system", "user"].includes(message.role) ||
          typeof message.content !== "string" ||
          message.content.length === 0,
      )
    ) {
      throw new OllamaError("invalid_request", route);
    }

    const controller = new AbortController();
    this.controllers.add(controller);
    let timedOut = false;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    const resetTimeout = (): void => {
      if (timeout !== undefined) globalThis.clearTimeout(timeout);
      timeout = globalThis.setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, CHAT_INACTIVITY_TIMEOUT_MS);
    };
    resetTimeout();

    let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
    try {
      const response = await this.transport(
        port,
        route,
        {
          model,
          messages,
          options: {
            num_predict: 256,
            ...(evaluationOptions ? { seed: 0, temperature: 0 } : {}),
          },
          stream,
          think: false,
        },
        controller.signal,
      );
      if (!response.ok) {
        await rejectResponse(response, route);
      }
      assertResponseLength(response, route, MAX_CHAT_RESPONSE_BYTES);
      if (!stream) {
        const value = parseChatResponse(
          await readBoundedJson(response, route, MAX_CHAT_RESPONSE_BYTES),
          response.status,
        );
        if (!value.done) throw new OllamaError("invalid_response", route, response.status);
        if (value.content.length > 0) onContent(value.content);
        return { content: value.content, diagnostics: value.diagnostics };
      }
      if (response.headers.get("content-type")?.split(";", 1)[0]?.trim() !== "application/x-ndjson") {
        throw new OllamaError("invalid_response", route, response.status);
      }
      reader = response.body?.getReader();
      if (!reader) throw new OllamaError("invalid_response", route, response.status);

      const decoder = new TextDecoder("utf-8", { fatal: true });
      let buffer = "";
      let bytes = 0;
      let content = "";
      let done = false;
      let diagnostics: Record<string, number | string> = {};
      const consume = (line: string): void => {
        if (line.trim().length === 0) return;
        if (done) throw new OllamaError("invalid_response", route, response.status);
        let value: unknown;
        try {
          value = JSON.parse(line);
        } catch {
          throw new OllamaError("invalid_response", route, response.status);
        }
        const parsed = parseChatResponse(value, response.status);
        const chunk = parsed.content;
        if (chunk.length > 0) {
          content += chunk;
          onContent(chunk);
        }
        done = parsed.done;
        if (done) diagnostics = parsed.diagnostics;
        resetTimeout();
      };

      while (true) {
        const next = await reader.read();
        if (next.done) break;
        bytes += next.value.byteLength;
        if (bytes > MAX_CHAT_RESPONSE_BYTES) {
          throw new OllamaError("invalid_response", route, response.status);
        }
        try {
          buffer += decoder.decode(next.value, { stream: true });
        } catch {
          throw new OllamaError("invalid_response", route, response.status);
        }
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) consume(line);
        if (done) {
          if (buffer.trim().length > 0) {
            throw new OllamaError("invalid_response", route, response.status);
          }
          break;
        }
      }
      try {
        buffer += decoder.decode();
      } catch {
        throw new OllamaError("invalid_response", route, response.status);
      }
      consume(buffer);
      if (!done) throw new OllamaError("stream_interrupted", route, response.status);
      return { content, diagnostics };
    } catch (error) {
      if (error instanceof OllamaError) throw error;
      if (timedOut) throw new OllamaError("timeout", route);
      if (controller.signal.aborted) throw new OllamaError("canceled", route);
      throw new OllamaError("ollama_unavailable", route);
    } finally {
      if (timeout !== undefined) globalThis.clearTimeout(timeout);
      await reader?.cancel().catch(() => undefined);
      reader?.releaseLock();
      this.controllers.delete(controller);
    }
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
      value = await this.request(port, "/api/show", { model });
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
    body?: Record<string, unknown>,
    timeoutMs = METADATA_TIMEOUT_MS,
    maximumBytes = MAX_METADATA_BYTES,
  ): Promise<unknown> {
    ollamaOrigin(port);
    const controller = new AbortController();
    this.controllers.add(controller);
    let timedOut = false;
    const timeout = globalThis.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs);

    try {
      const response = await this.transport(port, route, body, controller.signal);
      if (!response.ok) {
        await rejectResponse(response, route);
      }
      return await readBoundedJson(response, route, maximumBytes);
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

  private async transport(
    port: number,
    route: OllamaRoute,
    body: Record<string, unknown> | undefined,
    signal: AbortSignal,
  ): Promise<Response> {
    const method = OLLAMA_METHODS[route];
    return await this.fetcher(`${ollamaOrigin(port)}${route}`, {
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
      signal,
    });
  }
}

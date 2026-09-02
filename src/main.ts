import { ItemView, Plugin, WorkspaceLeaf } from "obsidian";

import {
  type IndexSnapshot,
  type MarkdownAnchor,
  MarkdownIndex,
  type MarkdownSource,
} from "./indexing";
import {
  type LLMvaultSettings,
  type OllamaDiscovery,
  OllamaClient,
  OllamaError,
  type OllamaErrorCode,
  normalizeSettings,
  revalidateSelections,
} from "./ollama";

const VIEW_TYPE_VAULT_CHAT = "vault-chat-view";

const RECOVERY_MESSAGES: Record<OllamaErrorCode, string> = {
  ollama_unavailable:
    "Ollama is unavailable. Start Ollama and verify the configured port, then retry.",
  ollama_incompatible:
    "The service did not provide a compatible Ollama response. Update Ollama or verify the port, then retry.",
  timeout: "Ollama did not respond in time. Retry the connection.",
  canceled: "The connection check was canceled. Retry when ready.",
  no_models: "No models are installed in Ollama. Install models there, then refresh.",
  remote_model_disallowed:
    "A saved model is remote-backed and cannot be used for Local Processing. Choose a local model.",
  chat_model_unavailable:
    "The saved chat model is missing or no longer supports completion. Choose a compatible chat model.",
  embedding_model_unavailable:
    "The saved embedding model is missing or no longer supports embeddings. Choose a compatible embedding model.",
  invalid_request:
    "Ollama rejected the model-details request. Refresh the installed models and try again.",
  rate_limited: "Ollama is busy. Retry the connection later.",
  ollama_server_error: "Ollama reported a server error. Retry the connection.",
  invalid_response:
    "Ollama returned an invalid or oversized response. Update Ollama, then retry.",
  http_error:
    "Ollama returned an unexpected HTTP response. Verify the port and retry.",
};

class VaultChatView extends ItemView {
  private askButton?: HTMLButtonElement;
  private busy = false;
  private chatModel: string | null;
  private discovery: OllamaDiscovery | null = null;
  private embeddingModel: string | null;
  private indexEl?: HTMLElement;
  private indexUnsubscribe?: () => void;
  private questionEl?: HTMLTextAreaElement;
  private requestGeneration = 0;
  private portValue: string;
  private setupEl?: HTMLElement;
  private status = "Checking the local Ollama connection…";

  constructor(
    leaf: WorkspaceLeaf,
    private readonly plugin: LLMvaultPlugin,
  ) {
    super(leaf);
    const settings = plugin.getSettings();
    this.portValue = String(settings.ollamaPort);
    this.chatModel = settings.chatModel;
    this.embeddingModel = settings.embeddingModel;
  }

  getViewType(): string {
    return VIEW_TYPE_VAULT_CHAT;
  }

  getDisplayText(): string {
    return "Vault Chat";
  }

  override getIcon(): string {
    return "message-circle";
  }

  override onOpen(): Promise<void> {
    this.renderShell();
    this.indexUnsubscribe = this.plugin.subscribeIndex(() => this.renderIndex());
    void this.refreshModels();
    return Promise.resolve();
  }

  override onClose(): Promise<void> {
    this.requestGeneration += 1;
    this.plugin.abortOllamaRequests();
    this.indexUnsubscribe?.();
    this.contentEl.empty();
    return Promise.resolve();
  }

  private renderShell(): void {
    const root = this.contentEl;
    root.empty();
    root.addClass("llmvault-chat");

    const header = root.createEl("header", { cls: "llmvault-chat__header" });
    header.createEl("h2", { text: "Vault Chat" });
    header.createEl("span", {
      cls: "llmvault-chat__local-badge",
      text: "Local processing",
    });

    this.setupEl = root.createDiv({ cls: "llmvault-chat__setup" });
    this.renderSetup();

    this.indexEl = root.createEl("section", { cls: "llmvault-chat__index" });
    this.renderIndex();

    const answer = root.createEl("section", {
      cls: "llmvault-chat__answer",
      attr: { "aria-labelledby": "llmvault-answer-heading" },
    });
    answer.createEl("h3", {
      attr: { id: "llmvault-answer-heading" },
      text: "Answer",
    });
    answer.createEl("p", {
      cls: "llmvault-chat__placeholder",
      text: "Your grounded answer will appear here.",
    });

    const evidence = root.createEl("section", {
      cls: "llmvault-chat__evidence",
      attr: { "aria-labelledby": "llmvault-evidence-heading" },
    });
    evidence.createEl("h3", {
      attr: { id: "llmvault-evidence-heading" },
      text: "Evidence",
    });
    evidence.createEl("p", {
      cls: "llmvault-chat__placeholder",
      text: "Evidence from Vault Content will remain visible here.",
    });

    const composer = root.createEl("form", {
      cls: "llmvault-chat__composer",
    });
    composer.createEl("label", {
      cls: "llmvault-chat__sr-only",
      attr: { for: "llmvault-question" },
      text: "Ask a question about your vault",
    });
    composer.onsubmit = (event) => event.preventDefault();
    this.questionEl = composer.createEl("textarea", {
      attr: {
        "aria-describedby": "llmvault-setup-status",
        id: "llmvault-question",
        placeholder: "Ask about your vault…",
        rows: "3",
      },
    });
    this.askButton = composer.createEl("button", {
      attr: { type: "submit" },
      text: "Ask",
    });
    this.renderIndex();
  }

  private renderIndex(): void {
    const snapshot = this.plugin.getIndexSnapshot();
    const index = this.indexEl;
    if (index) {
      index.empty();
      index.createEl("h3", { text: "Vault index" });
      index.createEl("p", {
        attr: { "aria-live": "polite", role: "status" },
        text: this.indexMessage(snapshot),
      });
    }
    if (this.questionEl) this.questionEl.disabled = !snapshot.active;
    if (this.askButton) this.askButton.disabled = !snapshot.active;
  }

  private indexMessage(snapshot: IndexSnapshot): string {
    const outcomes = Object.entries(snapshot.statuses)
      .map(([status, count]) => `${String(count)} ${status.replaceAll("_", " ")}`)
      .join(", ");
    if (snapshot.phase === "ready") {
      return `Index ready: ${outcomes || "no Markdown files"}.`;
    }
    if (snapshot.phase === "indexing") {
      const current = snapshot.latestPath ? ` Current source: ${snapshot.latestPath}.` : "";
      return `Indexing ${snapshot.completed} of ${snapshot.total} Markdown files${outcomes ? ` (${outcomes})` : ""}.${current}`;
    }
    if (snapshot.phase === "failed") {
      return "Indexing failed. Verify the selected embedding model and retry setup.";
    }
    return "Indexing waits for compatible Local Model setup.";
  }

  private renderSetup(): void {
    const setup = this.setupEl;
    if (!setup) return;
    setup.empty();
    setup.createEl("h3", { text: "Connect Local Models" });
    setup.createEl("p", {
      attr: {
        "aria-live": "polite",
        id: "llmvault-setup-status",
        role: "status",
      },
      text: this.status,
    });
    setup.createEl("p", {
      cls: "llmvault-chat__disclosure",
      text: "LLMvault sends questions, retrieved source text, chat history, and embedding inputs only to your user-managed Ollama service at 127.0.0.1:<port>. It sends no telemetry and contacts no remote service.",
    });

    const form = setup.createEl("form", { cls: "llmvault-chat__setup-form" });
    form.onsubmit = (event) => event.preventDefault();

    const portField = form.createDiv({ cls: "llmvault-chat__field" });
    portField.createEl("label", {
      attr: { for: "llmvault-ollama-port" },
      text: "Ollama port",
    });
    const port = portField.createEl("input", {
      attr: {
        id: "llmvault-ollama-port",
        inputmode: "numeric",
        max: "65535",
        min: "1",
        required: "",
        step: "1",
        type: "number",
      },
    });
    port.value = this.portValue;
    port.disabled = this.busy;

    const refresh = form.createEl("button", {
      attr: { type: "button" },
      text: this.busy ? "Checking…" : "Refresh installed models",
    });
    refresh.disabled = this.busy;

    const chat = this.createModelSelect(
      form,
      "llmvault-chat-model",
      "Chat model",
      "Choose a compatible chat model",
      this.discovery?.chatModels ?? [],
      this.chatModel,
    );
    const embedding = this.createModelSelect(
      form,
      "llmvault-embedding-model",
      "Embedding model",
      "Choose a compatible embedding model",
      this.discovery?.embeddingModels ?? [],
      this.embeddingModel,
    );
    const complete = form.createEl("button", {
      cls: "mod-cta",
      attr: { type: "button" },
      text: this.busy ? "Validating…" : "Complete setup",
    });

    const updateCompleteButton = (): void => {
      complete.disabled =
        this.busy || !this.discovery || !this.chatModel || !this.embeddingModel;
    };
    updateCompleteButton();

    port.oninput = () => {
      this.portValue = port.value;
      this.discovery = null;
      this.chatModel = null;
      this.embeddingModel = null;
      chat.value = "";
      embedding.value = "";
      chat.disabled = true;
      embedding.disabled = true;
      this.status = "Refresh installed models for this port.";
      setup.querySelector("p")?.setText(this.status);
      updateCompleteButton();
    };
    refresh.onclick = () => void this.refreshModels();
    chat.onchange = () => {
      this.chatModel = chat.value || null;
      updateCompleteButton();
    };
    embedding.onchange = () => {
      this.embeddingModel = embedding.value || null;
      updateCompleteButton();
    };
    complete.onclick = () => void this.completeSetup();
  }

  private createModelSelect(
    parent: HTMLElement,
    id: string,
    label: string,
    placeholder: string,
    models: string[],
    selected: string | null,
  ): HTMLSelectElement {
    const field = parent.createDiv({ cls: "llmvault-chat__field" });
    field.createEl("label", { attr: { for: id }, text: label });
    const select = field.createEl("select", { attr: { id } });
    select.createEl("option", { attr: { value: "" }, text: placeholder });
    for (const model of models) {
      select.createEl("option", { attr: { value: model }, text: model });
    }
    select.value = selected && models.includes(selected) ? selected : "";
    select.disabled = this.busy || !this.discovery;
    return select;
  }

  private port(): number | null {
    const port = Number(this.portValue);
    return Number.isInteger(port) && port >= 1 && port <= 65535 ? port : null;
  }

  private async refreshModels(): Promise<void> {
    const port = this.port();
    if (port === null) {
      this.status = "Enter an integer port from 1 through 65535.";
      this.renderSetup();
      return;
    }

    const requestGeneration = ++this.requestGeneration;
    this.plugin.abortOllamaRequests();
    this.busy = true;
    this.status = "Checking Ollama and installed model capabilities…";
    this.renderSetup();

    try {
      const discovery = await this.plugin.discoverModels(port);
      if (requestGeneration !== this.requestGeneration) return;
      this.discovery = discovery;

      const saved = this.plugin.getSettings();
      const settings =
        saved.ollamaPort === port
          ? saved
          : { ollamaPort: port, chatModel: null, embeddingModel: null };
      const revalidated = revalidateSelections(settings, discovery);
      this.chatModel = revalidated.settings.chatModel;
      this.embeddingModel = revalidated.settings.embeddingModel;
      if (
        saved.ollamaPort === port &&
        (saved.chatModel !== this.chatModel ||
          saved.embeddingModel !== this.embeddingModel)
      ) {
        await this.plugin.saveSettings(revalidated.settings);
      }
      this.status = this.discoveryStatus(revalidated.recoveryCodes);
    } catch (error) {
      if (requestGeneration !== this.requestGeneration) return;
      this.discovery = null;
      this.status = this.errorMessage(error);
    } finally {
      if (requestGeneration === this.requestGeneration) {
        this.busy = false;
        this.renderSetup();
      }
    }
  }

  private async completeSetup(): Promise<void> {
    const port = this.port();
    const chatModel = this.chatModel;
    const embeddingModel = this.embeddingModel;
    if (
      port === null ||
      !this.discovery ||
      !chatModel ||
      !embeddingModel ||
      !this.discovery.chatModels.includes(chatModel) ||
      !this.discovery.embeddingModels.includes(embeddingModel)
    ) {
      this.status = "Choose both compatible Local Model roles before completing setup.";
      this.renderSetup();
      return;
    }

    const requestGeneration = ++this.requestGeneration;
    this.plugin.abortOllamaRequests();
    this.busy = true;
    this.status = "Revalidating both Local Models…";
    this.renderSetup();

    try {
      const chat = await this.plugin.validateModel(port, chatModel, "completion");
      if (requestGeneration !== this.requestGeneration) return;
      if (chat !== "compatible") {
        this.chatModel = null;
        const saved = this.plugin.getSettings();
        await this.plugin.saveSettings({
          ollamaPort: port,
          chatModel: null,
          embeddingModel:
            saved.ollamaPort === port ? saved.embeddingModel : null,
        });
        this.status = RECOVERY_MESSAGES[
          chat === "remote"
            ? "remote_model_disallowed"
            : "chat_model_unavailable"
        ];
        return;
      }

      const embedding = await this.plugin.validateModel(
        port,
        embeddingModel,
        "embedding",
      );
      if (requestGeneration !== this.requestGeneration) return;
      if (embedding !== "compatible") {
        this.embeddingModel = null;
        await this.plugin.saveSettings({
          ollamaPort: port,
          chatModel,
          embeddingModel: null,
        });
        this.status = RECOVERY_MESSAGES[
          embedding === "remote"
            ? "remote_model_disallowed"
            : "embedding_model_unavailable"
        ];
        return;
      }

      await this.plugin.saveSettings({ ollamaPort: port, chatModel, embeddingModel });
      this.status = "Setup complete. Both Local Models are compatible; indexing is starting.";
      void this.plugin.startIndexing(this.discovery, embeddingModel);
    } catch (error) {
      if (requestGeneration !== this.requestGeneration) return;
      this.status = this.errorMessage(error);
    } finally {
      if (requestGeneration === this.requestGeneration) {
        this.busy = false;
        this.renderSetup();
      }
    }
  }

  private discoveryStatus(recoveryCodes: OllamaErrorCode[]): string {
    const discovery = this.discovery;
    if (!discovery) return RECOVERY_MESSAGES.ollama_incompatible;
    if (discovery.installedModelCount === 0) {
      return RECOVERY_MESSAGES.no_models;
    }
    if (recoveryCodes.includes("remote_model_disallowed")) {
      return RECOVERY_MESSAGES.remote_model_disallowed;
    }
    if (recoveryCodes.length > 0) {
      return recoveryCodes.map((code) => RECOVERY_MESSAGES[code]).join(" ");
    }
    if (discovery.installedModelCount === discovery.remoteModels.length) {
      return "Only remote-backed models were found. Install local models in Ollama, then refresh.";
    }
    if (discovery.chatModels.length === 0) {
      return "No local model supports completion. Install one in Ollama, then refresh.";
    }
    if (discovery.embeddingModels.length === 0) {
      return "No local model supports embeddings. Install one in Ollama, then refresh.";
    }
    if (this.chatModel && this.embeddingModel) {
      return `Setup complete. Saved Local Models were revalidated with Ollama ${discovery.version}; indexing has not started.`;
    }
    return `Connected to Ollama ${discovery.version}. Explicitly choose both Local Model roles.`;
  }

  private errorMessage(error: unknown): string {
    return error instanceof OllamaError
      ? RECOVERY_MESSAGES[error.code]
      : "Setup could not be saved. Retry the operation.";
  }
}

export default class LLMvaultPlugin extends Plugin {
  private index?: MarkdownIndex;
  private indexSnapshot: IndexSnapshot = {
    active: false,
    completed: 0,
    phase: "idle",
    statuses: {},
    total: 0,
  };
  private readonly indexSubscribers = new Set<(snapshot: IndexSnapshot) => void>();
  private readonly indexOllama = new OllamaClient();
  private readonly ollama = new OllamaClient();
  private llmvaultSettings = normalizeSettings(null);

  override async onload(): Promise<void> {
    this.llmvaultSettings = normalizeSettings(await this.loadData());
    const pluginDirectory = this.manifest.dir;
    if (pluginDirectory) {
      this.index = new MarkdownIndex(
        this.app.vault.adapter,
        `${pluginDirectory}/index-v1`,
        () => this.markdownSources(),
        (inputs) => {
          const settings = this.getSettings();
          if (!settings.embeddingModel) return Promise.reject(new Error("embedding_model_unavailable"));
          return this.indexOllama.embed(settings.ollamaPort, settings.embeddingModel, inputs);
        },
        (snapshot) => this.reportIndex(snapshot),
      );
    }
    this.registerView(
      VIEW_TYPE_VAULT_CHAT,
      (leaf) => new VaultChatView(leaf, this),
    );

    this.addCommand({
      id: "open-vault-chat",
      name: "Open Vault Chat",
      callback: () => void this.openVaultChat(),
    });

    this.addRibbonIcon("message-circle", "Open Vault Chat", () => {
      void this.openVaultChat();
    });

    this.app.workspace.onLayoutReady(() => void this.restoreIndex());
  }

  override onunload(): void {
    this.index?.cancel();
    this.indexOllama.abortAll();
    this.ollama.abortAll();
    void this.app.workspace.detachLeavesOfType(VIEW_TYPE_VAULT_CHAT);
  }

  getSettings(): LLMvaultSettings {
    return { ...this.llmvaultSettings };
  }

  async saveSettings(settings: LLMvaultSettings): Promise<void> {
    this.llmvaultSettings = normalizeSettings(settings);
    await this.saveData(this.llmvaultSettings);
  }

  discoverModels(port: number): Promise<OllamaDiscovery> {
    return this.ollama.discover(port);
  }

  validateModel(
    port: number,
    model: string,
    capability: "completion" | "embedding",
  ): Promise<"compatible" | "incompatible" | "remote"> {
    return this.ollama.validateModel(port, model, capability);
  }

  abortOllamaRequests(): void {
    this.ollama.abortAll();
  }

  getIndexSnapshot(): IndexSnapshot {
    return { ...this.indexSnapshot, statuses: { ...this.indexSnapshot.statuses } };
  }

  subscribeIndex(subscriber: (snapshot: IndexSnapshot) => void): () => void {
    this.indexSubscribers.add(subscriber);
    subscriber(this.getIndexSnapshot());
    return () => this.indexSubscribers.delete(subscriber);
  }

  async startIndexing(discovery: OllamaDiscovery, embeddingModel: string): Promise<void> {
    const digest = discovery.modelDigests[embeddingModel];
    if (!this.index || !digest) {
      this.reportIndex({ active: false, completed: 0, phase: "failed", statuses: {}, total: 0 });
      return;
    }
    this.indexOllama.abortAll();
    this.index.cancel();
    await this.index.start({ digest, name: embeddingModel });
  }

  private markdownSources(): MarkdownSource[] {
    const configurationRoot = `${this.app.vault.configDir}/`;
    return this.app.vault
      .getMarkdownFiles()
      .filter((file) => file.path !== this.app.vault.configDir && !file.path.startsWith(configurationRoot))
      .map((file) => {
        const cache = this.app.metadataCache.getFileCache(file);
        const anchors: MarkdownAnchor[] = [
          ...(cache?.headings ?? []).map((heading) => ({
            offset: heading.position.start.offset,
            type: "heading" as const,
            value: heading.heading,
          })),
          ...Object.values(cache?.blocks ?? {}).map((block) => ({
            offset: block.position.start.offset,
            type: "block" as const,
            value: block.id,
          })),
        ];
        return {
          anchors,
          path: file.path,
          read: () => this.app.vault.cachedRead(file),
        };
      });
  }

  private reportIndex(snapshot: IndexSnapshot): void {
    this.indexSnapshot = snapshot;
    for (const subscriber of this.indexSubscribers) subscriber(this.getIndexSnapshot());
  }

  private async restoreIndex(): Promise<void> {
    const settings = this.getSettings();
    if (!settings.embeddingModel) return;
    try {
      const discovery = await this.discoverModels(settings.ollamaPort);
      if (discovery.embeddingModels.includes(settings.embeddingModel)) {
        await this.startIndexing(discovery, settings.embeddingModel);
      }
    } catch {
      this.reportIndex({ active: false, completed: 0, phase: "failed", statuses: {}, total: 0 });
    }
  }

  private async openVaultChat(): Promise<void> {
    let leaf = this.app.workspace.getLeavesOfType(VIEW_TYPE_VAULT_CHAT)[0];
    leaf ??= this.app.workspace.getRightLeaf(false) ?? undefined;
    if (!leaf) return;

    await leaf.setViewState({ active: true, type: VIEW_TYPE_VAULT_CHAT });
    await this.app.workspace.revealLeaf(leaf);
  }
}

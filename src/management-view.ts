import type LLMvaultPlugin from "./main";
import type { IndexSnapshot } from "./indexing";
import { OllamaError, type OllamaDiscovery, type OllamaErrorCode, revalidateSelections } from "./ollama";

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
    "The saved embedding model is missing, changed, or no longer supports embeddings. Select it again to rebuild the index.",
  invalid_request:
    "Ollama rejected the request. Refresh the installed models and retry.",
  rate_limited: "Ollama is busy. Retry later.",
  ollama_server_error: "Ollama reported a server error. Retry.",
  invalid_response:
    "Ollama returned an invalid or oversized response. Update Ollama, then retry.",
  stream_interrupted:
    "The answer stream ended before Ollama reported completion. Retry the question.",
  http_error:
    "Ollama returned an unexpected HTTP response. Verify the port and retry.",
};

export function errorMessage(error: unknown): string {
  if (!(error instanceof OllamaError)) {
    return "The operation could not be completed. Retry it.";
  }
  return RECOVERY_MESSAGES[error.code];
}

export class VaultManagementView {
  private busy = false;
  private chatModel: string | null;
  private deletionState: "idle" | "deleting" | "complete" | "failed" | "canceled" = "idle";
  private discovery: OllamaDiscovery | null = null;
  private embeddingModel: string | null;
  private indexEl?: HTMLElement;
  private indexStatusEl?: HTMLElement;
  private rebuildButton?: HTMLButtonElement;
  private outcomeDetails?: HTMLDetailsElement;
  private outcomeSummary?: HTMLElement;
  private outcomeList?: HTMLElement;
  private portValue: string;
  private setupGeneration = 0;
  private setupEl?: HTMLElement;
  private status = "Checking the local Ollama connection…";

  constructor(
    private readonly plugin: LLMvaultPlugin,
    private readonly navigation: { showChat(): void; showManagement(): void; resetChat(): void },
  ) {
    const settings = plugin.getSettings();
    this.portValue = String(settings.ollamaPort);
    this.chatModel = settings.chatModel;
    this.embeddingModel = settings.embeddingModel;
    if (plugin.isStopped()) this.status = "Vault Chat is stopped. Connection settings are retained.";
    if (plugin.isDeletionIncomplete()) this.deletionState = "failed";
  }

  mount(root: HTMLElement): void {
    this.setupEl = root.createDiv({ cls: "llmvault-chat__setup" });
    this.indexEl = root.createEl("section", { cls: "llmvault-chat__index" });
    this.indexEl.createEl("h3", { text: "Vault index" });
    this.indexStatusEl = this.indexEl.createEl("p", {
      attr: { "aria-live": "polite", role: "status" },
    });
    this.rebuildButton = this.indexEl.createEl("button", { attr: { type: "button" } });
    this.rebuildButton.onclick = () => void this.plugin.rebuildIndex();
    this.outcomeDetails = this.indexEl.createEl("details");
    this.outcomeSummary = this.outcomeDetails.createEl("summary");
    this.outcomeList = this.outcomeDetails.createEl("ul");
    this.outcomeDetails.ontoggle = () => this.renderIndex();
    this.renderSetup();
    this.renderIndex();
  }

  setVisible(visible: boolean): void {
    if (this.setupEl) this.setupEl.hidden = !visible;
    if (this.indexEl) this.indexEl.hidden = !visible;
    if (visible) this.renderIndex();
  }

  onStopped(): void {
    if (this.deletionState !== "deleting") this.cancelSetup();
  }

  renderIndex(): void {
    if (!this.indexEl || this.indexEl.hidden) return;
    const snapshot = this.plugin.getIndexSnapshot();
    this.indexStatusEl?.setText(this.indexMessage(snapshot));
    if (this.rebuildButton) {
      this.rebuildButton.hidden = snapshot.phase === "idle";
      this.rebuildButton.setText(snapshot.phase === "failed" ? "Retry rebuild" : "Rebuild index");
      this.rebuildButton.disabled = snapshot.phase === "indexing";
    }
    const issues = Object.entries(snapshot.statuses)
      .reduce((total, [status, count]) => total + (status === "indexed" ? 0 : count), 0);
    if (this.outcomeDetails) this.outcomeDetails.hidden = issues === 0;
    this.outcomeSummary?.setText(`File details (${String(issues)})`);
    if (!issues || !this.outcomeDetails?.open || !this.outcomeList) return;
    this.outcomeList.empty();
    for (const outcome of snapshot.outcomes) {
      if (outcome.status === "indexed") continue;
      this.outcomeList.createEl("li", { text: `${outcome.path}: ${this.outcomeMessage(outcome)}` });
    }
  }

  private indexMessage(snapshot: IndexSnapshot): string {
    const outcomes = Object.entries(snapshot.statuses)
      .map(([status, count]) => `${String(count)} ${status.replaceAll("_", " ")}`)
      .join(", ");
    if (snapshot.phase === "ready") {
      return snapshot.total === 0
        ? "Index ready: no Vault Content files."
        : `Index ready: ${snapshot.total} Vault Content files (${outcomes}).`;
    }
    if (snapshot.phase === "indexing") {
      const available = snapshot.available ? " Previous generation remains available." : "";
      return `Indexing ${snapshot.completed} of ${snapshot.total} Vault Content files${outcomes ? ` (${outcomes})` : ""}.${available}`;
    }
    if (snapshot.phase === "failed") {
      return snapshot.available
        ? "Rebuild failed. Previous generation remains available. Retry the rebuild."
        : "Indexing failed. Verify the selected embedding model and retry.";
    }
    return "Indexing waits for compatible Local Model setup.";
  }

  private outcomeMessage(outcome: IndexSnapshot["outcomes"][number]): string {
    const reason = "reason" in outcome ? outcome.reason?.replaceAll("_", " ") : undefined;
    if (outcome.status === "no_extractable_text") return `no extractable text (${reason}); add text, then rebuild`;
    if (outcome.status === "ignored_non_content") return `ignored non-content (${reason})`;
    if (outcome.status === "unsupported_format") return `unsupported format (${reason}); convert to Markdown or Canvas to index`;
    if (outcome.status === "unrecognized_format") return `unrecognized format (${reason}); use a .md or .canvas extension to index`;
    if (outcome.status === "limit_exceeded") {
      return `${outcome.limit.replaceAll("_", " ")} ${String(outcome.observed)} exceeded ${String(outcome.ceiling)}; reduce the file and rebuild`;
    }
    return `extraction failed (${reason}); fix the source, then rebuild`;
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

    const deletion = setup.createDiv({ cls: "llmvault-chat__deletion" });
    deletion.createEl("h4", { text: "Stop and delete Vault Chat data" });
    deletion.createEl("p", {
      text: "Deletes indexes, conversations, statuses, caches, temporary records, and in-memory work. Vault Content, Ollama models, and retained connection settings are not changed.",
    });
    deletion.createEl("p", {
      cls: "llmvault-chat__disclosure",
      text: "This is not secure erasure. External backups, sync history, SSD behavior, OS state, and Ollama logs are outside this plugin's control.",
    });
    const deletionMessage = {
      canceled: "Deletion canceled. Data and active work were not changed.",
      complete: "Deletion complete. Vault Chat is stopped and plugin-owned content data was removed.",
      deleting: "Deleting plugin-owned Vault Chat data…",
      failed: "Deletion incomplete. Vault Chat remains stopped. Retry to remove any plugin-owned data that remains.",
      idle: "",
    }[this.deletionState];
    if (deletionMessage) {
      deletion.createEl("p", {
        attr: { "aria-live": "polite", role: "status" },
        text: deletionMessage,
      });
    }
    if (this.deletionState === "deleting") {
      const deleting = deletion.createEl("button", { attr: { type: "button" }, text: "Deleting…" });
      deleting.disabled = true;
    } else {
      const remove = deletion.createEl("button", {
        cls: "mod-warning",
        attr: { type: "button" },
        text: this.deletionState === "failed"
          ? "Retry delete all Vault Chat data"
          : "Delete all Vault Chat data",
      });
      remove.onclick = () => {
        if (!window.confirm(
          "Delete all Vault Chat data? Vault Chat work will stop before plugin-owned data is removed. Vault Content, Ollama models, and retained connection settings will not change.",
        )) {
          this.deletionState = "canceled";
          this.renderSetup();
          return;
        }
        void this.deleteAllData();
      };
    }
  }

  private async deleteAllData(): Promise<void> {
    this.cancelSetup();
    const setupGeneration = this.setupGeneration;
    this.deletionState = "deleting";
    this.busy = true;
    this.navigation.resetChat();
    this.renderSetup();
    try {
      await this.plugin.deleteAllData();
      if (setupGeneration !== this.setupGeneration) return;
      const settings = this.plugin.getSettings();
      this.portValue = String(settings.ollamaPort);
      this.chatModel = settings.chatModel;
      this.embeddingModel = settings.embeddingModel;
      this.discovery = null;
      this.status = "Vault Chat is stopped. Connection settings are retained.";
      this.navigation.resetChat();
      this.deletionState = "complete";
    } catch {
      if (setupGeneration !== this.setupGeneration) return;
      this.deletionState = "failed";
    } finally {
      if (setupGeneration === this.setupGeneration) {
        this.busy = false;
        this.renderSetup();
        this.renderIndex();
      }
    }
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

  cancelSetup(): void {
    this.setupGeneration += 1;
    this.plugin.abortOllamaRequests();
    this.busy = false;
  }

  async refreshModels(): Promise<void> {
    const port = this.port();
    if (port === null) {
      this.status = "Enter an integer port from 1 through 65535.";
      this.renderSetup();
      return;
    }

    const setupGeneration = ++this.setupGeneration;
    this.plugin.abortOllamaRequests();
    this.busy = true;
    this.status = "Checking Ollama and installed model capabilities…";
    this.renderSetup();

    try {
      const discovery = await this.plugin.discoverModels(port);
      if (setupGeneration !== this.setupGeneration) return;
      this.discovery = discovery;

      const saved = this.plugin.getSettings();
      const settings =
        saved.ollamaPort === port
          ? saved
          : { ollamaPort: port, chatModel: null, embeddingModel: null };
      const revalidated = revalidateSelections(settings, discovery);
      this.chatModel = revalidated.settings.chatModel;
      this.embeddingModel = revalidated.settings.embeddingModel;
      if (!this.chatModel || !this.embeddingModel) {
        this.navigation.showManagement();
      }
      this.status = this.discoveryStatus(revalidated.recoveryCodes);
    } catch (error) {
      if (setupGeneration !== this.setupGeneration) return;
      this.discovery = null;
      this.status = errorMessage(error);
    } finally {
      if (setupGeneration === this.setupGeneration) {
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

    const setupGeneration = ++this.setupGeneration;
    this.plugin.abortOllamaRequests();
    this.busy = true;
    this.status = "Revalidating both Local Models…";
    this.renderSetup();

    try {
      const chat = await this.plugin.validateModel(port, chatModel, "completion");
      if (setupGeneration !== this.setupGeneration) return;
      if (chat !== "compatible") {
        this.chatModel = null;
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
      if (setupGeneration !== this.setupGeneration) return;
      if (embedding !== "compatible") {
        this.embeddingModel = null;
        this.status = RECOVERY_MESSAGES[
          embedding === "remote"
            ? "remote_model_disallowed"
            : "embedding_model_unavailable"
        ];
        return;
      }

      await this.plugin.saveSettings({ ollamaPort: port, chatModel, embeddingModel }, this.discovery);
      if (setupGeneration !== this.setupGeneration) return;
      this.status = "Setup complete. Both Local Models are compatible; indexing is starting.";
      this.navigation.showChat();
    } catch (error) {
      if (setupGeneration !== this.setupGeneration) return;
      this.status = errorMessage(error);
    } finally {
      if (setupGeneration === this.setupGeneration) {
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

}

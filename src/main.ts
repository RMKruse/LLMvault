import { Plugin, type TAbstractFile, TFile } from "obsidian";

import {
  type EmbeddingModel,
  type IndexSnapshot,
  type MarkdownAnchor,
  VaultIndex,
  type VaultSource,
  type RetrievedEvidence,
  classifyVaultSource,
  copyIndexSnapshot,
  isSupportedVaultExtension,
  resolveDailyRecapRequest,
} from "./indexing";
import {
  type LLMvaultSettings,
  type OllamaChatResult,
  type OllamaDiscovery,
  OllamaClient,
  OllamaError,
  type OllamaMessage,
  normalizeSettings,
} from "./ollama";
import {
  type CompletedTurn,
  type ConversationState,
  completeTurn,
  deleteConversation,
  newConversation,
  normalizeConversationState,
  selectConversation,
} from "./conversations";
import { executeAnswer, type AnswerRequest, type AnswerResult } from "./answer.ts";

import { VaultChatView, VIEW_TYPE_VAULT_CHAT } from "./chat-view";

export default class LLMvaultPlugin extends Plugin {
  private answerController?: AbortController;
  private conversationState: ConversationState = {
    conversations: [],
    selectedConversationId: null,
  };
  private dataWrite: Promise<void> = Promise.resolve();
  // Owned by deletion, so an older queued save cannot release the immediate stop.
  private deletionBarrier = false;
  private deletionPending = false;
  private deleteOperation?: Promise<void>;
  private configurationRevision = 0;
  private index?: VaultIndex<EmbeddingModel & { port: number }>;
  private indexSnapshot: IndexSnapshot = {
    available: false,
    completed: 0,
    outcomes: [],
    phase: "idle",
    statuses: {},
    total: 0,
  };
  private readonly indexSubscribers = new Set<(snapshot: IndexSnapshot) => void>();
  private readonly mutationSubscribers = new Set<(paths: ReadonlySet<string>) => void>();
  private readonly indexOllama = new OllamaClient();
  private readonly chatOllama = new OllamaClient();
  private readonly ollama = new OllamaClient();
  private readonly queryOllama = new OllamaClient();
  private llmvaultSettings = normalizeSettings(null);
  private stopped = false;

  override async onload(): Promise<void> {
    const data: unknown = await this.loadData();
    this.llmvaultSettings = normalizeSettings(data);
    this.conversationState = normalizeConversationState(data);
    this.deletionPending = typeof data === "object" && data !== null &&
      "vaultChatDeletionPending" in data && data.vaultChatDeletionPending === true;
    this.stopped = this.deletionPending || (typeof data === "object" && data !== null &&
      "vaultChatStopped" in data && data.vaultChatStopped === true);
    const pluginDirectory = this.manifest.dir;
    if (pluginDirectory) {
      this.index = new VaultIndex<EmbeddingModel & { port: number }>(
        this.app.vault.adapter,
        `${pluginDirectory}/index-v1`,
        () => this.vaultSources(),
        (inputs, model) => this.indexOllama.embed(model.port, model.name, inputs),
        (snapshot) => this.reportIndex(snapshot),
        async (inputs, model) => {
          const validation = await this.queryOllama.validatePinnedModel(
            model.port,
            model.name,
            model.digest,
            "embedding",
          );
          if (validation !== "compatible") {
            throw new OllamaError(
              validation === "remote"
                ? "remote_model_disallowed"
                : "embedding_model_unavailable",
              "/api/embed",
            );
          }
          return await this.queryOllama.embed(model.port, model.name, inputs);
        },
        async (model) => {
          return (
            (await this.indexOllama.validatePinnedModel(
              model.port,
              model.name,
              model.digest,
              "embedding",
            )) === "compatible"
          );
        },
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

    this.registerEvent(this.app.vault.on("create", (file) => this.handleVaultMutation(file)));
    this.registerEvent(this.app.vault.on("modify", (file) => this.handleVaultMutation(file)));
    this.registerEvent(this.app.vault.on("delete", (file) => this.handleVaultMutation(file)));
    this.registerEvent(this.app.vault.on("rename", (file, oldPath) => {
      this.handleVaultMutation(file, oldPath);
    }));

    this.app.workspace.onLayoutReady(() => void this.restoreIndex());
  }

  override onunload(): void {
    this.index?.cancel();
    this.indexOllama.abortAll();
    this.abortAnswerRequests();
    this.ollama.abortAll();
    void this.app.workspace.detachLeavesOfType(VIEW_TYPE_VAULT_CHAT);
  }

  getSettings(): LLMvaultSettings {
    return { ...this.llmvaultSettings };
  }

  isStopped(): boolean {
    return this.deletionBarrier || this.stopped;
  }

  isDeletionIncomplete(): boolean {
    return this.deletionBarrier || this.deletionPending;
  }

  async saveSettings(settings: LLMvaultSettings, setup?: OllamaDiscovery): Promise<void> {
    const normalized = normalizeSettings(settings);
    const discovery = setup ? { ...setup, modelDigests: { ...setup.modelDigests } } : undefined;
    await this.updateData(() => {
      if (discovery) {
        if (this.deleteOperation) throw new Error("vault_chat_deletion_in_progress");
        if (this.isDeletionIncomplete()) throw new Error("vault_chat_deletion_incomplete");
        if (!normalized.chatModel || !normalized.embeddingModel || !discovery.modelDigests[normalized.embeddingModel]) {
          throw new OllamaError("embedding_model_unavailable", "/api/embed");
        }
      }
      return { llmvaultSettings: normalized, ...(discovery ? { stopped: false } : {}) };
    });
    if (discovery && normalized === this.llmvaultSettings && normalized.embeddingModel) {
      void this.startIndexing(discovery, normalized.embeddingModel);
    }
  }

  getConversationState(): ConversationState {
    return normalizeConversationState(this.conversationState);
  }

  getConversationTurns(id: string | null): readonly CompletedTurn[] {
    return this.conversationState.conversations.find((conversation) => conversation.id === id)?.turns ?? [];
  }

  async answerQuestion(
    question: string,
    conversationId: string | null,
    callbacks: Pick<AnswerRequest, "onEvidence" | "onContent">,
  ): Promise<AnswerResult> {
    this.abortAnswerRequests();
    const controller = new AbortController();
    this.answerController = controller;
    const requestedAt = new Date();
    const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    try {
      const result = await executeAnswer(this, {
        question, requestedAt, timeZone,
        history: this.getConversationTurns(conversationId),
        dailyRecapDate: this.dailyRecapRequest(question, requestedAt, timeZone)?.date,
        signal: controller.signal,
        ...callbacks,
      });
      controller.signal.throwIfAborted();
      if (result.status === "complete") {
        await this.saveCompletedTurn(conversationId, result.turn, controller.signal);
      }
      controller.signal.throwIfAborted();
      return result;
    } finally {
      if (this.answerController === controller) this.answerController = undefined;
    }
  }

  async saveCompletedTurn(
    conversationId: string | null,
    turn: Omit<CompletedTurn, "completedAt" | "status">,
    signal?: AbortSignal,
  ): Promise<void> {
    this.requireRunning();
    const completed: CompletedTurn = {
      ...turn,
      completedAt: Date.now(),
      status: "complete",
    };
    const id = globalThis.crypto.randomUUID();
    await this.updateData(() => {
      signal?.throwIfAborted();
      this.requireRunning();
      return {
        conversationState: completeTurn(this.conversationState, conversationId, completed, id),
      };
    });
  }

  async startNewConversation(): Promise<void> {
    this.requireRunning();
    this.abortAnswerRequests();
    await this.updateData(() => ({
      conversationState: newConversation(this.conversationState),
    }));
  }

  async selectConversation(id: string): Promise<void> {
    this.requireRunning();
    this.abortAnswerRequests();
    await this.updateData(() => ({
      conversationState: selectConversation(this.conversationState, id),
    }));
  }

  async deleteConversation(id: string): Promise<void> {
    this.requireRunning();
    if (this.conversationState.selectedConversationId === id) this.abortAnswerRequests();
    await this.updateData(() => ({
      conversationState: deleteConversation(this.conversationState, id),
    }));
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

  abortAnswerRequests(): void {
    this.answerController?.abort(new OllamaError("canceled", "/api/chat"));
    this.chatOllama.abortAll();
    this.queryOllama.abortAll();
  }

  async retrieve(
    question: string,
    now = new Date(),
    timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone,
  ): Promise<RetrievedEvidence[]> {
    if (this.isStopped() || !this.index) return [];
    const request = this.dailyRecapRequest(question, now, timeZone);
    if (!request) return this.index.retrieve(question);
    if (!request.targetPath) return [];

    const dailyFile = this.app.vault.getMarkdownFiles()
      .find(({ path }) => path === request.targetPath);
    if (!dailyFile) return [];
    const directPaths = (this.app.metadataCache.getFileCache(dailyFile)?.links ?? [])
      .map(({ link }) => this.app.metadataCache
        .getFirstLinkpathDest(link, dailyFile.path)?.path)
      .filter((path): path is string => path !== undefined);
    return this.index.retrievePaths([dailyFile.path, ...directPaths]);
  }

  async chat(
    messages: OllamaMessage[],
    onContent: (content: string) => void,
  ): Promise<OllamaChatResult> {
    this.requireRunning();
    const settings = this.getSettings();
    if (!settings.chatModel) {
      throw new OllamaError("chat_model_unavailable", "/api/chat");
    }
    return await this.chatOllama.chat(
      settings.ollamaPort,
      settings.chatModel,
      messages,
      onContent,
    );
  }

  async validateChatModel(): Promise<void> {
    this.requireRunning();
    const settings = this.getSettings();
    if (!settings.chatModel) {
      throw new OllamaError("chat_model_unavailable", "/api/show");
    }
    const validation = await this.chatOllama.validateModel(
      settings.ollamaPort,
      settings.chatModel,
      "completion",
    );
    if (validation !== "compatible") {
      throw new OllamaError(
        validation === "remote"
          ? "remote_model_disallowed"
          : "chat_model_unavailable",
        "/api/show",
      );
    }
  }

  async resolveEvidence(evidence: RetrievedEvidence): Promise<RetrievedEvidence | null> {
    if (this.isStopped()) return null;
    const current = await (this.index?.resolveEvidence(evidence) ?? Promise.resolve(null));
    return current && this.evidenceFile(current) ? current : null;
  }

  async revalidateEvidence(evidence: RetrievedEvidence[]): Promise<RetrievedEvidence[]> {
    if (this.isStopped()) return [];
    const current = await this.index?.resolveEvidenceBatch(evidence) ?? [];
    return this.isStopped() ? [] : current.filter((item) => this.evidenceFile(item) !== null);
  }

  async openSource(evidence: RetrievedEvidence): Promise<boolean> {
    const current = await this.resolveEvidence(evidence);
    if (!current) return false;
    const file = this.evidenceFile(current);
    if (!file) return false;
    await this.app.workspace.getLeaf(false).openFile(
      file,
      current.format === "markdown" && current.startLine !== undefined
        ? { eState: { line: current.startLine - 1 } }
        : undefined,
    );
    return true;
  }

  private evidenceFile(evidence: RetrievedEvidence): TFile | null {
    const file = this.app.vault.getFileByPath(evidence.path);
    const extension = evidence.format === "markdown" ? "md" : "canvas";
    return file instanceof TFile && file.extension.toLowerCase() === extension ? file : null;
  }

  private updateData(update: () => {
    llmvaultSettings?: LLMvaultSettings;
    conversationState?: ConversationState;
    deletionPending?: boolean;
    stopped?: boolean;
  }): Promise<void> {
    const operation = this.dataWrite.then(async () => {
      const proposed = {
        llmvaultSettings: this.llmvaultSettings,
        conversationState: this.conversationState,
        deletionPending: this.isDeletionIncomplete(),
        stopped: this.isStopped(),
        ...update(),
      };
      await this.saveData({
        ...proposed.llmvaultSettings,
        ...proposed.conversationState,
        vaultChatDeletionPending: proposed.deletionPending,
        vaultChatStopped: proposed.stopped,
      });
      if (proposed.llmvaultSettings !== this.llmvaultSettings) {
        // Invalidate work before publishing a durably saved connection.
        this.configurationRevision += 1;
        this.index?.cancel();
        this.indexOllama.abortAll();
        this.abortAnswerRequests();
      }
      this.llmvaultSettings = proposed.llmvaultSettings;
      this.conversationState = proposed.conversationState;
      this.deletionPending = proposed.deletionPending;
      this.stopped = proposed.stopped;
    });
    this.dataWrite = operation.then(
      () => undefined,
      () => undefined,
    );
    return operation;
  }

  getIndexSnapshot(): IndexSnapshot {
    return copyIndexSnapshot(this.indexSnapshot);
  }

  subscribeIndex(subscriber: (snapshot: IndexSnapshot) => void): () => void {
    this.indexSubscribers.add(subscriber);
    subscriber(this.getIndexSnapshot());
    return () => this.indexSubscribers.delete(subscriber);
  }

  subscribeMutations(subscriber: (paths: ReadonlySet<string>) => void): () => void {
    this.mutationSubscribers.add(subscriber);
    return () => this.mutationSubscribers.delete(subscriber);
  }

  async startIndexing(
    discovery: OllamaDiscovery,
    embeddingModel: string,
    rebuild = false,
  ): Promise<void> {
    if (this.isStopped()) return;
    const digest = discovery.modelDigests[embeddingModel];
    if (!this.index || !digest) {
      this.reportIndex({ ...this.indexSnapshot, phase: "failed" });
      return;
    }
    this.indexOllama.abortAll();
    this.index.cancel();
    const model = { digest, name: embeddingModel, port: this.getSettings().ollamaPort };
    await (rebuild ? this.index.rebuild(model) : this.index.start(model));
  }

  async rebuildIndex(): Promise<void> {
    await this.restoreIndex(true);
  }

  async resumeVaultChat(
    discovery: OllamaDiscovery,
    embeddingModel: string,
  ): Promise<void> {
    await this.saveSettings({ ...this.getSettings(), embeddingModel }, discovery);
  }

  deleteAllData(): Promise<void> {
    if (this.deleteOperation) return this.deleteOperation;
    this.deletionBarrier = true;
    this.configurationRevision += 1;
    this.index?.cancel();
    this.indexOllama.abortAll();
    this.abortAnswerRequests();
    this.ollama.abortAll();
    const operation = (async () => {
      await this.updateData(() => ({
        conversationState: { conversations: [], selectedConversationId: null },
        deletionPending: true,
        stopped: true,
      }));
      await this.index?.deleteAll();
      await this.updateData(() => ({ deletionPending: false }));
      this.deletionBarrier = false;
    })();
    this.deleteOperation = operation;
    operation.then(
      () => { if (this.deleteOperation === operation) this.deleteOperation = undefined; },
      () => { if (this.deleteOperation === operation) this.deleteOperation = undefined; },
    );
    return operation;
  }

  private vaultSources(): VaultSource[] {
    const configurationRoot = `${this.app.vault.configDir}/`;
    return this.app.vault
      .getFiles()
      .filter((file) => file.path !== this.app.vault.configDir && !file.path.startsWith(configurationRoot))
      .map((file) => {
        const cache = file.extension.toLowerCase() === "md"
          ? this.app.metadataCache.getFileCache(file)
          : null;
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
        return classifyVaultSource(
          file.path,
          file.extension,
          file.stat.size,
          () => this.app.vault.cachedRead(file),
          anchors,
        );
      });
  }

  dailyRecapRequest(question: string, now: Date, timeZone: string) {
    return resolveDailyRecapRequest(
      question,
      now,
      timeZone,
      this.app.vault.getMarkdownFiles().map(({ path }) => path),
    );
  }

  private handleVaultMutation(file: TAbstractFile, oldPath?: string): void {
    if (this.isStopped()) return;
    if (!(file instanceof TFile)) return;
    const paths = new Set(
      [oldPath, file.path].filter(
        (path): path is string => path !== undefined && this.isSupportedPath(path),
      ),
    );
    if (paths.size === 0 || !this.index) return;

    const rebuild = this.index.invalidate(paths);
    this.indexOllama.abortAll();
    this.abortAnswerRequests();
    for (const subscriber of this.mutationSubscribers) subscriber(paths);
    void rebuild;
  }

  private isSupportedPath(path: string): boolean {
    const configurationRoot = `${this.app.vault.configDir}/`;
    const extension = path.slice(path.lastIndexOf(".") + 1);
    return path !== this.app.vault.configDir &&
      !path.startsWith(configurationRoot) &&
      isSupportedVaultExtension(extension);
  }

  private reportIndex(snapshot: IndexSnapshot): void {
    this.indexSnapshot = snapshot;
    for (const subscriber of this.indexSubscribers) subscriber(this.getIndexSnapshot());
  }

  private async restoreIndex(rebuild = false): Promise<void> {
    if (this.isStopped()) return;
    const revision = this.configurationRevision;
    const settings = this.getSettings();
    if (!settings.embeddingModel) return;
    try {
      const discovery = await this.discoverModels(settings.ollamaPort);
      if (revision !== this.configurationRevision || this.isStopped()) return;
      if (discovery.embeddingModels.includes(settings.embeddingModel)) {
        await this.startIndexing(discovery, settings.embeddingModel, rebuild);
      } else {
        this.reportIndex({ ...this.indexSnapshot, phase: "failed" });
      }
    } catch {
      if (revision !== this.configurationRevision || this.isStopped()) return;
      this.reportIndex({ ...this.indexSnapshot, phase: "failed" });
    }
  }

  private async openVaultChat(): Promise<void> {
    let leaf = this.app.workspace.getLeavesOfType(VIEW_TYPE_VAULT_CHAT)[0];
    leaf ??= this.app.workspace.getRightLeaf(false) ?? undefined;
    if (!leaf) return;

    await leaf.setViewState({ active: true, type: VIEW_TYPE_VAULT_CHAT });
    await this.app.workspace.revealLeaf(leaf);
  }

  private requireRunning(): void {
    if (this.isStopped() || this.deleteOperation) throw new Error("vault_chat_stopped");
  }
}

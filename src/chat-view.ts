import { ItemView, type WorkspaceLeaf } from "obsidian";
import type LLMvaultPlugin from "./main";
import type { RetrievedEvidence } from "./indexing";
import type { Conversation } from "./conversations";
import { OllamaError } from "./ollama";
import { answerParts } from "./quality.ts";
import { VaultManagementView, errorMessage } from "./management-view";

export const VIEW_TYPE_VAULT_CHAT = "vault-chat-view";

type DisplayOwner =
  | { kind: "saved"; conversationId: string | null; pendingSelection?: boolean }
  | { kind: "draft"; running: boolean };

export class VaultChatView extends ItemView {
  private readonly management: VaultManagementView;
  private answerEl?: HTMLElement;
  private askButton?: HTMLButtonElement;
  private readonly citationRegistry = new Map<string, RetrievedEvidence>();
  private conversationTitleEl?: HTMLElement;
  private displayOwner: DisplayOwner | null = null;
  private evidenceEl?: HTMLElement;
  private historyEl?: HTMLElement;
  private historyOpen = false;
  private historyStatus = "";
  private indexUnsubscribe?: () => void;
  private managementOpen: boolean;
  private mutationUnsubscribe?: () => void;
  private questionEl?: HTMLTextAreaElement;
  private readonly unavailableCitations = new Set<string>();

  constructor(
    leaf: WorkspaceLeaf,
    private readonly plugin: LLMvaultPlugin,
  ) {
    super(leaf);
    const settings = plugin.getSettings();
    this.managementOpen =
      !settings.chatModel || !settings.embeddingModel || plugin.isStopped();
    this.management = new VaultManagementView(plugin, {
      showChat: () => this.showChat(),
      showManagement: () => this.showManagement(),
      resetChat: () => {
        this.managementOpen = true;
        this.resetChat();
      },
    });
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
    this.claimDisplay(null);
    let previous = this.plugin.getIndexSnapshot();
    this.renderShell();
    this.indexUnsubscribe = this.plugin.subscribeIndex((snapshot) => {
      const evidenceChanged = snapshot.generationId !== previous.generationId ||
        snapshot.available !== previous.available;
      previous = snapshot;
      this.management.renderIndex();
      this.renderComposerState(snapshot);
      if (this.plugin.isStopped()) {
        this.management.onStopped();
        this.resetChat();
        return;
      }
      if (evidenceChanged && !this.answering && !this.managementOpen) {
        void this.renderSelectedConversation();
      }
    });
    this.mutationUnsubscribe = this.plugin.subscribeMutations((paths) => {
      this.vaultContentChanged(paths);
    });
    if (!this.plugin.isStopped()) void this.management.refreshModels();
    return Promise.resolve();
  }

  override onClose(): Promise<void> {
    this.management.cancelSetup();
    this.claimDisplay(null);
    this.indexUnsubscribe?.();
    this.mutationUnsubscribe?.();
    this.contentEl.empty();
    return Promise.resolve();
  }

  private renderShell(): void {
    const root = this.contentEl;
    root.empty();
    root.addClass("llmvault-chat");

    const header = root.createEl("header", { cls: "llmvault-chat__header" });
    const history = header.createEl("button", {
      attr: {
        "aria-expanded": "false",
        "aria-label": "Open Vault Chat menu",
        title: "Open Vault Chat menu",
        type: "button",
      },
      text: "☰",
    });
    history.onclick = () => {
      this.historyOpen = !this.historyOpen;
      this.renderHistory();
    };
    this.conversationTitleEl = header.createEl("h2", { text: "Vault Chat" });
    header.createEl("span", {
      cls: "llmvault-chat__local-badge",
      text: "Local processing",
    });
    const newConversationButton = header.createEl("button", {
      attr: { "aria-label": "New conversation", title: "New conversation", type: "button" },
      text: "+",
    });
    newConversationButton.onclick = () => void this.startNewConversation();

    this.historyEl = root.createEl("section", { cls: "llmvault-chat__history" });
    this.renderHistory();

    this.management.mount(root);

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
    this.evidenceEl = evidence;

    const answer = root.createEl("section", {
      cls: "llmvault-chat__answer",
      attr: {
        "aria-labelledby": "llmvault-answer-heading",
        "aria-live": "polite",
      },
    });
    answer.createEl("h3", {
      attr: { id: "llmvault-answer-heading" },
      text: "Grounded Answer · quality not evaluated for this model",
    });
    answer.createEl("p", {
      cls: "llmvault-chat__placeholder",
      text: "Your grounded answer will appear here.",
    });
    this.answerEl = answer;

    const composer = root.createEl("form", {
      cls: "llmvault-chat__composer",
    });
    composer.createEl("label", {
      cls: "llmvault-chat__sr-only",
      attr: { for: "llmvault-question" },
      text: "Ask a question about your vault",
    });
    composer.onsubmit = (event) => {
      event.preventDefault();
      if (this.answering) this.stopAnswer();
      else void this.askQuestion();
    };
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
    this.renderComposerState();
    this.renderLayout();
    void this.renderSelectedConversation();
  }

  private renderHistory(): void {
    const history = this.historyEl;
    if (!history) return;
    this.contentEl
      .querySelector('button[aria-label="Open Vault Chat menu"]')
      ?.setAttribute("aria-expanded", String(this.historyOpen));
    history.hidden = !this.historyOpen;
    history.empty();
    const navigation = history.createDiv({ cls: "llmvault-chat__navigation" });
    const chat = navigation.createEl("button", {
      attr: { type: "button" },
      cls: this.managementOpen ? "" : "is-active",
      text: "Chat",
    });
    const settings = this.plugin.getSettings();
    chat.disabled =
      !settings.chatModel || !settings.embeddingModel || this.plugin.isStopped();
    chat.onclick = () => this.showChat();
    const management = navigation.createEl("button", {
      attr: { type: "button" },
      cls: this.managementOpen ? "is-active" : "",
      text: "Vault index & Local Models",
    });
    management.onclick = () => this.showManagement();
    history.createEl("h3", { text: "Local conversations" });
    if (this.historyStatus) {
      history.createEl("p", {
        attr: { "aria-live": "polite", role: "status" },
        text: this.historyStatus,
      });
    }
    const state = this.plugin.getConversationState();
    if (state.conversations.length === 0) {
      history.createEl("p", {
        cls: "llmvault-chat__placeholder",
        text: "No saved conversations yet.",
      });
      return;
    }
    const list = history.createEl("ul");
    for (const conversation of state.conversations) {
      const item = list.createEl("li", { cls: "llmvault-chat__history-item" });
      const select = item.createEl("button", {
        attr: { type: "button" },
        cls: conversation.id === state.selectedConversationId ? "is-active" : "",
      });
      select.createEl("strong", { text: conversation.title });
      select.createSpan({ text: this.conversationExcerpt(conversation) });
      select.onclick = () => void this.resumeConversation(conversation.id);
      const remove = item.createEl("button", {
        attr: { "aria-label": `Delete conversation ${conversation.title}`, type: "button" },
        text: "Delete",
      });
      remove.onclick = () => void this.removeConversation(conversation.id);
    }
  }

  private conversationExcerpt(conversation: Conversation): string {
    const answer = conversation.turns.at(-1)?.answer ?? "";
    const excerpt = Array.from(answer.trim()).slice(0, 120).join("");
    return excerpt.length < answer.trim().length ? `${excerpt}…` : excerpt;
  }

  private renderLayout(): void {
    this.contentEl.classList.toggle("llmvault-chat--management", this.managementOpen);
    this.management.setVisible(this.managementOpen);
    if (this.evidenceEl) this.evidenceEl.hidden = this.managementOpen;
    if (this.answerEl) this.answerEl.hidden = this.managementOpen;
    const composer = this.questionEl?.form;
    if (composer) composer.hidden = this.managementOpen;
    if (this.managementOpen) {
      this.conversationTitleEl?.setText("Vault index & Local Models");
    }
  }

  private showChat(): void {
    this.revealChat();
    void this.renderSelectedConversation();
  }

  private revealChat(): void {
    this.managementOpen = false;
    this.historyOpen = false;
    this.renderHistory();
    this.renderLayout();
    const state = this.plugin.getConversationState();
    this.conversationTitleEl?.setText(
      state.conversations.find(({ id }) => id === state.selectedConversationId)?.title ?? "Vault Chat",
    );
  }

  private showManagement(): void {
    this.cancelAnswer();
    this.managementOpen = true;
    this.historyOpen = false;
    this.renderHistory();
    this.renderLayout();
  }

  private cancelAnswer(): void {
    if (this.answering) this.renderIncomplete();
    this.claimDisplay(this.displayOwner?.kind === "draft" ? { kind: "draft", running: false } : null);
  }

  private get answering(): boolean {
    return this.displayOwner?.kind === "draft" && this.displayOwner.running;
  }

  // Identity is shared by restoration, streaming callbacks, and navigation.
  // Replacing the owner invalidates all old display work at this one boundary.
  private claimDisplay(owner: DisplayOwner | null): void {
    if (this.answering) this.plugin.abortAnswerRequests();
    this.displayOwner = owner;
    this.renderComposerState();
  }

  private resetChat(): void {
    this.claimDisplay(null);
    this.historyOpen = false;
    this.historyStatus = "";
    this.clearConversation();
    this.renderHistory();
    this.renderLayout();
  }

  private async startNewConversation(): Promise<void> {
    const owner: DisplayOwner = { kind: "saved", conversationId: null, pendingSelection: true };
    this.claimDisplay(owner);
    this.revealChat();
    try {
      await this.plugin.startNewConversation();
      if (this.displayOwner !== owner) return;
      owner.pendingSelection = false;
      this.historyStatus = "";
      this.clearConversation();
    } catch {
      if (this.displayOwner !== owner) return;
      owner.pendingSelection = false;
      this.historyStatus = "A new conversation could not be saved. The prior conversation remains selected.";
    }
    this.renderHistory();
  }

  private async resumeConversation(id: string): Promise<void> {
    const owner: DisplayOwner = { kind: "saved", conversationId: id, pendingSelection: true };
    this.claimDisplay(owner);
    this.revealChat();
    try {
      await this.plugin.selectConversation(id);
      if (this.displayOwner !== owner) return;
      owner.pendingSelection = false;
      this.historyStatus = "";
      await this.renderSelectedConversation();
    } catch {
      if (this.displayOwner !== owner) return;
      owner.pendingSelection = false;
      this.historyStatus = "The selected conversation could not be saved. Retry opening it.";
    }
    this.renderHistory();
  }

  private async removeConversation(id: string): Promise<void> {
    const selected = this.plugin.getConversationState().selectedConversationId === id;
    const owner: DisplayOwner | null = selected ? { kind: "saved", conversationId: null, pendingSelection: true } : null;
    if (owner) this.claimDisplay(owner);
    try {
      await this.plugin.deleteConversation(id);
      this.historyStatus = "Conversation deleted locally.";
      if (owner && this.displayOwner === owner) {
        owner.pendingSelection = false;
        await this.renderSelectedConversation();
      }
    } catch {
      if (owner) owner.pendingSelection = false;
      this.historyStatus = "Conversation deletion failed. It remains in local plugin data; retry deletion.";
    }
    this.renderHistory();
  }

  private clearConversation(): void {
    this.conversationTitleEl?.setText("Vault Chat");
    this.renderEvidence([]);
    this.renderAnswer("Your grounded answer will appear here.", "Grounded Answer · quality not evaluated for this model");
    if (this.questionEl) this.questionEl.value = "";
  }

  private async renderSelectedConversation(): Promise<void> {
    if (this.managementOpen || this.displayOwner?.kind === "draft" || this.displayOwner?.pendingSelection) return;
    const state = this.plugin.getConversationState();
    const owner: DisplayOwner = { kind: "saved", conversationId: state.selectedConversationId };
    this.claimDisplay(owner);
    const conversation = state.conversations.find(({ id }) => id === state.selectedConversationId);
    if (!conversation) {
      this.clearConversation();
      return;
    }
    this.conversationTitleEl?.setText(conversation.title);
    const turn = conversation.turns.at(-1);
    if (!turn) {
      this.clearConversation();
      this.conversationTitleEl?.setText(conversation.title);
      return;
    }
    this.renderEvidence([]);
    this.renderAnswer("Checking historical evidence…", "Restoring conversation", turn.question);
    const current = new Map(
      (await this.plugin.revalidateEvidence(turn.evidence)).map((item) => [item.citationId, item]),
    );
    if (
      owner !== this.displayOwner ||
      this.plugin.getConversationState().selectedConversationId !== conversation.id
    ) return;
    const unavailable = new Set(
      turn.evidence.filter((item) => !current.has(item.citationId)).map((item) => item.citationId),
    );
    this.renderEvidence(turn.evidence.map((item) => current.get(item.citationId) ?? item), unavailable);
    this.renderAnswer(
      turn.answer,
      turn.kind === "answer"
        ? "Grounded Answer · quality not evaluated for this model"
        : "Not enough evidence in Vault Content",
      turn.question,
    );
  }

  private async askQuestion(): Promise<void> {
    const question = this.questionEl?.value.trim() ?? "";
    if (!question) return;
    const conversationId = this.plugin.getConversationState().selectedConversationId;
    const owner: DisplayOwner = { kind: "draft", running: true };
    this.claimDisplay(owner);
    this.renderEvidence([]);
    this.renderAnswer("Finding current evidence…", "Answering…", question);

    try {
      const result = await this.plugin.answerQuestion(question, conversationId, {
        onEvidence: (evidence) => {
          if (owner === this.displayOwner) this.renderEvidence(evidence);
        },
        onContent: (content) => {
          if (owner === this.displayOwner) this.renderAnswer(content, "Answering…", question);
        },
      });
      if (owner !== this.displayOwner) return;
      if (result.status === "evidence_changed") {
        this.renderAnswer(
          "A selected source changed before chat started. The index is rebuilding; retry the question when it is ready.",
          "Evidence changed",
          question,
        );
        return;
      }
      if (result.turn.kind === "insufficient") {
        this.renderInsufficient(result.turn.answer);
      } else {
        this.renderAnswer(result.turn.answer, undefined, question);
        if (this.questionEl) this.questionEl.value = "";
      }
      this.renderHistory();
      const state = this.plugin.getConversationState();
      const selected = state.conversations.find(
        ({ id }) => id === state.selectedConversationId,
      );
      if (selected) this.conversationTitleEl?.setText(selected.title);
      owner.running = false;
      this.claimDisplay({ kind: "saved", conversationId: state.selectedConversationId });
    } catch (error) {
      if (owner !== this.displayOwner) return;
      if (error instanceof OllamaError && error.code === "canceled") {
        this.renderIncomplete();
      } else {
        this.renderAnswer(
          errorMessage(error),
          "Vault Chat could not complete this answer",
          question,
        );
      }
    } finally {
      if (owner === this.displayOwner) {
        owner.running = false;
        this.renderComposerState();
      }
    }
  }

  private stopAnswer(): void {
    this.cancelAnswer();
    this.askButton?.focus();
  }

  private renderComposerState(snapshot = this.plugin.getIndexSnapshot()): void {
    const ready = snapshot.available;
    if (this.questionEl) this.questionEl.disabled = this.answering || !ready;
    if (this.askButton) {
      this.askButton.disabled = !this.answering && !ready;
      this.askButton.setText(this.answering ? "Stop" : "Ask");
      if (this.answering) this.askButton.focus();
    }
  }

  private vaultContentChanged(paths: ReadonlySet<string>): void {
    if (this.answering) {
      this.claimDisplay({ kind: "draft", running: false });
      this.renderAnswer(
        "Vault Content changed while this answer was running. Its evidence is no longer current; retry when indexing is ready.",
        "Evidence changed",
      );
    }
    for (const [citationId, evidence] of this.citationRegistry) {
      if (!paths.has(evidence.path)) continue;
      this.unavailableCitations.add(citationId);
      for (const element of this.contentEl.querySelectorAll<HTMLElement>("[data-citation-id]")) {
        if (element.dataset.citationId !== citationId) continue;
        element.setAttribute("aria-disabled", "true");
        if (element instanceof HTMLButtonElement) element.disabled = true;
        if (element.classList.contains("llmvault-chat__citation")) {
          element.setText(`${citationId} unavailable`);
        } else if (!element.querySelector(".llmvault-chat__unavailable")) {
          element.createSpan({
            cls: "llmvault-chat__unavailable",
            text: "Unavailable — source changed.",
          });
        }
      }
    }
    if (!this.managementOpen && this.displayOwner?.kind === "saved") {
      const state = this.plugin.getConversationState();
      const turn = state.conversations.find(({ id }) => id === state.selectedConversationId)?.turns.at(-1);
      if (turn?.evidence.some(({ path }) => paths.has(path))) void this.renderSelectedConversation();
    }
    this.renderComposerState();
  }

  private renderAnswer(
    text: string,
    heading = "Grounded Answer",
    question?: string,
  ): void {
    const answer = this.answerEl;
    if (!answer) return;
    answer.empty();
    answer.createEl("h3", {
      attr: { id: "llmvault-answer-heading" },
      text: heading,
    });
    if (question) {
      answer.createEl("p", {
        cls: "llmvault-chat__question",
        text: `Question: ${question}`,
      });
    }
    const body = answer.createEl("p", { cls: "llmvault-chat__answer-text" });
    if (!text) return;

    for (const part of answerParts(
      text,
      new Set(this.citationRegistry.keys()),
      this.unavailableCitations,
    )) {
      if (part.kind === "text") {
        body.createSpan({ text: part.text });
      } else if (part.kind === "unavailable") {
        body.createSpan({
          cls: "llmvault-chat__unavailable",
          text: `${part.citationId} unavailable`,
        });
      } else {
        const citation = body.createEl("button", {
          cls: "llmvault-chat__citation",
          attr: { "data-citation-id": part.citationId, type: "button" },
          text: part.citationId,
        });
        citation.onclick = () => void this.showEvidence(part.citationId);
      }
    }
  }

  private renderEvidence(
    evidence: RetrievedEvidence[],
    unavailable = new Set<string>(),
  ): void {
    const evidenceEl = this.evidenceEl;
    if (!evidenceEl) return;
    this.citationRegistry.clear();
    this.unavailableCitations.clear();
    for (const citationId of unavailable) this.unavailableCitations.add(citationId);
    evidenceEl.empty();
    evidenceEl.scrollTop = 0;
    evidenceEl.createEl("h3", { text: `Evidence used · ${evidence.length}` });
    if (evidence.length === 0) {
      evidenceEl.createEl("p", {
        cls: "llmvault-chat__placeholder",
        text: "No current indexed source could support this question.",
      });
      return;
    }
    for (const item of evidence) {
      const isUnavailable = unavailable.has(item.citationId);
      if (!isUnavailable) this.citationRegistry.set(item.citationId, item);
      const card = isUnavailable
        ? evidenceEl.createDiv({ cls: "llmvault-chat__source" })
        : evidenceEl.createEl("button", {
            cls: "llmvault-chat__source",
            attr: { "data-citation-id": item.citationId, type: "button" },
          });
      card.createEl("strong", { text: `${item.citationId} · ${item.path}` });
      card.createSpan({ text: this.evidenceLocation(item) });
      if (isUnavailable) {
        card.createSpan({
          cls: "llmvault-chat__unavailable",
          text: "Unavailable — source changed or is missing.",
        });
      } else {
        card.onclick = () => void this.showEvidence(item.citationId);
      }
    }
  }

  private async showEvidence(citationId: string): Promise<void> {
    const evidence = this.citationRegistry.get(citationId);
    const evidenceEl = this.evidenceEl;
    if (!evidence || !evidenceEl) return;
    evidenceEl.querySelector(".llmvault-chat__preview")?.remove();
    const preview = evidenceEl.createDiv({ cls: "llmvault-chat__preview" });
    const current = await this.plugin.resolveEvidence(evidence);
    if (!current) {
      preview.setText("This source changed or is unavailable. Rebuild the index and retry.");
      return;
    }
    preview.createEl("strong", {
      text: `${current.path} · ${this.evidenceLocation(current).toLowerCase()}`,
    });
    preview.createEl("pre", { text: current.text });
    const open = preview.createEl("button", {
      cls: "mod-cta",
      attr: { type: "button" },
      text: "Open source",
    });
    open.onclick = async () => {
      if (!(await this.plugin.openSource(current))) {
        preview.setText("This source changed or is unavailable. Rebuild the index and retry.");
      }
    };
  }

  private evidenceLocation(evidence: RetrievedEvidence): string {
    return evidence.format === "canvas"
      ? `Card “${evidence.excerpt ?? ""}”`
      : `Lines ${String(evidence.startLine)}–${String(evidence.endLine)}`;
  }

  private renderInsufficient(message: string): void {
    this.renderAnswer(
      message,
      "Not enough evidence in Vault Content",
      this.questionEl?.value.trim() || undefined,
    );
    const revise = this.answerEl?.createEl("button", {
      attr: { type: "button" },
      text: "Revise question",
    });
    revise?.addEventListener("click", () => {
      this.questionEl?.focus();
      this.questionEl?.select();
    });
  }

  private renderIncomplete(): void {
    this.answerEl?.createEl("p", {
      cls: "llmvault-chat__incomplete",
      text: "Incomplete — stopped. This answer was not saved as complete.",
    });
  }

}

import { ItemView, Plugin } from "obsidian";

const VIEW_TYPE_VAULT_CHAT = "vault-chat-view";

class VaultChatView extends ItemView {
  getViewType(): string {
    return VIEW_TYPE_VAULT_CHAT;
  }

  getDisplayText(): string {
    return "Vault Chat";
  }

  override getIcon(): string {
    return "message-circle";
  }

  override async onOpen(): Promise<void> {
    const root = this.contentEl;
    root.empty();
    root.addClass("llmvault-chat");

    const header = root.createEl("header", { cls: "llmvault-chat__header" });
    header.createEl("h2", { text: "Vault Chat" });
    header.createEl("span", {
      cls: "llmvault-chat__local-badge",
      text: "Local processing",
    });

    const setup = root.createDiv({ cls: "llmvault-chat__setup" });
    setup.id = "llmvault-setup-status";
    setup.setAttrs({ "aria-live": "polite", role: "status" });
    setup.createEl("h3", { text: "Finish setup to ask your vault" });
    setup.createEl("p", {
      text: "Connect a Local Model to begin a grounded Vault Chat.",
    });

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
    composer.createEl("textarea", {
      attr: {
        "aria-describedby": setup.id,
        disabled: "",
        id: "llmvault-question",
        placeholder: "Ask about your vault…",
        rows: "3",
      },
    });
    composer.createEl("button", {
      attr: { disabled: "", type: "submit" },
      text: "Ask",
    });
  }
}

export default class LLMvaultPlugin extends Plugin {
  override onload(): void {
    this.registerView(
      VIEW_TYPE_VAULT_CHAT,
      (leaf) => new VaultChatView(leaf),
    );

    this.addCommand({
      id: "open-vault-chat",
      name: "Open Vault Chat",
      callback: () => void this.openVaultChat(),
    });

    this.addRibbonIcon("message-circle", "Open Vault Chat", () => {
      void this.openVaultChat();
    });
  }

  override onunload(): void {
    void this.app.workspace.detachLeavesOfType(VIEW_TYPE_VAULT_CHAT);
  }

  private async openVaultChat(): Promise<void> {
    let leaf = this.app.workspace.getLeavesOfType(VIEW_TYPE_VAULT_CHAT)[0];
    leaf ??= this.app.workspace.getRightLeaf(false) ?? undefined;
    if (!leaf) return;

    await leaf.setViewState({ active: true, type: VIEW_TYPE_VAULT_CHAT });
    await this.app.workspace.revealLeaf(leaf);
  }
}

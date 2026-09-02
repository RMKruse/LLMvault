# LLMvault

Use Local Models to ask questions grounded in your Obsidian vault.

LLMvault sends questions, retrieved source text, chat history, and embedding inputs only to your user-managed Ollama service at `127.0.0.1:<port>`. It sends no telemetry and contacts no remote service.

Indexes live under the plugin's `index-v1` directory, while conversations and retained connection settings live in the plugin's `data.json`. Vault or configuration-folder sync and backup software may copy this local derived data. User-created vault symlinks may point outside the vault even though LLMvault uses only Obsidian's vault-relative file identities.

Local Processing is a plugin-scoped claim: Obsidian, other plugins, Ollama, operating-system services, sync, and backup software remain outside LLMvault's control.

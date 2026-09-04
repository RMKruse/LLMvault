# LLMvault

Use Local Models to ask questions grounded in your Obsidian vault. LLMvault is a desktop-only Obsidian plugin for version 1.13.7 or newer.

## Requirements and setup

1. Install and start [Ollama](https://ollama.com/) with at least one chat model and one embedding model.
2. Install `main.js`, `manifest.json`, and `styles.css` in the LLMvault plugin directory, then enable LLMvault in Obsidian.
3. Open **Vault Chat** from the ribbon or command palette.
4. Enter the Ollama port, refresh the installed models, explicitly choose one compatible model for chat and one for embeddings, and complete setup.

After setup, the chat stays focused on conversations. Open the menu and choose **Vault index & Local Models** to inspect or rebuild the index or change the local-model connection.

LLMvault indexes Markdown (`.md`) and the text nodes of Canvas (`.canvas`) files. It reports empty, unsupported, unrecognized, oversized, or failed files instead of silently indexing them. Bases (`.base`) are ignored as non-content. PDF, images, media, office files, archives, OCR, and transcription are not supported.

## Local Processing and network use

LLMvault constructs only `http://127.0.0.1:<port>` and contacts no remote service. During indexing it sends Vault Content chunks and the selected embedding-model identifier to your user-managed Ollama service. When you ask a question, it sends the question for embedding, then sends the question, retrieved source excerpts, relevant local chat history, the selected chat-model identifier, and LLMvault's fixed grounding prompt for generation.

The plugin sends no telemetry, analytics, crash reports, update checks, remote logs, or remote assets. This loopback-only claim covers LLMvault itself; Obsidian, other plugins, Ollama, operating-system services, sync software, and backup software remain outside its control. Ollama and the operating system may retain their own request or diagnostic logs.

## Local data and deletion

Chat history and retained connection settings live in `<vault>/<config-folder>/plugins/llmvault/data.json`; the default Obsidian configuration folder is `.obsidian`. The derived index lives beside it in `index-v1/` and contains source identities, locations, statuses, and embeddings, but no persisted plaintext source copy.

The full conversation archive is retained locally. Model messages use a separate 32 KiB budget, measured as the UTF-8 JSON message array. The system instructions and current question/evidence take priority; remaining space holds whole recent turns, in order, stopping at the first turn that does not fit. If the current request alone exceeds the budget, Vault Chat asks you to shorten or narrow the question. This byte limit bounds request growth; it does not guarantee a fit in every Local Model's token context window.

Vault-level or configuration-folder sync and backup software may copy this plugin data. User-created vault symlinks or junctions may also point to files physically outside the vault even though LLMvault uses only vault-relative identities exposed by Obsidian.

**Delete all Vault Chat data** stops current work and removes plugin-owned indexes, conversations, statuses, caches, temporary records, and in-memory content. It keeps the Ollama port and model selections so the connection can be reused. It does not delete Vault Content, Ollama models or logs, Obsidian/OS state, sync history, external backups, or storage-device remnants, and it is not a secure-erasure feature.

## Scope

LLMvault reads but never creates, edits, moves, or deletes Vault Content. It does not support Obsidian Mobile, hosted inference, non-loopback or non-Ollama backends, model installation or management, arbitrary file extraction, folder/tag/current-note filters, tools or agent actions, telemetry, or secure-erasure guarantees. Grounding controls reduce risk but do not claim that prompt injection is impossible.

## Reproducible release build

Install Node.js 22 or newer, then run:

```sh
npm ci
npm test
npm run lint
npm run quality:run
npm run acceptance:run -- --vault=/path/to/vault --obsidian=/path/to/Obsidian --chat=CHAT_MODEL --embedding=EMBEDDING_MODEL
npm run release:build
```

The acceptance command runs against a disposable copy of the supplied vault. The last command performs two production builds, requires byte-identical `main.js` output, verifies the package, manifest, and minimum-version metadata, and writes exactly `main.js`, `manifest.json`, and `styles.css` to `release-candidate/<version>/`. It does not create a tag, publish a release, submit to the Community directory, or contact a service. After all release gates pass, the Git tag must exactly match the manifest version (for example `0.1.0`, not `v0.1.0`). Current qualification evidence and pending platform gates are recorded in [RELEASE_CANDIDATE.md](RELEASE_CANDIDATE.md).

The released code is licensed under the [MIT License](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md).

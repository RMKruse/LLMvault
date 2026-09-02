# Synthetic quality-suite fixture

`npm run quality:run` checks the committed fixture manifest, builds the production plugin, and runs all 24 cases three times as fresh conversations through production extraction, retrieval, prompting, streaming, and citation parsing. Human review is still required for semantic claim support, claim-to-citation association, and UI behavior.

The runner tries `gemma4:12b-mlx` with `qwen3-embedding:0.6b` first and tries `qwen3-embedding:4b` only after a failed candidate. It never weakens a gate. The required models and Ollama must already be installed and running on `127.0.0.1:11434`.

The plugin never runs this suite, selects these models, or recommends them. Users choose any compatible installed models based on their own hardware.

## Reference Vault acceptance

Run the production-build acceptance harness on the release Reference Vault from the same machine used for the release claim:

```sh
npm run acceptance:run -- \
  --vault=/absolute/path/to/reference-vault \
  --obsidian=/Applications/Obsidian.app/Contents/MacOS/Obsidian \
  --chat=gemma4:12b-mlx \
  --embedding=qwen3-embedding:0.6b
```

The command copies the source into a disposable vault and launches the production plugin in a separate Obsidian profile. It performs one cold/warm run through real Obsidian vault events, DOM controls, indexing, prompt, streaming, and cancellation paths. It also exercises mutation isolation, one-file and fatal failures, and sends `SIGKILL` to only that disposable Obsidian process after record, catalog, pointer, and cleanup writes. Ollama time is subtracted from plugin-controlled time. The source vault and the user's running Obsidian session are untouched; the disposable vault is removed when the run ends. Pass `--keep-temp` only when the isolated copy is needed for diagnosis.

The harness reads the Obsidian version from the supplied application bundle and records it with OS, hardware, plugin, Ollama, and exact model digests. The Reference Vault shape gate accepts the documented approximate workload with 20% tolerance. A report passes only when all gates pass in the production run. Progress is saved to `evaluation/results/acceptance.json`, which is ignored by Git. Recheck a retained report with `npm run acceptance:check -- path/to/report.json`.

Regenerate `reference-vault-v1/manifest.json` only after an intentional fixture edit:

```sh
node evaluation/reference-vault-v1/build-manifest.mjs
```

Run `npm run quality:calibrate` after an embedding digest or retrieval constant changes, update the digest-scoped cutoff, then rerun the full suite. Passing evidence is written to `evaluation/results/evaluated-configuration.json`.

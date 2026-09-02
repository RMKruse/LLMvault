# Reference Evaluation Vault

`npm run quality:run` checks the committed fixture manifest, builds the production plugin, and runs all 24 cases three times as fresh conversations through the production extraction, retrieval, prompt, streaming, citation, and rendering logic.

The runner tries `gemma4:12b-mlx` with `qwen3-embedding:0.6b` first and tries `qwen3-embedding:4b` only after a failed candidate. It never weakens a gate. The required models and Ollama must already be installed and running on `127.0.0.1:11434`.

Regenerate `reference-vault-v1/manifest.json` only after an intentional fixture edit:

```sh
node evaluation/reference-vault-v1/build-manifest.mjs
```

Run `npm run quality:calibrate` after an embedding digest or retrieval constant changes, update the digest-scoped cutoff, then rerun the full suite. Passing evidence is written to `evaluation/results/evaluated-configuration.json`.

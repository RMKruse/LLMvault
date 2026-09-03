# Synthetic quality-suite fixture

`npm run quality:run` checks the committed fixture manifest, builds the production plugin, and runs all 24 cases three times as fresh conversations through production extraction, retrieval, prompting, streaming, and citation parsing. Human review is still required for semantic claim support, claim-to-citation association, and UI behavior.

The prototype proof pins `gemma4:12b-mlx` with `qwen3-embedding:4b`, including their exact digests. `--chat` and `--embedding` are checked; a mismatch fails without trying another model. The required models and Ollama must already be installed and running on `127.0.0.1:11434`.

The plugin never runs this suite, selects these models, or recommends them. Users choose any compatible installed models based on their own hardware.

## Reference Vault acceptance

Run the production-build acceptance harness on the release Reference Vault from the same machine used for the release claim:

```sh
npm run acceptance:run -- \
  --vault=/absolute/path/to/reference-vault \
  --obsidian=/Applications/Obsidian.app/Contents/MacOS/Obsidian \
  --chat=gemma4:12b-mlx \
  --embedding=qwen3-embedding:4b
```

The command copies the source into a disposable vault and launches the production plugin in a separate Obsidian profile. It performs one cold/warm run through real Obsidian vault events, DOM controls, indexing, prompt, streaming, and cancellation paths. It also exercises mutation isolation, one-file and fatal failures, and sends `SIGKILL` to only that disposable Obsidian process after record, catalog, pointer, and cleanup writes. Ollama time is subtracted from plugin-controlled time. The source vault and the user's running Obsidian session are untouched; the disposable vault is removed when the run ends. Pass `--keep-temp` only when the isolated copy is needed for diagnosis.

The harness reads the Obsidian version from the supplied application bundle and records it with OS, hardware, plugin, Ollama, and exact model digests. The Reference Vault shape gate accepts the documented approximate workload with 20% tolerance. A report passes only when all gates pass in the production run. Progress is saved to `evaluation/results/prototype-acceptance.json`, which is ignored by Git. Recheck a retained report with `npm run acceptance:check -- path/to/report.json`.

Regenerate `reference-vault-v1/manifest.json` only after an intentional fixture edit:

```sh
node evaluation/reference-vault-v1/build-manifest.mjs
```

Run `npm run quality:calibrate` after an embedding digest or retrieval constant changes, update the digest-scoped cutoff, then rerun the full suite. The current run, passing or failing, is written to `evaluation/results/prototype-quality.json`. The previously checked-in release reports remain historical evidence.


## Prototype release verdict

Run `npm test`, `npm run quality:run`, then the acceptance command above. `npm test` records the successful unit/smoke build in `evaluation/results/prototype-tests.json`. Acceptance combines that receipt, the complete 24×3 quality report, the production Grounded Answer phase, and every existing performance/storage gate into a conjunctive verdict. Reports from different builds, commits, model digests, prompts, signatures or cutoffs cannot qualify together. A nonzero exit is expected while any gate, including human support approval, is missing.

Calibration now explicitly hydrates uncapped candidates. Product semantic retrieval stays at six with the unchanged `0.5710371502360156` cutoff. The nine insufficient-evidence identities are pinned in `reference-vault-v1/insufficient-identities.json`; each run compares them both to cap-four retrieval and to the committed baseline. Update this file only after investigating and explicitly accepting a changed retrieval baseline.

After cold/warm indexing, acceptance runs Daily Recap and Direct Evidence three times each through the production view, then Direct Evidence with one short prior insufficient-evidence turn (an ambiguous Daily Recap naming both today and yesterday). Each fresh repetition reloads the plugin so query-scoped citation IDs start in a fresh namespace. The harness fixes only the Daily Recap clock (`2026-09-03T10:00:00Z`, `Europe/Berlin`) and proof sampling (`temperature: 0`, `seed: 0`). It captures serialized requests, the client's allowlisted diagnostics, registered citations, post-request `/api/ps` context and model digest. The effective context must be 4,096 without a `num_ctx` request, output must stop within 256 tokens and the 180-second watchdog, and fresh repetitions must have identical response/message/retrieval hashes. Daily Recap must use no query embedding and exactly the expected one-layer selection. Direct Evidence must retrieve and cite the explicit local/partial-derivative proposition within six chunks.

The report contains hashes, opaque source identities, ranks, offsets/lines, controls, diagnostics, timings and pass flags. It never contains private questions, answers or source text. The source-vault file/content manifest (including configuration and Git state) must match before and after acceptance. Only the disposable copy is mutated and it is deleted by default.

### Local human support review

Unredacted answers, messages and cited excerpts are saved separately to the Git-ignored, owner-readable `evaluation/results/local-review.json`. This file stays on this computer; do not attach or commit it. Open it locally and verify that every factual claim is supported by its associated registered citation. Review both fresh answers and the short-history control. The automatic substring and registry checks cannot make this semantic judgement.

For each approved entry, copy its `binding` hash and record the human decision:

```sh
npm run acceptance:check -- evaluation/results/prototype-acceptance.json --approve=REVIEW_BINDING_HASH
```

Identical repetitions share an approval. Each binding covers the exact configuration, response, messages, cited locator hashes and selected retrieval identities. A changed binding requires another local review. The command does not generate or infer approvals; without explicit approvals the verdict stays red. Recheck without `--approve` at any time. Old performance-only reports cannot claim the new prototype release verdict. Delete `local-review.json` after review if it is no longer needed.

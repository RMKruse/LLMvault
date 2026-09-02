import assert from "node:assert/strict";
import test from "node:test";

import { evaluateAcceptance } from "../evaluation/acceptance.mjs";

const run = () => ({
  accounting: { complete: true, discovered: 1_200, statuses: { indexed: 1_000, unsupported_format: 200 }, terminal: 1_200 },
  cancellation: {
    controlsUsableMs: 20,
    lateCompleted: false,
    lateOutputInjected: true,
    latePersisted: false,
    noNewTextAfterMs: 10,
  },
  failures: {
    fatalFailedSourceAbsent: true,
    fatalGenerationIsolated: true,
    fatalUnrelatedAvailable: true,
    oneFileFailedSourceAbsent: true,
    oneFileIsolated: true,
    oneFileUnrelatedAvailable: true,
  },
  indexing: { ollamaMs: 70_000, partialActivated: false, pluginMs: 50_000, totalMs: 120_000 },
  mutation: { queryIneligibleMs: 10, replacementAfterEmbeddingMs: 20 },
  question: { embeddingDispatchMs: 2, postEmbeddingMs: 30 },
  recovery: ["record", "catalog", "pointer", "cleanup"].map((boundary) => ({
    boundary,
    canonical: true,
    converged: true,
    exposedInvalidContent: false,
    pluginMs: 50_000,
  })),
  storage: {
    containsPlaintextSourceCopy: false,
    peakRebuildBytes: 900 * 1024 * 1024,
    settledBytes: 500 * 1024 * 1024,
  },
  ui: Object.fromEntries(["indexing", "rebuild", "reconciliation", "streaming"].map((phase) => [phase, {
    maximumMs: 99,
    p95Ms: 95,
    samples: 11,
  }])),
  warm: { ollamaMs: 0, pluginMs: 5_000, totalMs: 5_000 },
});

const report = () => ({
  environment: {
    hardware: { architecture: "arm64", cpu: "Example", memoryBytes: 16_000_000_000 },
    models: {
      chat: { digest: "sha256:chat", name: "chat" },
      embedding: { digest: "sha256:embed", name: "embed" },
    },
    os: { platform: "darwin", release: "25.0.0" },
    vault: { bytes: 110 * 1024 * 1024, contentFiles: 1_000, totalFiles: 1_200 },
    versions: { node: "v22.0.0", obsidian: "1.13.7", ollama: "0.11.0", plugin: "0.1.0" },
  },
  runs: [run()],
});

test("acceptance report passes only when the production run meets every budget", () => {
  const result = evaluateAcceptance(report());

  assert.equal(result.pass, true);
  assert.deepEqual(Object.values(result.gates), Array(Object.keys(result.gates).length).fill(true));
});

test("acceptance report names the failed gate", () => {
  const input = report();
  input.runs[0].question.postEmbeddingMs = 501;

  const result = evaluateAcceptance(input);

  assert.equal(result.pass, false);
  assert.equal(result.gates.question, false);
});

test("terminal accounting must add up", () => {
  const input = report();
  input.runs[0].accounting.statuses.indexed -= 1;

  assert.equal(evaluateAcceptance(input).gates.accounting, false);
});

test("missing numeric evidence cannot pass by coercion", () => {
  const input = report();
  input.runs[0].indexing.pluginMs = null;

  assert.equal(evaluateAcceptance(input).gates.indexing, false);
});

test("a small fixture cannot claim Reference Vault acceptance", () => {
  const input = report();
  input.environment.vault = { bytes: 4_462, contentFiles: 10, totalFiles: 10 };

  assert.equal(evaluateAcceptance(input).gates.referenceVault, false);
});

test("a missing production recording fails closed", () => {
  assert.doesNotThrow(() => evaluateAcceptance({ runs: [{}] }));
  assert.equal(evaluateAcceptance({ runs: [] }).pass, false);
});

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

test("prototype release requires all proof layers and hash-bound human support", async () => {
  const { evaluateGroundedProof, supportBinding } = await import("../evaluation/proof.mjs");
  assert.equal(evaluateGroundedProof({}).pass, false);
  const config = { pluginBuildSha256: "a".repeat(64) };
  const item = { id: "direct", responseSha256: "b".repeat(64), request: { messageSha256: ["c".repeat(64)] },
    citedLocatorSha256: ["d".repeat(64)], retrieval: [{ identity: "e".repeat(64) }] };
  const binding = supportBinding(config, item);
  for (const mutate of [
    (c, i) => { i.responseSha256 = "f".repeat(64); },
    (c, i) => { i.request.messageSha256[0] = "f".repeat(64); },
    (c, i) => { i.citedLocatorSha256[0] = "f".repeat(64); },
    (c) => { c.pluginBuildSha256 = "f".repeat(64); },
  ]) {
    const c = structuredClone(config), i = structuredClone(item); mutate(c, i);
    assert.notEqual(supportBinding(c, i), binding);
  }
});

test("a complete Grounded Answer proof fails closed on missing, changed or unapproved evidence", async () => {
  const { evaluateGroundedProof, supportBinding, PINNED_CHAT, PINNED_EMBEDDING, PINNED_INDEX_SIGNATURE, PROMPT_SHA256 } = await import("../evaluation/proof.mjs");
  const configuration = {
    pluginBuildSha256: "a".repeat(64), pluginCommit: "b".repeat(40), chatModel: PINNED_CHAT,
    embeddingModel: PINNED_EMBEDDING,
    indexSignature: PINNED_INDEX_SIGNATURE,
    promptSha256: PROMPT_SHA256, cutoff: 0.5710371502360156, ollamaVersion: "0.33.2",
  };
  const cases = ["daily", "direct", "direct-history"].flatMap((id) =>
    Array.from({ length: id === "direct-history" ? 1 : 3 }, (_, i) => {
      const messageSha256 = [PROMPT_SHA256, ...(id === "direct-history" ? ["e".repeat(64), "f".repeat(64)] : []), "d".repeat(64)];
      return {
        id, repetition: i + 1, responseSha256: "a".repeat(64), expectedMessageSha256: messageSha256,
        request: { origin: "http://127.0.0.1:11434", path: "/api/chat", method: "POST", model: PINNED_CHAT.name,
          stream: true, think: false, keys: ["messages", "model", "options", "stream", "think"],
          optionKeys: ["num_predict", "seed", "temperature"], options: { num_predict: 256, seed: 0, temperature: 0 },
          roles: ["system", ...(id === "direct-history" ? ["user", "assistant"] : []), "user"], messageSha256 },
        diagnostics: { eval_count: 42, prompt_eval_count: 100, total_duration: 1, done_reason: "stop" },
        contextLength: 4096, effectiveModelDigest: PINNED_CHAT.digest, durationMs: 20,
        answerPass: true, pathPass: true, registryPass: true, locatorPass: true, modelDigestsPass: true, streamedExactly: true,
        retrieval: [{ identity: "b".repeat(64), rank: 1 }], citedLocatorSha256: ["b".repeat(64)],
      };
    }));
  const proof = { configuration, cases, sourceBeforeSha256: "f".repeat(64), sourceAfterSha256: "f".repeat(64),
    requestCount: 7, egressPass: true, approvedBindings: cases.map((item) => supportBinding(configuration, item)) };
  assert.equal(evaluateGroundedProof(proof).pass, true);
  for (const [gate, change] of [
    ["humanSupport", (p) => { p.approvedBindings = []; }],
    ["repeatability", (p) => { p.cases[1].responseSha256 = "c".repeat(64); }],
    ["requests", (p) => { delete p.cases[0].request; }],
    ["effectiveContext", (p) => { p.cases[0].contextLength = 8192; }],
    ["evidence", (p) => { p.cases[0].citedLocatorSha256 = []; }],
    ["sourceUnchanged", (p) => { delete p.sourceAfterSha256; }],
    ["egress", (p) => { p.egressPass = false; }],
  ]) {
    const changed = structuredClone(proof); change(changed);
    assert.equal(evaluateGroundedProof(changed).gates[gate], false, gate);
    assert.equal(evaluateGroundedProof(changed).pass, false, gate);
  }
});

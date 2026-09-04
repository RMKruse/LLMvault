import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, readdir } from "node:fs/promises";
import test from "node:test";

import { VaultIndex } from "../src/indexing.ts";
import {
  answerParts,
  calibrateCutoff,
  GROUNDING_SYSTEM_PROMPT,
} from "../src/quality.ts";
import { REFERENCE_CONFIGURATION } from "../src/retrieval-calibration.ts";

const fixtureRoot = new URL("../evaluation/reference-vault-v1/", import.meta.url);

class MemoryAdapter {
  files = new Map();
  folders = new Set();
  async exists(path) { return this.files.has(path) || this.folders.has(path); }
  async mkdir(path) { this.folders.add(path); }
  async list(path) {
    const prefix = `${path}/`;
    return {
      files: [...this.files.keys()].filter((item) => item.startsWith(prefix) && !item.slice(prefix.length).includes("/")),
      folders: [...this.folders].filter((item) => item.startsWith(prefix) && !item.slice(prefix.length).includes("/")),
    };
  }
  async read(path) {
    if (!this.files.has(path)) throw new Error(`missing ${path}`);
    return this.files.get(path);
  }
  async write(path, value) { this.files.set(path, value); }
  async process(path, update) {
    const value = update(this.files.get(path) ?? "");
    this.files.set(path, value);
    return value;
  }
  async rmdir(path) {
    for (const item of [...this.files.keys()]) if (item === path || item.startsWith(`${path}/`)) this.files.delete(item);
    for (const item of [...this.folders]) if (item === path || item.startsWith(`${path}/`)) this.folders.delete(item);
  }
}

test("calibration keeps every gold unit and is scoped to one embedding digest", () => {
  const cutoff = calibrateCutoff("sha256:embed-a", [
    { gold: ["g1"], ranked: [{ evidence: ["g1"], score: 0.81 }] },
    {
      gold: ["g2", "g3"],
      ranked: [
        { evidence: ["g2"], score: 0.92 },
        { evidence: ["g3"], score: 0.73 },
      ],
    },
  ]);

  assert.deepEqual(cutoff, { embeddingDigest: "sha256:embed-a", minimumScore: 0.73 });
  assert.equal(cutoff.embeddingDigest === "sha256:embed-b" ? cutoff.minimumScore : undefined, undefined);
  assert.throws(
    () => calibrateCutoff("sha256:embed-a", [{ gold: ["missing"], ranked: [] }]),
    /calibration evidence missing/,
  );
});

test("model selection remains user-controlled without an unearned quality claim", async () => {
  const source = await readFile(new URL("../src/management-view.ts", import.meta.url), "utf8");
  assert.doesNotMatch(source, /evaluated configuration|recommended model/i);
  assert.match(source, /Choose a compatible chat model/);
  assert.match(source, /Choose a compatible embedding model/);
});

test("production answer parsing activates only query-registry citations", () => {
  assert.deepEqual(answerParts("Known [S1-1], invented [S9-9].", new Set(["S1-1"])), [
    { kind: "text", text: "Known " },
    { citationId: "S1-1", kind: "citation" },
    { kind: "text", text: ", invented " },
    { kind: "text", text: "[S9-9]" },
    { kind: "text", text: "." },
  ]);
  assert.deepEqual(answerParts("Old [S1-1].", new Set(["S1-1"]), new Set(["S1-1"])), [
    { kind: "text", text: "Old " },
    { citationId: "S1-1", kind: "unavailable" },
    { kind: "text", text: "." },
  ]);
});

test("the production prompt treats false-premise questions as insufficient", () => {
  assert.match(GROUNDING_SYSTEM_PROMPT, /exact \[S1-1\] form/);
  assert.match(GROUNDING_SYSTEM_PROMPT, /never put backticks/);
  assert.match(GROUNDING_SYSTEM_PROMPT, /negated, contradicted, or impossible requested fact is insufficient/);
  assert.match(GROUNDING_SYSTEM_PROMPT, /a bare correction or "none" answer is invalid/);
  assert.match(GROUNDING_SYSTEM_PROMPT, /begin exactly with INSUFFICIENT_EVIDENCE:/);
});

test("retrieval applies a cutoff only to its exact calibrated embedding digest", async () => {
  const cutoff = REFERENCE_CONFIGURATION.minimumScore;
  const sources = [
    { path: "high.md", read: async () => "high" },
    { path: "low.md", read: async () => "low" },
  ];
  const embed = async (inputs) => inputs.map((input) => [
    input === "question" ? 1 : input === "high" ? cutoff + 0.1 : cutoff - 0.1,
    ...Array(REFERENCE_CONFIGURATION.indexSignature.vectorDimension - 1).fill(0),
  ]);
  const exact = new VaultIndex(new MemoryAdapter(), "index", () => sources, embed);
  await exact.start(REFERENCE_CONFIGURATION.embeddingModel);

  assert.deepEqual((await exact.retrieve("question")).map(({ path }) => path), ["high.md"]);

  const other = new VaultIndex(new MemoryAdapter(), "index", () => sources, embed);
  await other.start({ ...REFERENCE_CONFIGURATION.embeddingModel, digest: "sha256:other" });
  assert.deepEqual((await other.retrieve("question")).map(({ path }) => path), ["high.md", "low.md"]);
});

test("synthetic quality-suite fixture is complete and checksummed", async () => {
  const manifest = JSON.parse(await readFile(new URL("manifest.json", fixtureRoot), "utf8"));
  const sourceNames = (await readdir(new URL("sources/", fixtureRoot))).sort();

  assert.equal(manifest.suiteVersion, 1);
  assert.equal(sourceNames.filter((name) => name.endsWith(".md")).length, 8);
  assert.equal(sourceNames.filter((name) => name.endsWith(".canvas")).length, 2);
  assert.equal(manifest.cases.filter(({ category }) => category === "answerable").length, 12);
  assert.equal(manifest.cases.filter(({ category }) => category === "insufficient").length, 6);
  assert.equal(manifest.cases.filter(({ category }) => category === "poisoned").length, 6);
  assert.deepEqual(manifest.calibrationCaseIds, manifest.cases.slice(0, 6).map(({ id }) => id));
  assert.equal(manifest.poisonedVariants.length, 6);

  for (const [path, expected] of Object.entries(manifest.sourceSha256)) {
    const source = await readFile(new URL(`sources/${path}`, fixtureRoot));
    assert.equal(createHash("sha256").update(source).digest("hex"), expected, path);
  }
  for (const item of manifest.cases.flatMap(({ goldEvidence }) => goldEvidence)) {
    const source = await readFile(new URL(`sources/${item.path}`, fixtureRoot), "utf8");
    const text = item.nodeId
      ? JSON.parse(source).nodes.find(({ id }) => id === item.nodeId)?.text
      : source;
    assert.equal(text.slice(item.start, item.end), item.quote, `${item.path}:${item.start}`);
  }
});


test("calibration sees rank six and uncapped candidates without changing product cutoff", async () => {
  const scores = [0.91, 0.85, 0.79, 0.73, 0.67, 0.61, 0.55];
  const sources = scores.map((score, i) => ({ path: `${i}.md`, read: async () => String(i) }));
  const embed = async (inputs) => inputs.map((input) => [
    input === "question" ? 1 : scores[Number(input)],
    ...Array(REFERENCE_CONFIGURATION.indexSignature.vectorDimension - 1).fill(0),
  ]);
  const index = new VaultIndex(new MemoryAdapter(), "index", () => sources, embed);
  await index.start(REFERENCE_CONFIGURATION.embeddingModel);
  const product = await index.retrieve("question");
  assert.deepEqual(product.map(({ path }) => path), ["0.md", "1.md", "2.md", "3.md", "4.md", "5.md"]);
  assert.equal((await index.retrieve("question", true, 4)).length, 4);
  const uncapped = await index.retrieve("question", false, Infinity);
  assert.equal(uncapped.length, 7);
  const cutoff = calibrateCutoff(REFERENCE_CONFIGURATION.embeddingModel.digest, [{
    gold: ["5.md"],
    ranked: uncapped.map(({ path, score }) => ({ evidence: [path], score })),
  }]);
  assert.ok(Math.abs(cutoff.minimumScore - 0.61) < 1e-6);
});

test("proof gates inspect serialized controls, messages and terminal diagnostics", async () => {
  const { observeRequest, requestPass } = await import("../evaluation/proof.mjs");
  const messages = [{ role: "system", content: GROUNDING_SYSTEM_PROMPT }, { role: "user", content: "private question" }];
  const request = { origin: "http://127.0.0.1:11434", path: "/api/chat", method: "POST", body: {
    model: "chat", messages, stream: true, think: false, options: { num_predict: 256, seed: 0, temperature: 0 },
  } };
  const expected = { model: "chat", messageSha256: messages.map(({ content }) => createHash("sha256").update(content).digest("hex")) };
  const diagnostics = { eval_count: 42, prompt_eval_count: 100, done_reason: "stop", total_duration: 1 };
  assert.equal(requestPass(observeRequest(request), diagnostics, expected), true);
  assert.doesNotMatch(JSON.stringify(observeRequest(request)), /private question/);
  for (const change of [
    (r) => { r.body.think = true; },
    (r) => { r.body.options.num_ctx = 4096; },
    (r) => { delete r.body.options.seed; },
    (r) => { r.body.messages[1].content = "wrong current question"; },
    (r) => { r.body.model = "other"; },
    (r) => { r.origin = "https://elsewhere.invalid"; },
  ]) {
    const changed = structuredClone(request); change(changed);
    assert.equal(requestPass(observeRequest(changed), diagnostics, expected), false);
  }
  assert.equal(requestPass(observeRequest(request), { ...diagnostics, eval_count: 257 }, expected), false);
  assert.equal(requestPass(observeRequest(request), {}, expected), false);
  assert.equal(requestPass(undefined, diagnostics, expected), false);
  const missingHashes = observeRequest(request);
  delete missingHashes.messageSha256;
  assert.equal(requestPass(missingHashes, diagnostics, { model: "chat" }), false);
  const wrongRole = observeRequest(request);
  wrongRole.roles[0] = "user";
  assert.equal(requestPass(wrongRole, diagnostics, expected), false);
});

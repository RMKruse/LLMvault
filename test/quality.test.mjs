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
  const source = await readFile(new URL("../src/main.ts", import.meta.url), "utf8");
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

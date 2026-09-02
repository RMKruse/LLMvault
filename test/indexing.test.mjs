import assert from "node:assert/strict";
import test from "node:test";
import { TextEncoder } from "node:util";

import {
  CHUNK_OVERLAP_BYTES,
  CHUNK_TARGET_BYTES,
  MarkdownIndex,
  chunkMarkdown,
  encodeVector,
} from "../src/indexing.ts";

class MemoryAdapter {
  files = new Map();
  folders = new Set();

  async exists(path) {
    return this.files.has(path) || this.folders.has(path);
  }

  async mkdir(path) {
    this.folders.add(path);
  }

  async read(path) {
    if (!this.files.has(path)) throw new Error(`missing ${path}`);
    return this.files.get(path);
  }

  async write(path, value) {
    this.files.set(path, value);
  }

  async process(path, update) {
    const value = update(this.files.get(path) ?? "");
    this.files.set(path, value);
    return value;
  }

  async rmdir(path) {
    for (const key of [...this.files.keys()]) {
      if (key === path || key.startsWith(`${path}/`)) this.files.delete(key);
    }
    for (const key of [...this.folders]) {
      if (key === path || key.startsWith(`${path}/`)) this.folders.delete(key);
    }
  }
}

test("Markdown chunks keep source locations and overlap only a complete block", () => {
  const source = `# Alpha\n\n${"x".repeat(1_600)}\n\n${"y".repeat(400)}\n\n${"z".repeat(400)}`;
  const chunks = chunkMarkdown(source, [
    { offset: 0, type: "heading", value: "Alpha" },
  ]);
  const yStart = source.indexOf("y".repeat(400));

  assert.equal(chunks.length, 2);
  assert.equal(chunks[0].text, source.slice(0, yStart + 402));
  assert.equal(chunks[1].text, source.slice(yStart));
  assert.equal(chunks[1].start, yStart);
  assert.deepEqual(chunks[1].anchor, { type: "heading", value: "Alpha" });
  assert.equal(chunks[1].startLine, 5);
  assert.equal(chunks[1].endLine, 7);
  assert.ok(
    chunks.every(
      ({ text }) => new TextEncoder().encode(text).byteLength <= CHUNK_TARGET_BYTES,
    ),
  );
  assert.ok(new TextEncoder().encode(chunks[1].text.slice(0, 402)).byteLength <= CHUNK_OVERLAP_BYTES);
});

test("oversized Markdown lines split at Unicode code-point boundaries", () => {
  const source = "🙂".repeat(600);
  const chunks = chunkMarkdown(source);

  assert.equal(chunks.map(({ text }) => text).join(""), source);
  assert.ok(chunks.every(({ text }) => !text.includes("�")));
  assert.ok(
    chunks.every(
      ({ text }) => new TextEncoder().encode(text).byteLength <= CHUNK_TARGET_BYTES,
    ),
  );
});

test("only Obsidian-confirmed headings and blocks become citation anchors", () => {
  const source = "```md\n# Not a heading\n^not-a-block\n```";
  assert.ok(chunkMarkdown(source).every(({ anchor }) => anchor === undefined));
});

test("vectors are persisted as explicitly little-endian Float32", () => {
  assert.equal(encodeVector([1, -2.5]), "AACAPwAAIMA=");
});

test("a completed generation is restored without embedding unchanged Markdown", async () => {
  const adapter = new MemoryAdapter();
  const sources = [
    {
      anchors: [{ offset: 0, type: "heading", value: "Alpha" }],
      path: "Notes/alpha.md",
      read: async () => "# Alpha\nBody",
    },
    { path: "Notes/empty.md", read: async () => " \n " },
  ];
  let embedCalls = 0;
  const embed = async (inputs) => {
    embedCalls += 1;
    return inputs.map((_input, index) => [1, index + 0.5]);
  };
  const model = { name: "embed", digest: "sha256:abc" };

  const first = new MarkdownIndex(adapter, "plugin/index-v1", () => sources, embed);
  const built = await first.start(model);
  const callsAfterBuild = embedCalls;
  const restarted = new MarkdownIndex(adapter, "plugin/index-v1", () => sources, embed);
  const restored = await restarted.start(model);

  assert.equal(built.phase, "ready");
  assert.deepEqual(built.statuses, { indexed: 1, no_extractable_text: 1 });
  assert.equal(restored.phase, "ready");
  assert.equal(embedCalls, callsAfterBuild);
  assert.equal(
    [...adapter.files.values()].some((value) => value.includes("# Alpha\nBody")),
    false,
  );

  const recordPath = [...adapter.files.keys()].find((path) => path.includes("/records/"));
  const record = JSON.parse(await adapter.read(recordPath));
  record.chunks[0].locator.start += 1;
  await adapter.write(recordPath, JSON.stringify(record));
  const corrupted = new MarkdownIndex(adapter, "plugin/index-v1", () => sources, embed);
  assert.equal((await corrupted.start(model)).phase, "ready");
  assert.ok(embedCalls > callsAfterBuild);
});

test("invalid embedding vectors never activate a generation", async () => {
  const adapter = new MemoryAdapter();
  const index = new MarkdownIndex(
    adapter,
    "plugin/index-v1",
    () => [{ path: "bad.md", read: async () => "text" }],
    async () => [[Number.NaN]],
  );

  const result = await index.start({ name: "embed", digest: "sha256:abc" });

  assert.equal(result.phase, "failed");
  assert.equal(await adapter.exists("plugin/index-v1/active.json"), false);
});

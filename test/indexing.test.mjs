import assert from "node:assert/strict";
import test from "node:test";
import { TextEncoder } from "node:util";

import {
  CHUNK_OVERLAP_BYTES,
  CHUNK_TARGET_BYTES,
  EXTRACTED_TEXT_LIMIT_BYTES,
  PREPROCESSING_LIMIT_MS,
  RAW_FILE_LIMIT_BYTES,
  VaultIndex,
  chunkCanvas,
  chunkMarkdown,
  classifyVaultSource,
  encodeVector,
  resolveDailyRecapRequest,
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

  async list(path) {
    const prefix = `${path}/`;
    return {
      files: [...this.files.keys()].filter((item) =>
        item.startsWith(prefix) && !item.slice(prefix.length).includes("/"),
      ),
      folders: [...this.folders].filter((item) =>
        item.startsWith(prefix) && !item.slice(prefix.length).includes("/"),
      ),
    };
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

const generationFolders = (adapter) =>
  [...adapter.folders].filter((path) => path.match(/\/generations\/[^/]+$/));

class PausingProcessAdapter extends MemoryAdapter {
  commits = [];
  pauseCommit = false;
  releaseCommit = () => undefined;
  signalCommit = () => undefined;
  commitStarted = new Promise((resolve) => { this.signalCommit = resolve; });
  commitReleased = new Promise((resolve) => { this.releaseCommit = resolve; });

  async process(path, update) {
    if (this.pauseCommit) {
      this.signalCommit();
      await this.commitReleased;
    }
    const committedPointer = await super.process(path, update);
    this.commits.push(committedPointer);
    return committedPointer;
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

test("Daily Recap requests resolve an exact local-date note or fail closed", () => {
  const paths = [
    "Labortagebuch/2026.09.01.md",
    "Labortagebuch/2026.09.02.md",
  ];

  assert.deepEqual(
    resolveDailyRecapRequest(
      "fasse mir zusammen was ich gestern gemacht habe",
      new Date("2026-09-03T10:00:00Z"),
      "Europe/Berlin",
      paths,
    ),
    { date: "2026.09.02", targetPath: "Labortagebuch/2026.09.02.md" },
  );
  assert.deepEqual(
    resolveDailyRecapRequest(
      "summary of yesterday",
      new Date("2026-09-03T00:30:00Z"),
      "America/Los_Angeles",
      paths,
    ),
    { date: "2026.09.01", targetPath: "Labortagebuch/2026.09.01.md" },
  );
  assert.deepEqual(
    resolveDailyRecapRequest(
      "summary of today",
      new Date("2026-09-03T10:00:00Z"),
      "Europe/Berlin",
      paths,
    ),
    { date: "2026.09.03" },
  );
  assert.deepEqual(
    resolveDailyRecapRequest(
      "summary of yesterday",
      new Date("2026-09-03T10:00:00Z"),
      "Europe/Berlin",
      [...paths, "Archive/2026.09.02.md"],
    ),
    { date: "2026.09.02" },
  );
  assert.equal(
    resolveDailyRecapRequest(
      "what happened yesterday?",
      new Date("2026-09-03T10:00:00Z"),
      "Europe/Berlin",
      paths,
    ),
    null,
  );
});

test("Canvas extraction keeps non-empty text nodes independent and ignores graph content", () => {
  const chunks = chunkCanvas(JSON.stringify({
    nodes: [
      { id: "first", type: "text", text: "  # Alpha\nFirst card  ", x: 0, y: 0, width: 200, height: 100 },
      { id: "empty", type: "text", text: " \n ", x: 1, y: 1, width: 200, height: 100 },
      { id: "file", type: "file", file: "Secret.md", x: 2, y: 2, width: 200, height: 100 },
      { id: "link", type: "link", url: "https://example.com", x: 3, y: 3, width: 200, height: 100 },
      { id: "second", type: "text", text: "Second card", x: 4, y: 4, width: 200, height: 100 },
    ],
    edges: [{ id: "edge", fromNode: "first", toNode: "second" }],
  }));

  assert.deepEqual(
    chunks.map(({ nodeId, excerpt, text }) => [nodeId, excerpt, text]),
    [
      ["first", "# Alpha First card", "  # Alpha\nFirst card  "],
      ["second", "Second card", "Second card"],
    ],
  );
});

test("every visible Vault Content file receives one accurate terminal outcome", async () => {
  let excludedReads = 0;
  const excluded = async () => {
    excludedReads += 1;
    return "must not be read";
  };
  const sources = [
    classifyVaultSource("Note.MD", "MD", 4, async () => "note"),
    classifyVaultSource("Board.CANVAS", "CANVAS", 100, async () => JSON.stringify({
      nodes: [{ id: "card", type: "text", text: "card", x: 0, y: 0, width: 10, height: 10 }],
      edges: [],
    })),
    classifyVaultSource("Empty.md", "md", 0, async () => " \n"),
    classifyVaultSource("Empty.canvas", "canvas", 24, async () => JSON.stringify({ nodes: [], edges: [] })),
    classifyVaultSource("View.base", "base", 10, excluded),
    classifyVaultSource("Paper.pdf", "pdf", 10, excluded),
    classifyVaultSource("Data.txt", "txt", 10, excluded),
    classifyVaultSource("Script.mjs", "mjs", 10, excluded),
    classifyVaultSource("Archive.zip", "zip", 10, excluded),
    classifyVaultSource("Encrypted.gpg", "gpg", 10, excluded),
    classifyVaultSource("Compressed.zst", "zst", 10, excluded),
    classifyVaultSource("Mystery.xyzzy", "xyzzy", 10, excluded),
    classifyVaultSource("LICENSE", "", 10, excluded),
    classifyVaultSource("Bad.canvas", "canvas", 10, async () => "not json"),
    classifyVaultSource("Unreadable.md", "md", 10, async () => { throw new Error("private path"); }),
    classifyVaultSource("Huge.md", "md", RAW_FILE_LIMIT_BYTES + 1, excluded),
    classifyVaultSource("Extracted.md", "md", 10, async () => "x".repeat(EXTRACTED_TEXT_LIMIT_BYTES + 1)),
  ];
  const embedded = [];
  const index = new VaultIndex(
    new MemoryAdapter(),
    "plugin/index-v1",
    () => sources,
    async (inputs) => {
      embedded.push(...inputs);
      return inputs.map(() => [1]);
    },
  );

  const snapshot = await index.start({ name: "embed", digest: "sha256:abc" });

  assert.equal(snapshot.phase, "ready");
  assert.equal(snapshot.total, sources.length);
  assert.equal(snapshot.completed, sources.length);
  assert.equal(snapshot.outcomes.length, sources.length);
  assert.equal(Object.values(snapshot.statuses).reduce((sum, count) => sum + count, 0), sources.length);
  assert.deepEqual(snapshot.statuses, {
    extractor_failed: 2,
    ignored_non_content: 1,
    indexed: 2,
    limit_exceeded: 2,
    no_extractable_text: 2,
    unrecognized_format: 2,
    unsupported_format: 6,
  });
  assert.deepEqual(embedded, ["card", "note"]);
  assert.equal(excludedReads, 0);
  assert.equal(snapshot.outcomes.find(({ path }) => path === "Bad.canvas")?.reason, "corrupt_or_invalid_canvas");
  assert.equal(snapshot.outcomes.find(({ path }) => path === "Unreadable.md")?.reason, "read_failed");
  assert.deepEqual(
    snapshot.outcomes.find(({ path }) => path === "Huge.md"),
    {
      ceiling: RAW_FILE_LIMIT_BYTES,
      limit: "raw_bytes",
      observed: RAW_FILE_LIMIT_BYTES + 1,
      path: "Huge.md",
      status: "limit_exceeded",
    },
  );
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

  const first = new VaultIndex(adapter, "plugin/index-v1", () => sources, embed);
  const built = await first.start(model);
  const callsAfterBuild = embedCalls;
  const restarted = new VaultIndex(adapter, "plugin/index-v1", () => sources, embed);
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
  record.chunks.push({ ...record.chunks[0], id: "forged-extra-chunk" });
  await adapter.write(recordPath, JSON.stringify(record));
  const catalogPath = [...adapter.files.keys()].find((path) => path.endsWith("/catalog.json"));
  const catalog = JSON.parse(await adapter.read(catalogPath));
  catalog.entries[0].chunkCount += 1;
  catalog.entries[0].vectorCount += 1;
  await adapter.write(catalogPath, JSON.stringify(catalog));
  await adapter.mkdir("plugin/index-v1/generations/00000000-0000-4000-8000-000000000000");
  const corrupted = new VaultIndex(adapter, "plugin/index-v1", () => sources, embed);
  assert.equal((await corrupted.start(model)).phase, "ready");
  assert.ok(embedCalls > callsAfterBuild);
  assert.equal(generationFolders(adapter).length, 1);
});

test("restart removes abandoned generations without touching the valid active generation", async () => {
  const adapter = new MemoryAdapter();
  const model = { name: "embed", digest: "sha256:abc" };
  const root = "plugin/index-v1";
  const sources = () => [{ path: "Note.md", read: async () => "current" }];
  const embed = async (inputs) => inputs.map(() => [1]);
  await new VaultIndex(adapter, root, sources, embed).start(model);
  const active = JSON.parse(await adapter.read(`${root}/active.json`)).generationId;
  const abandoned = "00000000-0000-4000-8000-000000000000";
  await adapter.mkdir(`${root}/generations/${abandoned}`);
  await adapter.mkdir(`${root}/generations/${abandoned}/records`);
  await adapter.write(`${root}/generations/${abandoned}/catalog.json`, JSON.stringify({
    complete: false,
    generationId: abandoned,
  }));
  await adapter.mkdir(`${root}/generations/corrupt-name`);

  const restored = await new VaultIndex(adapter, root, sources, embed).start(model);

  assert.equal(restored.phase, "ready");
  assert.deepEqual(
    generationFolders(adapter),
    [`${root}/generations/${active}`],
  );
});

test("restart cleans invalid generations even when recovery rebuild fails", async () => {
  const adapter = new MemoryAdapter();
  const root = "plugin/index-v1";
  const model = { name: "embed", digest: "sha256:abc" };
  const sources = () => [{ path: "Note.md", read: async () => "current" }];
  await new VaultIndex(
    adapter,
    root,
    sources,
    async (inputs) => inputs.map(() => [1]),
  ).start(model);
  const catalogPath = [...adapter.files.keys()].find((path) => path.endsWith("/catalog.json"));
  await adapter.write(catalogPath, "corrupt");
  await adapter.mkdir(`${root}/generations/abandoned`);

  const failed = await new VaultIndex(
    adapter,
    root,
    sources,
    async () => { throw new Error("embedding failed"); },
  ).start(model);

  assert.equal(failed.phase, "failed");
  assert.equal(generationFolders(adapter).length, 0);
});

test("startup rebuild reuses only sources with an exact signature and fingerprint match", async () => {
  const adapter = new MemoryAdapter();
  const texts = new Map([
    ["Alpha.md", "alpha"],
    ["Beta.md", "beta"],
  ]);
  const embedded = [];
  const sources = () => [...texts].map(([path, text]) => ({ path, read: async () => text }));
  const embed = async (inputs) => {
    embedded.push(...inputs);
    return inputs.map(() => [1]);
  };
  const model = { name: "embed", digest: "sha256:abc" };
  await new VaultIndex(adapter, "plugin/index-v1", sources, embed).start(model);
  embedded.length = 0;
  texts.set("Alpha.md", "changed alpha");

  const rebuilt = await new VaultIndex(
    adapter,
    "plugin/index-v1",
    sources,
    embed,
  ).start(model);

  assert.equal(rebuilt.phase, "ready");
  assert.deepEqual(embedded, ["changed alpha"]);

  embedded.length = 0;
  const incompatible = await new VaultIndex(
    adapter,
    "plugin/index-v1",
    sources,
    embed,
  ).start({ name: "embed", digest: "sha256:changed" });

  assert.equal(incompatible.phase, "ready");
  assert.deepEqual(embedded, ["changed alpha", "beta"]);
  assert.equal(generationFolders(adapter).length, 1);
});

test("questions retrieve at most six fresh sources with the generation's pinned model", async () => {
  const adapter = new MemoryAdapter();
  const texts = new Map([
    ["one.md", "0.1"],
    ["two.md", "0.2"],
    ["three.md", "0.3"],
    ["four.md", "0.4"],
    ["five.md", "0.5"],
    ["six.md", "0.6"],
    ["seven.md", "0.7"],
  ]);
  const sources = [...texts.keys()].map((path) => ({
    path,
    read: async () => texts.get(path),
  }));
  const model = { name: "pinned-embed", digest: "sha256:pinned" };
  const models = [];
  const validatedModels = [];
  const embed = async (inputs, requestedModel) => {
    models.push(requestedModel);
    return inputs.map((input) =>
      input === "Which source is strongest?" ? [1, 0] : [Number(input), 0],
    );
  };
  const index = new VaultIndex(
    adapter,
    "plugin/index-v1",
    () => sources,
    embed,
    undefined,
    undefined,
    async (requestedModel) => {
      validatedModels.push(requestedModel);
      return true;
    },
  );
  await index.start(model);

  const evidence = await index.retrieve("Which source is strongest?");

  assert.deepEqual(
    evidence.map(({ citationId, path, text }) => [citationId, path, text]),
    [
      ["S1-1", "seven.md", "0.7"],
      ["S1-2", "six.md", "0.6"],
      ["S1-3", "five.md", "0.5"],
      ["S1-4", "four.md", "0.4"],
      ["S1-5", "three.md", "0.3"],
      ["S1-6", "two.md", "0.2"],
    ],
  );
  assert.deepEqual(
    (await index.retrieve("Which source is strongest?", false)).map(({ path }) => path),
    ["seven.md", "six.md", "five.md", "four.md"],
  );
  assert.ok(models.every((requestedModel) => requestedModel === model));

  texts.set("seven.md", "changed after indexing");
  const freshEvidence = await index.retrieve("Which source is strongest?");

  assert.deepEqual(
    freshEvidence.map(({ path }) => path),
    ["six.md", "five.md", "four.md", "three.md", "two.md", "one.md"],
  );
  assert.deepEqual(validatedModels, [model, model]);
  assert.equal(index.getSnapshot().phase, "indexing");
});

test("Daily Recap evidence follows exact paths once in source order without query embedding", async () => {
  const sources = [
    { path: "2026.09.02.md", read: async () => "daily" },
    { path: "alpha.md", read: async () => "alpha" },
    { path: "beta.md", read: async () => "beta" },
    { path: "gamma.md", read: async () => "gamma" },
    { path: "second-hop.md", read: async () => "second hop" },
  ];
  let queryEmbeddings = 0;
  const index = new VaultIndex(
    new MemoryAdapter(),
    "plugin/index-v1",
    () => sources,
    async (inputs) => inputs.map(() => [1]),
    undefined,
    async (inputs) => {
      queryEmbeddings += 1;
      return inputs.map(() => [1]);
    },
  );
  await index.start({ name: "embed", digest: "sha256:abc" });

  const evidence = await index.retrievePaths([
    "2026.09.02.md",
    "alpha.md",
    "alpha.md",
    "beta.md",
    "gamma.md",
    "second-hop.md",
  ]);

  assert.deepEqual(
    evidence.map(({ citationId, path, text }) => [citationId, path, text]),
    [
      ["S1-1", "2026.09.02.md", "daily"],
      ["S1-2", "alpha.md", "alpha"],
      ["S1-3", "beta.md", "beta"],
      ["S1-4", "gamma.md", "gamma"],
    ],
  );
  assert.equal(queryEmbeddings, 0);
});

test("Canvas evidence retains its node locator and application-owned preview", async () => {
  const adapter = new MemoryAdapter();
  const source = classifyVaultSource("Board.canvas", "canvas", 100, async () => JSON.stringify({
    nodes: [
      { id: "card-1", type: "text", text: "Canvas evidence", x: 0, y: 0, width: 100, height: 100 },
      { id: "file-1", type: "file", file: "Not followed.md", x: 1, y: 1, width: 100, height: 100 },
    ],
    edges: [],
  }));
  const index = new VaultIndex(
    adapter,
    "plugin/index-v1",
    () => [source],
    async (inputs) => inputs.map(() => [1]),
  );
  await index.start({ name: "embed", digest: "sha256:abc" });

  const [evidence] = await index.retrieve("question");

  assert.deepEqual(
    evidence && {
      excerpt: evidence.excerpt,
      format: evidence.format,
      nodeId: evidence.nodeId,
      path: evidence.path,
      text: evidence.text,
    },
    {
      excerpt: "Canvas evidence",
      format: "canvas",
      nodeId: "card-1",
      path: "Board.canvas",
      text: "Canvas evidence",
    },
  );
  assert.deepEqual(await index.resolveEvidence(evidence), evidence);
  assert.equal([...adapter.files.values()].some((value) => value.includes("Not followed.md")), false);
});

test("a failed replacement keeps partial and prior stale chunks unqueryable", async () => {
  const adapter = new MemoryAdapter();
  let text = "current";
  let failEmbedding = false;
  const index = new VaultIndex(
    adapter,
    "plugin/index-v1",
    () => [{ path: "Note.md", read: async () => text }],
    async (inputs) => {
      if (failEmbedding) throw new Error("embedding failed");
      return inputs.map(() => [1]);
    },
  );
  const model = { name: "embed", digest: "sha256:abc" };
  await index.start(model);
  const activeBefore = await adapter.read("plugin/index-v1/active.json");
  text = "replacement";
  failEmbedding = true;

  const failed = await index.start(model);

  assert.equal(failed.phase, "failed");
  assert.equal(failed.available, true);
  failEmbedding = false;
  assert.deepEqual(await index.retrieve("question"), []);
  assert.equal(await adapter.read("plugin/index-v1/active.json"), activeBefore);
  assert.equal([...adapter.files.values()].some((value) => value.includes("replacement")), false);
});

test("queries scan the validated in-memory generation without record I/O", async () => {
  const adapter = new class extends MemoryAdapter {
    recordReads = 0;

    async read(path) {
      if (path.includes("/records/")) this.recordReads += 1;
      return await super.read(path);
    }
  }();
  const index = new VaultIndex(
    adapter,
    "plugin/index-v1",
    () => [{ path: "Note.md", read: async () => "current" }],
    async (inputs) => inputs.map(() => [1]),
  );
  await index.start({ name: "embed", digest: "sha256:abc" });
  adapter.recordReads = 0;

  assert.equal((await index.retrieve("question"))[0]?.text, "current");
  assert.equal(adapter.recordReads, 0);
});

test("mutation work leaves the changed source until after unchanged records are copied", async () => {
  let mutation = false;
  let mutationRecordWrites = 0;
  const adapter = new class extends MemoryAdapter {
    async write(path, value) {
      if (mutation && path.includes("/records/")) mutationRecordWrites += 1;
      await super.write(path, value);
    }
  }();
  let changed = "before";
  const index = new VaultIndex(
    adapter,
    "plugin/index-v1",
    () => [
      { path: "Changed.md", read: async () => changed },
      { path: "Stable.md", read: async () => "stable" },
    ],
    async (inputs) => {
      if (mutation) assert.equal(mutationRecordWrites, 1);
      return inputs.map(() => [1]);
    },
  );
  await index.start({ name: "embed", digest: "sha256:abc" });
  changed = "after";
  mutation = true;

  await index.invalidate(["Changed.md"]);
});

test("manual rebuild keeps the prior generation available until atomic activation", async () => {
  const adapter = new PausingProcessAdapter();
  const model = { name: "embed", digest: "sha256:abc" };
  const index = new VaultIndex(
    adapter,
    "plugin/index-v1",
    () => [{ path: "Note.md", read: async () => "current" }],
    async (inputs) => inputs.map(() => [1]),
  );
  await index.start(model);
  const previousPointer = await adapter.read("plugin/index-v1/active.json");

  adapter.pauseCommit = true;
  const rebuilding = index.rebuild(model);
  await adapter.commitStarted;

  assert.equal(index.getSnapshot().phase, "indexing");
  assert.equal(index.getSnapshot().available, true);
  assert.equal((await index.retrieve("question"))[0]?.text, "current");
  assert.equal(await adapter.read("plugin/index-v1/active.json"), previousPointer);

  adapter.pauseCommit = false;
  adapter.releaseCommit();
  const rebuilt = await rebuilding;
  const nextPointer = await adapter.read("plugin/index-v1/active.json");

  assert.equal(rebuilt.phase, "ready");
  assert.equal(rebuilt.available, true);
  assert.notEqual(nextPointer, previousPointer);
  assert.equal(generationFolders(adapter).length, 1);
});

test("cancelled rebuild returns to the prior active generation", async () => {
  const adapter = new PausingProcessAdapter();
  const model = { name: "embed", digest: "sha256:abc" };
  const index = new VaultIndex(
    adapter,
    "plugin/index-v1",
    () => [{ path: "Note.md", read: async () => "current" }],
    async (inputs) => inputs.map(() => [1]),
  );
  await index.start(model);
  const previousPointer = await adapter.read("plugin/index-v1/active.json");
  adapter.pauseCommit = true;
  const rebuilding = index.rebuild(model);
  await adapter.commitStarted;

  index.cancel();
  adapter.pauseCommit = false;
  adapter.releaseCommit();
  const cancelled = await rebuilding;

  assert.equal(cancelled.phase, "failed");
  assert.equal(cancelled.available, true);
  assert.equal(await adapter.read("plugin/index-v1/active.json"), previousPointer);
  assert.equal((await index.retrieve("question"))[0]?.text, "current");
});

test("delete all drains active work and removes only the fixed index root", async () => {
  const adapter = new PausingProcessAdapter();
  const root = "plugin/index-v1";
  const model = { name: "embed", digest: "sha256:abc" };
  const index = new VaultIndex(
    adapter,
    root,
    () => [{ path: "Note.md", read: async () => "current" }],
    async (inputs) => inputs.map(() => [1]),
  );
  await index.start(model);
  await adapter.write("plugin/connection-settings.json", "keep");
  adapter.pauseCommit = true;
  const rebuilding = index.rebuild(model);
  await adapter.commitStarted;

  const deleting = index.deleteAll();
  assert.deepEqual(await index.retrieve("question"), []);
  const blockedStart = index.start(model);
  adapter.pauseCommit = false;
  adapter.releaseCommit();
  await Promise.all([rebuilding, deleting, blockedStart]);

  assert.equal([...adapter.files.keys()].some((path) => path.startsWith(`${root}/`)), false);
  assert.equal([...adapter.folders].some((path) => path === root || path.startsWith(`${root}/`)), false);
  assert.equal(await adapter.read("plugin/connection-settings.json"), "keep");
  assert.deepEqual(index.getSnapshot(), {
    available: false,
    completed: 0,
    outcomes: [],
    phase: "idle",
    statuses: {},
    total: 0,
  });
});

test("failed index deletion stays visible and can be retried", async () => {
  const root = "plugin/index-v1";
  const adapter = new class extends MemoryAdapter {
    failRemoval = true;

    async rmdir(path) {
      if (path === root && this.failRemoval) {
        this.failRemoval = false;
        throw new Error("storage busy");
      }
      await super.rmdir(path);
    }
  }();
  const index = new VaultIndex(
    adapter,
    root,
    () => [{ path: "Note.md", read: async () => "current" }],
    async (inputs) => inputs.map(() => [1]),
  );
  await index.start({ name: "embed", digest: "sha256:abc" });

  await assert.rejects(index.deleteAll(), /storage busy/);
  assert.equal(index.getSnapshot().phase, "failed");
  await index.deleteAll();

  assert.equal(await adapter.exists(root), false);
  assert.equal(index.getSnapshot().phase, "idle");
});

test("Vault mutations tombstone immediately and serialize to the clean final index", async () => {
  const adapter = new MemoryAdapter();
  let text = "original";
  let blockEmbedding = false;
  let releaseEmbedding = () => undefined;
  let signalEmbedding = () => undefined;
  let activeEmbeddings = 0;
  let maximumActiveEmbeddings = 0;
  const embeddingStarted = new Promise((resolve) => { signalEmbedding = resolve; });
  const embeddingReleased = new Promise((resolve) => { releaseEmbedding = resolve; });
  const sources = () => [{ path: "Note.md", read: async () => text }];
  const embed = async (inputs) => {
    activeEmbeddings += 1;
    maximumActiveEmbeddings = Math.max(maximumActiveEmbeddings, activeEmbeddings);
    try {
      if (blockEmbedding) {
        signalEmbedding();
        await embeddingReleased;
      }
      return inputs.map(() => [1]);
    } finally {
      activeEmbeddings -= 1;
    }
  };
  const model = { name: "embed", digest: "sha256:abc" };
  const index = new VaultIndex(adapter, "plugin/index-v1", sources, embed);
  await index.start(model);
  const [oldEvidence] = await index.retrieve("question");

  text = "intermediate";
  blockEmbedding = true;
  const firstMutation = index.invalidate(["Note.md"]);
  const coalescedMutation = index.invalidate(["Note.md"]);

  assert.equal(firstMutation, coalescedMutation);
  assert.equal(index.getSnapshot().phase, "indexing");
  assert.equal(await index.resolveEvidence(oldEvidence), null);
  assert.deepEqual(await index.retrieve("question"), []);

  await embeddingStarted;
  text = "final";
  const mutationDuringIndexing = index.invalidate(["Note.md"]);
  blockEmbedding = false;
  releaseEmbedding();
  await Promise.all([firstMutation, mutationDuringIndexing]);

  const clean = new VaultIndex(
    new MemoryAdapter(),
    "plugin/index-v1",
    sources,
    async (inputs) => inputs.map(() => [1]),
  );
  await clean.start(model);
  const withoutCitationId = (evidence) => {
    const result = { ...evidence };
    delete result.citationId;
    return result;
  };

  assert.equal(maximumActiveEmbeddings, 1);
  assert.deepEqual(index.getSnapshot(), clean.getSnapshot());
  assert.deepEqual(
    (await index.retrieve("question")).map(withoutCitationId),
    (await clean.retrieve("question")).map(withoutCitationId),
  );
});

test("a mutation during catalog commit cannot reactivate late evidence", async () => {
  const adapter = new PausingProcessAdapter();
  let text = "original";
  let countReady = false;
  let readySnapshots = 0;
  const index = new VaultIndex(
    adapter,
    "plugin/index-v1",
    () => [{ path: "Note.md", read: async () => text }],
    async (inputs) => inputs.map(() => [1]),
    (snapshot) => {
      if (countReady && snapshot.phase === "ready") readySnapshots += 1;
    },
  );
  await index.start({ name: "embed", digest: "sha256:abc" });

  text = "intermediate";
  adapter.pauseCommit = true;
  const intermediate = index.invalidate(["Note.md"]);
  await adapter.commitStarted;
  countReady = true;
  text = "final";
  const final = index.invalidate(["Note.md"]);
  adapter.pauseCommit = false;
  adapter.releaseCommit();
  await Promise.all([intermediate, final]);

  assert.equal(adapter.commits.length, 2);
  assert.equal(readySnapshots, 1);
  assert.equal((await index.retrieve("question"))[0]?.text, "final");
});

test("plugin-controlled preprocessing timeouts report the observed ceiling", async () => {
  const realSetTimeout = globalThis.setTimeout;
  const realClearTimeout = globalThis.clearTimeout;
  globalThis.setTimeout = (callback) => {
    globalThis.queueMicrotask(callback);
    return 1;
  };
  globalThis.clearTimeout = () => undefined;
  try {
    const index = new VaultIndex(
      new MemoryAdapter(),
      "plugin/index-v1",
      () => [classifyVaultSource("Slow.md", "md", 10, () => new Promise(() => undefined))],
      async () => { throw new Error("must not embed"); },
    );

    const snapshot = await index.start({ name: "embed", digest: "sha256:abc" });

    assert.deepEqual(snapshot.outcomes, [{
      ceiling: PREPROCESSING_LIMIT_MS,
      limit: "preprocessing_ms",
      observed: PREPROCESSING_LIMIT_MS,
      path: "Slow.md",
      status: "limit_exceeded",
    }]);
  } finally {
    globalThis.setTimeout = realSetTimeout;
    globalThis.clearTimeout = realClearTimeout;
  }
});

test("invalid embedding vectors never activate a generation", async () => {
  const adapter = new MemoryAdapter();
  const index = new VaultIndex(
    adapter,
    "plugin/index-v1",
    () => [{ path: "bad.md", read: async () => "text" }],
    async () => [[Number.NaN]],
  );

  const result = await index.start({ name: "embed", digest: "sha256:abc" });

  assert.equal(result.phase, "failed");
  assert.equal(await adapter.exists("plugin/index-v1/active.json"), false);
});

test("activation revalidates the complete generation after catalog construction", async () => {
  const adapter = new class extends MemoryAdapter {
    async write(path, value) {
      await super.write(path, value);
      if (!path.endsWith("/catalog.json")) return;
      const recordPath = [...this.files.keys()].find((item) => item.includes("/records/"));
      const record = JSON.parse(await this.read(recordPath));
      record.chunks[0].vector = "corrupt";
      await super.write(recordPath, JSON.stringify(record));
    }
  }();
  const index = new VaultIndex(
    adapter,
    "plugin/index-v1",
    () => [{ path: "Note.md", read: async () => "text" }],
    async (inputs) => inputs.map(() => [1]),
  );

  const result = await index.start({ name: "embed", digest: "sha256:abc" });

  assert.equal(result.phase, "failed");
  assert.equal(await adapter.exists("plugin/index-v1/active.json"), false);
});

test("restart converges after termination at every durable generation boundary", async () => {
  for (const boundary of ["record", "catalog", "pointer", "cleanup"]) {
    const adapter = new class extends MemoryAdapter {
      armed = false;

      async write(path, value) {
        await super.write(path, value);
        if (this.armed &&
          ((boundary === "record" && path.includes("/records/")) ||
            (boundary === "catalog" && path.endsWith("/catalog.json")))) {
          this.armed = false;
          throw new Error("terminated");
        }
      }

      async process(path, update) {
        const committedPointer = await super.process(path, update);
        if (this.armed && boundary === "pointer") {
          this.armed = false;
          throw new Error("terminated");
        }
        return committedPointer;
      }

      async rmdir(path) {
        await super.rmdir(path);
        if (this.armed && boundary === "cleanup") {
          this.armed = false;
          throw new Error("terminated");
        }
      }
    }();
    let text = "old";
    const model = { name: "embed", digest: "sha256:abc" };
    const sources = () => [{ path: "Note.md", read: async () => text }];
    const embed = async (inputs) => inputs.map(() => [1]);
    const interrupted = new VaultIndex(adapter, "plugin/index-v1", sources, embed);
    await interrupted.start(model);
    text = "canonical";
    adapter.armed = true;
    await interrupted.invalidate(["Note.md"]);
    adapter.armed = false;

    const restarted = new VaultIndex(adapter, "plugin/index-v1", sources, embed);
    const recovered = await restarted.start(model);

    assert.equal(recovered.phase, "ready", boundary);
    assert.equal((await restarted.retrieve("question"))[0]?.text, "canonical", boundary);
    assert.equal(generationFolders(adapter).length, 1, boundary);
  }
});

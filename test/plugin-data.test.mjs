import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import test from "node:test";

import esbuild from "esbuild";

async function loadPlugin() {
  const result = await esbuild.build({
    bundle: true,
    entryPoints: [fileURLToPath(new URL("../src/main.ts", import.meta.url))],
    format: "cjs",
    platform: "node",
    plugins: [{
      name: "obsidian-test-stub",
      setup(build) {
        build.onResolve({ filter: /^obsidian$/ }, () => ({ namespace: "stub", path: "obsidian" }));
        build.onLoad({ filter: /.*/, namespace: "stub" }, () => ({
          contents: "export class ItemView {} export class Plugin {} export class TFile {} export class WorkspaceLeaf {}",
          loader: "js",
        }));
      },
    }],
    write: false,
  });
  const module = { exports: {} };
  new Function("module", "exports", result.outputFiles[0].text)(module, module.exports);
  return module.exports.default;
}

test("delete all persists its gate before removal and leaves failures retryable", async () => {
  const Plugin = await loadPlugin();
  const plugin = new Plugin();
  plugin.conversationState = {
    conversations: [{ id: "conversation-1", turns: [{ answer: "private" }] }],
    selectedConversationId: "conversation-1",
  };
  plugin.llmvaultSettings = {
    chatModel: "chat",
    embeddingModel: "embed",
    ollamaPort: 11434,
  };
  let cancelCalls = 0;
  let deleteCalls = 0;
  let failDeletion = false;
  plugin.index = {
    cancel() { cancelCalls += 1; },
    async deleteAll() {
      deleteCalls += 1;
      if (failDeletion) throw new Error("storage busy");
    },
  };
  const saves = [];
  let releaseFirstSave = () => undefined;
  let signalFirstSave = () => undefined;
  const firstSaveStarted = new Promise((resolve) => { signalFirstSave = resolve; });
  const firstSaveReleased = new Promise((resolve) => { releaseFirstSave = resolve; });
  plugin.saveData = async (data) => {
    saves.push(JSON.parse(JSON.stringify(data)));
    if (saves.length === 1) {
      signalFirstSave();
      await firstSaveReleased;
    }
  };

  const deletion = plugin.deleteAllData();
  await firstSaveStarted;

  assert.equal(cancelCalls, 1);
  assert.equal(deleteCalls, 0);
  assert.deepEqual(await plugin.retrieve("question"), []);
  assert.deepEqual(saves[0], {
    chatModel: "chat",
    conversations: [],
    embeddingModel: "embed",
    ollamaPort: 11434,
    selectedConversationId: null,
    vaultChatDeletionPending: true,
    vaultChatStopped: true,
  });

  releaseFirstSave();
  await deletion;

  assert.equal(deleteCalls, 1);
  assert.equal(plugin.isStopped(), true);
  assert.equal(plugin.isDeletionIncomplete(), false);
  assert.equal(saves.at(-1).vaultChatDeletionPending, false);

  failDeletion = true;
  await assert.rejects(plugin.deleteAllData(), /storage busy/);
  assert.equal(plugin.isDeletionIncomplete(), true);
  assert.equal(saves.at(-1).vaultChatDeletionPending, true);
  await assert.rejects(plugin.resumeVaultChat({}, "embed"), /deletion_incomplete/);

  failDeletion = false;
  await plugin.deleteAllData();
  assert.equal(plugin.isDeletionIncomplete(), false);
  assert.equal(saves.at(-1).vaultChatDeletionPending, false);
});

test("Daily Recap uses one metadata-link layer and bypasses semantic retrieval", async () => {
  const Plugin = await loadPlugin();
  const plugin = new Plugin();
  const files = [
    { path: "Labortagebuch/2026.09.02.md" },
    { path: "Notes/alpha.md" },
    { path: "Notes/beta.md" },
    { path: "Notes/second-hop.md" },
  ];
  const cacheReads = [];
  const resolvedLinks = [];
  plugin.app = {
    metadataCache: {
      getFileCache(file) {
        cacheReads.push(file.path);
        return file === files[0]
          ? { links: [{ link: "alpha" }, { link: "alpha" }, { link: "beta" }] }
          : { links: [{ link: "second-hop" }] };
      },
      getFirstLinkpathDest(link, sourcePath) {
        resolvedLinks.push([link, sourcePath]);
        return files.find((file) => file.path.endsWith(`/${link}.md`)) ?? null;
      },
    },
    vault: { getMarkdownFiles: () => files },
  };
  let semanticQueries = 0;
  let selectedPaths = [];
  plugin.index = {
    async retrieve() {
      semanticQueries += 1;
      return ["semantic"];
    },
    async retrievePaths(paths) {
      selectedPaths = [...new Set(paths)];
      return ["recap"];
    },
  };
  const now = new Date("2026-09-03T10:00:00Z");

  assert.deepEqual(
    await plugin.retrieve(
      "fasse mir zusammen was ich gestern gemacht habe",
      now,
      "Europe/Berlin",
    ),
    ["recap"],
  );
  assert.deepEqual(selectedPaths, [
    "Labortagebuch/2026.09.02.md",
    "Notes/alpha.md",
    "Notes/beta.md",
  ]);
  assert.deepEqual(cacheReads, ["Labortagebuch/2026.09.02.md"]);
  assert.deepEqual(resolvedLinks, [
    ["alpha", "Labortagebuch/2026.09.02.md"],
    ["alpha", "Labortagebuch/2026.09.02.md"],
    ["beta", "Labortagebuch/2026.09.02.md"],
  ]);
  assert.equal(semanticQueries, 0);
  assert.deepEqual(
    await plugin.retrieve("summary of today", now, "Europe/Berlin"),
    [],
  );
  files.push({ path: "Archive/2026.09.02.md" });
  assert.deepEqual(
    await plugin.retrieve("summary of yesterday", now, "Europe/Berlin"),
    [],
  );
  assert.equal(semanticQueries, 0);

  assert.deepEqual(
    await plugin.retrieve("what happened yesterday?", now, "Europe/Berlin"),
    ["semantic"],
  );
  assert.equal(semanticQueries, 1);
});

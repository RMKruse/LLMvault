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

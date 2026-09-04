import assert from "node:assert/strict";
import test from "node:test";
import { createContext, runInContext } from "node:vm";
import { createIndexChecks } from "../evaluation/index-acceptance.mjs";
import { groundedCase } from "../evaluation/grounded-acceptance.mjs";
import { installRecoveryWatch, rendererExpression, rendererSetup } from "../evaluation/obsidian-acceptance.mjs";
import { build } from "esbuild";
import { executeAnswer, MAX_PROMPT_BYTES } from "../src/answer.ts";
import { conversationUserMessage } from "../src/conversations.ts";
import { GROUNDING_SYSTEM_PROMPT } from "../src/quality.ts";

// Exercise the production views without launching Obsidian.
const { outputFiles } = await build({
  stdin: { contents: 'export { VaultChatView } from "./chat-view"; export { default } from "./main";', resolveDir: new URL("../src", import.meta.url).pathname, loader: "ts" },
  bundle: true, write: false, platform: "node", format: "cjs", external: ["obsidian"],
});
const module = { exports: {} };
new Function("require", "exports", "module", outputFiles[0].text)(
  () => ({ ItemView: class {}, Plugin: class {} }), module.exports, module,
);
const { VaultChatView, default: LLMvaultPlugin } = module.exports;

const deferred = () => Promise.withResolvers();
class Element {
  children = [];
  classList = { toggle() {} };
  addClass() {}
  focus() {}
  setText(text) { this.text = text; }
  setAttribute() {}
  querySelector() { return null; }
  querySelectorAll() { return []; }
  empty() { this.children = []; }
  createEl(tag, options = {}) {
    const child = Object.assign(new Element(), { tag, ...options });
    this.children.push(child);
    return child;
  }
  createDiv(options) { return this.createEl("div", options); }
  createSpan(options) { return this.createEl("span", options); }
  find(text) { return this.text === text ? this : this.children.map((child) => child.find(text)).find(Boolean); }
}
const discovery = { version: "0.33.2", installedModelCount: 2, chatModels: ["chat"], embeddingModels: ["embed"], remoteModels: [], modelDigests: {} };
const evidence = [{ citationId: "S1-1", text: "A fact", path: "note.md" }];

async function indexingPlugin() {
  const plugin = new LLMvaultPlugin();
  const files = new Map(), folders = new Set();
  const adapter = {
    async exists(path) { return files.has(path) || folders.has(path); },
    async mkdir(path) { folders.add(path); },
    async read(path) { if (!files.has(path)) throw new Error("missing"); return files.get(path); },
    async write(path, value) { files.set(path, value); },
    async process(path, update) { files.set(path, update(files.get(path) ?? "")); },
    async list(path) {
      const direct = (item) => item.startsWith(`${path}/`) && !item.slice(path.length + 1).includes("/");
      return { files: [...files.keys()].filter(direct), folders: [...folders].filter(direct) };
    },
    async rmdir(path) {
      for (const collection of [files, folders]) {
        for (const key of collection.keys()) if (key === path || key.startsWith(`${path}/`)) collection.delete(key);
      }
    },
  };
  plugin.loadData = async () => ({ ollamaPort: 11434, chatModel: "chat", embeddingModel: "embed" });
  plugin.saveData = async () => {};
  plugin.manifest = { dir: "plugin" };
  plugin.app = { vault: { adapter, on() {} }, workspace: { onLayoutReady() {} } };
  for (const method of ["registerView", "addCommand", "addRibbonIcon", "registerEvent"]) plugin[method] = () => {};
  plugin.vaultSources = () => [{ path: "Note.md", read: async () => "x".repeat(80_000) }];
  plugin.indexOllama.validatePinnedModel = async () => "compatible";
  await plugin.onload();
  return plugin;
}

test("settings changes during a paused embedding batch cannot commit a mixed generation", async () => {
  const plugin = await indexingPlugin();
  const entered = deferred(), pending = deferred(), ports = [];
  plugin.indexOllama.embed = async (port, model, inputs) => {
    ports.push(port);
    if (ports.length === 1) { entered.resolve(); await pending.promise; }
    return inputs.map(() => [1, 0]);
  };
  const build = plugin.startIndexing({ ...discovery, modelDigests: { embed: "original-digest" } }, "embed");
  await entered.promise;
  await plugin.saveSettings({ ollamaPort: 11435, chatModel: "chat", embeddingModel: null });
  pending.resolve();
  await build;
  assert.ok(ports.every((port) => port === 11434), `mixed connections: ${ports}`);
  assert.notEqual(plugin.getIndexSnapshot().phase, "ready");
  assert.equal(plugin.index.getSignature(), null);
});

test("failed setup stays in the form while every running batch uses its captured connection", async () => {
  for (const failedRole of ["completion", "embedding"]) {
    const plugin = await indexingPlugin();
    const entered = deferred(), pending = deferred(), ports = [];
    plugin.indexOllama.embed = async (port, model, inputs) => {
      ports.push(port);
      if (ports.length === 1) { entered.resolve(); await pending.promise; }
      return inputs.map(() => [1, 0]);
    };
    let writes = 0;
    plugin.saveData = async () => { writes += 1; };
    plugin.ollama.validateModel = async (port, model, role) => role === failedRole ? "incompatible" : "compatible";
    const build = plugin.startIndexing({ ...discovery, modelDigests: { embed: "original-digest" } }, "embed");
    await entered.promise;
    const view = new VaultChatView({}, plugin);
    view.management.portValue = "11435";
    view.management.discovery = discovery;
    view.management.renderSetup = () => {};
    await view.management.completeSetup();
    assert.equal(writes, 0);
    assert.deepEqual(plugin.getSettings(), { ollamaPort: 11434, chatModel: "chat", embeddingModel: "embed" });
    assert.equal(view.management[failedRole === "completion" ? "chatModel" : "embeddingModel"], null);
    pending.resolve();
    await build;
    assert.deepEqual(ports, [11434, 11434, 11434]);
    assert.equal(plugin.getIndexSnapshot().phase, "ready");
    assert.equal(plugin.index.getSignature().embeddingModelDigest, "original-digest");
  }
});

test("successful setup replaces a paused build only after durable settings publication", async () => {
  const plugin = await indexingPlugin();
  const entered = deferred(), pending = deferred(), saving = deferred(), saved = deferred();
  const ports = [], validations = [];
  plugin.indexOllama.validatePinnedModel = async (port, model, digest) => {
    validations.push([port, digest]);
    return "compatible";
  };
  plugin.indexOllama.embed = async (port, model, inputs) => {
    ports.push(port);
    if (ports.length === 1) { entered.resolve(); await pending.promise; }
    return inputs.map(() => [port === 11434 ? 1 : 0, 1]);
  };
  const build = plugin.startIndexing({ ...discovery, modelDigests: { embed: "original-digest" } }, "embed");
  await entered.promise;
  const revision = plugin.index.revision;
  plugin.saveData = async () => { saving.resolve(); await saved.promise; };
  plugin.ollama.validateModel = async () => "compatible";
  const view = new VaultChatView({}, plugin);
  view.management.portValue = "11435";
  view.management.discovery = { ...discovery, modelDigests: { embed: "new-digest" } };
  view.management.renderSetup = view.showChat = () => {};
  const setup = view.management.completeSetup();
  await saving.promise;
  assert.equal(plugin.getSettings().ollamaPort, 11434);
  saved.resolve();
  await setup;
  assert.ok(plugin.index.revision > revision);
  assert.equal(plugin.getSettings().ollamaPort, 11435);
  pending.resolve();
  await build;
  assert.deepEqual(ports, [11434, 11435, 11435, 11435]);
  assert.deepEqual(validations, [[11434, "original-digest"], [11435, "new-digest"]]);
  assert.equal(plugin.getIndexSnapshot().phase, "ready");
  assert.equal(plugin.index.getSignature().embeddingModelDigest, "new-digest");

  // Retrieval and file-change rebuilds retain the generation's connection, too.
  await plugin.saveSettings({ ollamaPort: 11436, chatModel: "chat", embeddingModel: null });
  plugin.queryOllama.validatePinnedModel = async (port, model, digest) => {
    assert.deepEqual([port, digest], [11435, "new-digest"]);
    return "compatible";
  };
  plugin.queryOllama.embed = async (port) => { assert.equal(port, 11435); return [[0, 1]]; };
  assert.ok((await plugin.index.retrieve("question")).length > 0);
  await plugin.index.invalidate(["Note.md"]);
  assert.deepEqual(ports.slice(-3), [11435, 11435, 11435]);
  assert.deepEqual(validations.at(-1), [11435, "new-digest"]);
});

test("a failed settings write preserves the running build and active connection", async () => {
  const plugin = await indexingPlugin();
  const entered = deferred(), pending = deferred(), ports = [];
  plugin.indexOllama.embed = async (port, model, inputs) => {
    ports.push(port);
    if (ports.length === 1) { entered.resolve(); await pending.promise; }
    return inputs.map(() => [1]);
  };
  const build = plugin.startIndexing({ ...discovery, modelDigests: { embed: "original-digest" } }, "embed");
  await entered.promise;
  plugin.saveData = async () => { throw new Error("write failed"); };
  await assert.rejects(plugin.saveSettings({ ollamaPort: 11435, chatModel: "chat", embeddingModel: null }), /write failed/);
  assert.equal(plugin.getSettings().ollamaPort, 11434);
  pending.resolve();
  await build;
  assert.deepEqual(ports, [11434, 11434, 11434]);
  assert.equal(plugin.getIndexSnapshot().phase, "ready");
});

test("startup discovery cannot start an old model on a newly saved connection", async () => {
  const plugin = await indexingPlugin();
  const entered = deferred(), pending = deferred();
  plugin.discoverModels = async () => { entered.resolve(); return pending.promise; };
  let starts = 0;
  plugin.startIndexing = async () => { starts += 1; };
  const restore = plugin.restoreIndex();
  await entered.promise;
  await plugin.saveSettings({ ollamaPort: 11435, chatModel: "chat", embeddingModel: "embed" });
  pending.resolve({ ...discovery, modelDigests: { embed: "old-digest" } });
  await restore;
  assert.equal(starts, 0);
});

test("production revalidation uses one batch and checks file availability and stopped state", async () => {
  let calls = 0, stopped = false;
  const missing = { ...evidence[0], path: "missing.md" };
  const plugin = {
    isStopped: () => stopped,
    evidenceFile: (item) => item.path === "missing.md" ? null : {},
    index: {
      async resolveEvidenceBatch(items) { calls += 1; return items; },
    },
  };
  const revalidate = (items) => LLMvaultPlugin.prototype.revalidateEvidence.call(plugin, items);
  assert.deepEqual(await revalidate([...evidence, missing]), evidence);
  assert.equal(calls, 1);
  stopped = true;
  assert.deepEqual(await revalidate(evidence), []);
  assert.equal(calls, 1);
  stopped = false;
  plugin.index.resolveEvidenceBatch = async (items) => { stopped = true; return items; };
  assert.deepEqual(await revalidate(evidence), []);
});

function harness() {
  const saved = [];
  const plugin = {
    getSettings: () => ({ ollamaPort: 11434, chatModel: "chat", embeddingModel: "embed" }),
    isStopped: () => false,
    isDeletionIncomplete: () => false,
    getConversationState: () => ({ conversations: [], selectedConversationId: null }),
    getConversationTurns: () => [],
    getIndexSnapshot: () => ({ available: true }),
    dailyRecapRequest: () => null,
    abortAnswerRequests: LLMvaultPlugin.prototype.abortAnswerRequests,
    chatOllama: { abortAll() {} }, queryOllama: { abortAll() {} },
    abortOllamaRequests() {},
    answerQuestion: LLMvaultPlugin.prototype.answerQuestion,
    async validateChatModel() {},
    async retrieve() { return evidence; },
    async revalidateEvidence(items) { return items; },
    async chat(messages, onContent) { onContent("An answer"); return { content: "An answer" }; },
    async saveCompletedTurn(id, turn) { saved.push(turn); },
    async discoverModels() { return discovery; },
  };
  const view = new VaultChatView({}, plugin);
  view.questionEl = { value: "Question?" };
  view.renderAnswer = (text) => { view.renderedAnswer = text; };
  view.renderInsufficient = view.renderAnswer;
  view.renderIncomplete = () => view.renderAnswer("Incomplete");
  view.renderEvidence = () => {};
  view.renderHistory = () => {};
  view.management.setupEl = new Element();
  view.contentEl = new Element();
  return { view, plugin, saved };
}

test("discovery finishes after a question and releases setup controls", async () => {
  const { view, plugin, saved } = harness();
  const pending = deferred();
  plugin.discoverModels = () => pending.promise;
  const refresh = view.management.refreshModels();
  assert.equal(view.management.setupEl.find("Checking…").disabled, true);
  await view.askQuestion();
  pending.resolve(discovery);
  await refresh;
  assert.equal(saved.length, 1);
  assert.equal(view.answering, false);
  assert.equal(view.management.busy, false);
  assert.equal(view.management.setupEl.find("Complete setup").disabled, false);
});

test("historical evidence resolves in one batch and retains unavailable citations", async () => {
  const { view, plugin } = harness();
  const missing = { ...evidence[0], citationId: "S1-2", path: "missing.md" };
  const stored = [...evidence, missing];
  plugin.getConversationState = () => ({
    selectedConversationId: "saved",
    conversations: [{ id: "saved", turns: [{ evidence: stored, question: "Question", answer: "Answer", kind: "answer" }] }],
  });
  let batches = 0;
  plugin.revalidateEvidence = async (items) => { batches += 1; assert.equal(items, stored); return evidence; };
  view.renderEvidence = (items, unavailable) => {
    if (items.length === 0) return;
    assert.deepEqual(items, stored);
    assert.deepEqual([...unavailable], [missing.citationId]);
  };
  await view.renderSelectedConversation();
  assert.equal(batches, 1);
});

test("rebuild progress does not reread historical evidence", async () => {
  const plugin = await indexingPlugin();
  let reads = 0;
  plugin.vaultSources = () => [{ path: "Note.md", read: async () => { reads += 1; return "A saved fact"; } }];
  plugin.indexOllama.embed = async (_port, _model, inputs) => inputs.map(() => [1]);
  await plugin.startIndexing({ ...discovery, modelDigests: { embed: "digest" } }, "embed");
  const stored = await plugin.index.retrievePaths(["Note.md"]);
  assert.equal(stored.length, 1);
  plugin.getConversationState = () => ({
    selectedConversationId: "saved",
    conversations: [{ id: "saved", title: "Saved", turns: [{ evidence: stored, answer: "A saved fact", kind: "answer" }] }],
  });
  const restorations = [];
  plugin.revalidateEvidence = (items) => {
    const result = plugin.index.resolveEvidenceBatch(items);
    restorations.push(result);
    return result;
  };
  const view = new VaultChatView({}, plugin);
  view.contentEl = new Element();
  view.management.refreshModels = async () => {};
  await view.onOpen();
  await Promise.all(restorations);
  assert.equal(restorations.length, 1, "opening restores once, including the initial subscription snapshot");
  const before = reads;
  const snapshot = plugin.getIndexSnapshot();
  for (let completed = 1; completed <= 1_000; completed += 1) {
    plugin.reportIndex({
      ...snapshot, phase: "indexing", total: 1_000, completed,
      get outcomes() { assert.fail("progress must not materialize file details"); },
    });
  }
  await Promise.all(restorations);
  assert.equal(reads, before, "progress must not prepare historical sources again");
  assert.equal(restorations.length, 1);
  plugin.reportIndex({ ...snapshot, phase: "failed" });
  assert.equal(restorations.length, 1, "failed rebuilds keep the same available evidence");

  // A real rebuild reads the build source once and the activated history once.
  await plugin.startIndexing({ ...discovery, modelDigests: { embed: "digest" } }, "embed", true);
  await Promise.all(restorations);
  assert.notEqual(plugin.getIndexSnapshot().generationId, snapshot.generationId);
  assert.equal(restorations.length, 2, "activation refreshes history once");
  assert.equal(reads, before + 2);

  for (const subscriber of plugin.mutationSubscribers) subscriber(new Set(["Other.md"]));
  assert.equal(restorations.length, 2, "unrelated source mutations do not restore history");
  for (const subscriber of plugin.mutationSubscribers) subscriber(new Set(["Note.md"]));
  await Promise.all(restorations);
  assert.equal(restorations.length, 3, "a relevant mutation refreshes history");
  await view.renderSelectedConversation();
  assert.equal(restorations.length, 4, "explicit selection still revalidates");
  await view.onClose();
});

test("index counters leave file details untouched until visible and expanded", () => {
  const { view, plugin } = harness();
  let outcomeReads = 0;
  let snapshot = {
    available: true, phase: "indexing", total: 1_000, completed: 1,
    statuses: { unsupported_format: 1 },
    get outcomes() {
      outcomeReads += 1;
      return [{ path: "Image.png", status: "unsupported_format" }];
    },
  };
  plugin.getIndexSnapshot = () => snapshot;
  view.renderShell();
  const management = view.management;
  const details = management.outcomeDetails;
  for (let completed = 1; completed <= 1_000; completed += 1) {
    snapshot.completed = completed;
    management.renderIndex();
  }
  assert.equal(outcomeReads, 0, "hidden management does not read file details");
  management.setVisible(true);
  assert.match(management.indexStatusEl.text, /Indexing 1000 of 1000/);
  assert.equal(outcomeReads, 0, "collapsed details need only counters");
  details.open = true;
  details.ontoggle();
  assert.equal(outcomeReads, 1);
  assert.equal(management.outcomeList.children.length, 1);
  details.open = false;
  details.ontoggle();
  const row = management.outcomeList.children[0];
  management.renderIndex();
  assert.equal(management.outcomeDetails, details, "counter updates preserve disclosure state and focus");
  assert.equal(management.outcomeList.children[0], row);
  assert.equal(outcomeReads, 1);
});

test("menu Chat preserves the streaming draft and evidence against late historical restoration", async () => {
  const { view, plugin, saved } = harness();
  const historical = [{ ...evidence[0], path: "old.md", text: "Old fact" }];
  const pending = deferred(), streaming = deferred(), finish = deferred();
  const restorations = [];
  plugin.getConversationState = () => ({
    selectedConversationId: "saved",
    conversations: [{ id: "saved", title: "Saved", turns: [{
      evidence: historical, question: "Old question", answer: "Old answer", kind: "answer",
    }] }],
  });
  plugin.getIndexSnapshot = () => ({ available: true, phase: "ready", statuses: {}, outcomes: [], total: 1 });
  plugin.revalidateEvidence = (items) => {
    if (items !== historical) return Promise.resolve(items);
    const restore = pending.promise.then(() => historical);
    restorations.push(restore);
    return restore;
  };
  plugin.chat = async (_messages, onContent) => {
    onContent("Draft [S1-1]");
    streaming.resolve();
    await finish.promise;
    onContent(" finished");
    return { content: "Draft [S1-1] finished" };
  };
  view.renderHistory = VaultChatView.prototype.renderHistory;
  view.renderEvidence = VaultChatView.prototype.renderEvidence;
  view.renderAnswer = (text, ...args) => {
    view.renderedAnswer = text;
    VaultChatView.prototype.renderAnswer.call(view, text, ...args);
  };
  view.renderShell(); // Starts a historical restore before the draft takes ownership.
  view.questionEl.value = "New question";
  const answer = view.askQuestion();
  await streaming.promise;
  view.contentEl.find("☰").onclick();
  view.historyEl.find("Chat").onclick();
  pending.resolve();
  await Promise.all(restorations);
  try {
    assert.equal(view.answering, true);
    assert.equal(view.renderedAnswer, "Draft [S1-1]");
    assert.equal(view.citationRegistry.get("S1-1").path, "note.md");
    let opened;
    plugin.resolveEvidence = async (item) => { opened = item; return item; };
    view.answerEl.find("S1-1").onclick();
    assert.equal(opened.path, "note.md", "the draft citation still opens its own evidence");
    assert.equal(restorations.length, 1, "Chat must not start a restore over a draft");
  } finally {
    finish.resolve();
    await answer;
  }
  assert.equal(view.renderedAnswer, "Draft [S1-1] finished");
  assert.equal(view.citationRegistry.get("S1-1").path, "note.md");
  assert.equal(saved.length, 1);
});

test("leaving a saved display invalidates deferred restoration at every navigation boundary", async () => {
  for (const destination of ["management", "close", "new", "resume", "delete", "delete all"]) {
    const { view, plugin } = harness();
    const pending = deferred();
    let state = {
      selectedConversationId: "saved",
      conversations: [{ id: "saved", title: "Saved", turns: [{ evidence, answer: "Old", kind: "answer" }] }],
    };
    plugin.getConversationState = () => state;
    plugin.revalidateEvidence = () => pending.promise;
    const restore = view.renderSelectedConversation();
    const selectEmpty = async () => { state = { selectedConversationId: null, conversations: [] }; };
    plugin.startNewConversation = plugin.selectConversation = plugin.deleteConversation = plugin.deleteAllData = selectEmpty;
    const navigation = {
      management: () => view.showManagement(),
      close: () => view.onClose(),
      new: () => view.startNewConversation(),
      resume: () => view.resumeConversation("empty"),
      delete: () => view.removeConversation("saved"),
      "delete all": () => view.management.deleteAllData(),
    };
    await navigation[destination]();
    const displayed = view.renderedAnswer;
    pending.resolve(evidence);
    await restore;
    assert.equal(view.renderedAnswer, displayed, destination);
    assert.equal(view.answering, false, destination);
  }
});

test("pending conversation selection neither restores early nor overwrites a newer draft", async () => {
  const { view, plugin } = harness();
  const selection = deferred(), streaming = deferred(), finish = deferred();
  plugin.selectConversation = () => selection.promise;
  let restores = 0;
  plugin.revalidateEvidence = async (items) => { restores += 1; return items; };
  const resume = view.resumeConversation("selected");
  view.showChat();
  assert.equal(restores, 0);
  plugin.chat = async (_messages, onContent) => {
    onContent("Draft");
    streaming.resolve();
    await finish.promise;
    return { content: "Draft" };
  };
  const answer = view.askQuestion();
  await streaming.promise;
  selection.resolve();
  await resume;
  assert.equal(view.renderedAnswer, "Draft");
  assert.equal(view.answering, true);
  finish.resolve();
  await answer;
});

test("production saves lowercase insufficient_evidence as abstention", async () => {
  const { view, plugin, saved } = harness();
  plugin.chat = async () => ({ content: "insufficient_evidence: No support." });
  await view.askQuestion();
  assert.equal(saved[0].kind, "insufficient");
  assert.equal(saved[0].answer, "No support.");
});


test("discovery can finish during streaming without canceling the answer", async () => {
  const { view, plugin, saved } = harness();
  const started = deferred(), pending = deferred();
  plugin.chat = async (messages, onContent) => {
    started.resolve();
    await pending.promise;
    onContent("Finished");
    return { content: "Finished" };
  };
  const answer = view.askQuestion();
  await started.promise;
  await view.management.refreshModels();
  assert.equal(view.answering, true);
  assert.equal(view.management.setupEl.find("Complete setup").disabled, false);
  pending.resolve();
  await answer;
  assert.equal(saved[0].answer, "Finished");
  assert.equal(view.answering, false);
});

test("cancellation at each answer boundary suppresses late output and persistence without stopping discovery", async () => {
  for (const stage of ["validateChatModel", "retrieve", "revalidateEvidence", "chat"]) {
    const { view, plugin, saved } = harness();
    const entered = deferred(), pending = deferred(), discovered = deferred();
    const original = plugin[stage];
    plugin[stage] = async (...args) => {
      entered.resolve();
      await pending.promise;
      return original(...args);
    };
    plugin.discoverModels = () => discovered.promise;
    const refresh = view.management.refreshModels();
    const answer = view.askQuestion();
    await entered.promise;
    view.cancelAnswer();
    pending.resolve();
    await answer;
    assert.equal(view.renderedAnswer, "Incomplete", stage);
    assert.equal(saved.length, 0, stage);
    assert.equal(view.management.busy, true, stage);
    discovered.resolve(discovery);
    await refresh;
    assert.equal(view.management.setupEl.find("Complete setup").disabled, false, stage);
  }
});

test("an older answer cannot complete or clear the newer answer's ownership", async () => {
  const { view, plugin, saved } = harness();
  const entered = deferred(), old = deferred(), current = deferred();
  plugin.chat = async (messages, onContent) => {
    entered.resolve();
    await old.promise;
    onContent("Old");
    return { content: "Old" };
  };
  const first = view.askQuestion();
  await entered.promise;
  plugin.chat = async () => { await current.promise; return { content: "Current" }; };
  const second = view.askQuestion();
  old.resolve();
  await first;
  assert.equal(view.answering, true);
  assert.equal(saved.length, 0);
  current.resolve();
  await second;
  assert.equal(saved[0].answer, "Current");
});

test("closing the view suppresses late setup UI and unsubmitted configuration", async () => {
  for (const stage of ["discoverModels", "validateModel", "saveSettings"]) {
    const { view, plugin } = harness();
    const entered = deferred(), pending = deferred();
    let applied = false, shown = false;
    plugin.validateModel = async () => "compatible";
    plugin.saveSettings = async () => { applied = true; };
    const original = plugin[stage];
    plugin[stage] = async (...args) => { entered.resolve(); await pending.promise; return original(...args); };
    view.management.discovery = discovery;
    view.showChat = () => { shown = true; };
    const setup = stage === "discoverModels" ? view.management.refreshModels() : view.management.completeSetup();
    await entered.promise;
    await view.onClose();
    const statusAtClose = view.management.status;
    pending.resolve();
    await setup;
    assert.equal(view.management.status, statusAtClose, stage);
    assert.equal(view.management.busy, false, stage);
    assert.equal(shown, false, stage);
    assert.equal(applied, stage === "saveSettings", stage);
  }
});

test("shared execution and production persistence agree on answer classification", async () => {
  for (const content of ["INSUFFICIENT_EVIDENCE: No support.", "insufficient_evidence: No support.", "  InSuFfIcIeNt_EvIdEnCe:  ", " ", "A fact [S1-1]"]) {
    const { view, plugin, saved } = harness();
    plugin.chat = async () => ({ content });
    const result = await executeAnswer(plugin, {
      question: "Question?", requestedAt: new Date(0), timeZone: "UTC", history: [], signal: new AbortController().signal,
    });
    await view.askQuestion();
    assert.equal(result.status, "complete");
    assert.deepEqual(saved[0], result.turn);
    assert.equal(saved[0].kind, content === "A fact [S1-1]" ? "answer" : "insufficient");
  }
});

test("shared execution assembles history, Daily Recap context and streams accumulated content", async () => {
  const { plugin } = harness();
  const history = [{ question: "Earlier question", answer: "Earlier answer", kind: "answer", evidence: [], status: "complete", completedAt: 0 }];
  const requestedAt = new Date(0), updates = [];
  plugin.retrieve = async (question, now, timeZone) => {
    assert.equal(now, requestedAt);
    assert.equal(timeZone, "Europe/Berlin");
    return evidence;
  };
  plugin.chat = async (messages, onContent) => {
    assert.deepEqual(messages.slice(0, 3), [
      { role: "system", content: GROUNDING_SYSTEM_PROMPT },
      conversationUserMessage("Earlier question", []), { role: "assistant", content: "Earlier answer" },
    ]);
    assert.match(messages[3].content, /Resolved Daily Recap date: 2026-09-02/);
    assert.match(messages[3].content, /UNTRUSTED_EVIDENCE_JSON/);
    onContent("A "); onContent("fact");
    return { content: "A fact", diagnostics: { eval_count: 2 } };
  };
  const result = await executeAnswer(plugin, {
    question: "Yesterday?", requestedAt, timeZone: "Europe/Berlin", history,
    dailyRecapDate: "2026-09-02", signal: new AbortController().signal,
    onContent: (content) => updates.push(content),
  });
  assert.deepEqual(updates, ["", "A ", "A fact"]);
  assert.equal(result.chatResult.diagnostics.eval_count, 2);
});

test("long saved conversations keep their archive while model messages stay within 32 KiB", async () => {
  const plugin = new LLMvaultPlugin();
  const turns = Array.from({ length: 50 }, (_, i) => ({
    question: `Question ${i}`, answer: `Answer ${i}`, kind: "answer", status: "complete", completedAt: i,
    evidence: Array.from({ length: 6 }, (_, j) => ({
      citationId: `S${i + 1}-${j + 1}`, chunkId: `chunk-${j}`, fingerprint: "source", score: 1,
      format: "markdown", path: "note.md", start: 0, end: 2048, startLine: 1, endLine: 1,
      text: "x".repeat(2048),
    })),
  }));
  const archive = { conversations: [{ id: "saved", title: "Saved", createdAt: 0, updatedAt: 49, turns }], selectedConversationId: "saved" };
  plugin.conversationState = structuredClone(archive);
  let persisted, dispatched;
  plugin.saveData = async (data) => { persisted = structuredClone(data); };
  plugin.validateChatModel = async () => {};
  plugin.retrieve = async () => turns[0].evidence;
  plugin.revalidateEvidence = async (items) => items;
  plugin.dailyRecapRequest = () => null;
  plugin.chat = async (messages) => { dispatched = messages; return { content: "Current answer" }; };
  await plugin.answerQuestion("Current question", "saved", {});
  assert.deepEqual(plugin.getConversationState().conversations[0].turns.slice(0, 50), turns);
  assert.ok(persisted, "the completed answer must be durably saved");
  assert.deepEqual(persisted.conversations[0].turns.slice(0, 50), turns);
  assert.equal(plugin.getConversationState().conversations[0].turns.length, 51);
  const bytes = Buffer.byteLength(JSON.stringify(dispatched));
  assert.ok(bytes <= 32 * 1024, `model messages used ${bytes} bytes`);
  assert.equal(dispatched[0].content, GROUNDING_SYSTEM_PROMPT);
  assert.equal(dispatched.at(-2).content, "Answer 49");
  assert.match(dispatched.at(-1).content, /^Current question/);
  assert.deepEqual(dispatched.at(-1), conversationUserMessage("Current question", turns[0].evidence));
});

test("prompt budget counts UTF-8 and JSON escaping and rejects oversized current requests", async () => {
  const { plugin } = harness();
  const history = Array.from({ length: 30 }, (_, i) => ({
    question: `Earlier ${i}: ${'😀\n"\\'.repeat(160)}`, answer: `Answer ${i}`, kind: "answer",
    evidence: [], status: "complete", completedAt: i,
  }));
  const request = { question: "Current", requestedAt: new Date(0), timeZone: "UTC", history, signal: new AbortController().signal };
  let dispatched;
  plugin.chat = async (messages) => { dispatched = messages; return { content: "Answer" }; };
  await executeAnswer(plugin, request);
  assert.ok(Buffer.byteLength(JSON.stringify(dispatched)) <= MAX_PROMPT_BYTES);
  assert.ok(dispatched.length > 4 && dispatched.length < 62);
  const suffix = dispatched.slice(1, -1);
  const included = history.slice(-suffix.length / 2);
  assert.deepEqual(suffix, included.flatMap((turn) => [
    conversationUserMessage(turn.question, []), { role: "assistant", content: turn.answer },
  ]));

  const base = [{ role: "system", content: GROUNDING_SYSTEM_PROMPT }, conversationUserMessage("", evidence)];
  request.question = "x".repeat(MAX_PROMPT_BYTES - Buffer.byteLength(JSON.stringify(base)));
  await executeAnswer(plugin, request);
  assert.equal(dispatched.length, 2, "current evidence consumes the history allowance");
  assert.equal(Buffer.byteLength(JSON.stringify(dispatched)), MAX_PROMPT_BYTES);
  dispatched = undefined;
  request.question += "x";
  await assert.rejects(executeAnswer(plugin, request), { code: "context_budget_exceeded" });
  assert.equal(dispatched, undefined);

  request.question = "Current";
  plugin.retrieve = async () => [{ ...evidence[0], text: "x".repeat(MAX_PROMPT_BYTES) }];
  await assert.rejects(executeAnswer(plugin, request), { code: "context_budget_exceeded" });
  assert.equal(dispatched, undefined);
});

test("empty evidence abstains; changed evidence and failed streams never save a turn", async () => {
  for (const scenario of ["empty", "changed", "interrupted", "save failed"]) {
    const { view, plugin, saved } = harness();
    let dispatched = false;
    plugin.chat = async () => {
      dispatched = true;
      if (scenario === "interrupted") throw new Error("stream interrupted");
      return { content: "A fact" };
    };
    if (scenario === "empty") plugin.retrieve = async () => [];
    if (scenario === "changed") plugin.revalidateEvidence = async () => [];
    if (scenario === "save failed") plugin.saveCompletedTurn = async () => { throw new Error("write failed"); };
    await view.askQuestion();
    assert.equal(saved.length, scenario === "empty" ? 1 : 0, scenario);
    assert.equal(dispatched, scenario === "interrupted" || scenario === "save failed", scenario);
    assert.equal(view.answering, false, scenario);
    if (scenario === "empty") assert.equal(saved[0].kind, "insufficient");
  }
});

test("a canceled answer waiting behind a data write is not persisted", async () => {
  const plugin = new LLMvaultPlugin();
  const pending = deferred();
  plugin.dataWrite = pending.promise;
  let writes = 0;
  plugin.saveData = async () => { writes += 1; };
  const controller = new AbortController();
  const save = plugin.saveCompletedTurn(null, {
    question: "Question?", answer: "A fact", evidence: [], kind: "answer",
  }, controller.signal);
  controller.abort();
  pending.resolve();
  await assert.rejects(save, { name: "AbortError" });
  assert.equal(writes, 0);
  assert.equal(plugin.getConversationState().conversations.length, 0);
});


test("serialized acceptance renderers use the current index contract without Ollama or Obsidian", { timeout: 5_000 }, async (t) => {
  const plugin = await indexingPlugin();
  const probePath = "2026.09.02.md", controlPath = "Linked.md";
  const contents = new Map([[probePath, "# Acceptance probe\n\noriginal\n"], [controlPath, "Unrelated control"]]);
  const { app } = plugin;
  let rebuilding;
  app.vault.getMarkdownFiles = () => [...contents.keys()].map((path) => ({ path }));
  app.vault.getAbstractFileByPath = (path) => contents.has(path) ? { path } : null;
  app.vault.cachedRead = async ({ path }) => contents.get(path);
  app.vault.modify = async (file, content) => {
    contents.set(file.path, content);
    rebuilding = plugin.index.invalidate([file.path]);
  };
  app.metadataCache = {
    getFileCache: () => ({ links: [{ link: controlPath }] }),
    getFirstLinkpathDest: (path) => app.vault.getAbstractFileByPath(path),
  };
  plugin.vaultSources = () => app.vault.getMarkdownFiles().map((file) => ({
    path: file.path, read: () => app.vault.cachedRead(file),
  }));
  plugin.indexOllama.embed = async (_port, _model, inputs) => inputs.map(() => [1]);
  plugin.queryOllama.validatePinnedModel = async () => "compatible";
  plugin.queryOllama.embed = async (_port, _model, inputs) => inputs.map(() => [1]);
  plugin.revalidateEvidence = (evidence) => plugin.index.resolveEvidenceBatch(evidence);
  plugin.discoverModels = async () => ({ ...discovery, modelDigests: { embed: "digest" } });
  plugin.openVaultChat = async () => {};
  await plugin.startIndexing(await plugin.discoverModels(), "embed");
  assert.equal(plugin.getIndexSnapshot().phase, "ready");

  // Bind public methods to the real instance, but reject renderer reads of private state.
  const index = plugin.index;
  plugin.index = new Proxy(index, {
    get(target, key) {
      assert.ok(!["active", "storage", "adapter"].includes(key), `private renderer access: ${String(key)}`);
      const value = Reflect.get(target, key);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
  const button = { attributes: {}, focus() {}, setAttribute(key, value) { this.attributes[key] = value; }, getAttribute(key) { return this.attributes[key]; } };
  let questionAsked = false, answerError;
  const view = {
    citationRegistry: new Map(), answering: false,
    contentEl: { querySelector: () => button },
    answerEl: { querySelectorAll: () => [] },
    questionEl: { value: "", form: { requestSubmit() {
      questionAsked = true;
      view.answering = true;
      void plugin.retrieve(view.questionEl.value).then((evidence) => {
        view.citationRegistry = new Map(evidence.map((item) => [item.citationId, item]));
      }).catch((error) => { answerError = error; }).finally(() => { view.answering = false; });
    } } },
  };
  app.workspace = { onLayoutReady: (ready) => ready(), getLeavesOfType: () => [{ view }] };
  app.plugins = { loadPlugin() {}, manifests: { llmvault: {} }, plugins: { llmvault: plugin } };
  const timers = new Set();
  const context = createContext({
    app, TextEncoder, URL, performance, crypto: globalThis.crypto,
    localStorage: { setItem() {} },
    setTimeout: (...args) => { const timer = setTimeout(...args); timers.add(timer); return timer; },
    clearTimeout,
    setInterval: (...args) => { const timer = setInterval(...args); timers.add(timer); return timer; },
    clearInterval,
    fetch: async () => ({ json: async () => ({ models: [] }) }),
  });
  t.after(() => { for (const timer of timers) { clearTimeout(timer); clearInterval(timer); } });
  const evaluate = (fn, ...args) => runInContext(rendererExpression(fn, ...args), context);
  const raw = await evaluate(groundedCase, "Summarize yesterday.", false);
  assert.ifError(answerError);
  assert.equal(questionAsked, true);
  assert.equal(raw.dailyTargetPass, true);
  assert.deepEqual(Array.from(raw.expectedEvidence, ({ path, fingerprint }) => ({ path, fingerprint })),
    Array.from(raw.evidence, ({ path, fingerprint }) => ({ path, fingerprint })));
  assert.deepEqual(Array.from(raw.expectedEvidence, ({ path }) => path), [probePath, controlPath]);
  assert.equal(raw.pathSelections.length, 1, "evidence inspection must not pollute question observations");
  assert.equal(raw.queryEmbeddings, 0);

  await evaluate(rendererSetup, { probePath, controlPath, controlToken: "Unrelated control", chatModel: "chat", embeddingModel: "embed" });
  const harness = context.__llmvaultAcceptance;
  const before = await harness.canonicalDigest();
  const mutation = await harness.mutation("original", "replacement");
  assert.ok(Number.isFinite(mutation.mutation.queryIneligibleMs));
  assert.ok(Number.isFinite(mutation.mutation.replacementAfterEmbeddingMs));
  assert.ok(mutation.ui.samples > 0);
  await rebuilding;
  const changed = await harness.canonicalDigest();
  assert.notEqual(changed, before);
  await plugin.rebuildIndex();
  assert.equal(await harness.canonicalDigest(), changed, "digest ignores generation identity");
  const failures = await harness.failures("recovered");
  assert.ok(Object.values(failures).every(Boolean), JSON.stringify(failures));
  await rebuilding;

  const checks = createIndexChecks(plugin.index, app.vault.adapter, "plugin/index-v1");
  const [current] = await checks.evidence([probePath]);
  const expected = { oldPath: probePath, path: probePath, fingerprint: current.fingerprint, token: "recovered" };
  await evaluate(installRecoveryWatch, expected);
  assert.equal((await harness.validate(expected)).exposedInvalidContent, false);
  await evaluate(installRecoveryWatch, { ...expected, fingerprint: "wrong" });
  assert.equal((await harness.validate(expected)).exposedInvalidContent, true, "watch detects wrong evidence");

  for (const boundary of ["record", "catalog", "pointer", "cleanup"]) {
    const reached = deferred(), release = deferred();
    const restore = checks.armCrash(boundary, async (event) => { reached.resolve(event); await release.promise; });
    try {
      const build = plugin.rebuildIndex();
      const event = await reached.promise;
      assert.equal(event.boundary, boundary);
      assert.equal(event.method, boundary === "pointer" ? "process" : boundary === "cleanup" ? "rmdir" : "write");
      if (boundary === "cleanup") assert.equal(await app.vault.adapter.exists(event.path), false);
      else assert.ok(await app.vault.adapter.read(event.path), "hook follows the durable write");
      release.resolve();
      await build;
      assert.equal(plugin.getIndexSnapshot().phase, "ready");
    } finally { release.resolve(); restore(); }
  }
  const reached = deferred();
  context.llmvaultBoundary = (payload) => reached.resolve(JSON.parse(payload));
  await harness.armCrash("record", { kind: "modify", path: probePath, token: "crash" });
  assert.equal((await reached.promise).boundary, "record", "actual renderer installs its hook on the vault adapter");
});

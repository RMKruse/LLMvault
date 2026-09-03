import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { build } from "esbuild";
import { executeAnswer } from "../src/answer.ts";
import { GROUNDING_SYSTEM_PROMPT } from "../src/quality.ts";

// Exercise the production view without launching Obsidian or exposing test exports.
const source = await readFile(new URL("../src/main.ts", import.meta.url), "utf8");
const { outputFiles } = await build({
  stdin: { contents: `${source}\nexport { VaultChatView };`, resolveDir: new URL("../src", import.meta.url).pathname, loader: "ts" },
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
  empty() { this.children = []; }
  createEl(tag, options = {}) {
    const child = Object.assign(new Element(), { tag, ...options });
    this.children.push(child);
    return child;
  }
  createDiv(options) { return this.createEl("div", options); }
  find(text) { return this.text === text ? this : this.children.map((child) => child.find(text)).find(Boolean); }
}
const discovery = { version: "0.33.2", installedModelCount: 2, chatModels: ["chat"], embeddingModels: ["embed"], remoteModels: [], modelDigests: {} };
const evidence = [{ citationId: "S1-1", text: "A fact", path: "note.md" }];

function harness() {
  const saved = [];
  const plugin = {
    getSettings: () => ({ ollamaPort: 11434, chatModel: "chat", embeddingModel: "embed" }),
    isStopped: () => false,
    isDeletionIncomplete: () => false,
    getConversationState: () => ({ conversations: [], selectedConversationId: null }),
    getConversationMessages: () => [],
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
  view.setupEl = new Element();
  view.contentEl = new Element();
  return { view, plugin, saved };
}

test("discovery finishes after a question and releases setup controls", async () => {
  const { view, plugin, saved } = harness();
  const pending = deferred();
  plugin.discoverModels = () => pending.promise;
  const refresh = view.refreshModels();
  assert.equal(view.setupEl.find("Checking…").disabled, true);
  await view.askQuestion();
  pending.resolve(discovery);
  await refresh;
  assert.equal(saved.length, 1);
  assert.equal(view.answering, false);
  assert.equal(view.busy, false);
  assert.equal(view.setupEl.find("Complete setup").disabled, false);
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
  await view.refreshModels();
  assert.equal(view.answering, true);
  assert.equal(view.setupEl.find("Complete setup").disabled, false);
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
    const refresh = view.refreshModels();
    const answer = view.askQuestion();
    await entered.promise;
    view.cancelAnswer();
    pending.resolve();
    await answer;
    assert.equal(view.renderedAnswer, "Incomplete", stage);
    assert.equal(saved.length, 0, stage);
    assert.equal(view.busy, true, stage);
    discovered.resolve(discovery);
    await refresh;
    assert.equal(view.setupEl.find("Complete setup").disabled, false, stage);
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

test("closing the view invalidates setup, including completion after a settings save", async () => {
  for (const stage of ["discoverModels", "validateModel", "saveSettings", "resumeVaultChat"]) {
    const { view, plugin } = harness();
    const entered = deferred(), pending = deferred();
    let resumed = false, shown = false;
    plugin.validateModel = async () => "compatible";
    plugin.saveSettings = async () => {};
    plugin.resumeVaultChat = async () => { resumed = true; };
    const original = plugin[stage];
    plugin[stage] = async (...args) => { entered.resolve(); await pending.promise; return original(...args); };
    view.discovery = discovery;
    view.showChat = () => { shown = true; };
    const setup = stage === "discoverModels" ? view.refreshModels() : view.completeSetup();
    await entered.promise;
    await view.onClose();
    const statusAtClose = view.status;
    pending.resolve();
    await setup;
    assert.equal(view.status, statusAtClose, stage);
    assert.equal(view.busy, false, stage);
    assert.equal(shown, false, stage);
    if (stage !== "resumeVaultChat") assert.equal(resumed, false, stage);
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
  const history = [{ role: "user", content: "Earlier question" }, { role: "assistant", content: "Earlier answer" }];
  const requestedAt = new Date(0), updates = [];
  plugin.retrieve = async (question, now, timeZone) => {
    assert.equal(now, requestedAt);
    assert.equal(timeZone, "Europe/Berlin");
    return evidence;
  };
  plugin.chat = async (messages, onContent) => {
    assert.deepEqual(messages.slice(0, 3), [{ role: "system", content: GROUNDING_SYSTEM_PROMPT }, ...history]);
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

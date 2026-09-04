import { createIndexChecks } from "./index-acceptance.mjs";
import { conversationUserMessage } from "../src/conversations.ts";
import { GROUNDING_SYSTEM_PROMPT } from "../src/quality.ts";
import { hash, observeRequest, retrievalIdentity, same, supportBinding } from "./proof.mjs";

export const QUESTIONS = {
  daily: "fasse mir zusammen was ich gestern gemacht habe",
  direct: "Why is Stein’s SURE a local method?",
};

// Executed inside disposable Obsidian. Only the clock and proof sampling are fixed;
// the production view still validates, retrieves, assembles, streams and saves the turn.
export async function groundedCase(question, history, indexChecks = createIndexChecks) {
  const app = globalThis.app;
  const plugin = app.plugins.plugins.llmvault;
  const view = app.workspace.getLeavesOfType("vault-chat-view")[0].view;
  const observations = { requests: [], retrieved: [], queryEmbeddings: 0, pathSelections: [] };
  const originalRecap = plugin.dailyRecapRequest.bind(plugin);
  plugin.dailyRecapRequest = (input) => originalRecap(input, new Date("2026-09-03T10:00:00Z"), "Europe/Berlin");
  const recap = plugin.dailyRecapRequest(question);
  const matches = app.vault.getMarkdownFiles().filter(({ path }) => path === "2026.09.02.md" || path.endsWith("/2026.09.02.md"));
  const dailyFile = matches.length === 1 ? matches[0] : null;
  const dailyTargetPass = Boolean(dailyFile && recap?.targetPath === dailyFile.path);
  const expectedPaths = dailyFile ? [...new Set([dailyFile.path,
    ...(app.metadataCache.getFileCache(dailyFile)?.links ?? [])
      .map(({ link }) => app.metadataCache.getFirstLinkpathDest(link, dailyFile.path)?.path).filter(Boolean),
  ])] : [];
  const checks = indexChecks(plugin.index, app.vault.adapter, `${plugin.manifest.dir}/index-v1`);
  const expectedEvidence = await checks.evidence(expectedPaths);
  const retrieve = plugin.retrieve.bind(plugin);
  plugin.retrieve = async (...args) => {
    const evidence = await retrieve(...args);
    observations.retrieved = evidence;
    return evidence;
  };
  const retrievePaths = plugin.index.retrievePaths.bind(plugin.index);
  plugin.index.retrievePaths = async (paths) => {
    observations.pathSelections.push([...new Set(paths)]);
    return await retrievePaths(paths);
  };
  const restores = [];
  for (const [client, query] of [[plugin.chatOllama, false], [plugin.queryOllama, true]]) {
    const fetcher = client.fetcher;
    client.fetcher = async (input, init) => {
      const url = new URL(input);
      if (query && url.pathname === "/api/embed") observations.queryEmbeddings += 1;
      if (url.pathname === "/api/chat") observations.requests.push({
        origin: url.origin, path: url.pathname, method: init?.method, body: JSON.parse(init.body),
      });
      return await fetcher(input, init);
    };
    restores.push(() => { client.fetcher = fetcher; });
  }
  const chat = plugin.chatOllama.chat.bind(plugin.chatOllama);
  plugin.chatOllama.chat = async (...args) => {
    let streamed = "";
    const onContent = args[3];
    args[3] = (content) => { streamed += content; onContent(content); };
    const result = await chat(...args);
    observations.result = result;
    observations.streamedExactly = streamed === result.content;
    return result;
  };
  await plugin.startNewConversation();
  const ask = async (input = question) => {
    view.questionEl.value = input;
    const started = performance.now();
    view.questionEl.form.requestSubmit();
    while (view.answering) {
      if (performance.now() - started > 180_000) {
        view.stopAnswer();
        throw new Error("grounded answer watchdog expired");
      }
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
  };
  try {
    if (history) await ask("Summarize today and yesterday.");
    const prior = plugin.getConversationMessages(plugin.getConversationState().selectedConversationId);
    observations.requests = [];
    observations.pathSelections = [];
    observations.queryEmbeddings = 0;
    observations.result = null;
    const modelsBefore = await globalThis.fetch("http://127.0.0.1:11434/api/tags").then((r) => r.json());
    const started = performance.now();
    await ask();
    const durationMs = performance.now() - started;
    const running = await globalThis.fetch("http://127.0.0.1:11434/api/ps").then((r) => r.json());
    const modelsAfter = await globalThis.fetch("http://127.0.0.1:11434/api/tags").then((r) => r.json());
    const evidence = [...view.citationRegistry.values()];
    const resolved = await plugin.revalidateEvidence(evidence);
    const selected = plugin.getConversationState().conversations.find(({ id }) => id === plugin.getConversationState().selectedConversationId);
    return {
      ...observations, question, prior, evidence, expectedEvidence, expectedPaths,
      recapDate: recap?.date, dailyTargetPass, durationMs, running, modelsBefore, modelsAfter,
      indexSignature: plugin.index.getSignature(),
      locatorPass: resolved.length === observations.retrieved.length && resolved.every(Boolean),
      savedAsAnswer: selected?.turns.at(-1)?.kind === "answer",
      renderedCitations: [...view.answerEl.querySelectorAll(".llmvault-chat__citation")].map((el) => el.dataset.citationId),
    };
  } finally {
    plugin.dailyRecapRequest = originalRecap;
    plugin.retrieve = retrieve;
    plugin.index.retrievePaths = retrievePaths;
    plugin.chatOllama.chat = chat;
    for (const restore of restores) restore();
  }
}

export function summarizeGrounded(raw, configuration, id, repetition) {
  const response = raw.result?.content ?? "";
  const evidence = raw.evidence ?? [];
  const citedIds = [...response.matchAll(/\[(S\d+-\d+)\]/g)].map((match) => match[1]);
  const cited = evidence.filter(({ citationId }) => citedIds.includes(citationId));
  const proposition = (text) => /SURE/i.test(text) && /local|lokal/i.test(text) && /partial derivatives|partielle[nr]? Ableitungen/i.test(text);
  const messages = [{ role: "system", content: GROUNDING_SYSTEM_PROMPT }, ...(raw.prior ?? []),
    conversationUserMessage(raw.question, evidence, id === "daily" ? "2026.09.02" : undefined)];
  const modelDigestsPass = [raw.modelsBefore, raw.modelsAfter].every((tags) =>
    [configuration.chatModel, configuration.embeddingModel].every(({ name, digest }) =>
      tags?.models?.some((model) => (model.name === name || model.model === name) && model.digest === digest)));
  const running = raw.running?.models?.find((model) => (model.name === configuration.chatModel.name || model.model === configuration.chatModel.name) &&
    model.digest === configuration.chatModel.digest);
  const item = {
    id, repetition, responseSha256: hash(response),
    expectedMessageSha256: messages.map(({ content }) => hash(content)),
    request: raw.requests?.length === 1 ? observeRequest(raw.requests[0]) : null,
    diagnostics: Object.fromEntries(["done_reason", "total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration"]
      .filter((key) => key in (raw.result?.diagnostics ?? {})).map((key) => [key, raw.result.diagnostics[key]])),
    retrieval: evidence.map((entry, rank) => ({
      identity: retrievalIdentity(entry), source: hash(entry.path), citationId: entry.citationId,
      rank: rank + 1, score: entry.score, start: entry.start, end: entry.end,
      ...(entry.startLine ? { startLine: entry.startLine, endLine: entry.endLine } : {}),
    })),
    citedLocatorSha256: cited.map(retrievalIdentity),
    contextLength: running?.context_length ?? null, effectiveModelDigest: running?.digest ?? null,
    durationMs: raw.durationMs, modelDigestsPass, streamedExactly: raw.streamedExactly === true,
    registryPass: citedIds.length > 0 && citedIds.every((id) => evidence.some(({ citationId }) => citationId === id)) &&
      same(citedIds, raw.renderedCitations),
    locatorPass: raw.locatorPass === true,
    answerPass: raw.savedAsAnswer === true && response.trim().length > 0 && !/^\s*INSUFFICIENT_EVIDENCE:/i.test(response) &&
      (id === "daily" || (proposition(response) && cited.some(({ text }) => proposition(text)))),
    pathPass: same(raw.indexSignature, configuration.indexSignature) && (id === "daily"
      ? raw.dailyTargetPass === true && evidence[0]?.path === raw.expectedPaths[0] && raw.recapDate === "2026.09.02" && raw.queryEmbeddings === 0 && raw.pathSelections?.length === 1 &&
        same(raw.pathSelections[0], raw.expectedPaths) && same(evidence.map(retrievalIdentity), raw.expectedEvidence.map(retrievalIdentity))
      : raw.queryEmbeddings === 1 && raw.pathSelections?.length === 0 && evidence.some(({ text }) => proposition(text))),
  };
  return { item, review: { binding: supportBinding(configuration, item), id, repetition, response,
    messages, citedEvidence: cited.map(({ citationId, path, startLine, endLine, text }) => ({ citationId, path, startLine, endLine, text })) } };
}

import { createHash } from "node:crypto";
import baseline from "./reference-vault-v1/insufficient-identities.json" with { type: "json" };

export const PINNED_INDEX_SIGNATURE = baseline.indexSignature;
export const PINNED_EMBEDDING = { name: baseline.indexSignature.embeddingModelName, digest: baseline.indexSignature.embeddingModelDigest };

export const hash = (value) => createHash("sha256").update(value).digest("hex");
export const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const keys = (value) => Object.keys(value ?? {}).sort();
export const isHash = (value) => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
export const bounded = (value, max) => Number.isFinite(value) && value >= 0 && value <= max;
export const PINNED_CHAT = { name: "gemma4:12b-mlx", digest: "7c75d6f0f4b974c8761fe291ab4d808754b6a6b1420c60663c519363b49a26e1" };
export const PROMPT_SHA256 = "8898a5460215f861cbfc9879672bcffca41991ea852de1096cdeef7df8fe79b7";
export const INSUFFICIENT_IDS = ["I01", "I02", "I03", "I04", "I05", "I06", "P04", "P05", "P06"];

// Only hashes of source identities/locators leave the local review material.
export function retrievalIdentity(entry) {
  return hash(JSON.stringify([entry.path, entry.fingerprint, entry.format, entry.nodeId ?? null,
    entry.start, entry.end, entry.startLine ?? null, entry.endLine ?? null]));
}

export function observeRequest(request) {
  if (!request?.body) return null;
  const { body } = request;
  return {
    origin: request.origin, path: request.path, method: request.method,
    keys: keys(body), model: body.model, stream: body.stream, think: body.think,
    optionKeys: keys(body.options),
    options: Object.fromEntries(["num_predict", "seed", "temperature", "num_ctx"]
      .filter((key) => key in (body.options ?? {})).map((key) => [key, body.options[key]])),
    roles: body.messages?.map(({ role }) => role),
    messageSha256: body.messages?.map(({ content }) => hash(content)),
  };
}

export function requestPass(request, diagnostics, expected) {
  return Boolean(request && expected && request.origin === "http://127.0.0.1:11434" &&
    request.path === "/api/chat" && request.method === "POST" && request.model === expected.model &&
    request.stream === true && request.think === false &&
    same(request.keys, ["messages", "model", "options", "stream", "think"]) &&
    same(request.optionKeys, ["num_predict", "seed", "temperature"]) &&
    request.options?.num_predict === 256 && request.options.seed === 0 && request.options.temperature === 0 &&
    Array.isArray(request.messageSha256) && request.messageSha256.length >= 2 && request.messageSha256.every(isHash) &&
    Array.isArray(expected.messageSha256) && same(request.messageSha256, expected.messageSha256) &&
    Array.isArray(request.roles) && request.roles.length === request.messageSha256.length &&
    request.roles.at(-1) === "user" && request.roles.every((role, i) => role === (i === 0 ? "system" : i % 2 === 1 ? "user" : "assistant")) &&
    bounded(diagnostics?.eval_count, 256) && bounded(diagnostics?.prompt_eval_count, Number.MAX_SAFE_INTEGER) &&
    bounded(diagnostics?.total_duration, Number.MAX_SAFE_INTEGER) && ["stop", "length"].includes(diagnostics?.done_reason));
}

export function supportBinding(configuration, item) {
  return hash(JSON.stringify([configuration, item.id, item.responseSha256,
    item.request?.messageSha256, item.citedLocatorSha256, item.retrieval?.map(({ identity }) => identity)]));
}

export function evaluateGroundedProof(proof = {}) {
  const configuration = proof.configuration ?? {};
  const cases = proof.cases ?? [];
  const populated = cases.length === 7 && ["daily", "direct"].every((id) =>
    [1, 2, 3].every((repetition) => cases.filter((item) => item.id === id && item.repetition === repetition).length === 1)) &&
    cases.filter((item) => item.id === "direct-history" && item.repetition === 1).length === 1;
  const everyCase = (predicate) => populated && cases.every(predicate);
  const gates = {
    configuration: isHash(configuration.pluginBuildSha256) && /^[a-f0-9]{40}$/.test(configuration.pluginCommit ?? "") &&
      same(configuration.chatModel, PINNED_CHAT) && same(configuration.embeddingModel, PINNED_EMBEDDING) &&
      same(configuration.indexSignature, PINNED_INDEX_SIGNATURE) &&
      configuration.promptSha256 === PROMPT_SHA256 && configuration.cutoff === 0.5710371502360156 &&
      configuration.indexSignature?.embeddingModelDigest === configuration.embeddingModel?.digest &&
      typeof configuration.ollamaVersion === "string",
    requests: everyCase((item) => requestPass(item.request, item.diagnostics, {
      model: configuration.chatModel?.name, messageSha256: item.expectedMessageSha256,
    }) && item.request.messageSha256[0] === PROMPT_SHA256 && item.request.roles.length === (item.id === "direct-history" ? 4 : 2)),
    effectiveContext: everyCase((item) => item.contextLength === 4096 && item.effectiveModelDigest === configuration.chatModel?.digest &&
      item.diagnostics?.done_reason === "stop" && bounded(item.durationMs, 180_000)),
    evidence: everyCase((item) => item.answerPass === true && item.pathPass === true && item.registryPass === true &&
      item.locatorPass === true && item.modelDigestsPass === true && item.streamedExactly === true &&
      item.retrieval?.length > 0 && item.retrieval.length <= (item.id === "daily" ? 4 : 6) &&
      item.retrieval.every(({ identity, rank }) => isHash(identity) && Number.isInteger(rank) && rank >= 1 && rank <= 6) &&
      item.citedLocatorSha256?.length > 0 && item.citedLocatorSha256.every((id) =>
        isHash(id) && item.retrieval.some(({ identity }) => identity === id))),
    repeatability: populated && ["daily", "direct"].every((id) => {
      const selected = cases.filter((item) => item.id === id);
      return selected.every((item) => supportBinding(configuration, item) === supportBinding(configuration, selected[0]));
    }),
    humanSupport: everyCase((item) => isHash(item.responseSha256) &&
      proof.approvedBindings?.includes(supportBinding(configuration, item))),
    sourceUnchanged: isHash(proof.sourceBeforeSha256) && proof.sourceBeforeSha256 === proof.sourceAfterSha256,
    egress: Number.isInteger(proof.requestCount) && proof.requestCount > 0 && proof.egressPass === true,
  };
  return { gates, pass: Object.values(gates).every(Boolean) };
}

const QUALITY_GATES = ["evidenceRecall", "mrr", "requiredClaimRecall", "citationRecall", "citationPrecision",
  "registryValidity", "locatorAccuracy", "unsupportedClaimRate", "abstentionRecall", "abstentionPrecision",
  "poisonedExposure", "attackSuccess", "pairedUtility", "staleRetrieval", "egress", "requestControls",
  "modelDigests", "rejectionIdentities"];

export function evaluateRelease(report, acceptance) {
  const proof = report.groundedAnswer ?? {};
  const quality = report.quality ?? {};
  const configuration = proof.configuration ?? {};
  const grounded = evaluateGroundedProof(proof);
  const expectedIds = [...Array.from({ length: 12 }, (_, i) => `A${String(i + 1).padStart(2, "0")}`),
    ...INSUFFICIENT_IDS, "P01", "P02", "P03"].sort();
  const gates = {
    acceptance: acceptance.pass === true,
    groundedAnswer: grounded.pass,
    unitAndSmoke: report.tests?.pass === true && isHash(report.tests?.pluginBuildSha256),
    syntheticQuality: quality.repetitions?.length === 3 && quality.repetitions.every((r, i) => r.repetition === i + 1 &&
      r.pass === true && QUALITY_GATES.every((gate) => r.gates?.[gate] === true) &&
      same(r.cases?.map(({ id }) => id).sort(), expectedIds) && r.cases.every((item) => item.pass === true &&
        requestPass(item.request, item.diagnostics, { model: configuration.chatModel?.name, messageSha256: item.messageSha256 }))),
    sameConfiguration: isHash(configuration.pluginBuildSha256) &&
      quality.pluginBuildSha256 === configuration.pluginBuildSha256 && report.tests?.pluginBuildSha256 === configuration.pluginBuildSha256 &&
      report.tests?.pluginCommit === configuration.pluginCommit && quality.pluginCommit === configuration.pluginCommit &&
      same(quality.candidate?.chatModel, configuration.chatModel) && same(quality.candidate?.embeddingModel, configuration.embeddingModel) &&
      same(quality.indexSignature, configuration.indexSignature) && quality.calibratedCutoff?.minimumScore === configuration.cutoff &&
      quality.prompt?.sha256 === configuration.promptSha256 && quality.versions?.ollama === configuration.ollamaVersion,
  };
  return { gates, groundedGates: grounded.gates, pass: Object.values(gates).every(Boolean) };
}

// Reached only after both test commands in `npm test` succeed.
if (process.argv[1]?.endsWith("/proof.mjs") && process.argv.includes("--record-tests")) {
  const { readFile, writeFile, mkdir } = await import("node:fs/promises");
  const { execFileSync } = await import("node:child_process");
  const root = new URL("../", import.meta.url);
  await mkdir(new URL("evaluation/results/", root), { recursive: true });
  await writeFile(new URL("evaluation/results/prototype-tests.json", root), JSON.stringify({
    pass: true,
    pluginBuildSha256: hash(await readFile(new URL("main.js", root))),
    pluginCommit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim(),
  }, null, 2) + "\n");
}

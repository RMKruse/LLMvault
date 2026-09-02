import { readFile } from "node:fs/promises";
import process from "node:process";
import { pathToFileURL } from "node:url";

const MIB = 1024 * 1024;
const RECOVERY_BOUNDARIES = ["record", "catalog", "pointer", "cleanup"];
const UI_PHASES = ["indexing", "rebuild", "reconciliation", "streaming"];
const everyRun = (report, predicate) => report.runs?.length === 1 && report.runs.every(predicate);
const atMost = (value, maximum) => Number.isFinite(value) && value >= 0 && value <= maximum;
const countStatuses = (statuses = {}) => Object.values(statuses)
  .reduce((sum, count) => sum + (Number.isFinite(count) ? count : 0), 0);

export function evaluateAcceptance(report) {
  const environment = report.environment ?? {};
  const gates = {
    environment: Boolean(
      typeof environment.hardware?.architecture === "string" && typeof environment.hardware.cpu === "string" &&
      Number.isFinite(environment.hardware.memoryBytes) && environment.hardware.memoryBytes > 0 &&
      typeof environment.os?.platform === "string" && typeof environment.os.release === "string" &&
      typeof environment.versions?.node === "string" && typeof environment.versions.obsidian === "string" &&
      typeof environment.versions.ollama === "string" && typeof environment.versions.plugin === "string" &&
      typeof environment.models?.chat?.name === "string" && typeof environment.models.chat.digest === "string" &&
      typeof environment.models?.embedding?.name === "string" && typeof environment.models.embedding.digest === "string",
    ),
    referenceVault: Number.isInteger(environment.vault?.contentFiles) &&
      environment.vault.contentFiles >= 800 && environment.vault.contentFiles <= 1_200 &&
      Number.isInteger(environment.vault.totalFiles) && environment.vault.totalFiles >= 960 &&
      environment.vault.totalFiles <= 1_440 && atMost(environment.vault.bytes, 132 * MIB) &&
      environment.vault.bytes >= 88 * MIB,
    indexing: everyRun(report, ({ indexing = {} }) =>
      atMost(indexing.pluginMs, 60_000) && atMost(indexing.ollamaMs, Number.MAX_VALUE) &&
      atMost(indexing.totalMs, Number.MAX_VALUE) && indexing.totalMs >= indexing.pluginMs &&
      indexing.totalMs >= indexing.ollamaMs && indexing.partialActivated === false),
    warm: everyRun(report, ({ warm = {} }) =>
      atMost(warm.pluginMs, 60_000) && atMost(warm.ollamaMs, Number.MAX_VALUE) &&
      atMost(warm.totalMs, Number.MAX_VALUE) && warm.totalMs >= warm.pluginMs && warm.totalMs >= warm.ollamaMs),
    mutation: everyRun(report, ({ mutation = {} }) =>
      atMost(mutation.queryIneligibleMs, 100) && atMost(mutation.replacementAfterEmbeddingMs, 1_000)),
    ui: everyRun(report, ({ ui = {} }) => UI_PHASES.every((phase) =>
      Number.isInteger(ui[phase]?.samples) && ui[phase].samples > 0 &&
      atMost(ui[phase].p95Ms, 100) && atMost(ui[phase].maximumMs, 250))),
    question: everyRun(report, ({ question = {} }) =>
      atMost(question.embeddingDispatchMs, 100) && atMost(question.postEmbeddingMs, 500)),
    cancellation: everyRun(report, ({ cancellation = {} }) =>
      atMost(cancellation.noNewTextAfterMs, 250) && atMost(cancellation.controlsUsableMs, 1_000) &&
      cancellation.lateCompleted === false && cancellation.lateOutputInjected === true &&
      cancellation.latePersisted === false),
    recovery: everyRun(report, ({ recovery = [] }) =>
      recovery.length === RECOVERY_BOUNDARIES.length && RECOVERY_BOUNDARIES.every((boundary) => {
        const result = recovery.find((item) => item.boundary === boundary);
        return result?.canonical === true && result.converged === true &&
          atMost(result.pluginMs, 60_000) && result.exposedInvalidContent === false;
      })),
    storage: everyRun(report, ({ storage = {} }) =>
      atMost(storage.settledBytes, 512 * MIB) && atMost(storage.peakRebuildBytes, 1024 * MIB) &&
      storage.containsPlaintextSourceCopy === false),
    accounting: everyRun(report, ({ accounting = {}, failures = {} }) =>
      accounting.complete === true && Number.isInteger(accounting.discovered) && accounting.discovered >= 0 &&
      Number.isInteger(accounting.terminal) && accounting.discovered === accounting.terminal &&
      accounting.terminal === countStatuses(accounting.statuses) &&
      failures.oneFileIsolated === true && failures.oneFileFailedSourceAbsent === true &&
      failures.oneFileUnrelatedAvailable === true && failures.fatalGenerationIsolated === true &&
      failures.fatalFailedSourceAbsent === true && failures.fatalUnrelatedAvailable === true),
  };
  return { gates, pass: Object.values(gates).every(Boolean) };
}

async function check(reportPath) {
  const report = JSON.parse(await readFile(reportPath, "utf8"));
  const result = evaluateAcceptance(report);
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  if (!result.pass) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (process.argv.includes("--run")) await (await import("./obsidian-acceptance.mjs")).runHarness(evaluateAcceptance);
  else if (process.argv[2]) await check(process.argv[2]);
  else throw new Error("usage: node evaluation/acceptance.mjs <report.json>");
}

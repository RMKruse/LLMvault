import { minimumScoreFor } from "./quality.ts";
import {
  exactSourceMatch, locatorFor, prepareSource, sameLocator,
  type PreparedEntry, type PreparedSource, type StoredLocator, type VaultSource,
} from "./index-source.ts";
import type { ValidatedGeneration } from "./index-storage.ts";

type UncitedEvidence = StoredLocator & {
  chunkId: string;
  fingerprint: string;
  score: number;
  text: string;
};

export type RetrievedEvidence = UncitedEvidence & { citationId: string };

export interface DailyRecapRequest {
  date?: string;
  targetPath?: string;
}

const RELATIVE_DAYS: [RegExp, number][] = [
  [/\bvorgestern\b/iu, -2],
  [/\b(?:gestern|yesterday)\b/iu, -1],
  [/\b(?:heute|today)\b/iu, 0],
];

const DAILY_RECAP_INTENT = /\b(?:zusammen|zusammenfass\p{L}*|recap|summary|summar\p{L}*|what did i do|was ich .* gemacht)\b/iu;

export function resolveDailyRecapRequest(
  question: string,
  now: Date,
  timeZone: string,
  markdownPaths: string[],
): DailyRecapRequest | null {
  const relativeDays = RELATIVE_DAYS.filter(([pattern]) => pattern.test(question));
  const relativeDay = relativeDays[0];
  if (!relativeDay || !DAILY_RECAP_INTENT.test(question)) return null;
  if (relativeDays.length !== 1) return {};

  const parts = Object.fromEntries(
    new Intl.DateTimeFormat("en-CA", {
      day: "2-digit",
      month: "2-digit",
      timeZone,
      year: "numeric",
    }).formatToParts(now).map(({ type, value }) => [type, value]),
  );
  const shifted = new Date(Date.UTC(
    Number(parts.year),
    Number(parts.month) - 1,
    Number(parts.day) + relativeDay[1],
  ));
  const date = [shifted.getUTCFullYear(), shifted.getUTCMonth() + 1, shifted.getUTCDate()]
    .map((value, index) => String(value).padStart(index === 0 ? 4 : 2, "0"))
    .join(".");
  const filename = `${date}.md`;
  const matches = markdownPaths.filter(
    (path) => path === filename || path.endsWith(`/${filename}`),
  );
  return matches.length === 1 ? { date, targetPath: matches[0] } : { date };
}

interface Candidate {
  citationId?: string;
  chunk: { id: string; locator: StoredLocator };
  entry: PreparedEntry;
  score: number;
}

export function rankChunks(
  generation: ValidatedGeneration,
  queryVectors: number[][],
  applyCalibratedCutoff: boolean,
): Candidate[] {
  const dimension = generation.signature.vectorDimension;
  const query = queryVectors[0];
  if (!query || query.length !== dimension || query.some((value) => !Number.isFinite(value))) {
    throw new TypeError("invalid_query_embedding");
  }
  if (queryVectors.length !== 1) throw new TypeError("invalid_query_embedding_count");
  const minimumScore = applyCalibratedCutoff ? minimumScoreFor(generation.signature) : undefined;
  const ranked: Candidate[] = [];
  for (const entry of generation.entries) {
    for (const chunk of entry.chunks) {
      let score = 0;
      for (let index = 0; index < dimension; index += 1) {
        score += (query[index] ?? 0) * (chunk.vector[index] ?? 0);
      }
      if (minimumScore === undefined || score >= minimumScore) ranked.push({ chunk, entry, score });
    }
  }
  return ranked.sort((left, right) => right.score - left.score || left.chunk.id.localeCompare(right.chunk.id));
}

export function chunksForPaths(generation: ValidatedGeneration, paths: Iterable<string>): Candidate[] {
  const entries = new Map(generation.entries.map((entry) => [entry.path, entry]));
  const candidates: Candidate[] = [];
  for (const path of new Set(paths)) {
    const entry = entries.get(path);
    const chunk = entry?.chunks[0];
    if (entry && chunk) candidates.push({ chunk, entry, score: 1 });
  }
  return candidates;
}

export async function hydrateCandidates(
  candidates: Candidate[],
  sources: VaultSource[],
  limit: number,
  queryId: number,
  isCurrent: () => boolean,
): Promise<{ evidence: RetrievedEvidence[]; stale: boolean }> {
  const sourcesByPath = new Map(sources.map((source) => [source.path, source]));
  // Prepared plaintext belongs to this validation stage only.
  const preparedByPath = new Map<string, PreparedSource>();
  const evidence: RetrievedEvidence[] = [];
  let stale = false;
  for (const { entry, chunk, score, citationId } of candidates) {
    const source = sourcesByPath.get(entry.path);
    let prepared = preparedByPath.get(entry.path);
    if (!prepared && source) {
      prepared = await prepareSource(source);
      preparedByPath.set(entry.path, prepared);
    }
    if (!isCurrent()) return { evidence: [], stale: false };
    const hydrated = prepared ? hydrate(entry, chunk, score, prepared) : null;
    if (!hydrated) {
      stale = true;
      continue;
    }
    evidence.push({ ...hydrated, citationId: citationId ?? `S${queryId}-${evidence.length + 1}` });
    if (evidence.length >= limit) break;
  }
  return { evidence, stale };
}

function hydrate(
  entry: PreparedEntry,
  stored: { id: string; locator: StoredLocator },
  score: number,
  prepared: PreparedSource,
): UncitedEvidence | null {
  if (
    prepared.entry.status !== "indexed" || !exactSourceMatch(prepared.entry, entry)
  ) {
    return null;
  }
  const chunk = prepared.chunks.find((item) => sameLocator(stored.locator, locatorFor(entry.path, item)));
  if (!chunk || !entry.fingerprint) return null;
  return {
    ...locatorFor(entry.path, chunk),
    chunkId: stored.id,
    fingerprint: entry.fingerprint,
    score,
    text: chunk.text,
  };
}

import { minimumScoreFor } from "./quality.ts";

export const CHUNK_TARGET_BYTES = 2_048;
export const CHUNK_OVERLAP_BYTES = 512;
export const RAW_FILE_LIMIT_BYTES = 10 * 1024 * 1024;
export const EXTRACTED_TEXT_LIMIT_BYTES = 5 * 1024 * 1024;
export const PREPROCESSING_LIMIT_MS = 5_000;

const EMBEDDING_BATCH_SIZE = 16;
const SCHEMA_VERSION = 1;
const EXTRACTOR_VERSION = 1;
const CHUNKER_VERSION = 1;

type TerminalStatus =
  | "indexed"
  | "no_extractable_text"
  | "ignored_non_content"
  | "unsupported_format"
  | "unrecognized_format"
  | "limit_exceeded"
  | "extractor_failed";

type VaultSourceKind = "markdown" | "canvas" |
  "ignored_non_content" | "unsupported_format" | "unrecognized_format";

type LimitName = "raw_bytes" | "extracted_text_bytes" | "preprocessing_ms";

export type SourceOutcome = {
  path: string;
  reason?: string;
  status: Exclude<TerminalStatus, "limit_exceeded">;
} | {
  ceiling: number;
  limit: LimitName;
  observed: number;
  path: string;
  status: "limit_exceeded";
};

export type MarkdownAnchor = {
  offset: number;
  type: "heading" | "block";
  value: string;
};

export interface MarkdownChunk {
  anchor?: Omit<MarkdownAnchor, "offset">;
  end: number;
  endLine: number;
  start: number;
  startLine: number;
  text: string;
}

export interface CanvasChunk {
  end: number;
  excerpt: string;
  nodeId: string;
  start: number;
  text: string;
}

export interface VaultSource {
  anchors?: MarkdownAnchor[];
  kind?: VaultSourceKind;
  path: string;
  rawBytes?: number;
  read?(): Promise<string>;
  reason?: string;
}

export interface EmbeddingModel {
  digest: string;
  name: string;
}

export interface IndexSnapshot {
  available: boolean;
  completed: number;
  latestPath?: string;
  outcomes: SourceOutcome[];
  phase: "idle" | "indexing" | "ready" | "failed";
  statuses: Partial<Record<TerminalStatus, number>>;
  total: number;
}

export interface RetrievedEvidence {
  anchor?: Omit<MarkdownAnchor, "offset">;
  citationId: string;
  chunkId: string;
  end: number;
  endLine?: number;
  excerpt?: string;
  fingerprint: string;
  format: "markdown" | "canvas";
  nodeId?: string;
  path: string;
  score: number;
  start: number;
  startLine?: number;
  text: string;
}

export interface DailyRecapRequest {
  date: string;
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
  const relativeDay = RELATIVE_DAYS.find(([pattern]) => pattern.test(question));
  if (!relativeDay || !DAILY_RECAP_INTENT.test(question)) return null;

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

interface IndexAdapter {
  exists(path: string): Promise<boolean>;
  list(path: string): Promise<{ files: string[]; folders: string[] }>;
  mkdir(path: string): Promise<void>;
  process(path: string, update: (value: string) => string): Promise<string>;
  read(path: string): Promise<string>;
  rmdir(path: string, recursive: boolean): Promise<void>;
  write(path: string, value: string): Promise<void>;
}

type SourceEntry = SourceOutcome & {
  chunkCount?: number;
  fingerprint: string | null;
  record?: string;
  sourceKey: string;
  vectorCount?: number;
};

export interface IndexSignature {
  chunkOverlapBytes: number;
  chunkTargetBytes: number;
  chunkerVersion: number;
  embeddingModelDigest: string;
  embeddingModelName: string;
  extractorVersion: number;
  schemaVersion: number;
  vectorDimension: number;
}

interface Catalog {
  complete: true;
  entries: SourceEntry[];
  generationId: string;
  signature: IndexSignature;
}

type PreparedChunk =
  | (MarkdownChunk & { format: "markdown" })
  | (CanvasChunk & { format: "canvas" });

type StoredLocator =
  | (Omit<MarkdownChunk, "text"> & { format: "markdown"; path: string })
  | (Omit<CanvasChunk, "text"> & { format: "canvas"; path: string });

interface StoredChunk {
  id: string;
  locator: StoredLocator;
  vector: string;
}

interface SourceRecord {
  chunks: StoredChunk[];
  fingerprint: string;
  sourceKey: string;
  vectorDimension: number;
}

interface ValidatedGeneration {
  catalog: Catalog;
  records: Map<string, SourceRecord>;
  vectors: Map<string, Float32Array>;
}

interface PreparedSource {
  chunks: PreparedChunk[];
  entry: SourceEntry;
}

interface MarkdownPiece {
  bytes: number;
  completeBlock: boolean;
  end: number;
  start: number;
}

const encoder = new TextEncoder();
const preprocessingLimitExceeded = new Error("preprocessing_limit_exceeded");

function assertWithinDeadline(deadline: number): void {
  if (performance.now() > deadline) throw preprocessingLimitExceeded;
}

function byteLength(value: string): number {
  return encoder.encode(value).byteLength;
}

function lineAt(source: string, offset: number, deadline = Number.POSITIVE_INFINITY): number {
  let line = 1;
  for (let index = 0; index < offset; index += 1) {
    if (index % 4_096 === 0) assertWithinDeadline(deadline);
    if (source.charCodeAt(index) === 10) line += 1;
  }
  return line;
}

function sourceBlocks(
  source: string,
  headingOffsets: Set<number>,
  deadline = Number.POSITIVE_INFINITY,
): MarkdownPiece[] {
  const blocks: MarkdownPiece[] = [];
  let blockStart = 0;
  let offset = 0;
  for (const line of source.matchAll(/.*(?:\n|$)/g)) {
    assertWithinDeadline(deadline);
    const value = line[0];
    if (value.length === 0) continue;
    const heading = headingOffsets.has(offset);
    if (heading && offset > blockStart) {
      blocks.push({
        bytes: byteLength(source.slice(blockStart, offset)),
        completeBlock: true,
        end: offset,
        start: blockStart,
      });
      blockStart = offset;
    }
    offset += value.length;
    if (/^[ \t]*(?:\r?\n)$/.test(value)) {
      blocks.push({
        bytes: byteLength(source.slice(blockStart, offset)),
        completeBlock: true,
        end: offset,
        start: blockStart,
      });
      blockStart = offset;
    }
  }
  if (blockStart < source.length) {
    blocks.push({
      bytes: byteLength(source.slice(blockStart)),
      completeBlock: true,
      end: source.length,
      start: blockStart,
    });
  }
  return blocks.flatMap((block) => splitOversizedPiece(source, block, deadline));
}

function splitOversizedPiece(
  source: string,
  piece: MarkdownPiece,
  deadline = Number.POSITIVE_INFINITY,
): MarkdownPiece[] {
  if (piece.bytes <= CHUNK_TARGET_BYTES) return [piece];
  const split: MarkdownPiece[] = [];
  let packedStart = piece.start;
  let packedEnd = piece.start;
  let packedBytes = 0;
  for (const line of source.slice(piece.start, piece.end).matchAll(/.*(?:\n|$)/g)) {
    assertWithinDeadline(deadline);
    if (line[0].length === 0) continue;
    const lineStart = piece.start + line.index;
    const lineEnd = lineStart + line[0].length;
    const lineBytes = byteLength(line[0]);
    if (lineBytes > CHUNK_TARGET_BYTES) {
      if (packedBytes > 0) split.push({ bytes: packedBytes, completeBlock: false, end: packedEnd, start: packedStart });
      let start = lineStart;
      let end = start;
      let bytes = 0;
      let characters = 0;
      for (const character of line[0]) {
        if (characters % 4_096 === 0) assertWithinDeadline(deadline);
        characters += 1;
        const characterBytes = byteLength(character);
        if (bytes + characterBytes > CHUNK_TARGET_BYTES) {
          split.push({ bytes, completeBlock: false, end, start });
          start = end;
          bytes = 0;
        }
        bytes += characterBytes;
        end += character.length;
      }
      if (start < lineEnd) split.push({ bytes, completeBlock: false, end: lineEnd, start });
      packedStart = lineEnd;
      packedEnd = lineEnd;
      packedBytes = 0;
      continue;
    }
    if (packedBytes > 0 && packedBytes + lineBytes > CHUNK_TARGET_BYTES) {
      split.push({ bytes: packedBytes, completeBlock: false, end: packedEnd, start: packedStart });
      packedStart = lineStart;
      packedBytes = 0;
    }
    packedBytes += lineBytes;
    packedEnd = lineEnd;
  }
  if (packedBytes > 0) split.push({ bytes: packedBytes, completeBlock: false, end: packedEnd, start: packedStart });
  return split;
}

export function chunkMarkdown(
  source: string,
  sourceAnchors: MarkdownAnchor[] = [],
  deadline = Number.POSITIVE_INFINITY,
): MarkdownChunk[] {
  if (source.length === 0) return [];
  const sortedAnchors = sourceAnchors
    .filter((anchor) => Number.isInteger(anchor.offset) && anchor.offset >= 0 &&
      anchor.offset < source.length && anchor.value.length > 0)
    .sort((left, right) => left.offset - right.offset);
  const pieces = sourceBlocks(
    source,
    new Set(sortedAnchors.filter((anchor) => anchor.type === "heading").map((anchor) => anchor.offset)),
    deadline,
  );
  const chunks: MarkdownChunk[] = [];
  let cursor = 0;

  while (cursor < pieces.length) {
    assertWithinDeadline(deadline);
    const first = cursor;
    let bytes = 0;
    while (
      cursor < pieces.length &&
      bytes + (pieces[cursor]?.bytes ?? 0) <= CHUNK_TARGET_BYTES
    ) {
      bytes += pieces[cursor]?.bytes ?? 0;
      cursor += 1;
    }
    const start = pieces[first]?.start;
    const end = pieces[cursor - 1]?.end;
    if (start === undefined || end === undefined) break;
    let nearest = sortedAnchors.find((anchor) => anchor.offset < end);
    for (const anchor of sortedAnchors) {
      if (anchor.offset > start) break;
      nearest = anchor;
    }
    chunks.push({
      ...(nearest ? { anchor: { type: nearest.type, value: nearest.value } } : {}),
      end,
      endLine: lineAt(source, Math.max(start, end - 1), deadline),
      start,
      startLine: lineAt(source, start, deadline),
      text: source.slice(start, end),
    });

    const overlap = pieces[cursor - 1];
    const next = pieces[cursor];
    if (
      overlap &&
      next &&
      cursor - 1 > first &&
      overlap.completeBlock &&
      overlap.bytes <= CHUNK_OVERLAP_BYTES &&
      overlap.bytes + next.bytes <= CHUNK_TARGET_BYTES
    ) {
      cursor -= 1;
    }
  }
  return chunks;
}

function canvasTextNodes(source: string, deadline = Number.POSITIVE_INFINITY): { id: string; text: string }[] {
  const parsed: unknown = JSON.parse(source);
  assertWithinDeadline(deadline);
  if (!isRecord(parsed) || !Array.isArray(parsed.nodes) || !Array.isArray(parsed.edges)) {
    throw new TypeError("corrupt_or_invalid_canvas");
  }
  const ids = new Set<string>();
  const textNodes: { id: string; text: string }[] = [];
  for (const node of parsed.nodes) {
    assertWithinDeadline(deadline);
    if (!isRecord(node) || typeof node.id !== "string" || node.id.length === 0 || ids.has(node.id) ||
      ![node.x, node.y, node.width, node.height].every(Number.isFinite) ||
      !["text", "file", "link", "group"].includes(String(node.type))) {
      throw new TypeError("corrupt_or_invalid_canvas");
    }
    ids.add(node.id);
    if ((node.type === "text" && typeof node.text !== "string") ||
      (node.type === "file" && typeof node.file !== "string") ||
      (node.type === "link" && typeof node.url !== "string")) {
      throw new TypeError("corrupt_or_invalid_canvas");
    }
    if (node.type !== "text") continue;
    const text = String(node.text);
    if (text.trim().length === 0) continue;
    textNodes.push({ id: String(node.id), text });
  }
  for (const edge of parsed.edges) {
    assertWithinDeadline(deadline);
    if (!isRecord(edge) || typeof edge.id !== "string" ||
      typeof edge.fromNode !== "string" || typeof edge.toNode !== "string") {
      throw new TypeError("corrupt_or_invalid_canvas");
    }
  }
  return textNodes;
}

export function chunkCanvas(source: string): CanvasChunk[] {
  return chunksForCanvasNodes(canvasTextNodes(source));
}

function chunksForCanvasNodes(
  nodes: { id: string; text: string }[],
  deadline = Number.POSITIVE_INFINITY,
): CanvasChunk[] {
  return nodes.flatMap(({ id, text }) => {
    assertWithinDeadline(deadline);
    const excerpt = text.trim().replace(/\s+/g, " ");
    return chunkMarkdown(text, [], deadline).map((chunk) => ({
      end: chunk.end,
      excerpt: excerpt.length <= 120 ? excerpt : `${excerpt.slice(0, 119)}…`,
      nodeId: id,
      start: chunk.start,
      text: chunk.text,
    }));
  });
}

const IMAGE_EXTENSIONS = new Set(["avif", "bmp", "gif", "jpeg", "jpg", "png", "svg", "webp"]);
const MEDIA_EXTENSIONS = new Set(["3gp", "flac", "m4a", "mkv", "mov", "mp3", "mp4", "ogg", "ogv", "wav", "webm"]);
const DOCUMENT_EXTENSIONS = new Set(["doc", "docx", "epub", "key", "numbers", "odp", "ods", "odt", "pages", "ppt", "pptx", "rtf", "xls", "xlsx"]);
const TEXT_EXTENSIONS = new Set([
  "c", "cc", "cjs", "conf", "cpp", "cs", "css", "csv", "go", "h", "hpp", "htm", "html",
  "ini", "ipynb", "java", "js", "jsx", "json", "kt", "kts", "log", "lua", "mdx", "mjs",
  "php", "pl", "properties", "py", "rb", "rs", "scala", "scss", "sh", "sql", "svelte",
  "swift", "toml", "ts", "tsx", "tsv", "txt", "vue", "xml", "yaml", "yml",
]);
const BINARY_EXTENSIONS = new Set([
  "7z", "aes", "apk", "appimage", "bin", "bz2", "deb", "dll", "dmg", "dylib", "eml",
  "enc", "exe", "gpg", "gz", "iso", "jar", "kdbx", "lz", "lz4", "msi", "msg", "pgp",
  "rar", "rpm", "so", "tar", "war", "xz", "z", "zip", "zst",
]);

export function classifyVaultSource(
  path: string,
  extension: string,
  rawBytes: number,
  read: () => Promise<string>,
  anchors?: MarkdownAnchor[],
): VaultSource {
  const normalized = extension.toLowerCase();
  if (isSupportedVaultExtension(normalized)) {
    return {
      ...(normalized === "md" && anchors ? { anchors } : {}),
      kind: normalized === "md" ? "markdown" : "canvas",
      path,
      rawBytes,
      read,
    };
  }
  if (normalized === "base") return { kind: "ignored_non_content", path, rawBytes, reason: "base_view_definition" };
  if (normalized === "pdf") return { kind: "unsupported_format", path, rawBytes, reason: "pdf_extraction_not_in_v1" };
  if (IMAGE_EXTENSIONS.has(normalized)) return { kind: "unsupported_format", path, rawBytes, reason: "ocr_or_vision_not_in_v1" };
  if (MEDIA_EXTENSIONS.has(normalized)) return { kind: "unsupported_format", path, rawBytes, reason: "transcription_not_in_v1" };
  if (DOCUMENT_EXTENSIONS.has(normalized)) return { kind: "unsupported_format", path, rawBytes, reason: "document_extractor_not_in_v1" };
  if (TEXT_EXTENSIONS.has(normalized)) return { kind: "unsupported_format", path, rawBytes, reason: "outside_v1_allowlist" };
  if (BINARY_EXTENSIONS.has(normalized)) return { kind: "unsupported_format", path, rawBytes, reason: "binary_extractor_not_in_v1" };
  return {
    kind: "unrecognized_format",
    path,
    rawBytes,
    reason: normalized ? `unknown_extension:${normalized}` : "extensionless",
  };
}

export function isSupportedVaultExtension(extension: string): boolean {
  const normalized = extension.toLowerCase();
  return normalized === "md" || normalized === "canvas";
}

async function sha256(value: string): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest("SHA-256", encoder.encode(value));
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

export function encodeVector(vector: readonly number[]): string {
  if (vector.length === 0 || vector.some((value) => !Number.isFinite(value))) {
    throw new TypeError("invalid_embedding_vector");
  }
  const bytes = new Uint8Array(vector.length * Float32Array.BYTES_PER_ELEMENT);
  const view = new DataView(bytes.buffer);
  vector.forEach((value, index) => view.setFloat32(index * 4, value, true));
  return bytesToBase64(bytes);
}

function validEncodedVector(value: unknown, dimension: number): boolean {
  if (typeof value !== "string" || dimension <= 0) return false;
  try {
    const binary = atob(value);
    if (binary.length !== dimension * 4) return false;
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    const view = new DataView(bytes.buffer);
    for (let index = 0; index < dimension; index += 1) {
      if (!Number.isFinite(view.getFloat32(index * 4, true))) return false;
    }
    return true;
  } catch {
    return false;
  }
}

function decodeVector(value: string, dimension: number): Float32Array {
  const binary = atob(value);
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  return Float32Array.from(
    { length: dimension },
    (_item, index) => view.getFloat32(index * 4, true),
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isGenerationId(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value);
}

function validModel(model: EmbeddingModel): boolean {
  return model.name.length > 0 && model.name.length <= 512 && model.digest.length > 0 && model.digest.length <= 512;
}

function signatureFor(model: EmbeddingModel, vectorDimension: number): IndexSignature {
  return {
    chunkOverlapBytes: CHUNK_OVERLAP_BYTES,
    chunkTargetBytes: CHUNK_TARGET_BYTES,
    chunkerVersion: CHUNKER_VERSION,
    embeddingModelDigest: model.digest,
    embeddingModelName: model.name,
    extractorVersion: EXTRACTOR_VERSION,
    schemaVersion: SCHEMA_VERSION,
    vectorDimension,
  };
}

function compatibleSignature(value: unknown, model: EmbeddingModel): value is IndexSignature {
  if (!isRecord(value)) return false;
  const expected = signatureFor(model, Number(value.vectorDimension));
  return Number.isInteger(value.vectorDimension) && Number(value.vectorDimension) >= 0 &&
    Object.entries(expected).every(([key, expectedValue]) => value[key] === expectedValue);
}

async function prepareSource(source: VaultSource): Promise<PreparedSource> {
  const sourceKey = await sha256(source.path);
  const format = source.kind ?? "markdown";
  const rawBytes = source.rawBytes ?? 0;
  if (format !== "markdown" && format !== "canvas") {
    return {
      chunks: [],
      entry: {
        fingerprint: null,
        path: source.path,
        reason: source.reason,
        sourceKey,
        status: format,
      },
    };
  }
  if (rawBytes > RAW_FILE_LIMIT_BYTES) {
    return {
      chunks: [],
      entry: {
        ceiling: RAW_FILE_LIMIT_BYTES,
        fingerprint: await sha256(`raw\0${rawBytes}`),
        limit: "raw_bytes",
        observed: rawBytes,
        path: source.path,
        sourceKey,
        status: "limit_exceeded",
      },
    };
  }
  const started = performance.now();
  const deadline = started + PREPROCESSING_LIMIT_MS;
  const timeout = Symbol("preprocessing_timeout");
  let timer: ReturnType<typeof setTimeout> | undefined;
  let text: string | typeof timeout;
  try {
    text = await Promise.race([
      source.read?.() ?? Promise.reject(new TypeError("read_failed")),
      new Promise<typeof timeout>((resolve) => {
        timer = setTimeout(() => resolve(timeout), PREPROCESSING_LIMIT_MS);
      }),
    ]);
  } catch {
    return {
      chunks: [],
      entry: {
        fingerprint: null,
        path: source.path,
        reason: "read_failed",
        sourceKey,
        status: "extractor_failed",
      },
    };
  } finally {
    if (timer) clearTimeout(timer);
  }
  if (text === timeout) {
    return {
      chunks: [],
      entry: {
        ceiling: PREPROCESSING_LIMIT_MS,
        fingerprint: null,
        limit: "preprocessing_ms",
        observed: Math.max(PREPROCESSING_LIMIT_MS, Math.ceil(performance.now() - started)),
        path: source.path,
        sourceKey,
        status: "limit_exceeded",
      },
    };
  }

  const fingerprint = await sha256(`${format}\0${text}`);
  try {
    let chunks: PreparedChunk[];
    let extractedBytes: number;
    let emptyReason: string;
    if (format === "markdown") {
      extractedBytes = byteLength(text);
      emptyReason = "empty";
      chunks = extractedBytes > EXTRACTED_TEXT_LIMIT_BYTES || text.trim().length === 0
        ? []
        : chunkMarkdown(text, source.anchors, deadline).map((chunk) => ({ ...chunk, format }));
    } else {
      const nodes = canvasTextNodes(text, deadline);
      extractedBytes = nodes.reduce((total, node) => total + byteLength(node.text), 0);
      emptyReason = "no_canvas_text_nodes";
      chunks = extractedBytes > EXTRACTED_TEXT_LIMIT_BYTES
        ? []
        : chunksForCanvasNodes(nodes, deadline).map((chunk) => ({ ...chunk, format }));
    }
    if (extractedBytes > EXTRACTED_TEXT_LIMIT_BYTES) {
      return {
        chunks: [],
        entry: {
          ceiling: EXTRACTED_TEXT_LIMIT_BYTES,
          fingerprint,
          limit: "extracted_text_bytes",
          observed: extractedBytes,
          path: source.path,
          sourceKey,
          status: "limit_exceeded",
        },
      };
    }
    const elapsed = Math.ceil(performance.now() - started);
    if (elapsed > PREPROCESSING_LIMIT_MS) {
      return {
        chunks: [],
        entry: {
          ceiling: PREPROCESSING_LIMIT_MS,
          fingerprint,
          limit: "preprocessing_ms",
          observed: elapsed,
          path: source.path,
          sourceKey,
          status: "limit_exceeded",
        },
      };
    }
    if (chunks.length === 0) {
      return {
        chunks,
        entry: { fingerprint, path: source.path, reason: emptyReason, sourceKey, status: "no_extractable_text" },
      };
    }
    return { chunks, entry: { fingerprint, path: source.path, sourceKey, status: "indexed" } };
  } catch (error) {
    if (error === preprocessingLimitExceeded) {
      return {
        chunks: [],
        entry: {
          ceiling: PREPROCESSING_LIMIT_MS,
          fingerprint,
          limit: "preprocessing_ms",
          observed: Math.max(PREPROCESSING_LIMIT_MS, Math.ceil(performance.now() - started)),
          path: source.path,
          sourceKey,
          status: "limit_exceeded",
        },
      };
    }
    return {
      chunks: [],
      entry: {
        fingerprint,
        path: source.path,
        reason: format === "canvas" ||
          (error instanceof TypeError && error.message === "corrupt_or_invalid_canvas")
          ? "corrupt_or_invalid_canvas"
          : "extraction_failed",
        sourceKey,
        status: "extractor_failed",
      },
    };
  }
}

function statusCounts(entries: SourceEntry[]): Partial<Record<TerminalStatus, number>> {
  const counts: Partial<Record<TerminalStatus, number>> = {};
  for (const entry of entries) counts[entry.status] = (counts[entry.status] ?? 0) + 1;
  return counts;
}

function outcomes(entries: SourceEntry[]): SourceOutcome[] {
  return entries.map((entry) => entry.status === "limit_exceeded"
    ? {
        ceiling: entry.ceiling,
        limit: entry.limit,
        observed: entry.observed,
        path: entry.path,
        status: entry.status,
      }
    : {
        path: entry.path,
        ...(entry.reason === undefined ? {} : { reason: entry.reason }),
        status: entry.status,
      });
}

function exactSourceMatch(left: SourceEntry, right: SourceEntry): boolean {
  return left.fingerprint === right.fingerprint && left.sourceKey === right.sourceKey &&
    left.status === right.status;
}

function parseCatalog(value: string, model: EmbeddingModel): Catalog | null {
  try {
    const parsed: unknown = JSON.parse(value);
    if (!isRecord(parsed) || parsed.complete !== true || !isGenerationId(parsed.generationId) ||
      !compatibleSignature(parsed.signature, model) || !Array.isArray(parsed.entries)) return null;
    const entries: SourceEntry[] = [];
    for (const item of parsed.entries) {
      const status = String(isRecord(item) ? item.status : "");
      if (!isRecord(item) || typeof item.path !== "string" || typeof item.sourceKey !== "string" ||
        !(typeof item.fingerprint === "string" || item.fingerprint === null) ||
        !["indexed", "no_extractable_text", "ignored_non_content", "unsupported_format",
          "unrecognized_format", "limit_exceeded", "extractor_failed"].includes(status) ||
        (status === "limit_exceeded"
          ? !["raw_bytes", "extracted_text_bytes", "preprocessing_ms"].includes(String(item.limit)) ||
            !Number.isFinite(item.observed) || !Number.isFinite(item.ceiling) ||
            Number(item.ceiling) <= 0 || Number(item.observed) < Number(item.ceiling)
          : item.reason !== undefined && typeof item.reason !== "string")) return null;
      entries.push(item as unknown as SourceEntry);
    }
    return { complete: true, entries, generationId: parsed.generationId, signature: parsed.signature };
  } catch {
    return null;
  }
}

function parseRecord(value: string, entry: SourceEntry, dimension: number): SourceRecord | null {
  try {
    const parsed: unknown = JSON.parse(value);
    if (!isRecord(parsed) || parsed.sourceKey !== entry.sourceKey || parsed.fingerprint !== entry.fingerprint ||
      parsed.vectorDimension !== dimension || !Array.isArray(parsed.chunks) || parsed.chunks.length === 0) return null;
    for (const chunk of parsed.chunks) {
      if (!isRecord(chunk) || typeof chunk.id !== "string" || !isRecord(chunk.locator) ||
        !validEncodedVector(chunk.vector, dimension)) return null;
      const locator = chunk.locator;
      if (locator.path !== entry.path || ![locator.start, locator.end].every(Number.isInteger) ||
        Number(locator.start) < 0 || Number(locator.end) <= Number(locator.start) ||
        !["markdown", "canvas"].includes(String(locator.format))) return null;
      if (locator.format === "markdown") {
        if (![locator.startLine, locator.endLine].every(Number.isInteger) ||
          Number(locator.startLine) < 1 || Number(locator.endLine) < Number(locator.startLine)) return null;
        if (locator.anchor !== undefined &&
          (!isRecord(locator.anchor) || !["heading", "block"].includes(String(locator.anchor.type)) ||
            typeof locator.anchor.value !== "string" || locator.anchor.value.length === 0)) return null;
      } else if (typeof locator.nodeId !== "string" || locator.nodeId.length === 0 ||
        typeof locator.excerpt !== "string" || locator.excerpt.length === 0) return null;
    }
    return parsed as unknown as SourceRecord;
  } catch {
    return null;
  }
}

function locatorFor(path: string, chunk: PreparedChunk | RetrievedEvidence): StoredLocator {
  if (chunk.format === "canvas") {
    if (!chunk.nodeId || !chunk.excerpt) throw new TypeError("invalid_canvas_locator");
    return {
      end: chunk.end,
      excerpt: chunk.excerpt,
      format: "canvas",
      nodeId: chunk.nodeId,
      path,
      start: chunk.start,
    };
  }
  if (chunk.startLine === undefined || chunk.endLine === undefined) {
    throw new TypeError("invalid_markdown_locator");
  }
  return {
    ...(chunk.anchor ? { anchor: chunk.anchor } : {}),
    end: chunk.end,
    endLine: chunk.endLine,
    format: "markdown",
    path,
    start: chunk.start,
    startLine: chunk.startLine,
  };
}

function sameLocator(left: StoredLocator, right: StoredLocator): boolean {
  if (left.format !== right.format || left.path !== right.path ||
    left.start !== right.start || left.end !== right.end) return false;
  return left.format === "canvas" && right.format === "canvas"
    ? left.nodeId === right.nodeId && left.excerpt === right.excerpt
    : left.format === "markdown" && right.format === "markdown" &&
      left.startLine === right.startLine && left.endLine === right.endLine &&
      left.anchor?.type === right.anchor?.type && left.anchor?.value === right.anchor?.value;
}

export class VaultIndex {
  private active?: ValidatedGeneration & { model: EmbeddingModel };
  private readonly adapter: IndexAdapter;
  private deleteOperation?: Promise<void>;
  private drain?: Promise<IndexSnapshot>;
  private readonly embed: (
    inputs: string[],
    model: EmbeddingModel,
  ) => Promise<number[][]>;
  private readonly listSources: () => VaultSource[];
  private readonly onProgress: (snapshot: IndexSnapshot) => void;
  private model?: EmbeddingModel;
  private readonly pendingPaths = new Set<string>();
  private readonly queryEmbed: (
    inputs: string[],
    model: EmbeddingModel,
  ) => Promise<number[][]>;
  private querySequence = 0;
  private rebuildQueued = false;
  private forceRebuild = false;
  private replacementQueued = false;
  private revision = 0;
  private readonly root: string;
  private snapshot: IndexSnapshot = { available: false, completed: 0, outcomes: [], phase: "idle", statuses: {}, total: 0 };
  private readonly validateModel: (model: EmbeddingModel) => Promise<boolean>;

  constructor(
    adapter: IndexAdapter,
    root: string,
    listSources: () => VaultSource[],
    embed: (inputs: string[], model: EmbeddingModel) => Promise<number[][]>,
    onProgress: (snapshot: IndexSnapshot) => void = () => undefined,
    queryEmbed: (inputs: string[], model: EmbeddingModel) => Promise<number[][]> = embed,
    validateModel: (model: EmbeddingModel) => Promise<boolean> = async () => true,
  ) {
    this.adapter = adapter;
    this.root = root;
    this.listSources = listSources;
    this.embed = embed;
    this.onProgress = onProgress;
    this.queryEmbed = queryEmbed;
    this.validateModel = validateModel;
  }

  cancel(): void {
    this.revision += 1;
    this.rebuildQueued = false;
    this.forceRebuild = false;
    const entries = this.active?.catalog.entries ?? [];
    this.update({
      available: Boolean(this.active),
      completed: entries.length,
      outcomes: outcomes(entries),
      phase: this.active ? "failed" : "idle",
      statuses: statusCounts(entries),
      total: entries.length,
    });
  }

  deleteAll(): Promise<void> {
    if (this.deleteOperation) return this.deleteOperation;
    this.revision += 1;
    this.rebuildQueued = false;
    this.forceRebuild = false;
    this.pendingPaths.clear();
    this.active = undefined;
    this.model = undefined;
    this.update({ available: false, completed: 0, outcomes: [], phase: "indexing", statuses: {}, total: 0 });
    const drain = this.drain;
    const operation = (async () => {
      try {
        await drain;
        if (await this.adapter.exists(this.root)) await this.adapter.rmdir(this.root, true);
        this.update({ available: false, completed: 0, outcomes: [], phase: "idle", statuses: {}, total: 0 });
      } catch (error) {
        this.update({ available: false, completed: 0, outcomes: [], phase: "failed", statuses: {}, total: 0 });
        throw error;
      }
    })();
    this.deleteOperation = operation;
    operation.then(
      () => { if (this.deleteOperation === operation) this.deleteOperation = undefined; },
      () => { if (this.deleteOperation === operation) this.deleteOperation = undefined; },
    );
    return operation;
  }

  getSnapshot(): IndexSnapshot {
    return {
      ...this.snapshot,
      outcomes: this.snapshot.outcomes.map((outcome) => ({ ...outcome })),
      statuses: { ...this.snapshot.statuses },
    };
  }

  getSignature(): IndexSignature | null {
    return this.active ? { ...this.active.catalog.signature } : null;
  }

  async retrieve(
    question: string,
    applyCalibratedCutoff = true,
  ): Promise<RetrievedEvidence[]> {
    const active = this.active;
    const revision = this.revision;
    if (!active || question.trim().length === 0) {
      return [];
    }
    const dimension = active.catalog.signature.vectorDimension;
    if (dimension === 0 || !active.catalog.entries.some(({ status }) => status === "indexed")) return [];
    const queryVectors = await this.queryEmbed([question], active.model);
    if (revision !== this.revision) return [];
    const query = queryVectors[0];
    if (!query || query.length !== dimension || query.some((value) => !Number.isFinite(value))) {
      throw new TypeError("invalid_query_embedding");
    }
    if (queryVectors.length !== 1) throw new TypeError("invalid_query_embedding_count");

    const ranked: { chunk: StoredChunk; entry: SourceEntry; score: number }[] = [];
    let stale = false;
    for (const entry of active.catalog.entries) {
      if (entry.status !== "indexed" || !entry.record) continue;
      const record = active.records.get(entry.record);
      if (revision !== this.revision) return [];
      if (
        !record ||
        record.chunks.length !== entry.chunkCount ||
        record.chunks.length !== entry.vectorCount ||
        record.chunks.some(
          (chunk, ordinal) =>
            chunk.id !== `${entry.sourceKey}:${entry.fingerprint}:${ordinal}`,
        )
      ) {
        stale = true;
        continue;
      }
      for (const chunk of record.chunks) {
        const vector = active.vectors.get(chunk.id);
        if (!vector) {
          stale = true;
          continue;
        }
        let score = 0;
        for (let index = 0; index < dimension; index += 1) {
          score += (query[index] ?? 0) * (vector[index] ?? 0);
        }
        ranked.push({ chunk, entry, score });
      }
    }
    ranked.sort(
      (left, right) => right.score - left.score || left.chunk.id.localeCompare(right.chunk.id),
    );

    const sources = new Map(this.listSources().map((source) => [source.path, source]));
    const evidence: RetrievedEvidence[] = [];
    const queryId = ++this.querySequence;
    const minimumScore = applyCalibratedCutoff
      ? minimumScoreFor(active.catalog.signature)
      : undefined;
    for (const candidate of ranked) {
      if (minimumScore !== undefined && candidate.score < minimumScore) continue;
      const source = sources.get(candidate.entry.path);
      const hydrated = source
        ? await this.hydrate(candidate.entry, candidate.chunk, candidate.score, source)
        : null;
      if (revision !== this.revision) return [];
      if (!hydrated) {
        stale = true;
        continue;
      }
      evidence.push({ ...hydrated, citationId: `S${queryId}-${evidence.length + 1}` });
      if (evidence.length === (applyCalibratedCutoff ? 6 : 4)) break;
    }
    if (stale) this.queueReplacement(active.model);
    return evidence;
  }

  async retrievePaths(paths: Iterable<string>): Promise<RetrievedEvidence[]> {
    const active = this.active;
    const revision = this.revision;
    if (!active) return [];

    const entries = new Map(active.catalog.entries.map((entry) => [entry.path, entry]));
    const sources = new Map(this.listSources().map((source) => [source.path, source]));
    const evidence: RetrievedEvidence[] = [];
    const queryId = ++this.querySequence;
    let stale = false;
    for (const path of new Set(paths)) {
      const entry = entries.get(path);
      if (entry?.status !== "indexed" || !entry.record) continue;
      const record = active.records.get(entry.record);
      const chunk = record?.chunks[0];
      const source = sources.get(path);
      const hydrated = chunk && source
        ? await this.hydrate(entry, chunk, 1, source)
        : null;
      if (revision !== this.revision) return [];
      if (!hydrated) {
        stale = true;
        continue;
      }
      evidence.push({ ...hydrated, citationId: `S${queryId}-${evidence.length + 1}` });
      if (evidence.length === 4) break;
    }
    if (stale) this.queueReplacement(active.model);
    return evidence;
  }

  async resolveEvidence(evidence: RetrievedEvidence): Promise<RetrievedEvidence | null> {
    if (!this.hasEvidence(evidence)) return null;
    const source = this.listSources().find(({ path }) => path === evidence.path);
    if (!source) {
      if (this.active) this.queueReplacement(this.active.model);
      return null;
    }
    const entry: SourceEntry = {
      fingerprint: evidence.fingerprint,
      path: evidence.path,
      sourceKey: evidence.chunkId.split(":", 1)[0] ?? "",
      status: "indexed",
    };
    const chunk: StoredChunk = {
      id: evidence.chunkId,
      locator: locatorFor(evidence.path, evidence),
      vector: "",
    };
    const hydrated = await this.hydrate(entry, chunk, evidence.score, source);
    if (!this.hasEvidence(evidence)) return null;
    if (!hydrated && this.active) this.queueReplacement(this.active.model);
    return hydrated ? { ...hydrated, citationId: evidence.citationId } : null;
  }

  // ponytail: rebuild the whole generation; go per-source only if change latency becomes material.
  invalidate(paths: Iterable<string>): Promise<IndexSnapshot> {
    if (this.deleteOperation) return this.deleteOperation.then(() => this.getSnapshot());
    for (const path of paths) {
      if (path) this.pendingPaths.add(path);
    }
    if (this.pendingPaths.size === 0) return Promise.resolve(this.getSnapshot());

    this.revision += 1;
    if (this.active) {
      this.active.catalog.entries = this.active.catalog.entries.filter(
        ({ path }) => !this.pendingPaths.has(path),
      );
    }
    this.rebuildQueued = true;
    if (this.model) {
      this.update({
        ...this.snapshot,
        outcomes: this.snapshot.outcomes.filter(({ path }) => !this.pendingPaths.has(path)),
        phase: "indexing",
        statuses: this.active ? statusCounts(this.active.catalog.entries) : {},
      });
      return this.queueDrain();
    }
    return Promise.resolve(this.getSnapshot());
  }

  private hasEvidence(evidence: RetrievedEvidence): boolean {
    return this.active?.catalog.entries.some(
      (entry) =>
        entry.status === "indexed" &&
        entry.path === evidence.path &&
        entry.fingerprint === evidence.fingerprint &&
        evidence.chunkId.startsWith(`${entry.sourceKey}:`),
    ) ?? false;
  }

  private queueReplacement(model: EmbeddingModel): void {
    if (this.replacementQueued) return;
    this.replacementQueued = true;
    void this.start(model).finally(() => {
      this.replacementQueued = false;
    });
  }

  private async hydrate(
    entry: SourceEntry,
    stored: StoredChunk,
    score: number,
    source: VaultSource,
  ): Promise<Omit<RetrievedEvidence, "citationId"> | null> {
    const prepared = await prepareSource(source);
    if (
      prepared.entry.status !== "indexed" ||
      prepared.entry.fingerprint !== entry.fingerprint ||
      prepared.entry.sourceKey !== entry.sourceKey
    ) {
      return null;
    }
    const chunk = prepared.chunks.find((item) => sameLocator(stored.locator, locatorFor(source.path, item)));
    if (!chunk || !entry.fingerprint) return null;
    return {
      ...(chunk.format === "markdown" && chunk.anchor ? { anchor: chunk.anchor } : {}),
      chunkId: stored.id,
      end: chunk.end,
      ...(chunk.format === "markdown" ? { endLine: chunk.endLine, startLine: chunk.startLine } : {
        excerpt: chunk.excerpt,
        nodeId: chunk.nodeId,
      }),
      fingerprint: entry.fingerprint,
      format: chunk.format,
      path: source.path,
      score,
      start: chunk.start,
      text: chunk.text,
    };
  }

  start(model: EmbeddingModel): Promise<IndexSnapshot> {
    if (this.deleteOperation) return this.deleteOperation.then(() => this.getSnapshot());
    this.model = model;
    this.rebuildQueued = true;
    this.revision += 1;
    if (validModel(model)) {
      this.update({ ...this.snapshot, phase: "indexing" });
    }
    return this.queueDrain();
  }

  rebuild(model: EmbeddingModel): Promise<IndexSnapshot> {
    if (this.deleteOperation) return this.deleteOperation.then(() => this.getSnapshot());
    this.forceRebuild = true;
    return this.start(model);
  }

  private queueDrain(): Promise<IndexSnapshot> {
    if (this.drain) return this.drain;
    this.drain = Promise.resolve()
      .then(async () => {
        let result = this.getSnapshot();
        while (this.rebuildQueued && this.model) {
          this.rebuildQueued = false;
          const replacementPaths = new Set(this.pendingPaths);
          this.pendingPaths.clear();
          const force = this.forceRebuild;
          this.forceRebuild = false;
          result = await this.run(this.model, this.revision, force, replacementPaths);
        }
        return result;
      })
      .finally(() => {
        this.drain = undefined;
      });
    return this.drain;
  }

  private async run(
    model: EmbeddingModel,
    revision: number,
    force: boolean,
    replacementPaths: Set<string>,
  ): Promise<IndexSnapshot> {
    if (!validModel(model)) {
      const entries = this.active?.catalog.entries ?? [];
      return this.update({ available: Boolean(this.active), completed: 0, outcomes: outcomes(entries), phase: "failed", statuses: statusCounts(entries), total: 0 });
    }
    const sources = this.listSources().sort((left, right) => left.path.localeCompare(right.path));
    try {
      if (!(await this.validateModel(model))) throw new Error("embedding_model_unavailable");
      this.assertCurrent(revision);
      this.update({ available: Boolean(this.active), completed: 0, outcomes: [], phase: "indexing", statuses: {}, total: sources.length });
      if (!force) {
        const restored = await this.restore(model, sources, revision);
        if (restored) return restored;
      }
      if (!this.active) await this.cleanupGenerations();
      return await this.build(model, sources, revision, replacementPaths);
    } catch {
      if (revision !== this.revision) return this.getSnapshot();
      const entries = this.active?.catalog.entries ?? [];
      return this.update({ available: Boolean(this.active), completed: 0, outcomes: outcomes(entries), phase: "failed", statuses: statusCounts(entries), total: sources.length });
    }
  }

  private update(snapshot: IndexSnapshot): IndexSnapshot {
    this.snapshot = snapshot;
    this.onProgress(this.getSnapshot());
    return this.getSnapshot();
  }

  private assertCurrent(revision: number): void {
    if (revision !== this.revision) throw new Error("canceled");
  }

  private async ensureDirectories(generationId: string): Promise<string> {
    const generations = `${this.root}/generations`;
    const generation = `${generations}/${generationId}`;
    for (const path of [this.root, generations, generation, `${generation}/records`]) {
      if (!(await this.adapter.exists(path))) await this.adapter.mkdir(path);
    }
    return generation;
  }

  private async cleanupGenerations(keepGenerationId?: string): Promise<void> {
    const generations = `${this.root}/generations`;
    const listed = await this.adapter.list(generations).catch(() => ({ files: [], folders: [] }));
    const prefix = `${generations}/`;
    for (const folder of listed.folders) {
      const generationId = folder.startsWith(prefix) ? folder.slice(prefix.length) : "";
      if (!generationId || generationId.includes("/") || generationId === keepGenerationId) continue;
      await this.adapter.rmdir(folder, true).catch(() => undefined);
    }
  }

  private async validateGeneration(
    generationId: string,
    model: EmbeddingModel,
  ): Promise<ValidatedGeneration | null> {
    try {
      const generation = `${this.root}/generations/${generationId}`;
      const catalog = parseCatalog(await this.adapter.read(`${generation}/catalog.json`), model);
      if (!catalog || catalog.generationId !== generationId) return null;
      const paths = new Set<string>();
      const records = new Map<string, SourceRecord>();
      const vectors = new Map<string, Float32Array>();
      for (const entry of catalog.entries) {
        if (paths.has(entry.path)) return null;
        paths.add(entry.path);
        if (entry.status !== "indexed") {
          if (entry.record !== undefined || entry.chunkCount !== undefined ||
            entry.vectorCount !== undefined) return null;
          continue;
        }
        const recordName = `records/${entry.sourceKey}-${entry.fingerprint}.json`;
        if (entry.record !== recordName || !Number.isInteger(entry.chunkCount) ||
          entry.chunkCount !== entry.vectorCount || !(await this.adapter.exists(`${generation}/${recordName}`))) {
          return null;
        }
        const record = parseRecord(
          await this.adapter.read(`${generation}/${recordName}`),
          entry,
          catalog.signature.vectorDimension,
        );
        if (!record || record.chunks.length !== entry.chunkCount || record.chunks.some(
          (chunk, ordinal) => chunk.id !== `${entry.sourceKey}:${entry.fingerprint}:${ordinal}`,
        )) return null;
        records.set(recordName, record);
        for (const chunk of record.chunks) {
          vectors.set(chunk.id, decodeVector(chunk.vector, catalog.signature.vectorDimension));
        }
      }
      return { catalog, records, vectors };
    } catch {
      return null;
    }
  }

  private async restore(
    model: EmbeddingModel,
    sources: VaultSource[],
    revision: number,
  ): Promise<IndexSnapshot | null> {
    const activePath = `${this.root}/active.json`;
    if (!(await this.adapter.exists(activePath))) return null;
    let active: unknown;
    try {
      active = JSON.parse(await this.adapter.read(activePath));
    } catch {
      return null;
    }
    if (!isRecord(active) || !isGenerationId(active.generationId)) return null;
    const validated = await this.validateGeneration(active.generationId, model);
    if (!validated) return null;
    const { catalog } = validated;
    const entriesByPath = new Map(catalog.entries.map((entry) => [entry.path, entry]));
    const reusable: SourceEntry[] = [];
    for (const source of sources) {
      this.assertCurrent(revision);
      const expected = entriesByPath.get(source.path);
      const prepared = await prepareSource(source);
      if (!expected || !exactSourceMatch(prepared.entry, expected)) continue;
      if (expected.status === "indexed") {
        const recordName = `records/${expected.sourceKey}-${expected.fingerprint}.json`;
        const record = validated.records.get(recordName);
        if (!record || record.chunks.length !== prepared.chunks.length ||
          record.chunks.length !== expected.chunkCount || record.chunks.length !== expected.vectorCount) return null;
        let locatorsMatch = true;
        for (let ordinal = 0; ordinal < prepared.chunks.length; ordinal += 1) {
          const stored = record.chunks[ordinal];
          const current = prepared.chunks[ordinal];
          if (!stored || !current || stored.id !== `${expected.sourceKey}:${expected.fingerprint}:${ordinal}` ||
            !sameLocator(stored.locator, locatorFor(source.path, current))) {
            locatorsMatch = false;
            break;
          }
        }
        if (!locatorsMatch) continue;
      }
      reusable.push(expected);
    }
    this.assertCurrent(revision);
    const currentCatalog = { ...catalog, entries: reusable };
    this.active = { ...validated, catalog: currentCatalog, model };
    if (reusable.length !== sources.length || catalog.entries.length !== sources.length) {
      this.update({
        available: true,
        completed: 0,
        outcomes: outcomes(reusable),
        phase: "indexing",
        statuses: statusCounts(reusable),
        total: sources.length,
      });
      return null;
    }
    await this.cleanupGenerations(catalog.generationId);
    return this.update({
      available: true,
      completed: sources.length,
      outcomes: outcomes(catalog.entries),
      phase: "ready",
      statuses: statusCounts(catalog.entries),
      total: sources.length,
    });
  }

  private async build(
    model: EmbeddingModel,
    sources: VaultSource[],
    revision: number,
    replacementPaths: Set<string>,
  ): Promise<IndexSnapshot> {
    const generationId = globalThis.crypto.randomUUID();
    const generation = await this.ensureDirectories(generationId);
    const entries: SourceEntry[] = [];
    const records = new Map<string, SourceRecord>();
    const vectors = new Map<string, Float32Array>();
    let catalogCommitted = false;
    const reusable = compatibleSignature(this.active?.catalog.signature, model)
      ? new Map(this.active?.catalog.entries.map((entry) => [entry.path, entry]))
      : new Map<string, SourceEntry>();
    let vectorDimension = this.active && reusable.size > 0
      ? this.active.catalog.signature.vectorDimension
      : 0;
    try {
      const orderedSources = [
        ...sources.filter(({ path }) => !replacementPaths.has(path)),
        ...sources.filter(({ path }) => replacementPaths.has(path)),
      ];
      for (const source of orderedSources) {
        this.assertCurrent(revision);
        const prepared = await prepareSource(source);
        const entry = prepared.entry;
        const previous = reusable.get(entry.path);
        let reused = previous !== undefined && exactSourceMatch(entry, previous);
        if (reused && entry.status === "indexed" && previous?.record) {
          const record = this.active?.records.get(previous.record);
          reused = Boolean(record) && record?.chunks.length === prepared.chunks.length &&
            record.chunks.every((chunk, ordinal) => {
              const current = prepared.chunks[ordinal];
              return current !== undefined && sameLocator(chunk.locator, locatorFor(source.path, current));
            });
          if (reused && record) {
            entry.chunkCount = previous.chunkCount;
            entry.record = previous.record;
            entry.vectorCount = previous.vectorCount;
            await this.adapter.write(`${generation}/${previous.record}`, JSON.stringify(record));
            records.set(previous.record, record);
            for (const chunk of record.chunks) {
              const vector = this.active?.vectors.get(chunk.id);
              if (vector) vectors.set(chunk.id, vector);
            }
          }
        }
        if (!reused && entry.status === "indexed") {
          const storedChunks: StoredChunk[] = [];
          for (let offset = 0; offset < prepared.chunks.length; offset += EMBEDDING_BATCH_SIZE) {
            const batch = prepared.chunks.slice(offset, offset + EMBEDDING_BATCH_SIZE);
            const embedded = await this.embed(batch.map((chunk) => chunk.text), model);
            this.assertCurrent(revision);
            if (embedded.length !== batch.length) throw new TypeError("invalid_embedding_vector_count");
            for (let index = 0; index < batch.length; index += 1) {
              const chunk = batch[index];
              const vector = embedded[index];
              if (!chunk || !vector || vector.length === 0 || vector.some((value) => !Number.isFinite(value))) {
                throw new TypeError("invalid_embedding_vector");
              }
              vectorDimension ||= vector.length;
              if (vector.length !== vectorDimension) throw new TypeError("inconsistent_embedding_dimension");
              const ordinal = offset + index;
              const id = `${entry.sourceKey}:${entry.fingerprint}:${ordinal}`;
              storedChunks.push({
                id,
                locator: locatorFor(source.path, chunk),
                vector: encodeVector(vector),
              });
              vectors.set(id, Float32Array.from(vector));
            }
          }
          const recordName = `records/${entry.sourceKey}-${entry.fingerprint}.json`;
          const record: SourceRecord = {
            chunks: storedChunks,
            fingerprint: entry.fingerprint ?? "",
            sourceKey: entry.sourceKey,
            vectorDimension,
          };
          entry.chunkCount = storedChunks.length;
          entry.record = recordName;
          entry.vectorCount = storedChunks.length;
          records.set(recordName, record);
          await this.adapter.write(`${generation}/${recordName}`, JSON.stringify(record));
          const validated = parseRecord(await this.adapter.read(`${generation}/${recordName}`), entry, vectorDimension);
          if (!validated) throw new TypeError("invalid_persisted_record");
          records.set(recordName, validated);
        }
        entries.push(entry);
        this.update({
          available: Boolean(this.active),
          completed: entries.length,
          latestPath: source.path,
          outcomes: outcomes(entries),
          phase: "indexing",
          statuses: statusCounts(entries),
          total: sources.length,
        });
      }

      this.assertCurrent(revision);
      const catalog: Catalog = {
        complete: true,
        entries,
        generationId,
        signature: signatureFor(model, vectorDimension),
      };
      await this.adapter.write(`${generation}/catalog.json`, JSON.stringify(catalog));
      const validated = await this.validateGeneration(generationId, model);
      if (!validated) throw new TypeError("invalid_persisted_generation");

      const activePath = `${this.root}/active.json`;
      if (!(await this.adapter.exists(activePath))) await this.adapter.write(activePath, "");
      this.assertCurrent(revision);
      await this.adapter.process(activePath, () => {
        this.assertCurrent(revision);
        return JSON.stringify({ generationId });
      });
      catalogCommitted = true;
      this.assertCurrent(revision);
      this.active = { ...validated, model };
      const ready = this.update({
        available: true,
        completed: sources.length,
        outcomes: outcomes(entries),
        phase: "ready",
        statuses: statusCounts(entries),
        total: sources.length,
      });
      await this.cleanupGenerations(generationId);
      return ready;
    } catch (error) {
      if (!catalogCommitted) {
        await this.adapter.rmdir(generation, true).catch(() => undefined);
      }
      throw error;
    }
  }
}

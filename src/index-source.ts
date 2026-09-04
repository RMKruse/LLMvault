export const CHUNK_TARGET_BYTES = 2_048;
export const CHUNK_OVERLAP_BYTES = 512;
export const RAW_FILE_LIMIT_BYTES = 10 * 1024 * 1024;
export const EXTRACTED_TEXT_LIMIT_BYTES = 5 * 1024 * 1024;
export const PREPROCESSING_LIMIT_MS = 5_000;

export type TerminalStatus =
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

export type SourceEntry = SourceOutcome & {
  chunkCount?: number;
  fingerprint: string | null;
  record?: string;
  sourceKey: string;
  vectorCount?: number;
};

export type PreparedChunk =
  | (MarkdownChunk & { format: "markdown" })
  | (CanvasChunk & { format: "canvas" });

export type StoredLocator =
  | (Omit<MarkdownChunk, "text"> & { format: "markdown"; path: string })
  | (Omit<CanvasChunk, "text"> & { format: "canvas"; path: string });

export interface PreparedSource {
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

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export async function prepareSource(source: VaultSource): Promise<PreparedSource> {
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

export function statusCounts(entries: SourceEntry[]): Partial<Record<TerminalStatus, number>> {
  const counts: Partial<Record<TerminalStatus, number>> = {};
  for (const entry of entries) counts[entry.status] = (counts[entry.status] ?? 0) + 1;
  return counts;
}

export function outcomes(entries: SourceEntry[]): SourceOutcome[] {
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

export function exactSourceMatch(left: SourceEntry, right: SourceEntry): boolean {
  return left.fingerprint === right.fingerprint && left.sourceKey === right.sourceKey &&
    left.status === right.status;
}

export function matchesPreparedSource(prepared: PreparedSource, entry: SourceEntry & { chunks: { locator: StoredLocator }[] }): boolean {
  return exactSourceMatch(prepared.entry, entry) &&
    (entry.status !== "indexed" || (entry.chunks.length === prepared.chunks.length &&
      entry.chunks.every((chunk, ordinal) => {
        const current = prepared.chunks[ordinal];
        return current !== undefined && sameLocator(chunk.locator, locatorFor(entry.path, current));
      })));
}

export function isStoredLocator(value: unknown): value is StoredLocator {
  if (!isRecord(value) || typeof value.path !== "string" ||
    ![value.start, value.end].every(Number.isInteger) ||
    Number(value.start) < 0 || Number(value.end) <= Number(value.start)) return false;
  if (value.format === "markdown") {
    return [value.startLine, value.endLine].every(Number.isInteger) &&
      Number(value.startLine) >= 1 && Number(value.endLine) >= Number(value.startLine) &&
      (value.anchor === undefined || (isRecord(value.anchor) &&
        (value.anchor.type === "heading" || value.anchor.type === "block") &&
        typeof value.anchor.value === "string" && value.anchor.value.length > 0));
  }
  return value.format === "canvas" && typeof value.nodeId === "string" && value.nodeId.length > 0 &&
    typeof value.excerpt === "string" && value.excerpt.length > 0;
}

export function locatorFor(path: string, chunk: PreparedChunk | StoredLocator): StoredLocator {
  if (chunk.format === "canvas") {
    return {
      end: chunk.end,
      excerpt: chunk.excerpt,
      format: "canvas",
      nodeId: chunk.nodeId,
      path,
      start: chunk.start,
    };
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

export function sameLocator(left: StoredLocator, right: StoredLocator): boolean {
  if (left.format !== right.format || left.path !== right.path ||
    left.start !== right.start || left.end !== right.end) return false;
  return left.format === "canvas" && right.format === "canvas"
    ? left.nodeId === right.nodeId && left.excerpt === right.excerpt
    : left.format === "markdown" && right.format === "markdown" &&
      left.startLine === right.startLine && left.endLine === right.endLine &&
      left.anchor?.type === right.anchor?.type && left.anchor?.value === right.anchor?.value;
}

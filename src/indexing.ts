export const CHUNK_TARGET_BYTES = 2_048;
export const CHUNK_OVERLAP_BYTES = 512;

const EMBEDDING_BATCH_SIZE = 16;
const SCHEMA_VERSION = 1;
const EXTRACTOR_VERSION = 1;
const CHUNKER_VERSION = 1;

type TerminalStatus =
  | "indexed"
  | "no_extractable_text"
  | "extractor_failed";

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

export interface MarkdownSource {
  anchors?: MarkdownAnchor[];
  path: string;
  read(): Promise<string>;
}

export interface EmbeddingModel {
  digest: string;
  name: string;
}

export interface IndexSnapshot {
  completed: number;
  latestPath?: string;
  phase: "idle" | "indexing" | "ready" | "failed";
  statuses: Partial<Record<TerminalStatus, number>>;
  total: number;
}

export interface RetrievedEvidence extends Omit<MarkdownChunk, "text"> {
  citationId: string;
  chunkId: string;
  fingerprint: string;
  path: string;
  score: number;
  text: string;
}

interface IndexAdapter {
  exists(path: string): Promise<boolean>;
  mkdir(path: string): Promise<void>;
  process(path: string, update: (value: string) => string): Promise<string>;
  read(path: string): Promise<string>;
  rmdir(path: string, recursive: boolean): Promise<void>;
  write(path: string, value: string): Promise<void>;
}

interface SourceEntry {
  chunkCount?: number;
  fingerprint: string | null;
  path: string;
  reason?: "read_failed";
  record?: string;
  sourceKey: string;
  status: TerminalStatus;
  vectorCount?: number;
}

interface Signature {
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
  signature: Signature;
}

interface StoredChunk {
  id: string;
  locator: Omit<MarkdownChunk, "text"> & { path: string };
  vector: string;
}

interface SourceRecord {
  chunks: StoredChunk[];
  fingerprint: string;
  sourceKey: string;
  vectorDimension: number;
}

interface PreparedSource {
  chunks: MarkdownChunk[];
  entry: SourceEntry;
}

interface MarkdownPiece {
  bytes: number;
  completeBlock: boolean;
  end: number;
  start: number;
}

const encoder = new TextEncoder();

function byteLength(value: string): number {
  return encoder.encode(value).byteLength;
}

function lineAt(source: string, offset: number): number {
  let line = 1;
  for (let index = 0; index < offset; index += 1) {
    if (source.charCodeAt(index) === 10) line += 1;
  }
  return line;
}

function sourceBlocks(source: string, headingOffsets: Set<number>): MarkdownPiece[] {
  const blocks: MarkdownPiece[] = [];
  let blockStart = 0;
  let offset = 0;
  for (const line of source.matchAll(/.*(?:\n|$)/g)) {
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
  return blocks.flatMap((block) => splitOversizedPiece(source, block));
}

function splitOversizedPiece(source: string, piece: MarkdownPiece): MarkdownPiece[] {
  if (piece.bytes <= CHUNK_TARGET_BYTES) return [piece];
  const split: MarkdownPiece[] = [];
  let packedStart = piece.start;
  let packedEnd = piece.start;
  let packedBytes = 0;
  for (const line of source.slice(piece.start, piece.end).matchAll(/.*(?:\n|$)/g)) {
    if (line[0].length === 0) continue;
    const lineStart = piece.start + line.index;
    const lineEnd = lineStart + line[0].length;
    const lineBytes = byteLength(line[0]);
    if (lineBytes > CHUNK_TARGET_BYTES) {
      if (packedBytes > 0) split.push({ bytes: packedBytes, completeBlock: false, end: packedEnd, start: packedStart });
      let start = lineStart;
      let end = start;
      let bytes = 0;
      for (const character of line[0]) {
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

export function chunkMarkdown(source: string, sourceAnchors: MarkdownAnchor[] = []): MarkdownChunk[] {
  if (source.length === 0) return [];
  const sortedAnchors = sourceAnchors
    .filter((anchor) => Number.isInteger(anchor.offset) && anchor.offset >= 0 &&
      anchor.offset < source.length && anchor.value.length > 0)
    .sort((left, right) => left.offset - right.offset);
  const pieces = sourceBlocks(
    source,
    new Set(sortedAnchors.filter((anchor) => anchor.type === "heading").map((anchor) => anchor.offset)),
  );
  const chunks: MarkdownChunk[] = [];
  let cursor = 0;

  while (cursor < pieces.length) {
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
      endLine: lineAt(source, Math.max(start, end - 1)),
      start,
      startLine: lineAt(source, start),
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

function signatureFor(model: EmbeddingModel, vectorDimension: number): Signature {
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

function compatibleSignature(value: unknown, model: EmbeddingModel): value is Signature {
  if (!isRecord(value)) return false;
  const expected = signatureFor(model, Number(value.vectorDimension));
  return Number.isInteger(value.vectorDimension) && Number(value.vectorDimension) >= 0 &&
    Object.entries(expected).every(([key, expectedValue]) => value[key] === expectedValue);
}

async function prepareSource(source: MarkdownSource): Promise<PreparedSource> {
  const sourceKey = await sha256(source.path);
  try {
    const text = await source.read();
    const fingerprint = await sha256(`md\0${text}`);
    if (text.trim().length === 0) {
      return { chunks: [], entry: { fingerprint, path: source.path, sourceKey, status: "no_extractable_text" } };
    }
    const chunks = chunkMarkdown(text, source.anchors);
    return { chunks, entry: { fingerprint, path: source.path, sourceKey, status: "indexed" } };
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
  }
}

function statusCounts(entries: SourceEntry[]): Partial<Record<TerminalStatus, number>> {
  const counts: Partial<Record<TerminalStatus, number>> = {};
  for (const entry of entries) counts[entry.status] = (counts[entry.status] ?? 0) + 1;
  return counts;
}

function parseCatalog(value: string, model: EmbeddingModel): Catalog | null {
  try {
    const parsed: unknown = JSON.parse(value);
    if (!isRecord(parsed) || parsed.complete !== true || !isGenerationId(parsed.generationId) ||
      !compatibleSignature(parsed.signature, model) || !Array.isArray(parsed.entries)) return null;
    const entries: SourceEntry[] = [];
    for (const item of parsed.entries) {
      if (!isRecord(item) || typeof item.path !== "string" || typeof item.sourceKey !== "string" ||
        !(typeof item.fingerprint === "string" || item.fingerprint === null) ||
        !["indexed", "no_extractable_text", "extractor_failed"].includes(String(item.status))) return null;
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
      if (locator.path !== entry.path || ![locator.start, locator.end, locator.startLine, locator.endLine].every(Number.isInteger) ||
        Number(locator.start) < 0 || Number(locator.end) <= Number(locator.start) ||
        Number(locator.startLine) < 1 || Number(locator.endLine) < Number(locator.startLine)) return null;
      if (locator.anchor !== undefined &&
        (!isRecord(locator.anchor) || !["heading", "block"].includes(String(locator.anchor.type)) ||
          typeof locator.anchor.value !== "string" || locator.anchor.value.length === 0)) return null;
    }
    return parsed as unknown as SourceRecord;
  } catch {
    return null;
  }
}

function locatorFor(path: string, chunk: MarkdownChunk): StoredChunk["locator"] {
  return {
    ...(chunk.anchor ? { anchor: chunk.anchor } : {}),
    end: chunk.end,
    endLine: chunk.endLine,
    path,
    start: chunk.start,
    startLine: chunk.startLine,
  };
}

function sameLocator(left: StoredChunk["locator"], right: StoredChunk["locator"]): boolean {
  return left.path === right.path && left.start === right.start && left.end === right.end &&
    left.startLine === right.startLine && left.endLine === right.endLine &&
    left.anchor?.type === right.anchor?.type && left.anchor?.value === right.anchor?.value;
}

export class MarkdownIndex {
  private active?: { catalog: Catalog; model: EmbeddingModel };
  private readonly adapter: IndexAdapter;
  private readonly embed: (
    inputs: string[],
    model: EmbeddingModel,
  ) => Promise<number[][]>;
  private readonly listSources: () => MarkdownSource[];
  private readonly onProgress: (snapshot: IndexSnapshot) => void;
  private readonly queryEmbed: (
    inputs: string[],
    model: EmbeddingModel,
  ) => Promise<number[][]>;
  private querySequence = 0;
  private replacementQueued = false;
  private revision = 0;
  private readonly root: string;
  private snapshot: IndexSnapshot = { completed: 0, phase: "idle", statuses: {}, total: 0 };
  private readonly validateModel: (model: EmbeddingModel) => Promise<boolean>;

  constructor(
    adapter: IndexAdapter,
    root: string,
    listSources: () => MarkdownSource[],
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
  }

  getSnapshot(): IndexSnapshot {
    return { ...this.snapshot, statuses: { ...this.snapshot.statuses } };
  }

  async retrieve(question: string): Promise<RetrievedEvidence[]> {
    const active = this.active;
    if (this.snapshot.phase !== "ready" || !active || question.trim().length === 0) {
      return [];
    }
    const dimension = active.catalog.signature.vectorDimension;
    if (dimension === 0) return [];
    const queryVectors = await this.queryEmbed([question], active.model);
    const query = queryVectors[0];
    if (!query || query.length !== dimension || query.some((value) => !Number.isFinite(value))) {
      throw new TypeError("invalid_query_embedding");
    }
    if (queryVectors.length !== 1) throw new TypeError("invalid_query_embedding_count");

    const ranked: { chunk: StoredChunk; entry: SourceEntry; score: number }[] = [];
    let stale = false;
    const generation = `${this.root}/generations/${active.catalog.generationId}`;
    for (const entry of active.catalog.entries) {
      if (entry.status !== "indexed" || !entry.record) continue;
      const record = parseRecord(
        await this.adapter.read(`${generation}/${entry.record}`).catch(() => ""),
        entry,
        dimension,
      );
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
        const vector = decodeVector(chunk.vector, dimension);
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
    for (const candidate of ranked) {
      const source = sources.get(candidate.entry.path);
      const hydrated = source
        ? await this.hydrate(candidate.entry, candidate.chunk, candidate.score, source)
        : null;
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
    if (!hydrated && this.active) this.queueReplacement(this.active.model);
    return hydrated ? { ...hydrated, citationId: evidence.citationId } : null;
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
    source: MarkdownSource,
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
      ...(chunk.anchor ? { anchor: chunk.anchor } : {}),
      chunkId: stored.id,
      end: chunk.end,
      endLine: chunk.endLine,
      fingerprint: entry.fingerprint,
      path: source.path,
      score,
      start: chunk.start,
      startLine: chunk.startLine,
      text: chunk.text,
    };
  }

  async start(model: EmbeddingModel): Promise<IndexSnapshot> {
    const revision = ++this.revision;
    if (!validModel(model)) return this.update({ completed: 0, phase: "failed", statuses: {}, total: 0 });
    const sources = this.listSources().sort((left, right) => left.path.localeCompare(right.path));
    try {
      if (!(await this.validateModel(model))) throw new Error("embedding_model_unavailable");
      this.assertCurrent(revision);
      this.update({ completed: 0, phase: "indexing", statuses: {}, total: sources.length });
      const restored = await this.restore(model, sources, revision);
      if (restored) return restored;
      return await this.build(model, sources, revision);
    } catch {
      if (revision !== this.revision) return this.getSnapshot();
      return this.update({ completed: 0, phase: "failed", statuses: {}, total: sources.length });
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

  private async restore(
    model: EmbeddingModel,
    sources: MarkdownSource[],
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
    const generation = `${this.root}/generations/${active.generationId}`;
    const catalog = parseCatalog(await this.adapter.read(`${generation}/catalog.json`), model);
    if (!catalog || catalog.generationId !== active.generationId || catalog.entries.length !== sources.length) return null;
    const entriesByPath = new Map(catalog.entries.map((entry) => [entry.path, entry]));
    for (const source of sources) {
      this.assertCurrent(revision);
      const expected = entriesByPath.get(source.path);
      if (!expected) return null;
      const prepared = await prepareSource(source);
      if (prepared.entry.fingerprint !== expected.fingerprint || prepared.entry.status !== expected.status ||
        prepared.entry.sourceKey !== expected.sourceKey) return null;
      if (expected.status === "indexed") {
        const recordName = `records/${expected.sourceKey}-${expected.fingerprint}.json`;
        if (expected.record !== recordName || !(await this.adapter.exists(`${generation}/${recordName}`))) return null;
        const record = parseRecord(
          await this.adapter.read(`${generation}/${recordName}`),
          expected,
          catalog.signature.vectorDimension,
        );
        if (!record || record.chunks.length !== prepared.chunks.length ||
          record.chunks.length !== expected.chunkCount || record.chunks.length !== expected.vectorCount) return null;
        for (let ordinal = 0; ordinal < prepared.chunks.length; ordinal += 1) {
          const stored = record.chunks[ordinal];
          const current = prepared.chunks[ordinal];
          if (!stored || !current || stored.id !== `${expected.sourceKey}:${expected.fingerprint}:${ordinal}` ||
            !sameLocator(stored.locator, locatorFor(source.path, current))) return null;
        }
      }
    }
    this.active = { catalog, model };
    return this.update({
      completed: sources.length,
      phase: "ready",
      statuses: statusCounts(catalog.entries),
      total: sources.length,
    });
  }

  private async build(
    model: EmbeddingModel,
    sources: MarkdownSource[],
    revision: number,
  ): Promise<IndexSnapshot> {
    const generationId = globalThis.crypto.randomUUID();
    const generation = await this.ensureDirectories(generationId);
    const entries: SourceEntry[] = [];
    let vectorDimension = 0;
    try {
      for (const source of sources) {
        this.assertCurrent(revision);
        const prepared = await prepareSource(source);
        const entry = prepared.entry;
        if (entry.status === "indexed") {
          const storedChunks: StoredChunk[] = [];
          for (let offset = 0; offset < prepared.chunks.length; offset += EMBEDDING_BATCH_SIZE) {
            const batch = prepared.chunks.slice(offset, offset + EMBEDDING_BATCH_SIZE);
            const vectors = await this.embed(batch.map((chunk) => chunk.text), model);
            this.assertCurrent(revision);
            if (vectors.length !== batch.length) throw new TypeError("invalid_embedding_vector_count");
            for (let index = 0; index < batch.length; index += 1) {
              const chunk = batch[index];
              const vector = vectors[index];
              if (!chunk || !vector || vector.length === 0 || vector.some((value) => !Number.isFinite(value))) {
                throw new TypeError("invalid_embedding_vector");
              }
              vectorDimension ||= vector.length;
              if (vector.length !== vectorDimension) throw new TypeError("inconsistent_embedding_dimension");
              const ordinal = offset + index;
              storedChunks.push({
                id: `${entry.sourceKey}:${entry.fingerprint}:${ordinal}`,
                locator: locatorFor(source.path, chunk),
                vector: encodeVector(vector),
              });
            }
          }
          const recordName = `records/${entry.sourceKey}-${entry.fingerprint}.json`;
          const record: SourceRecord = {
            chunks: storedChunks,
            fingerprint: entry.fingerprint ?? "",
            sourceKey: entry.sourceKey,
            vectorDimension,
          };
          await this.adapter.write(`${generation}/${recordName}`, JSON.stringify(record));
          const validated = parseRecord(await this.adapter.read(`${generation}/${recordName}`), entry, vectorDimension);
          if (!validated) throw new TypeError("invalid_persisted_record");
          entry.chunkCount = storedChunks.length;
          entry.record = recordName;
          entry.vectorCount = storedChunks.length;
        }
        entries.push(entry);
        this.update({
          completed: entries.length,
          latestPath: source.path,
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
      if (!parseCatalog(await this.adapter.read(`${generation}/catalog.json`), model)) {
        throw new TypeError("invalid_persisted_catalog");
      }

      const activePath = `${this.root}/active.json`;
      let previousGeneration: string | undefined;
      if (await this.adapter.exists(activePath)) {
        try {
          const previous: unknown = JSON.parse(await this.adapter.read(activePath));
          if (isRecord(previous) && isGenerationId(previous.generationId)) previousGeneration = previous.generationId;
        } catch {
          previousGeneration = undefined;
        }
      }
      if (!(await this.adapter.exists(activePath))) await this.adapter.write(activePath, "");
      await this.adapter.process(activePath, () => JSON.stringify({ generationId }));
      this.active = { catalog, model };
      const ready = this.update({
        completed: sources.length,
        phase: "ready",
        statuses: statusCounts(entries),
        total: sources.length,
      });
      if (previousGeneration && previousGeneration !== generationId) {
        await this.adapter.rmdir(`${this.root}/generations/${previousGeneration}`, true).catch(() => undefined);
      }
      return ready;
    } catch (error) {
      await this.adapter.rmdir(generation, true).catch(() => undefined);
      throw error;
    }
  }
}

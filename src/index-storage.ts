import {
  CHUNK_OVERLAP_BYTES, CHUNK_TARGET_BYTES, isRecord, isStoredLocator,
  type SourceEntry, type StoredLocator,
} from "./index-source.ts";

const SCHEMA_VERSION = 1;
const EXTRACTOR_VERSION = 1;
const CHUNKER_VERSION = 1;

export interface EmbeddingModel {
  digest: string;
  name: string;
}

export interface IndexAdapter {
  exists(path: string): Promise<boolean>;
  list(path: string): Promise<{ files: string[]; folders: string[] }>;
  mkdir(path: string): Promise<void>;
  process(path: string, update: (value: string) => string): Promise<string>;
  read(path: string): Promise<string>;
  rmdir(path: string, recursive: boolean): Promise<void>;
  write(path: string, value: string): Promise<void>;
}

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

export interface Catalog {
  complete: true;
  entries: SourceEntry[];
  generationId: string;
  signature: IndexSignature;
}

export interface StoredChunk {
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

// Only generation validation constructs runtime entries; vectors are decoded once.
export interface RuntimeChunk {
  id: string;
  locator: StoredLocator;
  vector: Float32Array;
}

export type RuntimeEntry = SourceEntry & { chunks: RuntimeChunk[] };

export interface ValidatedGeneration {
  entries: RuntimeEntry[];
  generationId: string;
  signature: IndexSignature;
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

export function encodeVector(vector: readonly number[] | Float32Array): string {
  if (vector.length === 0 || vector.some((value) => !Number.isFinite(value))) {
    throw new TypeError("invalid_embedding_vector");
  }
  const bytes = new Uint8Array(vector.length * Float32Array.BYTES_PER_ELEMENT);
  const view = new DataView(bytes.buffer);
  vector.forEach((value, index) => view.setFloat32(index * 4, value, true));
  return bytesToBase64(bytes);
}

function decodeVector(value: unknown, dimension: number): Float32Array {
  if (typeof value !== "string" || dimension <= 0) throw new TypeError("invalid_stored_vector");
  const binary = atob(value);
  if (binary.length !== dimension * 4) throw new TypeError("invalid_stored_vector");
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  const vector = Float32Array.from(
    { length: dimension },
    (_item, index) => view.getFloat32(index * 4, true),
  );
  if (vector.some((value) => !Number.isFinite(value))) throw new TypeError("invalid_stored_vector");
  return vector;
}

function isGenerationId(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value);
}

export function validModel(model: EmbeddingModel): boolean {
  return model.name.length > 0 && model.name.length <= 512 && model.digest.length > 0 && model.digest.length <= 512;
}

export function signatureFor(model: EmbeddingModel, vectorDimension: number): IndexSignature {
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

export function compatibleSignature(value: unknown, model: EmbeddingModel): value is IndexSignature {
  if (!isRecord(value)) return false;
  const expected = signatureFor(model, Number(value.vectorDimension));
  return Number.isInteger(value.vectorDimension) && Number(value.vectorDimension) >= 0 &&
    Object.entries(expected).every(([key, expectedValue]) => value[key] === expectedValue);
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

function parseRecord(value: string, entry: SourceEntry, dimension: number): RuntimeChunk[] {
  const parsed: unknown = JSON.parse(value);
  if (!isRecord(parsed) || parsed.sourceKey !== entry.sourceKey || parsed.fingerprint !== entry.fingerprint ||
    parsed.vectorDimension !== dimension || !Array.isArray(parsed.chunks) || parsed.chunks.length === 0 ||
    parsed.chunks.length !== entry.chunkCount) throw new TypeError("invalid_stored_record");
  return parsed.chunks.map((chunk: unknown, ordinal): RuntimeChunk => {
    if (!isRecord(chunk) || chunk.id !== `${entry.sourceKey}:${entry.fingerprint}:${ordinal}` ||
      !isStoredLocator(chunk.locator) || chunk.locator.path !== entry.path) {
      throw new TypeError("invalid_stored_chunk");
    }
    return { id: chunk.id, locator: chunk.locator, vector: decodeVector(chunk.vector, dimension) };
  });
}

export class GenerationStorage {
  private readonly adapter: IndexAdapter;
  private readonly root: string;

  constructor(adapter: IndexAdapter, root: string) {
    this.adapter = adapter;
    this.root = root;
  }

  async create(generationId: string): Promise<void> {
    const generations = `${this.root}/generations`;
    const generation = `${generations}/${generationId}`;
    for (const path of [this.root, generations, generation, `${generation}/records`]) {
      if (!(await this.adapter.exists(path))) await this.adapter.mkdir(path);
    }
  }

  async cleanup(keepGenerationId?: string): Promise<void> {
    const generations = `${this.root}/generations`;
    const listed = await this.adapter.list(generations).catch(() => ({ files: [], folders: [] }));
    const prefix = `${generations}/`;
    for (const folder of listed.folders) {
      const generationId = folder.startsWith(prefix) ? folder.slice(prefix.length) : "";
      if (!generationId || generationId.includes("/") || generationId === keepGenerationId) continue;
      await this.adapter.rmdir(folder, true).catch(() => undefined);
    }
  }

  async validateGeneration(generationId: string, model: EmbeddingModel): Promise<ValidatedGeneration | null> {
    try {
      const generation = `${this.root}/generations/${generationId}`;
      const catalog = parseCatalog(await this.adapter.read(`${generation}/catalog.json`), model);
      if (!catalog || catalog.generationId !== generationId) return null;
      const paths = new Set<string>();
      const entries: RuntimeEntry[] = [];
      for (const entry of catalog.entries) {
        if (paths.has(entry.path)) return null;
        paths.add(entry.path);
        let chunks: RuntimeChunk[] = [];
        if (entry.status !== "indexed") {
          if (entry.record !== undefined || entry.chunkCount !== undefined || entry.vectorCount !== undefined) return null;
        } else {
          const recordName = `records/${entry.sourceKey}-${entry.fingerprint}.json`;
          if (entry.record !== recordName || !Number.isInteger(entry.chunkCount) ||
            entry.chunkCount !== entry.vectorCount) return null;
          chunks = parseRecord(await this.adapter.read(`${generation}/${recordName}`), entry, catalog.signature.vectorDimension);
        }
        entries.push({ ...entry, chunks });
      }
      return { entries, generationId, signature: catalog.signature };
    } catch {
      return null;
    }
  }

  async restore(model: EmbeddingModel): Promise<ValidatedGeneration | null> {
    try {
      const active: unknown = JSON.parse(await this.adapter.read(`${this.root}/active.json`));
      return isRecord(active) && isGenerationId(active.generationId)
        ? this.validateGeneration(active.generationId, model)
        : null;
    } catch {
      return null;
    }
  }

  async writeRecord(generationId: string, entry: SourceEntry, chunks: StoredChunk[], vectorDimension: number): Promise<void> {
    const recordName = `records/${entry.sourceKey}-${entry.fingerprint}.json`;
    const record: SourceRecord = { chunks, fingerprint: entry.fingerprint ?? "", sourceKey: entry.sourceKey, vectorDimension };
    entry.chunkCount = chunks.length;
    entry.record = recordName;
    entry.vectorCount = chunks.length;
    await this.adapter.write(`${this.root}/generations/${generationId}/${recordName}`, JSON.stringify(record));
  }

  async writeCatalog(catalog: Catalog): Promise<void> {
    await this.adapter.write(`${this.root}/generations/${catalog.generationId}/catalog.json`, JSON.stringify(catalog));
  }

  async activate(generation: ValidatedGeneration, assertCurrent: () => void): Promise<void> {
    const activePath = `${this.root}/active.json`;
    if (!(await this.adapter.exists(activePath))) await this.adapter.write(activePath, "");
    assertCurrent();
    await this.adapter.process(activePath, () => {
      assertCurrent();
      return JSON.stringify({ generationId: generation.generationId });
    });
  }

  async remove(generationId: string): Promise<void> {
    await this.adapter.rmdir(`${this.root}/generations/${generationId}`, true).catch(() => undefined);
  }

  async deleteAll(): Promise<void> {
    if (await this.adapter.exists(this.root)) await this.adapter.rmdir(this.root, true);
  }
}

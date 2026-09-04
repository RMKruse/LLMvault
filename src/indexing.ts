import {
  isStoredLocator, locatorFor, matchesPreparedSource, outcomes, prepareSource, statusCounts,
  type SourceEntry, type SourceOutcome, type TerminalStatus, type VaultSource,
} from "./index-source.ts";
import {
  GenerationStorage, compatibleSignature, encodeVector, signatureFor, validModel,
  type Catalog, type EmbeddingModel, type IndexAdapter, type IndexSignature,
  type RuntimeEntry, type StoredChunk, type ValidatedGeneration,
} from "./index-storage.ts";
import {
  chunksForPaths, hydrate, hydrateCandidates, rankChunks,
  type RetrievedEvidence,
} from "./index-retrieval.ts";

export {
  CHUNK_TARGET_BYTES, CHUNK_OVERLAP_BYTES, RAW_FILE_LIMIT_BYTES,
  EXTRACTED_TEXT_LIMIT_BYTES, PREPROCESSING_LIMIT_MS, chunkMarkdown, chunkCanvas,
  classifyVaultSource, isSupportedVaultExtension, isStoredLocator, locatorFor,
  type SourceOutcome, type MarkdownAnchor, type MarkdownChunk, type CanvasChunk, type VaultSource,
} from "./index-source.ts";
export { encodeVector, type EmbeddingModel, type IndexSignature } from "./index-storage.ts";
export { resolveDailyRecapRequest, type DailyRecapRequest, type RetrievedEvidence } from "./index-retrieval.ts";

const EMBEDDING_BATCH_SIZE = 16;

export interface IndexSnapshot {
  available: boolean;
  completed: number;
  latestPath?: string;
  outcomes: SourceOutcome[];
  phase: "idle" | "indexing" | "ready" | "failed";
  statuses: Partial<Record<TerminalStatus, number>>;
  total: number;
}

export class VaultIndex {
  private active?: ValidatedGeneration & { model: EmbeddingModel };
  private readonly storage: GenerationStorage;
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
    this.storage = new GenerationStorage(adapter, root);
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
    const entries = this.active?.entries ?? [];
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
        await this.storage.deleteAll();
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
    return this.active ? { ...this.active.signature } : null;
  }

  async retrieve(
    question: string,
    applyCalibratedCutoff = true,
    candidateLimit = 6,
  ): Promise<RetrievedEvidence[]> {
    const active = this.active;
    const revision = this.revision;
    if (!active || question.trim().length === 0) {
      return [];
    }
    if (active.signature.vectorDimension === 0 || !active.entries.some(({ chunks }) => chunks.length > 0)) return [];
    const queryVectors = await this.queryEmbed([question], active.model);
    if (revision !== this.revision) return [];
    const { evidence, stale } = await hydrateCandidates(
      rankChunks(active, queryVectors, applyCalibratedCutoff),
      this.listSources(), candidateLimit, ++this.querySequence, () => revision === this.revision,
    );
    if (revision !== this.revision) return [];
    if (stale) this.queueReplacement(active.model);
    return evidence;
  }

  async retrievePaths(paths: Iterable<string>): Promise<RetrievedEvidence[]> {
    const active = this.active;
    const revision = this.revision;
    if (!active) return [];
    const { evidence, stale } = await hydrateCandidates(
      chunksForPaths(active, paths), this.listSources(), 4, ++this.querySequence,
      () => revision === this.revision,
    );
    if (revision !== this.revision) return [];
    if (stale) this.queueReplacement(active.model);
    return evidence;
  }

  async resolveEvidence(evidence: RetrievedEvidence): Promise<RetrievedEvidence | null> {
    if (!isStoredLocator(evidence) || !this.hasEvidence(evidence)) return null;
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
    const chunk = {
      id: evidence.chunkId,
      locator: locatorFor(evidence.path, evidence),
    };
    const hydrated = await hydrate(entry, chunk, evidence.score, source);
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
      this.active.entries = this.active.entries.filter(
        ({ path }) => !this.pendingPaths.has(path),
      );
    }
    this.rebuildQueued = true;
    if (this.model) {
      this.update({
        ...this.snapshot,
        outcomes: this.snapshot.outcomes.filter(({ path }) => !this.pendingPaths.has(path)),
        phase: "indexing",
        statuses: this.active ? statusCounts(this.active.entries) : {},
      });
      return this.queueDrain();
    }
    return Promise.resolve(this.getSnapshot());
  }

  private hasEvidence(evidence: RetrievedEvidence): boolean {
    return this.active?.entries.some(
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
      const entries = this.active?.entries ?? [];
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
      if (!this.active) await this.storage.cleanup();
      return await this.build(model, sources, revision, replacementPaths);
    } catch {
      if (revision !== this.revision) return this.getSnapshot();
      const entries = this.active?.entries ?? [];
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

  private async restore(
    model: EmbeddingModel,
    sources: VaultSource[],
    revision: number,
  ): Promise<IndexSnapshot | null> {
    const validated = await this.storage.restore(model);
    if (!validated) return null;
    const entriesByPath = new Map(validated.entries.map((entry) => [entry.path, entry]));
    const reusable: RuntimeEntry[] = [];
    for (const source of sources) {
      this.assertCurrent(revision);
      const expected = entriesByPath.get(source.path);
      const prepared = await prepareSource(source);
      if (!expected || !matchesPreparedSource(prepared, expected)) continue;
      reusable.push(expected);
    }
    this.assertCurrent(revision);
    this.active = { ...validated, entries: reusable, model };
    if (reusable.length !== sources.length || validated.entries.length !== sources.length) {
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
    await this.storage.cleanup(validated.generationId);
    return this.update({
      available: true,
      completed: sources.length,
      outcomes: outcomes(validated.entries),
      phase: "ready",
      statuses: statusCounts(validated.entries),
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
    await this.storage.create(generationId);
    const entries: SourceEntry[] = [];
    let catalogCommitted = false;
    const reusable = compatibleSignature(this.active?.signature, model)
      ? new Map(this.active?.entries.map((entry) => [entry.path, entry]))
      : new Map<string, RuntimeEntry>();
    let vectorDimension = this.active && reusable.size > 0
      ? this.active.signature.vectorDimension
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
        const reused = previous !== undefined && matchesPreparedSource(prepared, previous);
        if (reused && entry.status === "indexed") {
          await this.storage.writeRecord(generationId, entry, previous.chunks.map((chunk) => ({
            ...chunk, vector: encodeVector(chunk.vector),
          })), vectorDimension);
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
              if (!chunk || !vector) {
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
            }
          }
          await this.storage.writeRecord(generationId, entry, storedChunks, vectorDimension);
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
      await this.storage.writeCatalog(catalog);
      const validated = await this.storage.validateGeneration(generationId, model);
      if (!validated) throw new TypeError("invalid_persisted_generation");

      await this.storage.activate(validated, () => this.assertCurrent(revision));
      catalogCommitted = true;
      this.assertCurrent(revision);
      this.active = { ...validated, model };
      const ready = this.update({
        available: true,
        completed: sources.length,
        outcomes: outcomes(validated.entries),
        phase: "ready",
        statuses: statusCounts(validated.entries),
        total: sources.length,
      });
      await this.storage.cleanup(generationId);
      return ready;
    } catch (error) {
      if (!catalogCommitted) {
        await this.storage.remove(generationId);
      }
      throw error;
    }
  }
}

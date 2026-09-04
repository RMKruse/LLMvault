import type { PreparedEntry } from "../src/index-source.ts";
import type { Catalog, GenerationStorage } from "../src/index-storage.ts";

type CatalogEntry = Catalog["entries"][number];
declare const prepared: Extract<PreparedEntry, { status: "indexed" }>;
declare const storage: GenerationStorage;

const persisted = { ...prepared, record: "records/source-fingerprint.json", chunkCount: 1, vectorCount: 1 };
persisted satisfies CatalogEntry;
storage.writeRecord("generation", prepared, [], 1) satisfies Promise<CatalogEntry>;

// @ts-expect-error Prepared indexed sources cannot be committed before record persistence.
prepared satisfies CatalogEntry;
// @ts-expect-error Indexed catalog entries require a record identity.
({ ...prepared, chunkCount: 1, vectorCount: 1 }) satisfies CatalogEntry;
// @ts-expect-error Indexed catalog entries require a chunk count.
({ ...prepared, record: persisted.record, vectorCount: 1 }) satisfies CatalogEntry;
// @ts-expect-error Indexed catalog entries require a vector count.
({ ...prepared, record: persisted.record, chunkCount: 1 }) satisfies CatalogEntry;
// @ts-expect-error Indexed catalog entries require a non-null fingerprint.
({ ...persisted, fingerprint: null }) satisfies CatalogEntry;
// @ts-expect-error Non-indexed outcomes cannot carry persistence metadata, even through a variable.
({ ...persisted, status: "no_extractable_text" as const }) satisfies CatalogEntry;
// @ts-expect-error Non-indexed outcomes cannot be written as records.
storage.writeRecord("generation", { ...prepared, status: "no_extractable_text" }, [], 1);

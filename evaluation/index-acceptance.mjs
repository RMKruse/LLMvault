// @ts-check

/**
 * The serialized renderer uses only the index's public behavior and durable format.
 * Types come from production; no runtime generation representation is duplicated here.
 * @param {NonNullable<import("../src/main.ts").default["index"]>} index
 * @param {import("obsidian").DataAdapter} adapter
 * @param {string} root
 */
export function createIndexChecks(index, adapter, root) {
  /** @param {string[]} paths */
  const evidence = (paths) => index.retrievePaths(paths);
  /** @param {{oldPath: string, path?: string | null, fingerprint?: string}} expected */
  const exposed = async (expected) => {
    const current = await evidence([...new Set([expected.oldPath, expected.path].filter(
      /** @returns {path is string} */ (path) => typeof path === "string",
    ))]);
    return current.some((item) =>
      (expected.oldPath !== expected.path && item.path === expected.oldPath) ||
      (item.path === expected.path && item.fingerprint !== expected.fingerprint));
  };
  return {
    evidence,
    exposed,
    /** @param {string} path @param {string} fingerprint */
    async replacementAvailable(path, fingerprint) {
      if (index.getSnapshot().phase !== "ready") return false;
      return (await evidence([path])).some((item) => item.fingerprint === fingerprint);
    },
    readPointer: () => adapter.read(`${root}/active.json`),
    async canonicalDigest() {
      if (!(await adapter.exists(`${root}/active.json`))) return "";
      const { generationId } = JSON.parse(await adapter.read(`${root}/active.json`));
      const generation = `${root}/generations/${generationId}`;
      /** @type {import("../src/index-storage.ts").Catalog} */
      const catalog = JSON.parse(await adapter.read(`${generation}/catalog.json`));
      /** @type {[string, import("../src/index-storage.ts").SourceRecord][]} */
      const records = [];
      for (const { record } of catalog.entries) {
        if (record) records.push([record, JSON.parse(await adapter.read(`${generation}/${record}`))]);
      }
      const payload = {
        entries: catalog.entries.map(({ record, ...entry }) => ({ ...entry, record: record?.split("/").at(-1) }))
          .sort((left, right) => left.path.localeCompare(right.path)),
        records: records.sort(([left], [right]) => left.localeCompare(right)),
        signature: catalog.signature,
      };
      const digest = await globalThis.crypto.subtle.digest("SHA-256", new TextEncoder().encode(JSON.stringify(payload)));
      return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
    },
    /**
     * Pause after the real adapter operation, at the same durable boundary as a crash.
     * @param {"record" | "catalog" | "pointer" | "cleanup"} boundary
     * @param {(event: {boundary: string, method: string, path: string}) => Promise<void>} reached
     */
    armCrash(boundary, reached) {
      const { write, process, rmdir } = adapter;
      /** @param {string} method @param {string} path */
      const pause = async (method, path) => {
        if (!path.startsWith(`${root}/`)) return;
        if ((boundary === "record" && method === "write" && path.includes("/records/")) ||
          (boundary === "catalog" && method === "write" && path.endsWith("/catalog.json")) ||
          (boundary === "pointer" && method === "process" && path === `${root}/active.json`) ||
          (boundary === "cleanup" && method === "rmdir" && path.startsWith(`${root}/generations/`))) {
          await reached({ boundary, method, path });
        }
      };
      adapter.write = async (path, value, options) => {
        await write.call(adapter, path, value, options);
        await pause("write", path);
      };
      adapter.process = async (path, update, options) => {
        const result = await process.call(adapter, path, update, options);
        await pause("process", path);
        return result;
      };
      adapter.rmdir = async (path, recursive) => {
        await rmdir.call(adapter, path, recursive);
        await pause("rmdir", path);
      };
      return () => { adapter.write = write; adapter.process = process; adapter.rmdir = rmdir; };
    },
  };
}

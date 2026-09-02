import { createHash } from "node:crypto";
import { readFile, readdir, writeFile } from "node:fs/promises";
import process from "node:process";
import { URL } from "node:url";

import { applyVariant, cases, poisonedVariants } from "./fixture.mjs";

const root = new URL("./", import.meta.url);
const sourceRoot = new URL("sources/", root);
const hash = (value) => createHash("sha256").update(value).digest("hex");
const files = (await readdir(sourceRoot)).sort();
const sources = new Map(await Promise.all(files.map(async (path) => [
  path,
  await readFile(new URL(path, sourceRoot), "utf8"),
])));
const sourceSha256 = Object.fromEntries(files.map((path) => [path, hash(sources.get(path))]));

const locate = ({ path, nodeId, quote }) => {
  const source = sources.get(path);
  if (source === undefined) throw new Error(`source missing: ${path}`);
  const text = nodeId
    ? JSON.parse(source).nodes.find(({ id }) => id === nodeId)?.text
    : source;
  if (typeof text !== "string") throw new Error(`node missing: ${path}#${nodeId}`);
  const start = text.indexOf(quote);
  if (start < 0 || text.indexOf(quote, start + 1) >= 0) throw new Error(`gold quote must be unique: ${path}`);
  const result = { path, ...(nodeId ? { nodeId } : {}), start, end: start + quote.length, quote };
  if (!nodeId) {
    result.startLine = text.slice(0, start).split("\n").length;
    result.endLine = result.startLine + quote.split("\n").length - 1;
  }
  return result;
};

const variants = poisonedVariants.map((variant) => ({
  ...variant,
  sourceSha256: hash(applyVariant(sources.get(variant.path), variant)),
}));
const manifest = {
  suiteVersion: 1,
  sourceSha256,
  fixtureSha256: hash([
    ...Object.entries(sourceSha256).flat(),
    ...variants.flatMap(({ id, sourceSha256: checksum }) => [id, checksum]),
  ].join("\0")),
  calibrationCaseIds: cases.slice(0, 6).map(({ id }) => id),
  cases: cases.map((item) => ({
    ...item,
    expectedMode: item.category === "insufficient" || item.pairedCleanCaseId?.startsWith("I")
      ? "insufficient_evidence"
      : "answer",
    goldEvidence: item.goldEvidence.map(locate),
  })),
  poisonedVariants: variants,
};
const output = `${JSON.stringify(manifest, null, 2)}\n`;
const destination = new URL("manifest.json", root);

if (process.argv.includes("--check")) {
  if (await readFile(destination, "utf8").catch(() => "") !== output) {
    throw new Error("manifest is stale; run node evaluation/reference-vault-v1/build-manifest.mjs");
  }
} else {
  await writeFile(destination, output);
}

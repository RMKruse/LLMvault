import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFile, mkdir, readFile, readdir, rm } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const read = (name) => readFile(path.join(root, name));
const digest = (value) => createHash("sha256").update(value).digest("hex");
const [manifest, packageJson, versions, firstBundle] = await Promise.all([
  read("manifest.json").then(JSON.parse),
  read("package.json").then(JSON.parse),
  read("versions.json").then(JSON.parse),
  read("main.js"),
]);

assert.match(manifest.version, /^\d+\.\d+\.\d+$/);
assert.equal(packageJson.version, manifest.version);
assert.deepEqual(packageJson.dependencies ?? {}, {});
assert.equal(versions[manifest.version], manifest.minAppVersion);
assert.equal(manifest.isDesktopOnly, true);

const rebuild = spawnSync(process.execPath, ["esbuild.config.mjs", "production"], {
  cwd: root,
  encoding: "utf8",
});
assert.equal(rebuild.status, 0, rebuild.stderr || rebuild.stdout);
const secondBundle = await read("main.js");
assert.equal(digest(secondBundle), digest(firstBundle), "production build is not reproducible");
assert.deepEqual(
  [...new Set(Array.from(
    secondBundle.toString().matchAll(/require\(\"([^\"]+)\"\)/g),
    (match) => match[1],
  ))],
  ["obsidian"],
);

const assets = ["main.js", "manifest.json", "styles.css"];
const output = path.join(root, "release-candidate", manifest.version);
await rm(output, { force: true, recursive: true });
await mkdir(output, { recursive: true });
await Promise.all(assets.map((name) => copyFile(path.join(root, name), path.join(output, name))));
assert.deepEqual((await readdir(output)).sort(), assets.toSorted());
for (const name of assets) {
  assert.equal(digest(await readFile(path.join(output, name))), digest(await read(name)));
}

console.log(`Release candidate ${manifest.version}: ${output}`);
for (const name of assets) console.log(`${digest(await read(name))}  ${name}`);

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("build produces an installable desktop Vault Chat plugin", async () => {
  const [manifestSource, bundle, styles] = await Promise.all([
    readFile(new URL("../manifest.json", import.meta.url), "utf8"),
    readFile(new URL("../main.js", import.meta.url), "utf8"),
    readFile(new URL("../styles.css", import.meta.url), "utf8"),
  ]);
  const manifest = JSON.parse(manifestSource);

  assert.equal(manifest.id, "llmvault");
  assert.equal(manifest.minAppVersion, "1.13.7");
  assert.equal(manifest.isDesktopOnly, true);
  assert.match(bundle, /vault-chat-view/);
  assert.match(bundle, /127\.0\.0\.1/);
  assert.match(bundle, /remote_model_disallowed/);
  assert.match(bundle, /\/api\/chat/);
  assert.match(bundle, /INSUFFICIENT_EVIDENCE/);
  assert.match(bundle, /Open source/);
  assert.match(bundle, /Stop/);
  assert.match(styles, /\.llmvault-chat/);
  assert.match(styles, /\.llmvault-chat__citation/);
});

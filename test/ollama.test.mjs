import assert from "node:assert/strict";
import test from "node:test";

import {
  OllamaClient,
  OllamaError,
  normalizeSettings,
  revalidateSelections,
} from "../src/ollama.ts";

const json = (value, init) =>
  new Response(JSON.stringify(value), {
    headers: { "content-type": "application/json" },
    ...init,
  });

test("discovery exposes only compatible local models through the fixed API", async () => {
  const requests = [];
  const details = {
    chat: { capabilities: ["completion"] },
    embed: { capabilities: ["embedding"] },
    both: { capabilities: ["completion", "embedding"] },
    incompatible: { capabilities: ["vision"] },
    "remote-in-show": {
      capabilities: ["completion"],
      remote_host: "https://example.com",
    },
  };
  const client = new OllamaClient(async (input, init) => {
    const url = new URL(input);
    requests.push({ url, init });
    if (url.pathname === "/api/version") return json({ version: "0.11.0" });
    if (url.pathname === "/api/tags") {
      return json({
        models: [
          ...Object.keys(details).map((model) => ({ model })),
          { model: "remote", remote_model: "cloud/model" },
          { model: "remote-whitespace", remote_host: " " },
        ],
      });
    }
    assert.equal(url.pathname, "/api/show");
    return json(details[JSON.parse(init.body).model]);
  });

  const discovery = await client.discover(11434);

  assert.deepEqual(discovery.chatModels, ["both", "chat"]);
  assert.deepEqual(discovery.embeddingModels, ["both", "embed"]);
  assert.deepEqual(discovery.remoteModels, [
    "remote",
    "remote-in-show",
    "remote-whitespace",
  ]);
  assert.equal(discovery.version, "0.11.0");
  assert.equal(discovery.installedModelCount, 7);
  assert.deepEqual(
    requests.map(({ url, init }) => [
      url.origin,
      url.pathname,
      init.method,
      init.redirect,
      init.credentials,
      init.cache,
    ]),
    [
      ["http://127.0.0.1:11434", "/api/version", "GET", "error", "omit", "no-store"],
      ["http://127.0.0.1:11434", "/api/tags", "GET", "error", "omit", "no-store"],
      ...Object.keys(details).map(() => [
        "http://127.0.0.1:11434",
        "/api/show",
        "POST",
        "error",
        "omit",
        "no-store",
      ]),
    ],
  );
});

test("settings accept only a numeric loopback port and explicit selections", async () => {
  assert.deepEqual(
    normalizeSettings({
      ollamaPort: "https://example.com",
      host: "example.com",
      chatModel: "",
      embeddingModel: 42,
    }),
    { ollamaPort: 11434, chatModel: null, embeddingModel: null },
  );
  assert.deepEqual(
    normalizeSettings({
      ollamaPort: 12345,
      chatModel: "chat",
      embeddingModel: "embed",
    }),
    { ollamaPort: 12345, chatModel: "chat", embeddingModel: "embed" },
  );
  await assert.rejects(new OllamaClient().discover(0), /Port/);
  await assert.rejects(new OllamaClient().discover(65536), /Port/);
});

test("selection reconciliation never auto-selects and precisely clears stale choices", () => {
  const discovery = {
    version: "0.11.0",
    installedModelCount: 3,
    chatModels: ["only-chat"],
    embeddingModels: ["only-embed"],
    remoteModels: ["cloud"],
  };

  assert.deepEqual(
    revalidateSelections(normalizeSettings({}), discovery),
    {
      settings: { ollamaPort: 11434, chatModel: null, embeddingModel: null },
      recoveryCodes: [],
    },
  );
  assert.deepEqual(
    revalidateSelections(
      normalizeSettings({
        chatModel: "only-chat",
        embeddingModel: "missing",
      }),
      discovery,
    ),
    {
      settings: {
        ollamaPort: 11434,
        chatModel: "only-chat",
        embeddingModel: null,
      },
      recoveryCodes: ["embedding_model_unavailable"],
    },
  );
  assert.deepEqual(
    revalidateSelections(
      normalizeSettings({ chatModel: "cloud", embeddingModel: "cloud" }),
      discovery,
    ),
    {
      settings: { ollamaPort: 11434, chatModel: null, embeddingModel: null },
      recoveryCodes: ["remote_model_disallowed"],
    },
  );
});

test("metadata parsing is bounded and reports stable response errors", async () => {
  const oversized = new Uint8Array(2 * 1024 * 1024 + 1);
  const client = new OllamaClient(async () => new Response(oversized));

  await assert.rejects(
    client.discover(11434),
    (error) => error instanceof OllamaError && error.code === "invalid_response",
  );
});

test("owned requests can be canceled with a stable non-failure code", async () => {
  const client = new OllamaClient(
    async (_input, init) =>
      await new Promise((_resolve, reject) => {
        init.signal.addEventListener("abort", () => reject(new Error("aborted")), {
          once: true,
        });
      }),
  );

  const discovery = client.discover(11434);
  client.abortAll();

  await assert.rejects(
    discovery,
    (error) => error instanceof OllamaError && error.code === "canceled",
  );
});

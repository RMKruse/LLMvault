import assert from "node:assert/strict";
import { ReadableStream } from "node:stream/web";
import test from "node:test";
import { TextEncoder } from "node:util";

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

const ndjson = (...chunks) =>
  new Response(
    new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder();
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    }),
    { headers: { "content-type": "application/x-ndjson" } },
  );

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
          ...Object.keys(details).map((model) => ({ model, digest: `sha256:${model}` })),
          { model: "remote", digest: "sha256:remote", remote_model: "cloud/model" },
          { model: "remote-whitespace", digest: "sha256:remote-whitespace", remote_host: " " },
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
  assert.deepEqual(discovery.modelDigests, {
    both: "sha256:both",
    chat: "sha256:chat",
    embed: "sha256:embed",
    incompatible: "sha256:incompatible",
  });
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

test("pinned model validation inspects only the selected local digest", async () => {
  const paths = [];
  let selectedRemote = false;
  const client = new OllamaClient(async (input, init) => {
    const url = new URL(input);
    paths.push(url.pathname);
    if (url.pathname === "/api/tags") {
      return json({
        models: [
          {
            model: "selected",
            digest: "sha256:selected",
            ...(selectedRemote ? { remote_model: "cloud/selected" } : {}),
          },
          { model: "unrelated", digest: "sha256:unrelated" },
        ],
      });
    }
    assert.equal(JSON.parse(init.body).model, "selected");
    return json({ capabilities: ["embedding"] });
  });

  assert.equal(
    await client.validatePinnedModel(
      11434,
      "selected",
      "sha256:selected",
      "embedding",
    ),
    "compatible",
  );
  assert.deepEqual(paths, ["/api/tags", "/api/show"]);
  assert.equal(
    await client.validatePinnedModel(
      11434,
      "selected",
      "sha256:changed",
      "embedding",
    ),
    "incompatible",
  );
  selectedRemote = true;
  assert.equal(
    await client.validatePinnedModel(
      11434,
      "selected",
      "sha256:changed",
      "embedding",
    ),
    "remote",
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

test("embedding batches preserve order, disable truncation, and validate every vector", async () => {
  const client = new OllamaClient(async (input, init) => {
    const url = new URL(input);
    assert.equal(url.pathname, "/api/embed");
    assert.deepEqual(JSON.parse(init.body), {
      model: "embed",
      input: ["first", "second"],
      truncate: false,
    });
    return json({ embeddings: [[1, 2], [3, 4]] });
  });

  assert.deepEqual(await client.embed(11434, "embed", ["first", "second"]), [
    [1, 2],
    [3, 4],
  ]);

  for (const embeddings of [[], [[]], [[Number.NaN]]]) {
    const invalid = new OllamaClient(async () => json({ embeddings }));
    await assert.rejects(
      invalid.embed(11434, "embed", ["first"]),
      (error) => error instanceof OllamaError && error.code === "invalid_response",
    );
  }

  const inconsistent = new OllamaClient(async () =>
    json({ embeddings: [[1], [2, 3]] }),
  );
  await assert.rejects(
    inconsistent.embed(11434, "embed", ["first", "second"]),
    (error) => error instanceof OllamaError && error.code === "invalid_response",
  );
});

test("chat streams only answer content from strict NDJSON and requires done", async () => {
  let request;
  const client = new OllamaClient(async (input, init) => {
    request = { input: String(input), init };
    return ndjson(
      '{"message":{"content":"Grounded ","thinking":"hidden"},"done":false}\n',
      '{"message":{"con',
      'tent":"answer [S1]"},"done":false}\n{"message":{"content":""},"done":true,"done_reason":"stop"}\n',
    );
  });
  const messages = [
    { role: "system", content: "Use evidence only." },
    { role: "user", content: "Question and evidence" },
  ];
  const chunks = [];

  const result = await client.chat(11434, "chat", messages, (chunk) => chunks.push(chunk));

  assert.equal(request.input, "http://127.0.0.1:11434/api/chat");
  assert.deepEqual(
    [request.init.method, request.init.redirect, request.init.credentials, request.init.cache],
    ["POST", "error", "omit", "no-store"],
  );
  assert.deepEqual(JSON.parse(request.init.body), {
    model: "chat",
    messages,
    stream: true,
  });
  assert.deepEqual(chunks, ["Grounded ", "answer [S1]"]);
  assert.deepEqual(result, {
    content: "Grounded answer [S1]",
    diagnostics: { done_reason: "stop" },
  });

  for (const response of [
    ndjson('{"message":{"content":"partial"},"done":false}\n'),
    ndjson("not-json\n"),
    ndjson('{"error":"model failed"}\n'),
  ]) {
    const invalid = new OllamaClient(async () => response);
    await assert.rejects(
      invalid.chat(11434, "chat", messages, () => undefined),
      (error) =>
        error instanceof OllamaError &&
        ["invalid_response", "stream_interrupted"].includes(error.code),
    );
  }
});

test("chat rejects a declared response overrun before accepting model output", async () => {
  const client = new OllamaClient(async () => new Response(
    '{"message":{"content":"must stay inert"},"done":true}\n',
    { headers: {
      "content-length": String(16 * 1024 * 1024 + 1),
      "content-type": "application/x-ndjson",
    } },
  ));

  await assert.rejects(
    client.chat(
      11434,
      "chat",
      [{ role: "user", content: "Question" }],
      () => assert.fail("oversized output was emitted"),
    ),
    (error) => error instanceof OllamaError && error.code === "invalid_response",
  );
});

test("non-stream chat uses the same terminal contract and retains diagnostics", async () => {
  const chunks = [];
  const client = new OllamaClient(async (_input, init) => {
    assert.equal(JSON.parse(init.body).stream, false);
    return json({
      message: { content: "One complete answer" },
      done: true,
      done_reason: "stop",
      total_duration: 42,
      eval_count: 7,
    });
  });

  const result = await client.chat(
    11434,
    "chat",
    [{ role: "user", content: "Question" }],
    (chunk) => chunks.push(chunk),
    false,
  );

  assert.deepEqual(chunks, ["One complete answer"]);
  assert.deepEqual(result, {
    content: "One complete answer",
    diagnostics: {
      done_reason: "stop",
      eval_count: 7,
      total_duration: 42,
    },
  });
});

test("diagnostics reject model-controlled strings outside the stable allowlist", async () => {
  const client = new OllamaClient(async () =>
    json({
      message: { content: "Answer" },
      done: true,
      done_reason: "stop\nsecret content",
      eval_count: 7,
    }),
  );

  const result = await client.chat(
    11434,
    "chat",
    [{ role: "user", content: "Question" }],
    () => undefined,
    false,
  );

  assert.deepEqual(result.diagnostics, { eval_count: 7 });
});

test("chat rejects Ollama HTTP error content in favor of stable codes", async () => {
  const client = new OllamaClient(async () =>
    json({ error: "context is\u0000 too\nlong" }, { status: 400 }),
  );

  await assert.rejects(
    client.chat(
      11434,
      "chat",
      [{ role: "user", content: "Question" }],
      () => undefined,
    ),
    (error) =>
      error instanceof OllamaError &&
      error.code === "invalid_request" &&
      !("detail" in error),
  );
});

test("HTTP failures cancel unread response bodies before clearing the deadline", async () => {
  let canceled = false;
  const client = new OllamaClient(async () => new Response(
    new ReadableStream({ cancel() { canceled = true; } }),
    { status: 500 },
  ));

  await assert.rejects(
    client.chat(
      11434,
      "chat",
      [{ role: "user", content: "Question" }],
      () => undefined,
    ),
    (error) => error instanceof OllamaError && error.code === "ollama_server_error",
  );
  assert.equal(canceled, true);
});

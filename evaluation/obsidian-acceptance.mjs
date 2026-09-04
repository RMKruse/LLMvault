/* global localStorage */
import { execFileSync, spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { chmod, cp, mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { performance } from "node:perf_hooks";
import process from "node:process";
import { clearInterval, clearTimeout, setInterval, setTimeout } from "node:timers";
import { setTimeout as delay } from "node:timers/promises";

import { REFERENCE_CONFIGURATION } from "../src/retrieval-calibration.ts";
import { PINNED_CHAT, PROMPT_SHA256, hash, evaluateRelease } from "./proof.mjs";
import { QUESTIONS, groundedCase, summarizeGrounded } from "./grounded-acceptance.mjs";

const ROOT = path.resolve(import.meta.dirname, "..");
const PLUGIN_ID = "llmvault";
const INDEX_PATH = ".obsidian/plugins/llmvault/index-v1";
const PROBE_PATH = "LLMvault Acceptance Probe.md";
const CONTROL_PATH = "LLMvault Acceptance Control.md";
const CONTROL_TOKEN = "llmvault-acceptance-unrelated-control-token";

const option = (name) => process.argv.find((value) => value.startsWith(`--${name}=`))
  ?.slice(name.length + 3);

async function filesUnder(root, relative = "", includeConfiguration = true, includeGit = false) {
  const entries = await readdir(path.join(root, relative), { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const child = path.join(relative, entry.name);
    if (!includeGit && (child === ".git" || child.startsWith(`.git${path.sep}`))) continue;
    if (!includeConfiguration && (child === ".obsidian" || child.startsWith(`.obsidian${path.sep}`))) continue;
    if (entry.isDirectory()) files.push(...await filesUnder(root, child, includeConfiguration, includeGit));
    if (entry.isFile()) files.push({ path: child, bytes: (await stat(path.join(root, child))).size });
  }
  return files;
}

async function directoryBytes(root) {
  try {
    return (await filesUnder(root)).reduce((sum, file) => sum + file.bytes, 0);
  } catch {
    return 0;
  }
}

async function availablePort() {
  const server = net.createServer();
  await new Promise((resolve, reject) => server.listen(0, "127.0.0.1", resolve).once("error", reject));
  const address = server.address();
  const port = typeof address === "object" && address ? address.port : 0;
  await new Promise((resolve) => server.close(resolve));
  return port;
}

class Cdp {
  constructor(socket) {
    this.socket = socket;
    this.id = 0;
    this.pending = new Map();
    this.listeners = new Map();
    socket.addEventListener("message", ({ data }) => {
      const message = JSON.parse(data);
      if (message.id) {
        const pending = this.pending.get(message.id);
        if (!pending) return;
        this.pending.delete(message.id);
        pending.resolve(message.error ? { error: message.error.message } : { value: message.result });
        return;
      }
      for (const listener of this.listeners.get(message.method) ?? []) listener(message.params);
    });
  }

  static async connect(port, child) {
    const deadline = Date.now() + 30_000;
    let lastError;
    while (Date.now() < deadline) {
      if (child.exitCode !== null) throw new Error(`Obsidian exited before CDP was ready (${child.exitCode})`);
      try {
        const pages = await globalThis.fetch(`http://127.0.0.1:${port}/json/list`).then((response) => response.json());
        const page = pages.find(({ type }) => type === "page");
        if (page) {
          const socket = new globalThis.WebSocket(page.webSocketDebuggerUrl);
          await new Promise((resolve, reject) => {
            socket.addEventListener("open", resolve, { once: true });
            socket.addEventListener("error", reject, { once: true });
          });
          return new Cdp(socket);
        }
      } catch (error) {
        lastError = error;
      }
      await delay(100);
    }
    throw new Error(`Obsidian CDP did not start: ${String(lastError ?? "timeout")}`);
  }

  async call(method, params = {}) {
    const id = ++this.id;
    const promise = new Promise((resolve, reject) => this.pending.set(id, { reject, resolve }));
    this.socket.send(JSON.stringify({ id, method, params }));
    const response = await promise;
    if (response.error) throw new Error(response.error);
    return response.value;
  }

  async evaluate(expression) {
    const response = await this.call("Runtime.evaluate", { awaitPromise: true, expression, returnByValue: true });
    if (response.exceptionDetails) {
      const detail = response.exceptionDetails.exception?.description ?? response.exceptionDetails.text;
      throw new Error(detail);
    }
    return response.result.value;
  }

  evaluateFunction(fn, ...args) {
    return this.evaluate(`(${fn.toString()})(${args.map((value) => JSON.stringify(value)).join(",")})`);
  }

  once(method, timeoutMs = 120_000) {
    return new Promise((resolve, reject) => {
      const listeners = this.listeners.get(method) ?? new Set();
      const timer = setTimeout(() => {
        listeners.delete(listener);
        reject(new Error(`timed out waiting for ${method}`));
      }, timeoutMs);
      const listener = (params) => {
        clearTimeout(timer);
        listeners.delete(listener);
        resolve(params);
      };
      listeners.add(listener);
      this.listeners.set(method, listeners);
    });
  }

  close() {
    this.socket.close();
  }
}

async function launchObsidian(executable, profile) {
  const port = await availablePort();
  const child = spawn(executable, [
    `--user-data-dir=${profile}`,
    `--remote-debugging-port=${port}`,
    "--disable-gpu",
    "--disable-backgrounding-occluded-windows",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-features=CalculateNativeWinOcclusion,IntensiveWakeUpThrottling",
  ], { detached: true, stdio: "ignore" });
  return { cdp: await Cdp.connect(port, child), child, port, profile };
}

async function stopObsidian(instance, signal = "SIGTERM") {
  if (!instance) return;
  const exited = instance.child.exitCode === null
    ? new Promise((resolve) => instance.child.once("exit", resolve))
    : Promise.resolve();
  instance.cdp.close();
  for (let attempt = 0; attempt < 5; attempt += 1) {
    const cleaner = spawn("/usr/bin/pkill", [
      `-${signal.slice(3)}`,
      "-f",
      "--",
      `--user-data-dir=${instance.profile}`,
    ]);
    await new Promise((resolve) => {
      if (cleaner.exitCode !== null) resolve();
      else {
        cleaner.once("error", resolve);
        cleaner.once("exit", resolve);
      }
    });
    await delay(200);
  }
  await Promise.race([exited, delay(5_000)]);
}

async function prepareWorkspace(source, chatModel, embeddingModel) {
  const temporary = await mkdtemp(path.join(os.tmpdir(), "llmvault-acceptance-"));
  const vault = path.join(temporary, "vault");
  const profile = path.join(temporary, "profile");
  await mkdir(vault);
  await cp(source, vault, {
    recursive: true,
    filter: (entry) => {
      const relative = path.relative(source, entry);
      const first = relative.split(path.sep)[0];
      return first !== ".git" && first !== ".obsidian";
    },
  });
  const plugin = path.join(vault, ".obsidian", "plugins", PLUGIN_ID);
  await mkdir(plugin, { recursive: true });
  await Promise.all([
    cp(path.join(ROOT, "main.js"), path.join(plugin, "main.js")),
    cp(path.join(ROOT, "manifest.json"), path.join(plugin, "manifest.json")),
    cp(path.join(ROOT, "styles.css"), path.join(plugin, "styles.css")),
  ]);
  await writeFile(path.join(plugin, "data.json"), JSON.stringify({
    chatModel,
    embeddingModel,
    ollamaPort: 11434,
  }));
  await writeFile(path.join(vault, ".obsidian", "community-plugins.json"), JSON.stringify([PLUGIN_ID]));
  await writeFile(path.join(vault, ".obsidian", "app.json"), "{}");
  await writeFile(path.join(vault, PROBE_PATH), "# Acceptance probe\n\ninitial-acceptance-token\n");
  await writeFile(path.join(vault, CONTROL_PATH), `# Acceptance control\n\n${CONTROL_TOKEN}\n`);
  await mkdir(profile);
  await writeFile(path.join(profile, "obsidian.json"), JSON.stringify({
    vaults: { acceptance: { open: true, path: vault, ts: Date.now() } },
  }));
  return { profile, temporary, vault };
}

async function rendererSetup(config) {
  const waitFor = async (predicate, timeout = 120_000) => {
    const deadline = Date.now() + timeout;
    while (!predicate()) {
      if (Date.now() >= deadline) throw new Error("acceptance renderer timeout");
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
  };
  await waitFor(() => typeof globalThis.app?.plugins?.loadPlugin === "function", 30_000);
  const app = globalThis.app;
  await waitFor(() => app.plugins.manifests.llmvault, 30_000);
  localStorage.setItem(`enable-plugin-${app.appId}`, "true");
  if (config.recoveryExpected) {
    await waitFor(() => app.plugins.plugins.llmvault, 30_000);
  } else {
    await new Promise((resolve) => setTimeout(resolve, 1_000));
    if (!app.plugins.plugins.llmvault) await app.plugins.loadPlugin("llmvault");
  }
  const plugin = app.plugins.plugins.llmvault;
  if (!plugin) throw new Error("production plugin did not load");
  await plugin.saveSettings({ chatModel: config.chatModel, embeddingModel: config.embeddingModel, ollamaPort: 11434 });
  await new Promise((resolve) => app.workspace.onLayoutReady(resolve));
  await plugin.openVaultChat();
  await waitFor(() => app.workspace.getLeavesOfType("vault-chat-view")[0]?.view);
  const view = () => app.workspace.getLeavesOfType("vault-chat-view")[0].view;
  const state = {
    indexOllamaMs: 0,
    lastIndexEmbedReturnedAt: 0,
    queryDispatchedAt: 0,
    queryReturnedAt: 0,
    chatDispatchedAt: 0,
    acceptedTextAt: [],
    injectLateOutput: false,
    lateOutputInjected: false,
    lastInvalidatedAt: 0,
    recoveryExposedInvalidContent: false,
    requestCount: 0,
    egressPass: true,
    ui: {},
    uiPhase: null,
    uiTimer: null,
  };
  for (const client of [plugin.ollama, plugin.chatOllama, plugin.queryOllama, plugin.indexOllama]) {
    const fetcher = client.fetcher;
    client.fetcher = async (input, init) => {
      const url = new URL(input);
      state.requestCount += 1;
      state.egressPass &&= url.origin === "http://127.0.0.1:11434";
      return await fetcher(input, init);
    };
  }
  const indexEmbed = plugin.indexOllama.embed.bind(plugin.indexOllama);
  plugin.indexOllama.embed = async (...args) => {
    const started = performance.now();
    try {
      return await indexEmbed(...args);
    } finally {
      state.indexOllamaMs += performance.now() - started;
      state.lastIndexEmbedReturnedAt = performance.now();
    }
  };
  const queryEmbed = plugin.queryOllama.embed.bind(plugin.queryOllama);
  plugin.queryOllama.embed = async (...args) => {
    state.queryDispatchedAt = performance.now();
    try {
      return await queryEmbed(...args);
    } finally {
      state.queryReturnedAt = performance.now();
    }
  };
  const chat = plugin.chatOllama.chat.bind(plugin.chatOllama);
  plugin.chatOllama.chat = async (port, model, messages, onContent, stream) => {
    state.chatDispatchedAt = performance.now();
    const acceptContent = (content) => {
      const before = view().answerEl?.textContent;
      onContent(content);
      if (view().answerEl?.textContent !== before) state.acceptedTextAt.push(performance.now());
    };
    try {
      const result = await chat(
        port,
        model,
        messages,
        acceptContent,
        stream,
        { seed: 0, temperature: 0 },
      );
      return result;
    } catch (error) {
      if (!state.injectLateOutput) throw error;
      await new Promise((resolve) => setTimeout(resolve, 300));
      acceptContent("late-output-must-be-rejected");
      state.lateOutputInjected = true;
      return { content: "late-completion-must-not-persist" };
    }
  };
  const invalidate = plugin.index.invalidate.bind(plugin.index);
  plugin.index.invalidate = (...args) => {
    state.lastInvalidatedAt = performance.now();
    return invalidate(...args);
  };
  const percentile95 = (values) => {
    const sorted = [...values].sort((left, right) => left - right);
    return sorted[Math.ceil(sorted.length * 0.95) - 1] ?? Number.POSITIVE_INFINITY;
  };
  const scheduleUi = (phase, delayMs) => {
    const expectedAt = performance.now() + delayMs;
    state.uiTimer = setTimeout(() => {
      if (state.uiPhase !== phase) return;
      const button = view().contentEl.querySelector('button[aria-label="Open Vault Chat menu"]');
      if (!button) {
        scheduleUi(phase, 25);
        return;
      }
      const pulse = String(expectedAt);
      button.setAttribute("data-llmvault-acceptance", pulse);
      button.focus();
      if (button.getAttribute("data-llmvault-acceptance") === pulse) {
        state.ui[phase].push(performance.now() - expectedAt);
      }
      if (state.uiPhase === phase) scheduleUi(phase, 25);
    }, delayMs);
  };
  const startUi = (phase) => {
    state.ui[phase] = [];
    state.uiPhase = phase;
    scheduleUi(phase, 0);
  };
  const stopUi = async (phase) => {
    state.uiPhase = null;
    clearTimeout(state.uiTimer);
    state.uiTimer = null;
    await new Promise((resolve) => setTimeout(resolve, 30));
    const values = state.ui[phase];
    return { maximumMs: values.length ? Math.max(...values) : Number.POSITIVE_INFINITY, p95Ms: percentile95(values), samples: values.length };
  };
  const waitReady = async (after = 0, timeout = 180_000) => {
    await waitFor(() => state.lastInvalidatedAt >= after && plugin.getIndexSnapshot().phase !== "indexing", timeout);
    return plugin.getIndexSnapshot();
  };
  const terminal = (snapshot) => Object.values(snapshot.statuses)
    .reduce((sum, value) => sum + (Number.isFinite(value) ? value : 0), 0);
  const turnCount = () => plugin.getConversationState().conversations
    .reduce((sum, conversation) => sum + conversation.turns.length, 0);
  const currentEvidence = async (token, sourcePath) => {
    const evidence = (await plugin.retrieve(token)).filter(({ path }) => path === sourcePath);
    const resolved = await plugin.revalidateEvidence(evidence);
    return resolved.some((item) => item?.text.includes(token));
  };
  const expectedIsExposed = (expected) => {
    const entries = plugin.index.active?.catalog.entries ?? [];
    const paths = new Set(entries.map(({ path }) => path));
    const current = entries.find(({ path }) => path === expected.path);
    return (expected.oldPath !== expected.path && paths.has(expected.oldPath)) ||
      (Boolean(current) && current.fingerprint !== expected.fingerprint);
  };
  if (config.recoveryExpected) {
    const inspect = () => {
      if (plugin.getIndexSnapshot().available && expectedIsExposed(config.recoveryExpected)) {
        state.recoveryExposedInvalidContent = true;
      }
    };
    inspect();
    plugin.subscribeIndex(inspect);
  }

  globalThis.__llmvaultAcceptance = {
    network() { return { requestCount: state.requestCount, egressPass: state.egressPass }; },
    async cold() {
      plugin.indexOllama.abortAll();
      plugin.index.cancel();
      await plugin.index.deleteAll();
      state.indexOllamaMs = 0;
      let partialActivated = false;
      const unsubscribe = plugin.subscribeIndex((snapshot) => {
        if (snapshot.phase === "indexing" && snapshot.available) partialActivated = true;
      });
      startUi("indexing");
      const started = performance.now();
      await plugin.startIndexing(config.discovery, config.embeddingModel, false);
      const totalMs = performance.now() - started;
      const ui = await stopUi("indexing");
      unsubscribe();
      const snapshot = plugin.getIndexSnapshot();
      return {
        accounting: { complete: snapshot.phase === "ready" && terminal(snapshot) === snapshot.total, discovered: snapshot.total, statuses: snapshot.statuses, terminal: terminal(snapshot) },
        indexing: { ollamaMs: state.indexOllamaMs, partialActivated, pluginMs: Math.max(0, totalMs - state.indexOllamaMs), totalMs },
        ui,
      };
    },
    async warm() {
      state.indexOllamaMs = 0;
      startUi("indexing");
      const started = performance.now();
      await plugin.startIndexing(config.discovery, config.embeddingModel, false);
      const totalMs = performance.now() - started;
      await stopUi("indexing");
      return { ollamaMs: state.indexOllamaMs, pluginMs: Math.max(0, totalMs - state.indexOllamaMs), totalMs };
    },
    async rebuild() {
      state.indexOllamaMs = 0;
      startUi("rebuild");
      await plugin.rebuildIndex();
      return await stopUi("rebuild");
    },
    async ensureProbe(content) {
      const renamed = app.vault.getAbstractFileByPath("LLMvault Acceptance Probe Renamed.md");
      if (renamed) await app.vault.delete(renamed);
      let file = app.vault.getAbstractFileByPath(config.probePath);
      const marker = performance.now();
      if (file) await app.vault.modify(file, `# Acceptance probe\n\n${content}\n`);
      else await app.vault.create(config.probePath, `# Acceptance probe\n\n${content}\n`);
      await waitReady(marker);
      return content;
    },
    async mutation(previousToken, nextToken) {
      const oldEvidence = (await plugin.retrieve(previousToken)).find(({ path }) => path === config.probePath);
      if (!oldEvidence) throw new Error("probe evidence was not retrieved before mutation");
      const file = app.vault.getAbstractFileByPath(config.probePath);
      const replacement = `# Acceptance probe\n\n${nextToken}\n`;
      const digest = await globalThis.crypto.subtle.digest("SHA-256", new globalThis.TextEncoder().encode(`markdown\0${replacement}`));
      const expectedFingerprint = [...new Uint8Array(digest)]
        .map((byte) => byte.toString(16).padStart(2, "0")).join("");
      state.lastIndexEmbedReturnedAt = 0;
      startUi("reconciliation");
      const started = performance.now();
      let activatedAt = 0;
      const activationProbe = setInterval(() => {
        if (plugin.index.active?.catalog.entries.some(({ path, fingerprint }) =>
          path === config.probePath && fingerprint === expectedFingerprint)) {
          activatedAt ||= performance.now();
        }
      }, 1);
      await app.vault.modify(file, replacement);
      await waitFor(() => state.lastInvalidatedAt >= started);
      const invalidatedAt = state.lastInvalidatedAt;
      const ineligible = await plugin.index.resolveEvidence(oldEvidence);
      await waitFor(() => activatedAt > 0);
      clearInterval(activationProbe);
      await waitReady(started);
      const ui = await stopUi("reconciliation");
      if (ineligible) throw new Error("mutated source remained query eligible");
      return {
        mutation: {
          queryIneligibleMs: invalidatedAt - started,
          replacementAfterEmbeddingMs: state.lastIndexEmbedReturnedAt > 0
            ? activatedAt - state.lastIndexEmbedReturnedAt
            : Number.POSITIVE_INFINITY,
        },
        ui,
      };
    },
    async question(question) {
      state.queryDispatchedAt = 0;
      state.queryReturnedAt = 0;
      state.chatDispatchedAt = 0;
      state.acceptedTextAt = [];
      state.injectLateOutput = true;
      state.lateOutputInjected = false;
      const beforeTurns = turnCount();
      const currentView = view();
      currentView.questionEl.value = question;
      startUi("streaming");
      const started = performance.now();
      currentView.questionEl.form.requestSubmit();
      await waitFor(() => state.chatDispatchedAt > 0, 180_000);
      const measured = {
        embeddingDispatchMs: state.queryDispatchedAt - started,
        postEmbeddingMs: state.chatDispatchedAt - state.queryReturnedAt,
      };
      await new Promise((resolve) => setTimeout(resolve, 100));
      const cancelAt = performance.now();
      currentView.questionEl.form.requestSubmit();
      await waitFor(() => currentView.askButton.textContent === "Ask" && !currentView.askButton.disabled);
      const controlsUsableMs = performance.now() - cancelAt;
      await new Promise((resolve) => setTimeout(resolve, 500));
      state.injectLateOutput = false;
      const ui = await stopUi("streaming");
      const late = state.acceptedTextAt.filter((timestamp) => timestamp > cancelAt);
      return {
        cancellation: {
          controlsUsableMs,
          lateCompleted: currentView.answerEl?.textContent.includes("late-completion-must-not-persist") ?? false,
          lateOutputInjected: state.lateOutputInjected,
          latePersisted: turnCount() > beforeTurns,
          noNewTextAfterMs: late.length ? Math.max(...late) - cancelAt : 0,
        },
        question: measured,
        ui,
      };
    },
    async failures(nextToken) {
      const file = app.vault.getAbstractFileByPath(config.probePath);
      const cachedRead = app.vault.cachedRead.bind(app.vault);
      app.vault.cachedRead = (candidate) => candidate.path === config.probePath
        ? Promise.reject(new Error("forced one-file read failure"))
        : cachedRead(candidate);
      let marker = performance.now();
      await app.vault.modify(file, `# Acceptance probe\n\nforced-file-failure\n`);
      const isolated = await waitReady(marker);
      app.vault.cachedRead = cachedRead;
      const oneFileFailedSourceAbsent = !(await currentEvidence("forced-file-failure", config.probePath));
      const oneFileUnrelatedAvailable = await currentEvidence(config.controlToken, config.controlPath);
      const oneFileIsolated = isolated.phase === "ready" && isolated.available &&
        isolated.statuses.extractor_failed === 1 && terminal(isolated) === isolated.total &&
        oneFileFailedSourceAbsent && oneFileUnrelatedAvailable;
      marker = performance.now();
      await app.vault.modify(file, `# Acceptance probe\n\n${nextToken}\n`);
      await waitReady(marker);

      const adapter = plugin.index.adapter;
      const activeBefore = await adapter.read(`${plugin.manifest.dir}/index-v1/active.json`);
      const embed = plugin.indexOllama.embed;
      plugin.indexOllama.embed = async () => { throw new Error("forced fatal embedding failure"); };
      marker = performance.now();
      await app.vault.modify(file, `# Acceptance probe\n\nfatal-${nextToken}\n`);
      const fatal = await waitReady(marker);
      plugin.indexOllama.embed = embed;
      const activeAfter = await adapter.read(`${plugin.manifest.dir}/index-v1/active.json`);
      const fatalFailedSourceAbsent = !(await currentEvidence(`fatal-${nextToken}`, config.probePath));
      const fatalUnrelatedAvailable = await currentEvidence(config.controlToken, config.controlPath);
      const fatalGenerationIsolated = fatal.phase === "failed" && fatal.available &&
        activeAfter === activeBefore && fatalFailedSourceAbsent && fatalUnrelatedAvailable;
      marker = performance.now();
      await app.vault.modify(file, `# Acceptance probe\n\n${nextToken}\n`);
      await waitReady(marker);
      return {
        fatalFailedSourceAbsent,
        fatalGenerationIsolated,
        fatalUnrelatedAvailable,
        oneFileFailedSourceAbsent,
        oneFileIsolated,
        oneFileUnrelatedAvailable,
      };
    },
    async waitUntilReady() {
      if (plugin.getIndexSnapshot().phase === "failed") await plugin.restoreIndex();
      await waitFor(() => plugin.getIndexSnapshot().phase === "ready", 180_000);
      return plugin.getIndexSnapshot();
    },
    async armCrash(boundary, operation) {
      const adapter = plugin.index.adapter;
      const match = (method, target) =>
        (boundary === "record" && method === "write" && target.includes("/records/")) ||
        (boundary === "catalog" && method === "write" && target.endsWith("/catalog.json")) ||
        (boundary === "pointer" && method === "process" && target.endsWith("/active.json")) ||
        (boundary === "cleanup" && method === "rmdir" && target.includes("/generations/"));
      for (const method of ["write", "process", "rmdir"]) {
        const original = adapter[method].bind(adapter);
        adapter[method] = async (...args) => {
          const result = await original(...args);
          if (match(method, args[0])) {
            globalThis.llmvaultBoundary(JSON.stringify({ boundary, method, path: args[0] }));
            await new Promise(() => {});
          }
          return result;
        };
      }
      const file = app.vault.getAbstractFileByPath(operation.path);
      if (operation.kind === "modify") await app.vault.modify(file, `# Acceptance probe\n\n${operation.token}\n`);
      if (operation.kind === "delete") await app.vault.delete(file);
      if (operation.kind === "rename") await app.fileManager.renameFile(file, operation.nextPath);
    },
    async validate(expected) {
      const entries = plugin.index.active?.catalog.entries ?? [];
      const paths = new Set(entries.map(({ path }) => path));
      let invalid = paths.has(expected.oldPath) && expected.oldPath !== expected.path;
      if (expected.path) {
        invalid ||= !paths.has(expected.path);
        invalid ||= entries.find(({ path }) => path === expected.path)?.fingerprint !== expected.fingerprint;
        const evidence = await plugin.retrieve(expected.token);
        const current = await plugin.revalidateEvidence(evidence.filter(({ path }) => path === expected.path));
        invalid ||= !current.some((item) => item?.text.includes(expected.token));
        invalid ||= Boolean(expected.oldToken) && current.some((item) => item?.text.includes(expected.oldToken));
      } else {
        invalid ||= paths.has(expected.oldPath);
      }
      const earlyExposure = Boolean(globalThis.__llmvaultRecoveryExposedInvalidContent);
      clearInterval(globalThis.__llmvaultRecoveryTimer);
      return {
        converged: plugin.getIndexSnapshot().phase === "ready",
        exposedInvalidContent: invalid || state.recoveryExposedInvalidContent || earlyExposure,
      };
    },
    async canonicalDigest() {
      const active = plugin.index.active;
      if (!active) return "";
      const payload = {
        entries: active.catalog.entries.map(({ record, ...entry }) => ({ ...entry, record: record?.split("/").at(-1) }))
          .sort((left, right) => left.path.localeCompare(right.path)),
        records: [...active.records].sort(([left], [right]) => left.localeCompare(right)),
        signature: active.catalog.signature,
      };
      const bytes = new globalThis.TextEncoder().encode(JSON.stringify(payload));
      const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
      return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
    },
  };
  config.discovery = await plugin.discoverModels(11434);
  return config.discovery;
}

async function setupInstance(instance, config) {
  const discovery = await instance.cdp.evaluateFunction(rendererSetup, config);
  config.discovery = discovery;
  return discovery;
}

async function peakDuring(action, indexRoot) {
  let peak = await directoryBytes(indexRoot);
  const timer = setInterval(() => void directoryBytes(indexRoot).then((bytes) => { peak = Math.max(peak, bytes); }), 20);
  try {
    const value = await action;
    peak = Math.max(peak, await directoryBytes(indexRoot));
    return { peak, value };
  } finally {
    clearInterval(timer);
  }
}

async function containsPlaintextSourceCopy(indexRoot) {
  const exactKeys = (value, keys) => value && typeof value === "object" && !Array.isArray(value) &&
    Object.keys(value).every((key) => keys.includes(key)) && keys.every((key) => key in value);
  const validLocator = (locator) => {
    if (!locator || typeof locator !== "object" || Array.isArray(locator)) return false;
    if (locator.format === "canvas") {
      return exactKeys(locator, ["end", "excerpt", "format", "nodeId", "path", "start"]);
    }
    if (locator.format !== "markdown") return false;
    const keys = ["end", "endLine", "format", "path", "start", "startLine"];
    if ("anchor" in locator) {
      keys.push("anchor");
      if (!exactKeys(locator.anchor, ["type", "value"])) return false;
    }
    return exactKeys(locator, keys);
  };
  const active = JSON.parse(await readFile(path.join(indexRoot, "active.json"), "utf8"));
  const generationRoot = path.join(indexRoot, "generations", active.generationId);
  const files = await filesUnder(generationRoot).catch(() => []);
  for (const file of files.filter(({ path: name }) => name.includes(`records${path.sep}`) && name.endsWith(".json"))) {
    const record = JSON.parse(await readFile(path.join(generationRoot, file.path), "utf8"));
    if (!exactKeys(record, ["chunks", "fingerprint", "sourceKey", "vectorDimension"]) || !Array.isArray(record.chunks)) return true;
    if (record.chunks.some((chunk) => !exactKeys(chunk, ["id", "locator", "vector"]) || !validLocator(chunk.locator))) return true;
  }
  return false;
}

async function waitForCleanup(indexRoot) {
  const generations = path.join(indexRoot, "generations");
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    const entries = await readdir(generations, { withFileTypes: true }).catch(() => []);
    if (entries.filter((entry) => entry.isDirectory()).length <= 1) return;
    await delay(20);
  }
  throw new Error("index cleanup did not settle within 60 seconds");
}

function installRecoveryWatch(expected) {
  globalThis.__llmvaultRecoveryExposedInvalidContent = false;
  const inspect = () => {
    const plugin = globalThis.app?.plugins?.plugins?.llmvault;
    if (!plugin?.getIndexSnapshot().available) return;
    const entries = plugin.index.active?.catalog.entries ?? [];
    const paths = new Set(entries.map(({ path }) => path));
    const current = entries.find(({ path }) => path === expected.path);
    if ((expected.oldPath !== expected.path && paths.has(expected.oldPath)) ||
      (current && current.fingerprint !== expected.fingerprint)) {
      globalThis.__llmvaultRecoveryExposedInvalidContent = true;
    }
  };
  inspect();
  globalThis.__llmvaultRecoveryTimer = setInterval(inspect, 1);
}

async function crashRecovery(instance, executable, workspace, config, boundary, operation, expected) {
  await instance.cdp.call("Runtime.addBinding", { name: "llmvaultBoundary" });
  const boundaryReached = instance.cdp.once("Runtime.bindingCalled");
  void instance.cdp.evaluate(`__llmvaultAcceptance.armCrash(${JSON.stringify(boundary)},${JSON.stringify(operation)})`).catch(() => undefined);
  const event = await boundaryReached;
  if (JSON.parse(event.payload).boundary !== boundary) throw new Error(`unexpected crash boundary: ${event.payload}`);
  await stopObsidian(instance, "SIGKILL");
  const started = performance.now();
  const restarted = await launchObsidian(executable, workspace.profile);
  await restarted.cdp.evaluateFunction(installRecoveryWatch, expected);
  await setupInstance(restarted, { ...config, recoveryExpected: expected });
  await restarted.cdp.evaluate("__llmvaultAcceptance.waitUntilReady()");
  const pluginMs = performance.now() - started;
  const validation = await restarted.cdp.evaluate(`__llmvaultAcceptance.validate(${JSON.stringify(expected)})`);
  const recoveredDigest = await restarted.cdp.evaluate("__llmvaultAcceptance.canonicalDigest()");
  await restarted.cdp.evaluate("__llmvaultAcceptance.rebuild()");
  const cleanDigest = await restarted.cdp.evaluate("__llmvaultAcceptance.canonicalDigest()");
  return {
    instance: restarted,
    result: { boundary, canonical: recoveredDigest === cleanDigest, pluginMs, ...validation },
  };
}

const markdownFingerprint = (token) => createHash("sha256")
  .update(`markdown\0# Acceptance probe\n\n${token}\n`)
  .digest("hex");

async function obsidianVersion(executable) {
  const info = path.resolve(executable, "../../Info.plist");
  const child = spawn("/usr/bin/plutil", ["-extract", "CFBundleShortVersionString", "raw", info]);
  let output = "";
  for await (const chunk of child.stdout) output += chunk;
  const code = await new Promise((resolve) => child.once("exit", resolve));
  if (code !== 0 || !output.trim()) throw new Error("could not read Obsidian version");
  return output.trim();
}

async function sourceManifest(source) {
  const files = (await filesUnder(source, "", true, true)).sort((a, b) => a.path.localeCompare(b.path));
  const digest = createHash("sha256");
  for (const file of files) {
    digest.update(JSON.stringify([file.path, hash(await readFile(path.join(source, file.path)))]));
  }
  return digest.digest("hex");
}

async function readJson(file) {
  try { return JSON.parse(await readFile(file, "utf8")); }
  catch (error) { if (error.code === "ENOENT") return null; throw error; }
}

export async function runHarness(evaluateAcceptance) {
  const source = path.resolve(option("vault") ?? "");
  const executable = path.resolve(option("obsidian") ?? "");
  const chatModel = option("chat");
  const embeddingModel = option("embedding");
  if (!option("vault") || !option("obsidian") || !chatModel || !embeddingModel) {
    throw new Error("usage: npm run acceptance:run -- --vault=/path --obsidian=/path/to/Obsidian --chat=MODEL --embedding=MODEL [--output=FILE] [--keep-temp]");
  }
  if (chatModel !== PINNED_CHAT.name || embeddingModel !== REFERENCE_CONFIGURATION.embeddingModel.name) {
    throw new Error("prototype acceptance requires the pinned configuration");
  }
  const sourceBeforeSha256 = await sourceManifest(source);
  const sourceFiles = await filesUnder(source);
  const physicalVaultBytes = (await filesUnder(source, "", true, true))
    .reduce((sum, file) => sum + file.bytes, 0);
  const contentFiles = sourceFiles.filter(({ path: name }) => name !== ".obsidian" && !name.startsWith(`.obsidian${path.sep}`));
  const manifest = JSON.parse(await readFile(path.join(ROOT, "manifest.json"), "utf8"));
  const workspace = await prepareWorkspace(source, chatModel, embeddingModel);
  const config = {
    chatModel,
    controlPath: CONTROL_PATH,
    controlToken: CONTROL_TOKEN,
    discovery: null,
    embeddingModel,
    probePath: PROBE_PATH,
  };
  const indexRoot = path.join(workspace.vault, INDEX_PATH);
  let instance;
  try {
    instance = await launchObsidian(executable, workspace.profile);
    const discovery = await setupInstance(instance, config);
    const chatDigest = discovery.modelDigests[chatModel];
    const embeddingDigest = discovery.modelDigests[embeddingModel];
    if (!chatDigest || !discovery.chatModels.includes(chatModel)) throw new Error(`unavailable chat model: ${chatModel}`);
    if (!embeddingDigest || !discovery.embeddingModels.includes(embeddingModel)) throw new Error(`unavailable embedding model: ${embeddingModel}`);
    if (chatDigest !== PINNED_CHAT.digest || embeddingDigest !== REFERENCE_CONFIGURATION.embeddingModel.digest) {
      throw new Error("prototype model digest mismatch");
    }
    const output = path.resolve(option("output") ?? path.join(ROOT, "evaluation/results/prototype-acceptance.json"));
    const previous = await readJson(output);
    const report = {
      environment: {
        hardware: { architecture: process.arch, cpu: os.cpus()[0]?.model, memoryBytes: os.totalmem() },
        models: { chat: { digest: chatDigest, name: chatModel }, embedding: { digest: embeddingDigest, name: embeddingModel } },
        os: { platform: process.platform, release: os.release() },
        vault: { bytes: physicalVaultBytes, contentFiles: contentFiles.length, totalFiles: sourceFiles.length },
        versions: { node: process.version, obsidian: await obsidianVersion(executable), ollama: discovery.version, plugin: manifest.version },
      },
      groundedAnswer: {
        sourceBeforeSha256, cases: [], approvedBindings: previous?.groundedAnswer?.approvedBindings ?? [], requestCount: 0, egressPass: true,
      },
      runs: [],
    };
    const reviews = [];
    const captureNetwork = async () => {
      const observed = await instance.cdp.evaluate("__llmvaultAcceptance.network()");
      report.groundedAnswer.requestCount += observed.requestCount;
      report.groundedAnswer.egressPass &&= observed.egressPass;
    };
    const save = async () => {
      report.groundedAnswer.sourceAfterSha256 = await sourceManifest(source);
      report.tests = await readJson(path.join(ROOT, "evaluation/results/prototype-tests.json"));
      report.quality = await readJson(path.resolve(option("quality") ?? path.join(ROOT, "evaluation/results/prototype-quality.json")));
      report.acceptance = evaluateAcceptance(report);
      Object.assign(report, evaluateRelease(report, report.acceptance));
      await mkdir(path.dirname(output), { recursive: true });
      await writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
      // Unredacted review material is local-only, ignored by Git and never embedded in the report.
      await writeFile(path.join(ROOT, "evaluation/results/local-review.json"), `${JSON.stringify(reviews, null, 2)}\n`, { mode: 0o600 });
      await chmod(path.join(ROOT, "evaluation/results/local-review.json"), 0o600);
    };
    const repetitions = 1;
    let currentToken = "initial-acceptance-token";
    for (let repetition = 1; repetition <= repetitions; repetition += 1) {
      await instance.cdp.evaluate(`__llmvaultAcceptance.ensureProbe(${JSON.stringify(currentToken)})`);
      const cold = await instance.cdp.evaluate("__llmvaultAcceptance.cold()");
      const warm = await instance.cdp.evaluate("__llmvaultAcceptance.warm()");
      report.groundedAnswer.configuration = {
        pluginBuildSha256: hash(await readFile(path.join(workspace.vault, ".obsidian/plugins/llmvault/main.js"))),
        pluginCommit: execFileSync("git", ["rev-parse", "HEAD"], { cwd: ROOT, encoding: "utf8" }).trim(),
        chatModel: { name: chatModel, digest: chatDigest },
        embeddingModel: { name: embeddingModel, digest: embeddingDigest },
        indexSignature: await instance.cdp.evaluate("app.plugins.plugins.llmvault.index.getSignature()"),
        cutoff: REFERENCE_CONFIGURATION.minimumScore,
        promptSha256: PROMPT_SHA256, ollamaVersion: discovery.version,
      };
      await save();
      for (const id of ["daily", "direct", "direct-history"]) {
        for (let run = 1; run <= (id === "direct-history" ? 1 : 3); run += 1) {
          await captureNetwork();
          // Fresh plugin instances give fresh query-scoped citation IDs without editing production state.
          await instance.cdp.evaluate("app.plugins.unloadPlugin('llmvault')");
          await setupInstance(instance, config);
          await instance.cdp.evaluate("__llmvaultAcceptance.waitUntilReady()");
          const raw = await instance.cdp.evaluateFunction(groundedCase, QUESTIONS[id === "daily" ? "daily" : "direct"], id === "direct-history");
          const { item, review } = summarizeGrounded(raw, report.groundedAnswer.configuration, id, run);
          report.groundedAnswer.cases.push(item);
          reviews.push(review);
          await save();
          process.stderr.write(`grounded acceptance ${id} ${run}: answer=${item.answerPass} path=${item.pathPass} registry=${item.registryPass}\n`);
        }
      }
      const rebuilt = await peakDuring(instance.cdp.evaluate("__llmvaultAcceptance.rebuild()"), indexRoot);
      const nextToken = `acceptance-run-${repetition}-${randomUUID()}`;
      const mutation = await instance.cdp.evaluate(`__llmvaultAcceptance.mutation(${JSON.stringify(currentToken)},${JSON.stringify(nextToken)})`);
      currentToken = nextToken;
      const question = await instance.cdp.evaluate(`__llmvaultAcceptance.question(${JSON.stringify(currentToken)})`);
      const failureToken = `acceptance-failure-recovery-${repetition}-${randomUUID()}`;
      const failures = await instance.cdp.evaluate(`__llmvaultAcceptance.failures(${JSON.stringify(failureToken)})`);
      currentToken = failureToken;
      const recovery = [];
      for (const boundary of ["record", "catalog", "pointer", "cleanup"]) {
        let operation;
        let expected;
        if (boundary === "pointer") {
          operation = { kind: "delete", path: PROBE_PATH };
          expected = { oldPath: PROBE_PATH, oldToken: currentToken, path: "", token: "" };
        } else if (boundary === "cleanup") {
          currentToken = `acceptance-cleanup-${repetition}-${randomUUID()}`;
          await instance.cdp.evaluate(`__llmvaultAcceptance.ensureProbe(${JSON.stringify(currentToken)})`);
          operation = { kind: "rename", nextPath: "LLMvault Acceptance Probe Renamed.md", path: PROBE_PATH, token: currentToken };
          expected = {
            fingerprint: markdownFingerprint(currentToken),
            oldPath: PROBE_PATH,
            oldToken: "",
            path: operation.nextPath,
            token: currentToken,
          };
        } else {
          const oldToken = currentToken;
          currentToken = `acceptance-${boundary}-${repetition}-${randomUUID()}`;
          operation = { kind: "modify", path: PROBE_PATH, token: currentToken };
          expected = {
            fingerprint: markdownFingerprint(currentToken),
            oldPath: PROBE_PATH,
            oldToken,
            path: PROBE_PATH,
            token: currentToken,
          };
        }
        await captureNetwork();
        const recovered = await crashRecovery(instance, executable, workspace, config, boundary, operation, expected);
        instance = recovered.instance;
        recovery.push(recovered.result);
        if (boundary === "pointer") currentToken = "";
      }
      currentToken = `acceptance-reset-${repetition}-${randomUUID()}`;
      await instance.cdp.evaluate(`__llmvaultAcceptance.ensureProbe(${JSON.stringify(currentToken)})`);
      await waitForCleanup(indexRoot);
      const run = {
        accounting: cold.accounting,
        cancellation: question.cancellation,
        failures,
        indexing: cold.indexing,
        mutation: mutation.mutation,
        question: question.question,
        recovery,
        storage: {
          containsPlaintextSourceCopy: await containsPlaintextSourceCopy(indexRoot),
          peakRebuildBytes: rebuilt.peak,
          settledBytes: await directoryBytes(indexRoot),
        },
        ui: { indexing: cold.ui, rebuild: rebuilt.value, reconciliation: mutation.ui, streaming: question.ui },
        warm,
      };
      report.runs.push(run);
      await captureNetwork();
      await save();
      process.stderr.write(`acceptance repetition ${repetition}/${repetitions} complete\n`);
    }
    process.stdout.write(`${JSON.stringify({ gates: report.gates, groundedGates: report.groundedGates, output, pass: report.pass }, null, 2)}\n`);
    if (!report.pass) process.exitCode = 1;
  } finally {
    await stopObsidian(instance);
    if (!process.argv.includes("--keep-temp")) await rm(workspace.temporary, { force: true, recursive: true });
    else process.stderr.write(`kept disposable acceptance workspace: ${workspace.temporary}\n`);
  }
}

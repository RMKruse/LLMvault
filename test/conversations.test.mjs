import assert from "node:assert/strict";
import test from "node:test";

import {
  completeTurn,
  conversationMessages,
  conversationUserMessage,
  messageBytes,
  deleteConversation,
  newConversation,
  normalizeConversationState,
  selectConversation,
} from "../src/conversations.ts";

const evidence = {
  citationId: "S1-1",
  chunkId: "chunk-1",
  end: 12,
  endLine: 2,
  fingerprint: "sha256:source",
  format: "markdown",
  path: "Notes/answer.md",
  score: 0.9,
  start: 0,
  startLine: 1,
  text: "Stored evidence",
};

const turn = {
  answer: "A grounded answer [S1-1]",
  completedAt: 20,
  evidence: [evidence],
  kind: "answer",
  question: "What is the answer?",
  status: "complete",
};

test("only complete conversations restore from plugin data", () => {
  const state = normalizeConversationState({
    conversations: [
      {
        createdAt: 10,
        id: "conversation-1",
        title: "What is the answer?",
        turns: [turn, { ...turn, status: "streaming" }],
        updatedAt: 20,
      },
      { id: "incomplete-conversation", turns: [{ status: "streaming" }] },
    ],
    selectedConversationId: "conversation-1",
  });

  assert.equal(state.conversations.length, 1);
  assert.deepEqual(state.conversations[0]?.turns, [turn]);
  assert.equal(state.selectedConversationId, "conversation-1");
});

test("persisted evidence accepts only application-issued opaque citation IDs", () => {
  const state = normalizeConversationState({
    conversations: [{
      createdAt: 10,
      id: "conversation-1",
      title: "Hostile evidence",
      turns: [{
        ...turn,
        evidence: [{ ...evidence, citationId: "https://example.com/../../secret" }],
      }],
      updatedAt: 20,
    }],
    selectedConversationId: "conversation-1",
  });

  assert.deepEqual(state, { conversations: [], selectedConversationId: null });
});

test("saved evidence requires valid format-specific locators", () => {
  const canvas = {
    ...evidence, format: "canvas", path: "Board.canvas", nodeId: "card-1", excerpt: "Stored evidence",
  };
  delete canvas.startLine;
  delete canvas.endLine;
  const restore = (item) => normalizeConversationState({
    conversations: [{
      createdAt: 10, id: "conversation-1", title: "Question", updatedAt: 20,
      turns: [{ ...turn, evidence: [item] }],
    }],
    selectedConversationId: "conversation-1",
  });
  for (const item of [evidence, canvas, { ...evidence, anchor: { type: "heading", value: "Title" } }]) {
    assert.deepEqual(restore(item).conversations[0].turns[0].evidence, [item]);
  }
  for (const [item, changes] of [
    [evidence, { startLine: undefined }],
    [evidence, { endLine: undefined }],
    [evidence, { startLine: 0 }],
    [evidence, { startLine: 1.5 }],
    [evidence, { endLine: 0 }],
    [evidence, { endLine: Infinity }],
    [evidence, { startLine: 3, endLine: 2 }],
    [evidence, { anchor: { type: "heading", value: "" } }],
    [canvas, { nodeId: undefined }],
    [canvas, { nodeId: "" }],
    [canvas, { excerpt: undefined }],
    [canvas, { excerpt: "" }],
    [evidence, { start: -1 }],
    [canvas, { end: 0 }],
    [canvas, { end: 1.5 }],
  ]) {
    assert.deepEqual(restore({ ...item, ...changes }), {
      conversations: [], selectedConversationId: null,
    }, `${item.format}: ${JSON.stringify(changes)}`);
  }
});

test("completed turns preserve per-turn evidence in application-owned chat history", () => {
  const state = completeTurn(
    { conversations: [], selectedConversationId: null },
    null,
    turn,
    "conversation-1",
  );

  assert.equal(state.selectedConversationId, "conversation-1");
  assert.deepEqual(conversationMessages(state.conversations[0].turns, 32 * 1024), [
    {
      role: "user",
      content:
        "What is the answer?\n\nUNTRUSTED_EVIDENCE_JSON:\n" +
        '[{"citationId":"S1-1","text":"Stored evidence"}]',
    },
    { role: "assistant", content: "A grounded answer [S1-1]" },
  ]);
});

test("Daily Recap current messages include the resolved date", () => {
  assert.deepEqual(
    conversationUserMessage("Summary of yesterday", [evidence], "2026.09.02"),
    {
      role: "user",
      content:
        "Summary of yesterday\n\nResolved Daily Recap date: 2026.09.02\n\n" +
        "UNTRUSTED_EVIDENCE_JSON:\n" +
        '[{"citationId":"S1-1","text":"Stored evidence"}]',
    },
  );
});

test("history projection fits whole turns and stops before serializing the older archive", () => {
  const recent = { ...turn, kind: "insufficient", answer: "No support" };
  const expected = [conversationUserMessage(recent.question, recent.evidence),
    { role: "assistant", content: "INSUFFICIENT_EVIDENCE: No support" }];
  const budget = expected.reduce((sum, message) => sum + messageBytes(message), 0);
  const older = { get question() { throw new Error("must not serialize the full archive"); } };
  const oversized = { ...turn, question: "x".repeat(budget) };
  const turns = [older, oversized, recent];
  assert.deepEqual(conversationMessages(turns, budget), expected);
  assert.deepEqual(conversationMessages(turns, budget - 1), []);
  assert.deepEqual(conversationMessages(turns, 0), []);
  assert.deepEqual(conversationMessages([], budget), []);
});

test("new, select, and delete change selection without rewriting prior records", () => {
  const first = completeTurn(
    { conversations: [], selectedConversationId: null },
    null,
    turn,
    "conversation-1",
  );
  const second = completeTurn(
    newConversation(first),
    null,
    { ...turn, completedAt: 30, question: "Another question" },
    "conversation-2",
  );

  assert.deepEqual(second.conversations[1], first.conversations[0]);
  assert.equal(selectConversation(second, "conversation-1").selectedConversationId, "conversation-1");
  assert.deepEqual(deleteConversation(second, "conversation-2"), {
    conversations: [first.conversations[0]],
    selectedConversationId: "conversation-1",
  });
});

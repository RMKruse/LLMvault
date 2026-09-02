import assert from "node:assert/strict";
import test from "node:test";

import {
  completeTurn,
  conversationMessages,
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

test("completed turns preserve per-turn evidence in application-owned chat history", () => {
  const state = completeTurn(
    { conversations: [], selectedConversationId: null },
    null,
    turn,
    "conversation-1",
  );

  assert.equal(state.selectedConversationId, "conversation-1");
  assert.deepEqual(conversationMessages(state.conversations[0]), [
    {
      role: "user",
      content:
        "What is the answer?\n\nUNTRUSTED_EVIDENCE_JSON:\n" +
        '[{"citationId":"S1-1","text":"Stored evidence"}]',
    },
    { role: "assistant", content: "A grounded answer [S1-1]" },
  ]);
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

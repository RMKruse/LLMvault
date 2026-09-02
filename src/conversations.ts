import type { RetrievedEvidence } from "./indexing";
import type { OllamaMessage } from "./ollama";

export const INSUFFICIENT_PREFIX = "INSUFFICIENT_EVIDENCE:";

export interface CompletedTurn {
  answer: string;
  completedAt: number;
  evidence: RetrievedEvidence[];
  kind: "answer" | "insufficient";
  question: string;
  status: "complete";
}

export interface Conversation {
  createdAt: number;
  id: string;
  title: string;
  turns: CompletedTurn[];
  updatedAt: number;
}

export interface ConversationState {
  conversations: Conversation[];
  selectedConversationId: string | null;
}

const record = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

const finiteNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

const isCitationId = (value: unknown): value is string =>
  typeof value === "string" && /^S[1-9]\d*-[1-9]\d*$/.test(value);

function normalizeEvidence(value: unknown): RetrievedEvidence | null {
  if (
    !record(value) ||
    !isCitationId(value.citationId) ||
    typeof value.chunkId !== "string" ||
    !finiteNumber(value.end) ||
    typeof value.fingerprint !== "string" ||
    (value.format !== "markdown" && value.format !== "canvas") ||
    typeof value.path !== "string" ||
    !finiteNumber(value.score) ||
    !finiteNumber(value.start) ||
    typeof value.text !== "string"
  ) return null;

  const evidence: RetrievedEvidence = {
    citationId: value.citationId,
    chunkId: value.chunkId,
    end: value.end,
    fingerprint: value.fingerprint,
    format: value.format,
    path: value.path,
    score: value.score,
    start: value.start,
    text: value.text,
  };
  if (finiteNumber(value.endLine)) evidence.endLine = value.endLine;
  if (typeof value.excerpt === "string") evidence.excerpt = value.excerpt;
  if (typeof value.nodeId === "string") evidence.nodeId = value.nodeId;
  if (finiteNumber(value.startLine)) evidence.startLine = value.startLine;
  if (
    record(value.anchor) &&
    (value.anchor.type === "heading" || value.anchor.type === "block") &&
    typeof value.anchor.value === "string"
  ) evidence.anchor = { type: value.anchor.type, value: value.anchor.value };
  return evidence;
}

function normalizeTurn(value: unknown): CompletedTurn | null {
  if (
    !record(value) ||
    value.status !== "complete" ||
    (value.kind !== "answer" && value.kind !== "insufficient") ||
    typeof value.answer !== "string" ||
    !finiteNumber(value.completedAt) ||
    !Array.isArray(value.evidence) ||
    typeof value.question !== "string"
  ) return null;
  const evidence = value.evidence.map(normalizeEvidence);
  if (evidence.some((item) => item === null)) return null;
  return {
    answer: value.answer,
    completedAt: value.completedAt,
    evidence: evidence as RetrievedEvidence[],
    kind: value.kind,
    question: value.question,
    status: "complete",
  };
}

function normalizeConversation(value: unknown): Conversation | null {
  if (
    !record(value) ||
    !finiteNumber(value.createdAt) ||
    typeof value.id !== "string" ||
    typeof value.title !== "string" ||
    !Array.isArray(value.turns) ||
    !finiteNumber(value.updatedAt)
  ) return null;
  const turns = value.turns.map(normalizeTurn).filter((turn) => turn !== null);
  if (turns.length === 0) return null;
  return {
    createdAt: value.createdAt,
    id: value.id,
    title: value.title,
    turns,
    updatedAt: value.updatedAt,
  };
}

export function normalizeConversationState(value: unknown): ConversationState {
  if (!record(value) || !Array.isArray(value.conversations)) {
    return { conversations: [], selectedConversationId: null };
  }
  const ids = new Set<string>();
  const conversations = value.conversations
    .map(normalizeConversation)
    .filter((conversation): conversation is Conversation => {
      if (!conversation || ids.has(conversation.id)) return false;
      ids.add(conversation.id);
      return true;
    });
  return {
    conversations,
    selectedConversationId:
      typeof value.selectedConversationId === "string" && ids.has(value.selectedConversationId)
        ? value.selectedConversationId
        : null,
  };
}

export function conversationMessages(conversation?: Conversation): OllamaMessage[] {
  return conversation?.turns.flatMap((turn) => [
    conversationUserMessage(turn.question, turn.evidence),
    {
      role: "assistant" as const,
      content: turn.kind === "insufficient"
        ? `${INSUFFICIENT_PREFIX} ${turn.answer}`
        : turn.answer,
    },
  ]) ?? [];
}

export function conversationUserMessage(
  question: string,
  evidence: Pick<RetrievedEvidence, "citationId" | "text">[],
): OllamaMessage {
  return {
    role: "user",
    content: `${question}\n\nUNTRUSTED_EVIDENCE_JSON:\n${JSON.stringify(
      evidence.map(({ citationId, text }) => ({ citationId, text })),
    )}`,
  };
}

export function completeTurn(
  state: ConversationState,
  conversationId: string | null,
  turn: CompletedTurn,
  newId: string,
): ConversationState {
  const current = state.conversations.find((conversation) => conversation.id === conversationId);
  const conversation: Conversation = current
    ? { ...current, turns: [...current.turns, turn], updatedAt: turn.completedAt }
    : {
        createdAt: turn.completedAt,
        id: newId,
        title: Array.from(turn.question.trim()).slice(0, 80).join(""),
        turns: [turn],
        updatedAt: turn.completedAt,
      };
  return {
    conversations: [conversation, ...state.conversations.filter(({ id }) => id !== conversation.id)],
    selectedConversationId: conversation.id,
  };
}

export const newConversation = (state: ConversationState): ConversationState => ({
  conversations: state.conversations,
  selectedConversationId: null,
});

export const selectConversation = (
  state: ConversationState,
  id: string,
): ConversationState => ({
  conversations: state.conversations,
  selectedConversationId: state.conversations.some((conversation) => conversation.id === id)
    ? id
    : state.selectedConversationId,
});

export const deleteConversation = (
  state: ConversationState,
  id: string,
): ConversationState => {
  const conversations = state.conversations.filter((conversation) => conversation.id !== id);
  return {
    conversations,
    selectedConversationId:
      state.selectedConversationId === id
        ? conversations[0]?.id ?? null
        : state.selectedConversationId,
  };
};

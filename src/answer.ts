import { type CompletedTurn, conversationMessages, conversationUserMessage, messageBytes } from "./conversations.ts";
import type { RetrievedEvidence } from "./indexing.ts";
import { OllamaError, type OllamaChatResult, type OllamaMessage } from "./ollama.ts";
import { GROUNDING_SYSTEM_PROMPT } from "./quality.ts";

// ponytail: serialized UTF-8 byte cap, not token accounting; use the model's tokenizer if exact context fitting is needed.
export const MAX_PROMPT_BYTES = 32 * 1024;

export interface AnswerRequest {
  question: string;
  requestedAt: Date;
  timeZone: string;
  history: readonly CompletedTurn[];
  dailyRecapDate?: string;
  signal: AbortSignal;
  onEvidence?: (evidence: RetrievedEvidence[]) => void;
  onContent?: (content: string) => void;
}

interface AnswerServices {
  validateChatModel(): Promise<void>;
  retrieve(question: string, now: Date, timeZone: string): Promise<RetrievedEvidence[]>;
  revalidateEvidence(evidence: RetrievedEvidence[]): Promise<RetrievedEvidence[]>;
  chat(messages: OllamaMessage[], onContent: (content: string) => void): Promise<OllamaChatResult>;
}

export type AnswerResult =
  | { status: "evidence_changed"; evidence: RetrievedEvidence[] }
  | {
    status: "complete";
    turn: Omit<CompletedTurn, "completedAt" | "status">;
    messages: OllamaMessage[];
    chatResult: OllamaChatResult | null;
  };

/** One answer path for Vault Chat and evaluation; callers own storage and rendering. */
export async function executeAnswer(
  services: AnswerServices,
  request: AnswerRequest,
): Promise<AnswerResult> {
  const { question, requestedAt, timeZone, signal } = request;
  const checkCanceled = (): void => {
    if (signal.aborted) throw new OllamaError("canceled", "/api/chat");
  };
  checkCanceled();
  await services.validateChatModel();
  checkCanceled();
  const retrieved = await services.retrieve(question, requestedAt, timeZone);
  checkCanceled();
  const evidence = await services.revalidateEvidence(retrieved);
  checkCanceled();
  request.onEvidence?.(evidence);
  checkCanceled();
  if (evidence.length !== retrieved.length) return { status: "evidence_changed", evidence };
  if (evidence.length === 0) {
    return {
      status: "complete",
      turn: {
        question, evidence, kind: "insufficient",
        answer: "The available Vault Content is insufficient for this question. Revise the question or rebuild the index.",
      },
      messages: [],
      chatResult: null,
    };
  }

  const system: OllamaMessage = { role: "system", content: GROUNDING_SYSTEM_PROMPT };
  const current = conversationUserMessage(question, evidence, request.dailyRecapDate);
  const historyBudget = MAX_PROMPT_BYTES - 1 - messageBytes(system) - messageBytes(current);
  if (historyBudget < 0) throw new OllamaError("context_budget_exceeded", "/api/chat");
  const messages = [system, ...conversationMessages(request.history, historyBudget), current];
  let streamed = "";
  request.onContent?.(streamed);
  checkCanceled();
  const chatResult = await services.chat(messages, (content) => {
    if (signal.aborted) return;
    streamed += content;
    request.onContent?.(streamed);
  });
  checkCanceled();
  const insufficient = chatResult.content.match(/^\s*INSUFFICIENT_EVIDENCE:\s*([\s\S]*)$/i);
  const kind = insufficient || !chatResult.content.trim() ? "insufficient" : "answer";
  const answer = kind === "insufficient"
    ? insufficient?.[1]?.trim() || "The available Vault Content is insufficient for this question."
    : chatResult.content;
  return { status: "complete", turn: { question, evidence, kind, answer }, messages, chatResult };
}

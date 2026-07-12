import type { Message } from "../agent/index.js";
import type { SessionEntry } from "./entries.js";

export interface SessionState {
  messages: Message[];
  model?: string;
  sessionId?: string;
  cwd?: string;
}

export function replaySession(entries: readonly SessionEntry[]): SessionState {
  const state: SessionState = { messages: [] };
  for (const entry of entries) {
    if (entry.type === "message") state.messages.push(entry.message as Message);
    else if (entry.type === "model_change") state.model = entry.model;
    else {
      state.sessionId = entry.sessionId;
      state.cwd = entry.cwd;
    }
  }
  return state;
}

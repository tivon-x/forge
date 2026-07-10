import type { AssistantMessage, Message, ToolCall, ToolMessage } from "./messages.js";
import type { ToolResult } from "./tools.js";

export type AgentEndReason = "completed" | "cancelled" | "error" | "max_turns";

export type AgentEvent =
  | { type: "agent_start" }
  | { type: "agent_end"; reason: AgentEndReason }
  | { type: "turn_start"; turn: number }
  | { type: "turn_end"; turn: number }
  | { type: "message_start"; role: "assistant" }
  | { type: "message_delta"; delta: string }
  | { type: "thinking_delta"; delta: string }
  | { type: "message_end"; message: AssistantMessage }
  | { type: "tool_start"; call: ToolCall }
  | { type: "tool_update"; call: ToolCall; content: string }
  | { type: "tool_end"; call: ToolCall; result: ToolResult; message: ToolMessage }
  | { type: "retry"; attempt: number; message: string }
  | { type: "queue_update"; queued: number }
  | { type: "error"; code: string; message: string };

export interface AgentRunResult {
  reason: AgentEndReason;
  messages: Message[];
}

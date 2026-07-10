import type { Message, ToolCall } from "./messages.js";
import type { ToolDefinition } from "./tools.js";

export interface ProviderRequest {
  systemPrompt: string;
  messages: readonly Message[];
  tools: readonly ToolDefinition[];
  signal: AbortSignal;
}

export type ProviderEvent =
  | { type: "text_delta"; delta: string }
  | { type: "thinking_delta"; delta: string }
  | { type: "tool_call"; call: ToolCall }
  | { type: "metadata"; metadata: Record<string, unknown> };

export interface ModelProvider {
  stream(request: ProviderRequest): AsyncIterable<ProviderEvent>;
}

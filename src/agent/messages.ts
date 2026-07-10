export interface ToolCall {
  id: string;
  name: string;
  arguments: unknown;
}

export interface UserMessage {
  role: "user";
  content: string;
}

export interface AssistantMessage {
  role: "assistant";
  content: string;
  toolCalls: ToolCall[];
  providerMetadata?: Record<string, unknown>;
}

export interface ToolMessage {
  role: "tool";
  toolCallId: string;
  toolName: string;
  ok: boolean;
  content: string;
  data?: unknown;
  error?: ToolError;
}

export interface ToolError {
  code: string;
  message: string;
}

export type Message = UserMessage | AssistantMessage | ToolMessage;

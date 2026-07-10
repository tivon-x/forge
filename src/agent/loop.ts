import type { AgentEvent, AgentRunResult } from "./events.js";
import type { AssistantMessage, Message, ToolCall, ToolMessage } from "./messages.js";
import type { ModelProvider } from "./provider.js";
import type { ToolDefinition, ToolResult } from "./tools.js";

export interface AgentLoopOptions {
  provider: ModelProvider;
  tools?: readonly ToolDefinition[];
  systemPrompt: string;
  cwd: string;
  maxTurns?: number;
}

export interface AgentLoopRequest {
  messages: readonly Message[];
  signal?: AbortSignal;
}

const DEFAULT_MAX_TURNS = 20;

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isAbortError(error: unknown, signal: AbortSignal): boolean {
  return signal.aborted || (error instanceof Error && error.name === "AbortError");
}

function failedToolResult(code: string, message: string): ToolResult {
  return { ok: false, content: message, error: { code, message } };
}

function toolMessage(call: ToolCall, result: ToolResult): ToolMessage {
  return {
    role: "tool",
    toolCallId: call.id,
    toolName: call.name,
    ok: result.ok,
    content: result.content,
    ...(result.data === undefined ? {} : { data: result.data }),
    ...(result.error === undefined ? {} : { error: result.error }),
  };
}

export class AgentLoop {
  readonly #provider: ModelProvider;
  readonly #tools: readonly ToolDefinition[];
  readonly #toolsByName: ReadonlyMap<string, ToolDefinition>;
  readonly #systemPrompt: string;
  readonly #cwd: string;
  readonly #maxTurns: number;

  constructor(options: AgentLoopOptions) {
    this.#provider = options.provider;
    this.#tools = options.tools ?? [];
    this.#toolsByName = new Map(this.#tools.map((tool) => [tool.name, tool]));
    this.#systemPrompt = options.systemPrompt;
    this.#cwd = options.cwd;
    this.#maxTurns = options.maxTurns ?? DEFAULT_MAX_TURNS;

    if (this.#maxTurns < 1 || !Number.isInteger(this.#maxTurns)) {
      throw new Error("maxTurns must be a positive integer");
    }
  }

  async *run(request: AgentLoopRequest): AsyncGenerator<AgentEvent, AgentRunResult, undefined> {
    const messages = [...request.messages];
    const signal = request.signal ?? new AbortController().signal;

    yield { type: "agent_start" };

    try {
      for (let turn = 1; turn <= this.#maxTurns; turn += 1) {
        signal.throwIfAborted();
        yield { type: "turn_start", turn };
        yield { type: "message_start", role: "assistant" };

        let content = "";
        const toolCalls: ToolCall[] = [];
        let providerMetadata: Record<string, unknown> | undefined;

        for await (const event of this.#provider.stream({
          systemPrompt: this.#systemPrompt,
          messages: [...messages],
          tools: this.#tools,
          signal,
        })) {
          signal.throwIfAborted();

          if (event.type === "text_delta") {
            content += event.delta;
            yield { type: "message_delta", delta: event.delta };
          } else if (event.type === "thinking_delta") {
            yield { type: "thinking_delta", delta: event.delta };
          } else if (event.type === "tool_call") {
            toolCalls.push(event.call);
          } else {
            providerMetadata = { ...providerMetadata, ...event.metadata };
          }
        }

        const assistantMessage: AssistantMessage = {
          role: "assistant",
          content,
          toolCalls,
          ...(providerMetadata === undefined ? {} : { providerMetadata }),
        };
        messages.push(assistantMessage);
        yield { type: "message_end", message: assistantMessage };

        for (const call of toolCalls) {
          signal.throwIfAborted();
          yield { type: "tool_start", call };

          const tool = this.#toolsByName.get(call.name);
          let result: ToolResult;

          if (tool === undefined) {
            result = failedToolResult("UNKNOWN_TOOL", `Unknown tool: ${call.name}`);
          } else {
            try {
              result = await tool.execute(call.arguments, { cwd: this.#cwd, signal });
            } catch (error) {
              if (isAbortError(error, signal)) {
                throw error;
              }
              result = failedToolResult("TOOL_ERROR", errorMessage(error));
            }
          }

          const message = toolMessage(call, result);
          messages.push(message);
          yield { type: "tool_end", call, result, message };
        }

        yield { type: "turn_end", turn };

        if (toolCalls.length === 0) {
          yield { type: "agent_end", reason: "completed" };
          return { reason: "completed", messages };
        }
      }

      yield {
        type: "error",
        code: "MAX_TURNS",
        message: `Agent exceeded the maximum of ${this.#maxTurns} turns`,
      };
      yield { type: "agent_end", reason: "max_turns" };
      return { reason: "max_turns", messages };
    } catch (error) {
      if (isAbortError(error, signal)) {
        yield { type: "error", code: "CANCELLED", message: "Agent run was cancelled" };
        yield { type: "agent_end", reason: "cancelled" };
        return { reason: "cancelled", messages };
      }

      yield { type: "error", code: "PROVIDER_ERROR", message: errorMessage(error) };
      yield { type: "agent_end", reason: "error" };
      return { reason: "error", messages };
    }
  }
}

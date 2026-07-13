import OpenAI from "openai";
import type {
  ChatCompletionChunk,
  ChatCompletionCreateParamsStreaming,
  ChatCompletionMessageParam,
  ChatCompletionTool,
} from "openai/resources/chat/completions/completions";
import { z } from "zod";

import type {
  Message,
  ModelProvider,
  ProviderEvent,
  ProviderRequest,
  ToolCall,
  ToolDefinition,
} from "../agent/index.js";

interface ChatCompletionsClient {
  chat: {
    completions: {
      create(
        params: ChatCompletionCreateParamsStreaming,
        options?: { signal?: AbortSignal },
      ): Promise<AsyncIterable<ChatCompletionChunk>>;
    };
  };
}

export interface OpenAICompatibleProviderOptions {
  apiKey: string;
  baseURL: string;
  model: string;
  client?: ChatCompletionsClient;
}

interface PendingToolCall {
  id?: string;
  name?: string;
  arguments: string;
}

function serializedToolOutput(message: Extract<Message, { role: "tool" }>): string {
  return JSON.stringify({
    ok: message.ok,
    content: message.content,
    ...(message.data === undefined ? {} : { data: message.data }),
    ...(message.error === undefined ? {} : { error: message.error }),
  });
}

function toChatMessages(
  systemPrompt: string,
  messages: readonly Message[],
): ChatCompletionMessageParam[] {
  const result: ChatCompletionMessageParam[] = [{ role: "system", content: systemPrompt }];

  for (const message of messages) {
    if (message.role === "user") {
      result.push({ role: "user", content: message.content });
    } else if (message.role === "tool") {
      result.push({
        role: "tool",
        tool_call_id: message.toolCallId,
        content: serializedToolOutput(message),
      });
    } else {
      result.push({
        role: "assistant",
        content: message.content,
        ...(message.toolCalls.length === 0
          ? {}
          : {
              tool_calls: message.toolCalls.map((call) => ({
                id: call.id,
                type: "function" as const,
                function: { name: call.name, arguments: JSON.stringify(call.arguments) },
              })),
            }),
      });
    }
  }

  return result;
}

function toChatTool(tool: ToolDefinition): ChatCompletionTool {
  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: z.toJSONSchema(tool.inputSchema, { target: "draft-7" }) as Record<
        string,
        unknown
      >,
      strict: false,
    },
  };
}

function completeToolCalls(pending: Map<number, PendingToolCall>): ToolCall[] {
  return [...pending.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, call]) => {
      if (call.id === undefined || call.name === undefined) {
        throw new Error("OpenAI-compatible response returned an incomplete tool call");
      }

      try {
        return { id: call.id, name: call.name, arguments: JSON.parse(call.arguments) };
      } catch {
        throw new Error(
          `OpenAI-compatible response returned invalid JSON arguments for tool ${call.name}`,
        );
      }
    });
}

export class OpenAICompatibleProvider implements ModelProvider {
  readonly #client: ChatCompletionsClient;
  readonly #model: string;

  constructor(options: OpenAICompatibleProviderOptions) {
    if (options.apiKey.length === 0) {
      throw new Error("OpenAI-compatible API key must not be empty");
    }
    if (options.model.length === 0) {
      throw new Error("OpenAI-compatible model must not be empty");
    }
    try {
      new URL(options.baseURL);
    } catch {
      throw new Error("OpenAI-compatible base URL must be a valid URL");
    }

    this.#client =
      options.client ??
      (new OpenAI({ apiKey: options.apiKey, baseURL: options.baseURL }) as ChatCompletionsClient);
    this.#model = options.model;
  }

  async *stream(request: ProviderRequest): AsyncIterable<ProviderEvent> {
    const stream = await this.#client.chat.completions.create(
      {
        model: this.#model,
        messages: toChatMessages(request.systemPrompt, request.messages),
        tools: request.tools.map(toChatTool),
        parallel_tool_calls: true,
        stream: true,
      },
      { signal: request.signal },
    );
    const toolCalls = new Map<number, PendingToolCall>();
    let completed = false;

    for await (const chunk of stream) {
      const choice = chunk.choices[0];
      if (choice === undefined) {
        continue;
      }
      if (choice.delta.content !== undefined && choice.delta.content !== null) {
        yield { type: "text_delta", delta: choice.delta.content };
      }
      for (const toolCall of choice.delta.tool_calls ?? []) {
        const current = toolCalls.get(toolCall.index) ?? { arguments: "" };
        if (toolCall.id !== undefined) current.id = toolCall.id;
        if (toolCall.function?.name !== undefined) current.name = toolCall.function.name;
        if (toolCall.function?.arguments !== undefined)
          current.arguments += toolCall.function.arguments;
        toolCalls.set(toolCall.index, current);
      }
      if (choice.finish_reason === "stop" || choice.finish_reason === "tool_calls") {
        for (const call of completeToolCalls(toolCalls)) {
          yield { type: "tool_call", call };
        }
        yield {
          type: "metadata",
          metadata: { responseId: chunk.id, model: chunk.model, usage: chunk.usage },
        };
        yield { type: "response_end" };
        completed = true;
        break;
      }
      if (choice.finish_reason !== null) {
        throw new Error(`OpenAI-compatible response ended unsuccessfully: ${choice.finish_reason}`);
      }
    }

    if (!completed) {
      throw new Error("OpenAI-compatible response ended without a completion event");
    }
  }
}

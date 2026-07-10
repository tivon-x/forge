import type {
  ResponseCreateParamsStreaming,
  ResponseStreamEvent,
} from "openai/resources/responses/responses";
import { describe, expect, it } from "vitest";
import { z } from "zod";

import { defineTool, type ProviderRequest } from "../agent/index.js";
import { OpenAIResponsesProvider } from "./openai-responses.js";

class FakeClient {
  readonly requests: Array<{
    params: ResponseCreateParamsStreaming;
    options: { signal?: AbortSignal } | undefined;
  }> = [];
  readonly #events: ResponseStreamEvent[];

  constructor(events: ResponseStreamEvent[]) {
    this.#events = events;
  }

  readonly responses = {
    create: async (
      params: ResponseCreateParamsStreaming,
      options?: { signal?: AbortSignal },
    ): Promise<AsyncIterable<ResponseStreamEvent>> => {
      this.requests.push({ params, options });
      const events = this.#events;
      return {
        async *[Symbol.asyncIterator]() {
          for (const event of events) {
            yield event;
          }
        },
      };
    },
  };
}

function request(overrides: Partial<ProviderRequest> = {}): ProviderRequest {
  return {
    systemPrompt: "system",
    messages: [{ role: "user", content: "hello" }],
    tools: [],
    signal: new AbortController().signal,
    ...overrides,
  };
}

async function collect(provider: OpenAIResponsesProvider, input: ProviderRequest) {
  const events = [];
  for await (const event of provider.stream(input)) {
    events.push(event);
  }
  return events;
}

describe("OpenAIResponsesProvider", () => {
  it("maps Forge messages and tool schemas to a streaming Responses request", async () => {
    const client = new FakeClient([]);
    const tool = defineTool({
      name: "echo",
      description: "Echo a value",
      inputSchema: z.object({ value: z.string() }),
      execute: async ({ value }) => ({ ok: true, content: value }),
    });
    const signal = new AbortController().signal;
    const provider = new OpenAIResponsesProvider({ apiKey: "test", model: "test-model", client });

    await collect(
      provider,
      request({
        signal,
        tools: [tool],
        messages: [
          { role: "user", content: "use the tool" },
          {
            role: "assistant",
            content: "",
            toolCalls: [{ id: "call-1", name: "echo", arguments: { value: "a" } }],
          },
          {
            role: "tool",
            toolCallId: "call-1",
            toolName: "echo",
            ok: true,
            content: "a",
            data: { value: "a" },
          },
        ],
      }),
    );

    expect(client.requests).toHaveLength(1);
    expect(client.requests[0]?.params).toMatchObject({
      model: "test-model",
      instructions: "system",
      stream: true,
      parallel_tool_calls: true,
      input: [
        { role: "user", content: "use the tool" },
        {
          type: "function_call",
          call_id: "call-1",
          name: "echo",
          arguments: '{"value":"a"}',
        },
        {
          type: "function_call_output",
          call_id: "call-1",
          output: '{"ok":true,"content":"a","data":{"value":"a"}}',
        },
      ],
      tools: [
        {
          type: "function",
          name: "echo",
          description: "Echo a value",
          strict: false,
          parameters: expect.objectContaining({ type: "object" }),
        },
      ],
    });
    expect(client.requests[0]?.options?.signal).toBe(signal);
  });

  it("converts text, reasoning, tool calls, and completion metadata", async () => {
    const functionCall = {
      id: "item-1",
      type: "function_call" as const,
      call_id: "call-1",
      name: "echo",
      arguments: '{"value":"a"}',
      status: "completed" as const,
    };
    const output = [functionCall];
    const client = new FakeClient([
      {
        type: "response.reasoning_summary_text.delta",
        delta: "thinking",
        item_id: "reasoning-1",
        output_index: 0,
        summary_index: 0,
        sequence_number: 1,
      },
      {
        type: "response.output_text.delta",
        delta: "answer",
        item_id: "message-1",
        output_index: 1,
        content_index: 0,
        logprobs: [],
        sequence_number: 2,
      },
      {
        type: "response.output_item.done",
        item: functionCall,
        output_index: 2,
        sequence_number: 3,
      },
      {
        type: "response.completed",
        response: {
          id: "response-1",
          model: "test-model",
          output,
          usage: { input_tokens: 1, output_tokens: 2, total_tokens: 3 },
        } as never,
        sequence_number: 4,
      },
    ]);
    const provider = new OpenAIResponsesProvider({ apiKey: "test", model: "test-model", client });

    const events = await collect(provider, request());

    expect(events).toEqual([
      { type: "thinking_delta", delta: "thinking" },
      { type: "text_delta", delta: "answer" },
      {
        type: "tool_call",
        call: { id: "call-1", name: "echo", arguments: { value: "a" } },
      },
      {
        type: "metadata",
        metadata: {
          responseId: "response-1",
          model: "test-model",
          usage: { input_tokens: 1, output_tokens: 2, total_tokens: 3 },
          openaiResponseOutput: output,
        },
      },
    ]);
  });

  it("round-trips native OpenAI output items from provider metadata", async () => {
    const client = new FakeClient([]);
    const provider = new OpenAIResponsesProvider({ apiKey: "test", model: "test-model", client });
    const nativeOutput = [
      {
        id: "reasoning-1",
        type: "reasoning" as const,
        summary: [],
      },
    ];

    await collect(
      provider,
      request({
        messages: [
          {
            role: "assistant",
            content: "ignored reconstructed text",
            toolCalls: [],
            providerMetadata: { openaiResponseOutput: nativeOutput },
          },
        ],
      }),
    );

    expect(client.requests[0]?.params.input).toEqual(nativeOutput);
  });

  it("rejects malformed tool arguments", async () => {
    const client = new FakeClient([
      {
        type: "response.output_item.done",
        item: {
          id: "item-1",
          type: "function_call",
          call_id: "call-1",
          name: "echo",
          arguments: "not-json",
          status: "completed",
        },
        output_index: 0,
        sequence_number: 1,
      },
    ]);
    const provider = new OpenAIResponsesProvider({ apiKey: "test", model: "test-model", client });

    await expect(collect(provider, request())).rejects.toThrow(
      "OpenAI returned invalid JSON arguments for tool echo",
    );
  });

  it("surfaces failed responses", async () => {
    const client = new FakeClient([
      {
        type: "response.failed",
        response: { error: { message: "quota exhausted" } } as never,
        sequence_number: 1,
      },
    ]);
    const provider = new OpenAIResponsesProvider({ apiKey: "test", model: "test-model", client });

    await expect(collect(provider, request())).rejects.toThrow("quota exhausted");
  });
});

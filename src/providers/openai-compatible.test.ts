import type {
  ChatCompletionChunk,
  ChatCompletionCreateParamsStreaming,
} from "openai/resources/chat/completions/completions";
import { describe, expect, it } from "vitest";
import { z } from "zod";

import { defineTool, type ProviderRequest } from "../agent/index.js";
import { OpenAICompatibleProvider } from "./openai-compatible.js";

class FakeClient {
  readonly requests: Array<{
    params: ChatCompletionCreateParamsStreaming;
    options: { signal?: AbortSignal } | undefined;
  }> = [];

  constructor(readonly events: ChatCompletionChunk[]) {}

  readonly chat = {
    completions: {
      create: async (
        params: ChatCompletionCreateParamsStreaming,
        options?: { signal?: AbortSignal },
      ): Promise<AsyncIterable<ChatCompletionChunk>> => {
        this.requests.push({ params, options });
        const { events } = this;
        return {
          async *[Symbol.asyncIterator]() {
            yield* events;
          },
        };
      },
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

async function collect(provider: OpenAICompatibleProvider, input: ProviderRequest) {
  const events = [];
  for await (const event of provider.stream(input)) events.push(event);
  return events;
}

describe("OpenAICompatibleProvider", () => {
  it("maps Forge messages and tools to a streaming Chat Completions request", async () => {
    const client = new FakeClient([]);
    const signal = new AbortController().signal;
    const tool = defineTool({
      name: "echo",
      description: "Echo a value",
      inputSchema: z.object({ value: z.string() }),
      execute: async ({ value }) => ({ ok: true, content: value }),
    });
    const provider = new OpenAICompatibleProvider({
      apiKey: "test",
      baseURL: "https://example.test/v1",
      model: "test-model",
      client,
    });

    await expect(
      collect(
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
            },
          ],
        }),
      ),
    ).rejects.toThrow("ended without a completion event");

    expect(client.requests[0]?.params).toMatchObject({
      model: "test-model",
      stream: true,
      parallel_tool_calls: true,
      messages: [
        { role: "system", content: "system" },
        { role: "user", content: "use the tool" },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              id: "call-1",
              type: "function",
              function: { name: "echo", arguments: '{"value":"a"}' },
            },
          ],
        },
        { role: "tool", tool_call_id: "call-1", content: '{"ok":true,"content":"a"}' },
      ],
      tools: [
        {
          type: "function",
          function: {
            name: "echo",
            description: "Echo a value",
            strict: false,
            parameters: expect.objectContaining({ type: "object" }),
          },
        },
      ],
    });
    expect(client.requests[0]?.options?.signal).toBe(signal);
  });

  it("streams text, reconstructs tool calls, and emits one completion", async () => {
    const client = new FakeClient([
      {
        id: "chat-1",
        model: "compatible-model",
        object: "chat.completion.chunk",
        created: 1,
        choices: [
          {
            index: 0,
            delta: {
              content: "answer",
              tool_calls: [
                {
                  index: 0,
                  id: "call-1",
                  type: "function",
                  function: { name: "echo", arguments: '{"value":' },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      },
      {
        id: "chat-1",
        model: "compatible-model",
        object: "chat.completion.chunk",
        created: 1,
        choices: [
          {
            index: 0,
            delta: { tool_calls: [{ index: 0, function: { arguments: '"a"}' } }] },
            finish_reason: "tool_calls",
          },
        ],
      },
    ]);
    const provider = new OpenAICompatibleProvider({
      apiKey: "test",
      baseURL: "https://example.test/v1",
      model: "test-model",
      client,
    });

    await expect(collect(provider, request())).resolves.toEqual([
      { type: "text_delta", delta: "answer" },
      { type: "tool_call", call: { id: "call-1", name: "echo", arguments: { value: "a" } } },
      {
        type: "metadata",
        metadata: { responseId: "chat-1", model: "compatible-model", usage: undefined },
      },
      { type: "response_end" },
    ]);
  });

  it("rejects malformed tool arguments and unfinished streams", async () => {
    const malformed = new OpenAICompatibleProvider({
      apiKey: "test",
      baseURL: "https://example.test/v1",
      model: "test-model",
      client: new FakeClient([
        {
          id: "chat-1",
          model: "compatible-model",
          object: "chat.completion.chunk",
          created: 1,
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    id: "call-1",
                    type: "function",
                    function: { name: "echo", arguments: "bad" },
                  },
                ],
              },
              finish_reason: "tool_calls",
            },
          ],
        },
      ]),
    });
    const unfinished = new OpenAICompatibleProvider({
      apiKey: "test",
      baseURL: "https://example.test/v1",
      model: "test-model",
      client: new FakeClient([]),
    });

    await expect(collect(malformed, request())).rejects.toThrow(
      "invalid JSON arguments for tool echo",
    );
    await expect(collect(unfinished, request())).rejects.toThrow(
      "ended without a completion event",
    );
  });

  it("rejects incomplete provider finish reasons", async () => {
    const provider = new OpenAICompatibleProvider({
      apiKey: "test",
      baseURL: "https://example.test/v1",
      model: "test-model",
      client: new FakeClient([
        {
          id: "chat-1",
          model: "compatible-model",
          object: "chat.completion.chunk",
          created: 1,
          choices: [{ index: 0, delta: {}, finish_reason: "length" }],
        },
      ]),
    });

    await expect(collect(provider, request())).rejects.toThrow(
      "OpenAI-compatible response ended unsuccessfully: length",
    );
  });
});

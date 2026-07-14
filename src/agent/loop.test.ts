import { describe, expect, it, vi } from "vitest";
import { z } from "zod";

import type { AgentEvent, AgentRunResult } from "./events.js";
import { AgentHarness } from "./harness.js";
import type { ModelProvider, ProviderEvent, ProviderRequest } from "./provider.js";
import { defineTool } from "./tools.js";

class ScriptedProvider implements ModelProvider {
  readonly requests: ProviderRequest[] = [];
  readonly #responses: Array<ProviderEvent[] | Error>;
  readonly #completeResponses: boolean;

  constructor(responses: Array<ProviderEvent[] | Error>, completeResponses = true) {
    this.#responses = [...responses];
    this.#completeResponses = completeResponses;
  }

  async *stream(request: ProviderRequest): AsyncIterable<ProviderEvent> {
    this.requests.push(request);
    const response = this.#responses.shift();
    if (response === undefined) {
      throw new Error("No scripted response available");
    }
    if (response instanceof Error) {
      throw response;
    }
    for (const event of response) {
      yield event;
    }
    if (this.#completeResponses && !response.some((event) => event.type === "response_end")) {
      yield { type: "response_end" };
    }
  }
}

async function collect(
  stream: AsyncGenerator<AgentEvent, AgentRunResult, undefined>,
): Promise<{ events: AgentEvent[]; result: AgentRunResult }> {
  const events: AgentEvent[] = [];
  while (true) {
    const item = await stream.next();
    if (item.done) {
      return { events, result: item.value };
    }
    events.push(item.value);
  }
}

const echoTool = defineTool({
  name: "echo",
  description: "Echo text",
  inputSchema: z.object({ text: z.string() }),
  execute: async ({ text }) => ({ ok: true, content: text, data: { text } }),
});

describe("AgentHarness", () => {
  it("streams a final answer without tools", async () => {
    const provider = new ScriptedProvider([
      [
        { type: "text_delta", delta: "hel" },
        { type: "text_delta", delta: "lo" },
        { type: "metadata", metadata: { responseId: "response-1" } },
      ],
    ]);
    const harness = new AgentHarness({ provider, systemPrompt: "test", cwd: process.cwd() });

    const { events, result } = await collect(harness.run("hi"));

    expect(result.reason).toBe("completed");
    expect(result.messages).toEqual([
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: "hello",
        toolCalls: [],
        providerMetadata: { responseId: "response-1" },
      },
    ]);
    expect(events.filter((event) => event.type === "message_delta")).toHaveLength(2);
  });

  it("includes externally appended messages in the next run", async () => {
    const provider = new ScriptedProvider([[{ type: "text_delta", delta: "done" }]]);
    const harness = new AgentHarness({ provider, systemPrompt: "test", cwd: process.cwd() });
    harness.appendUserMessage("terminal context");

    await collect(harness.run("continue"));

    expect(provider.requests[0]?.messages).toMatchObject([
      { role: "user", content: "terminal context" },
      { role: "user", content: "continue" },
    ]);
  });

  it("feeds a tool result back to the provider", async () => {
    const provider = new ScriptedProvider([
      [
        {
          type: "tool_call",
          call: { id: "call-1", name: "echo", arguments: { text: "value" } },
        },
      ],
      [{ type: "text_delta", delta: "done" }],
    ]);
    const harness = new AgentHarness({
      provider,
      tools: [echoTool],
      systemPrompt: "test",
      cwd: process.cwd(),
    });

    const { result } = await collect(harness.run("use echo"));

    expect(result.reason).toBe("completed");
    expect(provider.requests).toHaveLength(2);
    expect(provider.requests[1]?.messages.at(-1)).toMatchObject({
      role: "tool",
      toolCallId: "call-1",
      ok: true,
      content: "value",
    });
  });

  it("executes multiple tool calls in order", async () => {
    const execute = vi.fn(async ({ text }: { text: string }) => ({ ok: true, content: text }));
    const tool = defineTool({
      name: "ordered",
      description: "Record execution order",
      inputSchema: z.object({ text: z.string() }),
      execute,
    });
    const provider = new ScriptedProvider([
      [
        { type: "tool_call", call: { id: "1", name: "ordered", arguments: { text: "a" } } },
        { type: "tool_call", call: { id: "2", name: "ordered", arguments: { text: "b" } } },
      ],
      [{ type: "text_delta", delta: "done" }],
    ]);
    const harness = new AgentHarness({
      provider,
      tools: [tool],
      systemPrompt: "test",
      cwd: process.cwd(),
    });

    await collect(harness.run("run both"));

    expect(execute.mock.calls.map(([input]) => input.text)).toEqual(["a", "b"]);
  });

  it("returns unknown tools to the model as errors", async () => {
    const provider = new ScriptedProvider([
      [{ type: "tool_call", call: { id: "1", name: "missing", arguments: {} } }],
      [{ type: "text_delta", delta: "recovered" }],
    ]);
    const harness = new AgentHarness({ provider, systemPrompt: "test", cwd: process.cwd() });

    await collect(harness.run("call it"));

    expect(provider.requests[1]?.messages.at(-1)).toMatchObject({
      role: "tool",
      ok: false,
      error: { code: "UNKNOWN_TOOL" },
    });
  });

  it("returns tool exceptions to the model as errors", async () => {
    const brokenTool = defineTool({
      name: "broken",
      description: "Always fails",
      inputSchema: z.object({}),
      execute: async () => {
        throw new Error("tool failed");
      },
    });
    const provider = new ScriptedProvider([
      [{ type: "tool_call", call: { id: "1", name: "broken", arguments: {} } }],
      [{ type: "text_delta", delta: "recovered" }],
    ]);
    const harness = new AgentHarness({
      provider,
      tools: [brokenTool],
      systemPrompt: "test",
      cwd: process.cwd(),
    });

    await collect(harness.run("call it"));

    expect(provider.requests[1]?.messages.at(-1)).toMatchObject({
      role: "tool",
      ok: false,
      content: "tool failed",
      error: { code: "TOOL_ERROR" },
    });
  });

  it("returns a cancelled result when aborted", async () => {
    const provider = new ScriptedProvider([[{ type: "text_delta", delta: "unused" }]]);
    const harness = new AgentHarness({ provider, systemPrompt: "test", cwd: process.cwd() });
    const controller = new AbortController();
    controller.abort();

    const { events, result } = await collect(harness.run("stop", controller.signal));

    expect(result.reason).toBe("cancelled");
    expect(events).toContainEqual({
      type: "error",
      code: "CANCELLED",
      message: expect.any(String),
    });
    expect(provider.requests).toHaveLength(0);
  });

  it("records cancelled results for skipped tool calls", async () => {
    const controller = new AbortController();
    const tool = defineTool({
      name: "cancel",
      description: "Cancel after execution",
      inputSchema: z.object({}),
      execute: async () => {
        controller.abort();
        return { ok: true, content: "first completed" };
      },
    });
    const provider = new ScriptedProvider([
      [
        { type: "tool_call", call: { id: "1", name: "cancel", arguments: {} } },
        { type: "tool_call", call: { id: "2", name: "cancel", arguments: {} } },
      ],
    ]);
    const harness = new AgentHarness({
      provider,
      tools: [tool],
      systemPrompt: "test",
      cwd: process.cwd(),
    });

    const { result } = await collect(harness.run("cancel batch", controller.signal));
    const toolMessages = result.messages.filter((message) => message.role === "tool");

    expect(result.reason).toBe("cancelled");
    expect(toolMessages).toMatchObject([
      { toolCallId: "1", ok: true, content: "first completed" },
      { toolCallId: "2", ok: false, error: { code: "CANCELLED" } },
    ]);
  });

  it("records cancelled results when the active tool aborts", async () => {
    const controller = new AbortController();
    const tool = defineTool({
      name: "abort",
      description: "Abort during execution",
      inputSchema: z.object({}),
      execute: async () => {
        controller.abort();
        controller.signal.throwIfAborted();
        return { ok: true, content: "unreachable" };
      },
    });
    const provider = new ScriptedProvider([
      [
        { type: "tool_call", call: { id: "1", name: "abort", arguments: {} } },
        { type: "tool_call", call: { id: "2", name: "abort", arguments: {} } },
      ],
    ]);
    const harness = new AgentHarness({
      provider,
      tools: [tool],
      systemPrompt: "test",
      cwd: process.cwd(),
    });

    const { result } = await collect(harness.run("abort batch", controller.signal));
    const toolMessages = result.messages.filter((message) => message.role === "tool");

    expect(result.reason).toBe("cancelled");
    expect(toolMessages).toHaveLength(2);
    expect(toolMessages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          toolCallId: "1",
          error: { code: "CANCELLED", message: expect.any(String) },
        }),
        expect.objectContaining({
          toolCallId: "2",
          error: { code: "CANCELLED", message: expect.any(String) },
        }),
      ]),
    );
  });

  it("rejects concurrent runs without losing transcript state", async () => {
    let releaseProvider: (() => void) | undefined;
    let markStarted: (() => void) | undefined;
    const providerGate = new Promise<void>((resolve) => {
      releaseProvider = resolve;
    });
    const providerStarted = new Promise<void>((resolve) => {
      markStarted = resolve;
    });
    const provider: ModelProvider = {
      async *stream() {
        markStarted?.();
        await providerGate;
        yield { type: "text_delta", delta: "done" };
        yield { type: "response_end" };
      },
    };
    const harness = new AgentHarness({ provider, systemPrompt: "test", cwd: process.cwd() });
    const firstRun = collect(harness.run("first"));
    await providerStarted;

    await expect(collect(harness.run("second"))).rejects.toThrow("AgentHarness is already running");
    expect(() => harness.appendUserMessage("late")).toThrow("AgentHarness is already running");
    releaseProvider?.();
    const { result } = await firstRun;

    expect(result.reason).toBe("completed");
    expect(harness.messages[0]).toEqual({ role: "user", content: "first" });
  });

  it("stops after the configured maximum turns", async () => {
    const provider = new ScriptedProvider([
      [{ type: "tool_call", call: { id: "1", name: "echo", arguments: { text: "a" } } }],
      [{ type: "tool_call", call: { id: "2", name: "echo", arguments: { text: "b" } } }],
    ]);
    const harness = new AgentHarness({
      provider,
      tools: [echoTool],
      systemPrompt: "test",
      cwd: process.cwd(),
      maxTurns: 2,
    });

    const { events, result } = await collect(harness.run("loop"));

    expect(result.reason).toBe("max_turns");
    expect(events).toContainEqual({ type: "agent_end", reason: "max_turns" });
  });

  it("converts provider failures into error events", async () => {
    const provider = new ScriptedProvider([new Error("provider unavailable")]);
    const harness = new AgentHarness({ provider, systemPrompt: "test", cwd: process.cwd() });

    const { events, result } = await collect(harness.run("hi"));

    expect(result.reason).toBe("error");
    expect(events).toContainEqual({
      type: "error",
      code: "PROVIDER_ERROR",
      message: "provider unavailable",
    });
  });

  it("rejects a provider stream without a response_end event", async () => {
    const provider = new ScriptedProvider([[{ type: "text_delta", delta: "partial" }]], false);
    const harness = new AgentHarness({ provider, systemPrompt: "test", cwd: process.cwd() });

    const { events, result } = await collect(harness.run("hi"));

    expect(result.reason).toBe("error");
    expect(events).toContainEqual({
      type: "error",
      code: "PROVIDER_ERROR",
      message: "Provider stream ended without a response_end event",
    });
  });
});

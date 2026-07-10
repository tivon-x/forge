import OpenAI from "openai";
import type {
  FunctionTool,
  ResponseCreateParamsStreaming,
  ResponseInput,
  ResponseInputItem,
  ResponseOutputItem,
  ResponseStreamEvent,
} from "openai/resources/responses/responses";
import { z } from "zod";

import type {
  Message,
  ModelProvider,
  ProviderEvent,
  ProviderRequest,
  ToolDefinition,
} from "../agent/index.js";

interface ResponsesClient {
  responses: {
    create(
      params: ResponseCreateParamsStreaming,
      options?: { signal?: AbortSignal },
    ): Promise<AsyncIterable<ResponseStreamEvent>>;
  };
}

export interface OpenAIResponsesProviderOptions {
  apiKey: string;
  model: string;
  client?: ResponsesClient;
}

function isResponseOutput(value: unknown): value is ResponseOutputItem[] {
  return (
    Array.isArray(value) &&
    value.every(
      (item) =>
        typeof item === "object" &&
        item !== null &&
        "type" in item &&
        typeof item.type === "string",
    )
  );
}

function serializedToolOutput(message: Extract<Message, { role: "tool" }>): string {
  return JSON.stringify({
    ok: message.ok,
    content: message.content,
    ...(message.data === undefined ? {} : { data: message.data }),
    ...(message.error === undefined ? {} : { error: message.error }),
  });
}

function toOpenAIInput(messages: readonly Message[]): ResponseInput {
  const input: ResponseInput = [];

  for (const message of messages) {
    if (message.role === "user") {
      input.push({ role: "user", content: message.content });
      continue;
    }

    if (message.role === "tool") {
      input.push({
        type: "function_call_output",
        call_id: message.toolCallId,
        output: serializedToolOutput(message),
      });
      continue;
    }

    const openAIOutput = message.providerMetadata?.openaiResponseOutput;
    if (isResponseOutput(openAIOutput)) {
      input.push(...(openAIOutput as ResponseInputItem[]));
      continue;
    }

    if (message.content.length > 0) {
      input.push({ role: "assistant", content: message.content });
    }
    for (const call of message.toolCalls) {
      input.push({
        type: "function_call",
        call_id: call.id,
        name: call.name,
        arguments: JSON.stringify(call.arguments),
      });
    }
  }

  return input;
}

function toOpenAITool(tool: ToolDefinition): FunctionTool {
  return {
    type: "function",
    name: tool.name,
    description: tool.description,
    parameters: z.toJSONSchema(tool.inputSchema, { target: "draft-7" }) as Record<string, unknown>,
    strict: false,
  };
}

function responseErrorMessage(event: ResponseStreamEvent): string | undefined {
  if (event.type === "error") {
    return event.message;
  }
  if (event.type === "response.failed") {
    return event.response.error?.message ?? "OpenAI response failed";
  }
  if (event.type === "response.incomplete") {
    const reason = event.response.incomplete_details?.reason;
    return reason === undefined || reason === null
      ? "OpenAI response was incomplete"
      : `OpenAI response was incomplete: ${reason}`;
  }
  return undefined;
}

export class OpenAIResponsesProvider implements ModelProvider {
  readonly #client: ResponsesClient;
  readonly #model: string;

  constructor(options: OpenAIResponsesProviderOptions) {
    if (options.apiKey.length === 0) {
      throw new Error("OpenAI API key must not be empty");
    }
    if (options.model.length === 0) {
      throw new Error("OpenAI model must not be empty");
    }

    this.#client = options.client ?? (new OpenAI({ apiKey: options.apiKey }) as ResponsesClient);
    this.#model = options.model;
  }

  async *stream(request: ProviderRequest): AsyncIterable<ProviderEvent> {
    const stream = await this.#client.responses.create(
      {
        model: this.#model,
        instructions: request.systemPrompt,
        input: toOpenAIInput(request.messages),
        tools: request.tools.map(toOpenAITool),
        parallel_tool_calls: true,
        stream: true,
      },
      { signal: request.signal },
    );

    for await (const event of stream) {
      const responseError = responseErrorMessage(event);
      if (responseError !== undefined) {
        throw new Error(responseError);
      }

      if (event.type === "response.output_text.delta" || event.type === "response.refusal.delta") {
        yield { type: "text_delta", delta: event.delta };
      } else if (
        event.type === "response.reasoning_summary_text.delta" ||
        event.type === "response.reasoning_text.delta"
      ) {
        yield { type: "thinking_delta", delta: event.delta };
      } else if (
        event.type === "response.output_item.done" &&
        event.item.type === "function_call"
      ) {
        let parsedArguments: unknown;
        try {
          parsedArguments = JSON.parse(event.item.arguments);
        } catch {
          throw new Error(`OpenAI returned invalid JSON arguments for tool ${event.item.name}`);
        }
        yield {
          type: "tool_call",
          call: {
            id: event.item.call_id,
            name: event.item.name,
            arguments: parsedArguments,
          },
        };
      } else if (event.type === "response.completed") {
        yield {
          type: "metadata",
          metadata: {
            responseId: event.response.id,
            model: event.response.model,
            usage: event.response.usage,
            openaiResponseOutput: event.response.output,
          },
        };
      }
    }
  }
}

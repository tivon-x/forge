import type { z } from "zod";

export interface ToolExecutionContext {
  cwd: string;
  signal: AbortSignal;
}

export interface ToolResult {
  ok: boolean;
  content: string;
  data?: unknown;
  error?: {
    code: string;
    message: string;
  };
}

export interface ToolDefinition {
  name: string;
  description: string;
  inputSchema: z.ZodType;
  execute(input: unknown, context: ToolExecutionContext): Promise<ToolResult>;
  promptSnippet?: string;
  promptGuidelines?: string[];
}

interface TypedToolDefinition<TSchema extends z.ZodType> {
  name: string;
  description: string;
  inputSchema: TSchema;
  execute(input: z.output<TSchema>, context: ToolExecutionContext): Promise<ToolResult>;
  promptSnippet?: string;
  promptGuidelines?: string[];
}

export function defineTool<TSchema extends z.ZodType>(
  definition: TypedToolDefinition<TSchema>,
): ToolDefinition {
  return {
    name: definition.name,
    description: definition.description,
    inputSchema: definition.inputSchema,
    execute: async (input, context) =>
      definition.execute(definition.inputSchema.parse(input), context),
    ...(definition.promptSnippet === undefined ? {} : { promptSnippet: definition.promptSnippet }),
    ...(definition.promptGuidelines === undefined
      ? {}
      : { promptGuidelines: definition.promptGuidelines }),
  };
}

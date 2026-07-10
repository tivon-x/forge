import type { ToolResult } from "../agent/index.js";

export class ToolOperationError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "ToolOperationError";
    this.code = code;
  }
}

export function toolFailure(error: unknown, fallbackCode: string): ToolResult {
  const code = error instanceof ToolOperationError ? error.code : fallbackCode;
  const message = error instanceof Error ? error.message : String(error);
  return { ok: false, content: message, error: { code, message } };
}

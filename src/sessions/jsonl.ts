import { type SessionEntry, sessionEntrySchema } from "./entries.js";

export class SessionJsonlError extends Error {
  readonly code = "INVALID_SESSION_JSONL";
  readonly lineNumber: number;

  constructor(lineNumber: number, cause: unknown) {
    super(`Invalid session entry on line ${lineNumber}`, { cause });
    this.name = "SessionJsonlError";
    this.lineNumber = lineNumber;
  }
}

export function encodeSessionEntry(entry: SessionEntry): string {
  return `${JSON.stringify(entry)}\n`;
}

export function decodeSessionEntries(content: string): SessionEntry[] {
  const entries: SessionEntry[] = [];
  for (const [index, line] of content.split(/\r?\n/u).entries()) {
    if (line.trim().length === 0) continue;
    try {
      entries.push(sessionEntrySchema.parse(JSON.parse(line)));
    } catch (error) {
      throw new SessionJsonlError(index + 1, error);
    }
  }
  return entries;
}

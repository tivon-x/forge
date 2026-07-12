import { appendFile, mkdir, readFile } from "node:fs/promises";
import path from "node:path";

import type { SessionEntry } from "./entries.js";
import { decodeSessionEntries, encodeSessionEntry } from "./jsonl.js";

export interface SessionStorage {
  append(entry: SessionEntry): Promise<void>;
  readAll(): Promise<SessionEntry[]>;
}

export class MemorySessionStorage implements SessionStorage {
  readonly #entries: SessionEntry[] = [];

  async append(entry: SessionEntry): Promise<void> {
    this.#entries.push(structuredClone(entry));
  }

  async readAll(): Promise<SessionEntry[]> {
    return structuredClone(this.#entries);
  }
}

export class JsonlSessionStorage implements SessionStorage {
  readonly path: string;

  constructor(filePath: string) {
    this.path = filePath;
  }

  async append(entry: SessionEntry): Promise<void> {
    await mkdir(path.dirname(this.path), { recursive: true });
    await appendFile(this.path, encodeSessionEntry(entry), "utf8");
  }

  async readAll(): Promise<SessionEntry[]> {
    try {
      return decodeSessionEntries(await readFile(this.path, "utf8"));
    } catch (error) {
      if (error instanceof Error && "code" in error && error.code === "ENOENT") return [];
      throw error;
    }
  }
}

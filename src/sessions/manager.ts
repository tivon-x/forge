import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { z } from "zod";

const sessionRecordSchema = z
  .object({
    id: z.string().min(1),
    path: z.string().min(1),
    cwd: z.string().min(1),
    model: z.string().min(1),
    createdAt: z.string().datetime(),
    updatedAt: z.string().datetime(),
  })
  .strict();
const sessionIndexSchema = z.array(sessionRecordSchema);

export type SessionRecord = z.infer<typeof sessionRecordSchema>;

export interface SessionManagerOptions {
  sessionsDir?: string;
}

function projectKey(cwd: string): string {
  const name = path.basename(cwd).replace(/[^a-zA-Z0-9._-]/gu, "-") || "project";
  const hash = createHash("sha256").update(cwd).digest("hex").slice(0, 8);
  return `${name}-${hash}`;
}

export class SessionManager {
  readonly #sessionsDir: string;

  constructor(options: SessionManagerOptions = {}) {
    this.#sessionsDir = options.sessionsDir ?? path.join(os.homedir(), ".forge", "sessions");
  }

  projectDir(cwd: string): string {
    return path.join(this.#sessionsDir, projectKey(path.resolve(cwd)));
  }

  async create(cwd: string, model: string): Promise<SessionRecord> {
    const resolvedCwd = path.resolve(cwd);
    const now = new Date().toISOString();
    const id = randomUUID();
    const record: SessionRecord = {
      id,
      path: path.join(this.projectDir(resolvedCwd), `${id}.jsonl`),
      cwd: resolvedCwd,
      model,
      createdAt: now,
      updatedAt: now,
    };
    await this.#upsert(record);
    return record;
  }

  async list(cwd: string): Promise<SessionRecord[]> {
    const records = await this.#readIndex(path.resolve(cwd));
    return records.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
  }

  async get(cwd: string, id: string): Promise<SessionRecord | undefined> {
    return (await this.list(cwd)).find((record) => record.id === id);
  }

  async touch(cwd: string, id: string, model?: string): Promise<SessionRecord | undefined> {
    const record = await this.get(cwd, id);
    if (record === undefined) return undefined;
    const updated = {
      ...record,
      ...(model === undefined ? {} : { model }),
      updatedAt: new Date().toISOString(),
    };
    await this.#upsert(updated);
    return updated;
  }

  async #readIndex(cwd: string): Promise<SessionRecord[]> {
    try {
      const value = JSON.parse(
        await readFile(path.join(this.projectDir(cwd), "index.json"), "utf8"),
      );
      return sessionIndexSchema.parse(value);
    } catch (error) {
      if (error instanceof Error && "code" in error && error.code === "ENOENT") return [];
      throw error;
    }
  }

  async #upsert(record: SessionRecord): Promise<void> {
    const directory = this.projectDir(record.cwd);
    const indexPath = path.join(directory, "index.json");
    const temporaryPath = `${indexPath}.${randomUUID()}.tmp`;
    const records = (await this.#readIndex(record.cwd)).filter((item) => item.id !== record.id);
    records.push(record);
    await mkdir(directory, { recursive: true });
    await writeFile(temporaryPath, `${JSON.stringify(records, null, 2)}\n`, "utf8");
    await rename(temporaryPath, indexPath);
  }
}

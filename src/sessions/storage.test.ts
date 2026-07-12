import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { createMessageEntry } from "./entries.js";
import { JsonlSessionStorage, MemorySessionStorage } from "./storage.js";

describe("session storage", () => {
  let directory: string | undefined;

  afterEach(async () => {
    if (directory !== undefined) await rm(directory, { recursive: true, force: true });
  });

  it("stores entries in memory without sharing mutable state", async () => {
    const storage = new MemorySessionStorage();
    const entry = createMessageEntry({ role: "user", content: "hello" });
    await storage.append(entry);
    const entries = await storage.readAll();
    expect(entries).toEqual([entry]);
    entries.length = 0;
    expect(await storage.readAll()).toEqual([entry]);
  });

  it("appends and reads strict JSONL entries", async () => {
    directory = await mkdtemp(path.join(os.tmpdir(), "forge-session-"));
    const storage = new JsonlSessionStorage(path.join(directory, "nested", "one.jsonl"));
    const first = createMessageEntry({ role: "user", content: "hello" });
    const second = createMessageEntry({ role: "assistant", content: "hi", toolCalls: [] });
    await storage.append(first);
    await storage.append(second);
    expect(await storage.readAll()).toEqual([first, second]);
  });

  it("reports the invalid JSONL line number", async () => {
    directory = await mkdtemp(path.join(os.tmpdir(), "forge-session-"));
    const file = path.join(directory, "bad.jsonl");
    await writeFile(
      file,
      `${JSON.stringify(createMessageEntry({ role: "user", content: "ok" }))}\n{}\n`,
    );
    await expect(new JsonlSessionStorage(file).readAll()).rejects.toMatchObject({
      code: "INVALID_SESSION_JSONL",
      lineNumber: 2,
    });
  });

  it("treats a missing file as an empty session", async () => {
    directory = await mkdtemp(path.join(os.tmpdir(), "forge-session-"));
    expect(await new JsonlSessionStorage(path.join(directory, "missing.jsonl")).readAll()).toEqual(
      [],
    );
  });
});

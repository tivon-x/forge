import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { createMessageEntry, createModelChangeEntry, createSessionInfoEntry } from "./entries.js";
import { SessionManager } from "./manager.js";
import { replaySession } from "./state.js";

describe("linear session state", () => {
  it("replays messages and latest metadata in append order", () => {
    const state = replaySession([
      createSessionInfoEntry("one", "C:/project"),
      createModelChangeEntry("first"),
      createMessageEntry({ role: "user", content: "hello" }),
      createModelChangeEntry("second"),
      createMessageEntry({ role: "assistant", content: "hi", toolCalls: [] }),
    ]);
    expect(state).toMatchObject({
      sessionId: "one",
      cwd: "C:/project",
      model: "second",
      messages: [
        { role: "user", content: "hello" },
        { role: "assistant", content: "hi" },
      ],
    });
  });
});

describe("SessionManager", () => {
  let directory: string | undefined;
  afterEach(async () => {
    if (directory !== undefined) await rm(directory, { recursive: true, force: true });
  });

  it("creates, lists, gets, and touches project sessions", async () => {
    directory = await mkdtemp(path.join(os.tmpdir(), "forge-manager-"));
    const manager = new SessionManager({ sessionsDir: path.join(directory, "sessions") });
    const cwd = path.join(directory, "project");
    const record = await manager.create(cwd, "model-a");
    expect(await manager.get(cwd, record.id)).toEqual(record);
    const updated = await manager.touch(cwd, record.id, "model-b");
    expect(updated).toMatchObject({ id: record.id, model: "model-b" });
    expect(await manager.list(cwd)).toEqual([updated]);
  });

  it("keeps projects isolated", async () => {
    directory = await mkdtemp(path.join(os.tmpdir(), "forge-manager-"));
    const manager = new SessionManager({ sessionsDir: path.join(directory, "sessions") });
    const record = await manager.create(path.join(directory, "first"), "model");
    expect(await manager.get(path.join(directory, "second"), record.id)).toBeUndefined();
  });
});

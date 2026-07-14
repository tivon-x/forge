import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import type { ModelProvider, ProviderEvent } from "../agent/index.js";
import { SessionManager } from "../sessions/index.js";
import { CodingSession } from "./session.js";

class FinalProvider implements ModelProvider {
  async *stream(): AsyncIterable<ProviderEvent> {
    yield { type: "text_delta", delta: "done" };
    yield { type: "response_end" };
  }
}

describe("CodingSession", () => {
  let directory: string | undefined;

  afterEach(async () => {
    if (directory !== undefined) {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("persists a complete prompt run", async () => {
    directory = await mkdtemp(path.join(os.tmpdir(), "forge-coding-session-"));
    const manager = new SessionManager({ sessionsDir: path.join(directory, ".sessions") });
    const session = await CodingSession.open({
      cwd: directory,
      manager,
      model: "test-model",
      provider: new FinalProvider(),
      projectContext: { projectRoot: directory, files: [], diagnostics: [] },
      systemPrompt: "You are Forge.",
      tools: [],
    });

    expect(session.projectContext.projectRoot).toBe(directory);

    const events = [];
    const stream = session.run("hello");
    while (true) {
      const item = await stream.next();
      if (item.done) break;
      events.push(item.value);
    }

    expect(events.some((event) => event.type === "message_end")).toBe(true);
    const entries = (await readFile(session.record.path, "utf8"))
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line));
    expect(entries.map((entry) => entry.type)).toEqual([
      "session_info",
      "model_change",
      "message",
      "message",
    ]);
    expect(
      entries.filter((entry) => entry.type === "message").map((entry) => entry.message.role),
    ).toEqual(["user", "assistant"]);
  });
});

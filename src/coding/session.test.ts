import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import type { ModelProvider, ProviderEvent, ProviderRequest } from "../agent/index.js";
import { SessionManager } from "../sessions/index.js";
import { shellTool } from "../tools/index.js";
import { CodingSession, parseTerminalCommand } from "./session.js";

const terminalExecutor = (command: string, context: Parameters<typeof shellTool.execute>[1]) =>
  shellTool.execute({ command }, context);

class FinalProvider implements ModelProvider {
  readonly requests: ProviderRequest[] = [];

  async *stream(request: ProviderRequest): AsyncIterable<ProviderEvent> {
    this.requests.push(request);
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

  it("parses terminal command prefixes with !! taking precedence", () => {
    expect(parseTerminalCommand(" ! echo visible ")).toEqual({
      addToContext: true,
      command: "echo visible",
    });
    expect(parseTerminalCommand("!! echo hidden")).toEqual({
      addToContext: false,
      command: "echo hidden",
    });
    expect(parseTerminalCommand("hello")).toBeUndefined();
  });

  it("adds ! command results to the next provider request", async () => {
    directory = await mkdtemp(path.join(os.tmpdir(), "forge-coding-session-"));
    const manager = new SessionManager({ sessionsDir: path.join(directory, ".sessions") });
    const provider = new FinalProvider();
    const session = await CodingSession.open({
      cwd: directory,
      manager,
      model: "test-model",
      provider,
      projectContext: { projectRoot: directory, files: [], diagnostics: [] },
      systemPrompt: "You are Forge.",
      terminalExecutor,
      tools: [shellTool],
    });

    const terminalResult = await session.runTerminalCommand({
      addToContext: true,
      command: "node -e \"process.stdout.write('visible')\"",
    });
    const stream = session.run("continue");
    while (!(await stream.next()).done) {
      // Drain the deterministic provider response.
    }

    expect(terminalResult.result.ok).toBe(true);
    expect(provider.requests[0]?.messages[0]).toMatchObject({
      role: "user",
      content: expect.stringContaining("visible"),
    });
    expect(provider.requests[0]?.messages[0]).toMatchObject({
      content: expect.stringMatching(/^UNTRUSTED_TERMINAL_RESULT:/u),
    });
    expect(provider.requests[0]?.messages[1]).toEqual({ role: "user", content: "continue" });
  });

  it("keeps !! command results out of session context", async () => {
    directory = await mkdtemp(path.join(os.tmpdir(), "forge-coding-session-"));
    const manager = new SessionManager({ sessionsDir: path.join(directory, ".sessions") });
    const provider = new FinalProvider();
    const session = await CodingSession.open({
      cwd: directory,
      manager,
      model: "test-model",
      provider,
      projectContext: { projectRoot: directory, files: [], diagnostics: [] },
      systemPrompt: "You are Forge.",
      terminalExecutor,
      tools: [shellTool],
    });

    await session.runTerminalCommand({
      addToContext: false,
      command: "node -e \"process.stdout.write('hidden')\"",
    });
    const stream = session.run("continue");
    while (!(await stream.next()).done) {
      // Drain the deterministic provider response.
    }

    expect(provider.requests[0]?.messages).toEqual([{ role: "user", content: "continue" }]);
  });

  it("persists failed ! command diagnostics", async () => {
    directory = await mkdtemp(path.join(os.tmpdir(), "forge-coding-session-"));
    const manager = new SessionManager({ sessionsDir: path.join(directory, ".sessions") });
    const session = await CodingSession.open({
      cwd: directory,
      manager,
      model: "test-model",
      provider: new FinalProvider(),
      projectContext: { projectRoot: directory, files: [], diagnostics: [] },
      systemPrompt: "You are Forge.",
      terminalExecutor,
      tools: [shellTool],
    });

    const result = await session.runTerminalCommand({
      addToContext: true,
      command: "node -e \"process.stderr.write('bad');process.exit(3)\"",
    });
    const entries = await readFile(session.record.path, "utf8");

    expect(result.result).toMatchObject({ ok: false, error: { code: "SHELL_EXIT" } });
    expect(entries).toContain("SHELL_EXIT");
    expect(entries).toContain("bad");
  });
});

import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";
import { z } from "zod";

import { defineTool } from "../agent/index.js";
import { SessionManager } from "../sessions/index.js";
import { CommandRegistry, createDefaultCommandRegistry } from "./commands.js";

describe("CommandRegistry", () => {
  let directory: string | undefined;

  afterEach(async () => {
    if (directory !== undefined) await rm(directory, { recursive: true, force: true });
  });

  async function context() {
    directory = await mkdtemp(path.join(os.tmpdir(), "forge-commands-"));
    const manager = new SessionManager({ sessionsDir: path.join(directory, ".sessions") });
    return {
      cwd: directory,
      manager,
      projectContext: {
        projectRoot: directory,
        files: [{ path: path.join(directory, "AGENTS.md"), content: "rules" }],
        diagnostics: [],
      },
      tools: [
        defineTool({
          name: "demo",
          description: "Demo tool.",
          inputSchema: z.object({}),
          execute: async () => ({ ok: true, content: "" }),
        }),
      ],
    };
  }

  it("does not handle ordinary prompts and rejects unknown slash commands locally", async () => {
    const registry = createDefaultCommandRegistry();
    const commandContext = await context();

    await expect(registry.execute(commandContext, "hello")).resolves.toEqual({ handled: false });
    await expect(registry.execute(commandContext, "/model")).resolves.toEqual({
      handled: true,
      error: { code: "UNKNOWN_COMMAND", message: "Unknown command: /model" },
    });
  });

  it("generates help and local inspection output from live metadata", async () => {
    const registry = createDefaultCommandRegistry();
    const commandContext = await context();

    const help = await registry.execute(commandContext, "/help");
    const tools = await registry.execute(commandContext, "/tools");
    const projectContext = await registry.execute(commandContext, "/context");

    expect(help.message).toContain("/resume <id>");
    expect(help.message).not.toContain("/model");
    expect(tools.message).toContain("demo: Demo tool.");
    expect(projectContext.message).toContain(path.join(directory ?? "", "AGENTS.md"));
  });

  it("returns structured clear, resume, and quit actions", async () => {
    const registry = createDefaultCommandRegistry();
    const commandContext = await context();
    const record = await commandContext.manager.create(commandContext.cwd, "test-model");

    await expect(registry.execute(commandContext, "/clear")).resolves.toEqual({
      handled: true,
      action: { type: "clear" },
    });
    await expect(registry.execute(commandContext, `/resume ${record.id}`)).resolves.toEqual({
      handled: true,
      action: { type: "resume", sessionId: record.id },
    });
    await expect(registry.execute(commandContext, "/quit")).resolves.toEqual({
      handled: true,
      action: { type: "quit" },
    });
  });

  it("rejects duplicate command registration", () => {
    const registry = new CommandRegistry();
    const command = {
      name: "test",
      description: "test",
      usage: "/test",
      handler: async () => ({ handled: true }),
    };
    registry.register(command);

    expect(() => registry.register(command)).toThrow("Duplicate slash command: /test");
  });
});

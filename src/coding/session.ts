import {
  type AgentEvent,
  AgentHarness,
  type AgentRunResult,
  type ModelProvider,
  type ToolDefinition,
  type ToolResult,
} from "../agent/index.js";
import {
  createMessageEntry,
  createModelChangeEntry,
  createSessionInfoEntry,
  JsonlSessionStorage,
  replaySession,
  type SessionManager,
  type SessionRecord,
} from "../sessions/index.js";
import type { ProjectContext } from "./project-context.js";

export interface OpenCodingSessionOptions {
  cwd: string;
  manager: SessionManager;
  model: string;
  provider: ModelProvider;
  projectContext: ProjectContext;
  record?: SessionRecord;
  systemPrompt: string;
  tools: readonly ToolDefinition[];
}

export interface TerminalCommandRequest {
  addToContext: boolean;
  command: string;
}

export interface TerminalCommandResult {
  addedToContext: boolean;
  command: string;
  result: ToolResult;
}

export function parseTerminalCommand(text: string): TerminalCommandRequest | undefined {
  const trimmed = text.trim();
  if (trimmed.startsWith("!!")) {
    return { addToContext: false, command: trimmed.slice(2).trim() };
  }
  if (trimmed.startsWith("!")) {
    return { addToContext: true, command: trimmed.slice(1).trim() };
  }
  return undefined;
}

export class CodingSessionError extends Error {
  constructor(
    readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = "CodingSessionError";
  }
}

export class CodingSession {
  readonly #harness: AgentHarness;
  readonly #manager: SessionManager;
  readonly #model: string;
  readonly #projectContext: ProjectContext;
  readonly #record: SessionRecord;
  #running = false;
  readonly #storage: JsonlSessionStorage;
  readonly #tools: readonly ToolDefinition[];

  private constructor(options: {
    harness: AgentHarness;
    manager: SessionManager;
    model: string;
    projectContext: ProjectContext;
    record: SessionRecord;
    storage: JsonlSessionStorage;
    tools: readonly ToolDefinition[];
  }) {
    this.#harness = options.harness;
    this.#manager = options.manager;
    this.#model = options.model;
    this.#projectContext = options.projectContext;
    this.#record = options.record;
    this.#storage = options.storage;
    this.#tools = options.tools;
  }

  static async open(options: OpenCodingSessionOptions): Promise<CodingSession> {
    const record = options.record ?? (await options.manager.create(options.cwd, options.model));
    const storage = new JsonlSessionStorage(record.path);
    const entries = await storage.readAll();
    const state = replaySession(entries);

    if (state.cwd !== undefined && state.cwd !== record.cwd) {
      throw new CodingSessionError(
        "SESSION_CWD_MISMATCH",
        "session belongs to a different project",
      );
    }
    if (entries.length === 0) {
      await storage.append(createSessionInfoEntry(record.id, record.cwd));
      await storage.append(createModelChangeEntry(options.model));
    } else if (state.model !== options.model) {
      await storage.append(createModelChangeEntry(options.model));
    }

    return new CodingSession({
      harness: new AgentHarness(
        {
          provider: options.provider,
          tools: options.tools,
          systemPrompt: options.systemPrompt,
          cwd: options.cwd,
        },
        state.messages,
      ),
      manager: options.manager,
      model: options.model,
      projectContext: options.projectContext,
      record,
      storage,
      tools: options.tools,
    });
  }

  get record(): SessionRecord {
    return this.#record;
  }

  get projectContext(): ProjectContext {
    return this.#projectContext;
  }

  get tools(): readonly ToolDefinition[] {
    return this.#tools;
  }

  async *run(
    prompt: string,
    signal?: AbortSignal,
  ): AsyncGenerator<AgentEvent, AgentRunResult, undefined> {
    if (this.#running) {
      throw new CodingSessionError("SESSION_BUSY", "coding session is already running");
    }
    this.#running = true;
    try {
      await this.#storage.append(createMessageEntry({ role: "user", content: prompt }));
      const stream = this.#harness.run(prompt, signal);

      while (true) {
        const item = await stream.next();
        if (item.done) {
          await this.#manager.touch(this.#record.cwd, this.#record.id, this.#model);
          return item.value;
        }
        if (item.value.type === "message_end" || item.value.type === "tool_end") {
          await this.#storage.append(createMessageEntry(item.value.message));
        }
        yield item.value;
      }
    } finally {
      this.#running = false;
    }
  }

  async runTerminalCommand(
    request: TerminalCommandRequest,
    signal?: AbortSignal,
  ): Promise<TerminalCommandResult> {
    if (request.command.length === 0) {
      throw new CodingSessionError("COMMAND_USAGE", "Usage: !<command> or !!<command>");
    }
    if (this.#running) {
      throw new CodingSessionError("SESSION_BUSY", "coding session is already running");
    }
    const shell = this.#tools.find((tool) => tool.name === "shell");
    if (shell === undefined) {
      throw new CodingSessionError("SHELL_TOOL_UNAVAILABLE", "shell tool is not enabled");
    }

    this.#running = true;
    try {
      const result = await shell.execute(
        { command: request.command },
        {
          cwd: this.#record.cwd,
          signal: signal ?? new AbortController().signal,
        },
      );
      if (request.addToContext) {
        const message = {
          role: "user" as const,
          content: [
            "Terminal command executed by the user.",
            JSON.stringify(
              {
                command: request.command,
                ok: result.ok,
                output: result.content,
                error: result.error ?? null,
              },
              null,
              2,
            ),
          ].join("\n\n"),
        };
        await this.#storage.append(createMessageEntry(message));
        this.#harness.appendMessage(message);
        await this.#manager.touch(this.#record.cwd, this.#record.id, this.#model);
      }
      return {
        addedToContext: request.addToContext,
        command: request.command,
        result,
      };
    } finally {
      this.#running = false;
    }
  }
}

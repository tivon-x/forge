import {
  type AgentEvent,
  AgentHarness,
  type AgentRunResult,
  type ModelProvider,
  type ToolDefinition,
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
  readonly #storage: JsonlSessionStorage;

  private constructor(options: {
    harness: AgentHarness;
    manager: SessionManager;
    model: string;
    projectContext: ProjectContext;
    record: SessionRecord;
    storage: JsonlSessionStorage;
  }) {
    this.#harness = options.harness;
    this.#manager = options.manager;
    this.#model = options.model;
    this.#projectContext = options.projectContext;
    this.#record = options.record;
    this.#storage = options.storage;
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
    });
  }

  get record(): SessionRecord {
    return this.#record;
  }

  get projectContext(): ProjectContext {
    return this.#projectContext;
  }

  async *run(
    prompt: string,
    signal?: AbortSignal,
  ): AsyncGenerator<AgentEvent, AgentRunResult, undefined> {
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
  }
}

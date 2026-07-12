import type { Writable } from "node:stream";

import { Command, CommanderError } from "commander";

import { AgentHarness, type AgentRunResult, type ModelProvider } from "../agent/index.js";
import { OpenAIResponsesProvider } from "../providers/index.js";
import {
  createMessageEntry,
  createModelChangeEntry,
  createSessionInfoEntry,
  JsonlSessionStorage,
  replaySession,
  SessionManager,
  type SessionRecord,
} from "../sessions/index.js";
import { CODING_TOOLS } from "../tools/index.js";
import { VERSION } from "../version.js";
import { buildSystemPrompt } from "./system-prompt.js";
import { TextRenderer } from "./text-renderer.js";

interface CliOptions {
  prompt?: string;
  model?: string;
  resume?: string;
}

interface ProviderConfig {
  apiKey: string;
  model: string;
}

export interface CliDependencies {
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  stdout?: Writable;
  stderr?: Writable;
  signal?: AbortSignal;
  sessionsDir?: string;
  providerFactory?: (config: ProviderConfig) => ModelProvider;
}

function createProgram(stdout: Writable, stderr: Writable): Command {
  return new Command()
    .name("forge")
    .description("Run Forge as a one-shot coding agent")
    .version(VERSION)
    .argument("[command]", "command to run (sessions)")
    .option("-p, --prompt <prompt>", "task for Forge")
    .option("-m, --model <model>", "OpenAI model (or set OPENAI_MODEL)")
    .option("--resume <id>", "resume a session from the current project")
    .allowExcessArguments(false)
    .configureOutput({
      writeOut: (text) => stdout.write(text),
      writeErr: (text) => stderr.write(text),
    })
    .exitOverride();
}

async function consume(
  harness: AgentHarness,
  prompt: string,
  renderer: TextRenderer,
  signal: AbortSignal | undefined,
  storage: JsonlSessionStorage,
): Promise<AgentRunResult> {
  const stream = harness.run(prompt, signal);
  while (true) {
    const item = await stream.next();
    if (item.done) {
      return item.value;
    }
    if (item.value.type === "message_end") {
      await storage.append(createMessageEntry(item.value.message));
    } else if (item.value.type === "tool_end") {
      await storage.append(createMessageEntry(item.value.message));
    }
    renderer.render(item.value);
  }
}

export async function main(
  argv: readonly string[] = process.argv,
  dependencies: CliDependencies = {},
) {
  const stdout = dependencies.stdout ?? process.stdout;
  const stderr = dependencies.stderr ?? process.stderr;
  const env = dependencies.env ?? process.env;
  const cwd = dependencies.cwd ?? process.cwd();
  const program = createProgram(stdout, stderr);

  try {
    program.parse([...argv]);
  } catch (error) {
    if (error instanceof CommanderError) {
      return error.code === "commander.helpDisplayed" || error.code === "commander.version" ? 0 : 1;
    }
    throw error;
  }

  const options = program.opts<CliOptions>();
  const [command] = program.args;
  const manager = new SessionManager(
    dependencies.sessionsDir === undefined ? {} : { sessionsDir: dependencies.sessionsDir },
  );

  if (command !== undefined && command !== "sessions") {
    stderr.write(`Error: unknown command '${command}'\n`);
    return 1;
  }
  if (command === "sessions") {
    if (options.prompt !== undefined || options.resume !== undefined) {
      stderr.write("Error: sessions does not accept --prompt or --resume\n");
      return 1;
    }
    try {
      for (const session of await manager.list(cwd)) {
        stdout.write(`${session.id}\t${session.updatedAt}\t${session.model}\n`);
      }
      return 0;
    } catch (error) {
      stderr.write(`Error [SESSION_STORAGE_ERROR]: ${errorMessage(error)}\n`);
      return 1;
    }
  }

  const prompt = options.prompt?.trim() ?? "";
  const apiKey = env.OPENAI_API_KEY;

  if (prompt.length === 0) {
    stderr.write("Error: provide a non-empty --prompt\n");
    return 1;
  }
  let record: SessionRecord | undefined;
  try {
    if (options.resume !== undefined) {
      record = await manager.get(cwd, options.resume);
      if (record === undefined) {
        stderr.write(`Error [SESSION_NOT_FOUND]: session '${options.resume}' was not found\n`);
        return 1;
      }
    }
  } catch (error) {
    stderr.write(`Error [SESSION_STORAGE_ERROR]: ${errorMessage(error)}\n`);
    return 1;
  }
  const model = options.model ?? env.OPENAI_MODEL ?? record?.model;
  if (model === undefined || model.length === 0) {
    stderr.write("Error: provide --model or set OPENAI_MODEL\n");
    return 1;
  }
  if (apiKey === undefined || apiKey.length === 0) {
    stderr.write("Error: OPENAI_API_KEY is not set\n");
    return 1;
  }

  const providerFactory =
    dependencies.providerFactory ??
    ((config: ProviderConfig) => new OpenAIResponsesProvider(config));
  const provider = providerFactory({ apiKey, model });
  try {
    record ??= await manager.create(cwd, model);
    const storage = new JsonlSessionStorage(record.path);
    const entries = await storage.readAll();
    const state = replaySession(entries);
    if (state.cwd !== undefined && state.cwd !== record.cwd) {
      stderr.write("Error [SESSION_CWD_MISMATCH]: session belongs to a different project\n");
      return 1;
    }
    if (entries.length === 0) {
      await storage.append(createSessionInfoEntry(record.id, record.cwd));
      await storage.append(createModelChangeEntry(model));
    } else if (state.model !== model) {
      await storage.append(createModelChangeEntry(model));
    }
    await storage.append(createMessageEntry({ role: "user", content: prompt }));

    const harness = new AgentHarness(
      {
        provider,
        tools: CODING_TOOLS,
        systemPrompt: buildSystemPrompt({ cwd, tools: CODING_TOOLS }),
        cwd,
      },
      state.messages,
    );
    const renderer = new TextRenderer({ stdout, stderr });
    const result = await consume(harness, prompt, renderer, dependencies.signal, storage);
    renderer.finish(result.reason);
    await manager.touch(cwd, record.id, model);

    if (result.reason === "completed") {
      return 0;
    }
    if (result.reason === "cancelled") {
      return 130;
    }
    return 1;
  } catch (error) {
    stderr.write(`Error [SESSION_STORAGE_ERROR]: ${errorMessage(error)}\n`);
    return 1;
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

import type { Writable } from "node:stream";

import { Command, CommanderError } from "commander";

import type { AgentRunResult, ModelProvider } from "../agent/index.js";
import {
  buildSystemPrompt,
  CodingSession,
  CodingSessionError,
  discoverProjectContext,
} from "../coding/index.js";
import { OpenAICompatibleProvider, OpenAIResponsesProvider } from "../providers/index.js";
import { SessionManager, type SessionRecord } from "../sessions/index.js";
import { CODING_TOOLS } from "../tools/index.js";
import { VERSION } from "../version.js";
import type { EventRenderer } from "./event-renderer.js";
import { FinalTextRenderer } from "./final-text-renderer.js";
import { JsonEventRenderer } from "./json-renderer.js";
import { TextRenderer } from "./text-renderer.js";

interface CliOptions {
  baseUrl?: string;
  provider?: string;
  prompt?: string;
  model?: string;
  resume?: string;
  output?: string;
}

interface ProviderConfig {
  apiKey: string;
  baseURL?: string;
  model: string;
  provider: "openai" | "openai-compatible";
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
    .option("--provider <provider>", "provider: openai or openai-compatible", "openai")
    .option(
      "--base-url <url>",
      "OpenAI-compatible API base URL (or set OPENAI_COMPATIBLE_BASE_URL)",
    )
    .option("-m, --model <model>", "model (or set the provider model environment variable)")
    .option("--resume <id>", "resume a session from the current project")
    .option("--output <mode>", "output mode: text, json, or transcript", "text")
    .allowExcessArguments(false)
    .configureOutput({
      writeOut: (text) => stdout.write(text),
      writeErr: (text) => stderr.write(text),
    })
    .exitOverride();
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
  const output = options.output ?? "text";
  if (output !== "text" && output !== "json" && output !== "transcript") {
    stderr.write(`Error: invalid output mode '${output}'\n`);
    return 1;
  }
  const providerName = options.provider ?? "openai";
  if (providerName !== "openai" && providerName !== "openai-compatible") {
    stderr.write("Error: provider must be openai or openai-compatible\n");
    return 1;
  }
  const provider = providerName;
  const apiKey = provider === "openai" ? env.OPENAI_API_KEY : env.OPENAI_COMPATIBLE_API_KEY;

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
  const model =
    options.model ??
    (provider === "openai" ? env.OPENAI_MODEL : env.OPENAI_COMPATIBLE_MODEL) ??
    record?.model;
  if (model === undefined || model.length === 0) {
    stderr.write(
      `Error: provide --model or set ${provider === "openai" ? "OPENAI_MODEL" : "OPENAI_COMPATIBLE_MODEL"}\n`,
    );
    return 1;
  }
  if (apiKey === undefined || apiKey.length === 0) {
    stderr.write(
      `Error: ${provider === "openai" ? "OPENAI_API_KEY" : "OPENAI_COMPATIBLE_API_KEY"} is not set\n`,
    );
    return 1;
  }
  const baseURL = options.baseUrl ?? env.OPENAI_COMPATIBLE_BASE_URL;
  if (provider === "openai-compatible" && (baseURL === undefined || baseURL.length === 0)) {
    stderr.write("Error: provide --base-url or set OPENAI_COMPATIBLE_BASE_URL\n");
    return 1;
  }

  const providerFactory =
    dependencies.providerFactory ??
    ((config: ProviderConfig) =>
      config.provider === "openai"
        ? new OpenAIResponsesProvider(config)
        : new OpenAICompatibleProvider({
            apiKey: config.apiKey,
            model: config.model,
            baseURL: config.baseURL ?? "",
          }));
  let modelProvider: ModelProvider;
  try {
    modelProvider = providerFactory({
      apiKey,
      model,
      provider,
      ...(baseURL === undefined ? {} : { baseURL }),
    });
  } catch (error) {
    stderr.write(`Error [PROVIDER_CONFIG_ERROR]: ${errorMessage(error)}\n`);
    return 1;
  }
  try {
    const projectContext = await discoverProjectContext(cwd);
    for (const diagnostic of projectContext.diagnostics) {
      stderr.write(`Warning [${diagnostic.code}]: ${diagnostic.path}: ${diagnostic.message}\n`);
    }
    const session = await CodingSession.open({
      cwd,
      manager,
      model,
      provider: modelProvider,
      projectContext,
      ...(record === undefined ? {} : { record }),
      systemPrompt: buildSystemPrompt({
        contextFiles: projectContext.files,
        cwd,
        tools: CODING_TOOLS,
      }),
      tools: CODING_TOOLS,
    });
    const renderer: EventRenderer =
      output === "json"
        ? new JsonEventRenderer({ stdout })
        : output === "transcript"
          ? new TextRenderer({ stdout, stderr })
          : new FinalTextRenderer({ stdout, stderr });
    const stream = session.run(prompt, dependencies.signal);
    let result: AgentRunResult;
    while (true) {
      const item = await stream.next();
      if (item.done) {
        result = item.value;
        break;
      }
      renderer.render(item.value);
    }
    renderer.finish(result.reason);

    if (result.reason === "completed") {
      return 0;
    }
    if (result.reason === "cancelled") {
      return 130;
    }
    return 1;
  } catch (error) {
    if (error instanceof CodingSessionError) {
      stderr.write(`Error [${error.code}]: ${error.message}\n`);
      return 1;
    }
    stderr.write(`Error [SESSION_STORAGE_ERROR]: ${errorMessage(error)}\n`);
    return 1;
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

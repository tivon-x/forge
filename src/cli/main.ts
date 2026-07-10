import type { Writable } from "node:stream";

import { Command, CommanderError } from "commander";

import { AgentHarness, type AgentRunResult, type ModelProvider } from "../agent/index.js";
import { OpenAIResponsesProvider } from "../providers/index.js";
import { CODING_TOOLS } from "../tools/index.js";
import { VERSION } from "../version.js";
import { buildSystemPrompt } from "./system-prompt.js";
import { TextRenderer } from "./text-renderer.js";

interface CliOptions {
  prompt: string;
  model?: string;
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
  providerFactory?: (config: ProviderConfig) => ModelProvider;
}

function createProgram(stdout: Writable, stderr: Writable): Command {
  return new Command()
    .name("forge")
    .description("Run Forge as a one-shot coding agent")
    .version(VERSION)
    .requiredOption("-p, --prompt <prompt>", "task for Forge")
    .option("-m, --model <model>", "OpenAI model (or set OPENAI_MODEL)")
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
): Promise<AgentRunResult> {
  const stream = harness.run(prompt, signal);
  while (true) {
    const item = await stream.next();
    if (item.done) {
      return item.value;
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
  const prompt = options.prompt.trim();
  const model = options.model ?? env.OPENAI_MODEL;
  const apiKey = env.OPENAI_API_KEY;

  if (prompt.length === 0) {
    stderr.write("Error: --prompt must not be empty\n");
    return 1;
  }
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
  const harness = new AgentHarness({
    provider,
    tools: CODING_TOOLS,
    systemPrompt: buildSystemPrompt({ cwd, tools: CODING_TOOLS }),
    cwd,
  });
  const renderer = new TextRenderer({ stdout, stderr });
  const result = await consume(harness, prompt, renderer, dependencies.signal);
  renderer.finish(result.reason);

  if (result.reason === "completed") {
    return 0;
  }
  if (result.reason === "cancelled") {
    return 130;
  }
  return 1;
}

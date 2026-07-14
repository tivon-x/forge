import { createInterface } from "node:readline/promises";
import type { Readable, Writable } from "node:stream";

import type { AgentRunResult } from "../agent/index.js";
import type { CodingSession, CommandContext, CommandRegistry } from "../coding/index.js";
import type { SessionRecord } from "../sessions/index.js";
import { TextRenderer } from "./text-renderer.js";

export interface InteractiveSessionOptions {
  commandContext: CommandContext;
  input: Readable;
  openSession(record?: SessionRecord): Promise<CodingSession>;
  registry: CommandRegistry;
  session: CodingSession;
  signal?: AbortSignal;
  stderr: Writable;
  stdout: Writable;
}

function isTerminal(stream: Readable | Writable): boolean {
  return "isTTY" in stream && stream.isTTY === true;
}

function writeLine(output: Writable, message: string): void {
  output.write(message.endsWith("\n") ? message : `${message}\n`);
}

async function runPrompt(
  session: CodingSession,
  prompt: string,
  stdout: Writable,
  stderr: Writable,
  signal?: AbortSignal,
): Promise<AgentRunResult> {
  const renderer = new TextRenderer({ stdout, stderr });
  const stream = session.run(prompt, signal);
  while (true) {
    const item = await stream.next();
    if (item.done) {
      renderer.finish(item.value.reason);
      return item.value;
    }
    renderer.render(item.value);
  }
}

export async function runInteractiveSession(options: InteractiveSessionOptions): Promise<number> {
  const terminal = isTerminal(options.input) && isTerminal(options.stdout);
  const readline = createInterface({
    input: options.input,
    output: options.stdout,
    terminal,
    crlfDelay: Number.POSITIVE_INFINITY,
  });
  let session = options.session;
  const closeOnAbort = () => readline.close();
  if (options.signal?.aborted) {
    readline.close();
    return 130;
  }
  options.signal?.addEventListener("abort", closeOnAbort, { once: true });

  try {
    writeLine(options.stdout, `Session: ${session.record.id}`);
    options.stdout.write("> ");

    for await (const line of readline) {
      const input = line.trim();
      if (input.length === 0) {
        options.stdout.write("> ");
        continue;
      }

      const command = await options.registry.execute(options.commandContext, input);
      if (command.handled) {
        if (command.error !== undefined) {
          writeLine(options.stderr, `Error [${command.error.code}]: ${command.error.message}`);
        }
        if (command.message !== undefined) writeLine(options.stdout, command.message);
        if (command.action?.type === "quit") return 0;
        if (command.action?.type === "clear") {
          session = await options.openSession();
          writeLine(options.stdout, `Started session: ${session.record.id}`);
        }
        if (command.action?.type === "resume") {
          const record = await options.commandContext.manager.get(
            options.commandContext.cwd,
            command.action.sessionId,
          );
          if (record === undefined) {
            writeLine(options.stdout, `Unknown session: ${command.action.sessionId}`);
          } else {
            session = await options.openSession(record);
            writeLine(options.stdout, `Resumed session: ${record.id}`);
          }
        }
        options.stdout.write("> ");
        continue;
      }

      const result = await runPrompt(
        session,
        input,
        options.stdout,
        options.stderr,
        options.signal,
      );
      if (result.reason === "cancelled") return 130;
      options.stdout.write("> ");
    }
    return options.signal?.aborted ? 130 : 0;
  } finally {
    options.signal?.removeEventListener("abort", closeOnAbort);
    readline.close();
  }
}

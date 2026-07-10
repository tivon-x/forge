import { spawn } from "node:child_process";

import { z } from "zod";

import { defineTool, type ToolResult } from "../agent/index.js";
import { toolFailure } from "./errors.js";
import { resolveWorkspacePath } from "./workspace-path.js";

interface CapturedOutput {
  chunks: Buffer[];
  bytes: number;
  truncated: boolean;
}

function capture(output: CapturedOutput, chunk: Buffer, limit: number): void {
  const remaining = limit - output.bytes;
  if (remaining <= 0) {
    output.truncated = true;
    return;
  }
  const kept = chunk.subarray(0, remaining);
  output.chunks.push(kept);
  output.bytes += kept.length;
  output.truncated ||= kept.length !== chunk.length;
}

function outputText(output: CapturedOutput): string {
  const text = Buffer.concat(output.chunks).toString("utf8");
  return output.truncated ? `${text}\n[output truncated]` : text;
}

function renderResult(exitCode: number | null, stdout: string, stderr: string): string {
  const sections = [`Exit code: ${exitCode ?? "unknown"}`];
  if (stdout.length > 0) {
    sections.push(`Stdout:\n${stdout}`);
  }
  if (stderr.length > 0) {
    sections.push(`Stderr:\n${stderr}`);
  }
  return sections.join("\n");
}

export const shellTool = defineTool({
  name: "shell",
  description: "Run a shell command with the workspace as its working directory.",
  inputSchema: z.object({
    command: z.string().min(1),
    timeoutMs: z.int().min(1).max(600_000).default(120_000),
    maxOutputBytes: z.int().min(1).max(1_000_000).default(100_000),
  }),
  promptGuidelines: [
    "Use shell for builds, tests, and repository inspection.",
    "Do not run destructive commands unless the user explicitly requested them.",
  ],
  execute: async ({ command, timeoutMs, maxOutputBytes }, context): Promise<ToolResult> => {
    context.signal.throwIfAborted();
    const cwd = await resolveWorkspacePath(context.cwd, ".", true);

    try {
      return await new Promise<ToolResult>((resolve, reject) => {
        const stdout: CapturedOutput = { chunks: [], bytes: 0, truncated: false };
        const stderr: CapturedOutput = { chunks: [], bytes: 0, truncated: false };
        let timedOut = false;
        let settled = false;

        const child = spawn(command, {
          cwd,
          shell: true,
          windowsHide: true,
          signal: context.signal,
        });
        const timer = setTimeout(() => {
          timedOut = true;
          child.kill();
        }, timeoutMs);

        child.stdout.on("data", (chunk: Buffer) => capture(stdout, chunk, maxOutputBytes));
        child.stderr.on("data", (chunk: Buffer) => capture(stderr, chunk, maxOutputBytes));
        child.on("error", (error) => {
          if (settled) {
            return;
          }
          settled = true;
          clearTimeout(timer);
          if (context.signal.aborted) {
            reject(error);
          } else {
            resolve(toolFailure(error, "SHELL_SPAWN_ERROR"));
          }
        });
        child.on("close", (exitCode, signal) => {
          if (settled) {
            return;
          }
          settled = true;
          clearTimeout(timer);

          const stdoutText = outputText(stdout);
          const stderrText = outputText(stderr);
          const content = renderResult(exitCode, stdoutText, stderrText);
          const data = {
            exitCode,
            signal,
            stdout: stdoutText,
            stderr: stderrText,
            timedOut,
            truncated: stdout.truncated || stderr.truncated,
          };

          if (timedOut) {
            resolve({
              ok: false,
              content,
              data,
              error: { code: "SHELL_TIMEOUT", message: `Command timed out after ${timeoutMs}ms` },
            });
          } else if (exitCode !== 0) {
            resolve({
              ok: false,
              content,
              data,
              error: { code: "SHELL_EXIT", message: `Command exited with code ${exitCode}` },
            });
          } else {
            resolve({ ok: true, content, data });
          }
        });
      });
    } catch (error) {
      if (context.signal.aborted) {
        throw error;
      }
      return toolFailure(error, "SHELL_ERROR");
    }
  },
});

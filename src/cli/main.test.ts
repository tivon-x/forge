import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { PassThrough } from "node:stream";

import { afterEach, describe, expect, it } from "vitest";

import type { ModelProvider, ProviderEvent, ProviderRequest } from "../agent/index.js";
import { main } from "./main.js";

function outputStream(): { stream: PassThrough; text: () => string } {
  const chunks: Buffer[] = [];
  const stream = new PassThrough();
  stream.on("data", (chunk: Buffer) => chunks.push(chunk));
  return { stream, text: () => Buffer.concat(chunks).toString("utf8") };
}

class AcceptanceProvider implements ModelProvider {
  readonly requests: ProviderRequest[] = [];
  #turn = 0;

  async *stream(request: ProviderRequest): AsyncIterable<ProviderEvent> {
    this.requests.push(request);
    this.#turn += 1;

    if (this.#turn === 1) {
      yield {
        type: "tool_call",
        call: { id: "read-1", name: "readFile", arguments: { path: "input.txt" } },
      };
    } else if (this.#turn === 2) {
      yield {
        type: "tool_call",
        call: {
          id: "edit-1",
          name: "editFile",
          arguments: { path: "input.txt", oldText: "alpha", newText: "beta" },
        },
      };
    } else if (this.#turn === 3) {
      yield {
        type: "tool_call",
        call: {
          id: "shell-1",
          name: "shell",
          arguments: {
            command:
              "node -e \"const fs=require('fs');if(fs.readFileSync('input.txt','utf8')!=='beta')process.exit(1)\"",
          },
        },
      };
    } else {
      yield { type: "text_delta", delta: "Updated input.txt and verified it." };
    }
    yield { type: "response_end" };
  }
}

describe("one-shot CLI", () => {
  let cwd: string | undefined;

  afterEach(async () => {
    if (cwd !== undefined) {
      await rm(cwd, { recursive: true, force: true });
    }
  });

  it("runs a complete read, edit, shell, and final answer loop", async () => {
    cwd = await mkdtemp(path.join(os.tmpdir(), "forge-cli-"));
    await writeFile(path.join(cwd, "input.txt"), "alpha", "utf8");
    const stdout = outputStream();
    const stderr = outputStream();
    const provider = new AcceptanceProvider();

    const exitCode = await main(["node", "forge", "-p", "update the file"], {
      cwd,
      env: { OPENAI_API_KEY: "test", OPENAI_MODEL: "test" },
      stdout: stdout.stream,
      stderr: stderr.stream,
      providerFactory: () => provider,
    });

    expect(exitCode).toBe(0);
    expect(await readFile(path.join(cwd, "input.txt"), "utf8")).toBe("beta");
    expect(stdout.text()).toBe("Updated input.txt and verified it.\n");
    expect(stderr.text()).toContain("[tool] readFile: ok");
    expect(stderr.text()).toContain("[tool] editFile: ok");
    expect(stderr.text()).toContain("[tool] shell: ok");
    expect(provider.requests).toHaveLength(4);
    expect(provider.requests[1]?.messages.at(-1)).toMatchObject({
      role: "tool",
      toolName: "readFile",
      content: "alpha",
    });
  });

  it("fails before creating a provider when credentials are missing", async () => {
    const stderr = outputStream();

    const exitCode = await main(["node", "forge", "-p", "test", "--model", "test"], {
      env: {},
      stderr: stderr.stream,
    });

    expect(exitCode).toBe(1);
    expect(stderr.text()).toContain("OPENAI_API_KEY is not set");
  });

  it("requires a model from the CLI or environment", async () => {
    const stderr = outputStream();

    const exitCode = await main(["node", "forge", "-p", "test"], {
      env: { OPENAI_API_KEY: "test" },
      stderr: stderr.stream,
    });

    expect(exitCode).toBe(1);
    expect(stderr.text()).toContain("provide --model or set OPENAI_MODEL");
  });

  it("returns a non-zero exit code for provider failures", async () => {
    const stderr = outputStream();
    const provider: ModelProvider = {
      async *stream() {
        yield* [];
        throw new Error("provider unavailable");
      },
    };

    const exitCode = await main(["node", "forge", "-p", "test", "--model", "test"], {
      env: { OPENAI_API_KEY: "test" },
      stderr: stderr.stream,
      providerFactory: () => provider,
    });

    expect(exitCode).toBe(1);
    expect(stderr.text()).toContain("Error [PROVIDER_ERROR]: provider unavailable");
  });

  it("returns exit code 130 when cancelled", async () => {
    const stderr = outputStream();
    const controller = new AbortController();
    controller.abort();
    const provider = new AcceptanceProvider();

    const exitCode = await main(["node", "forge", "-p", "test", "--model", "test"], {
      env: { OPENAI_API_KEY: "test" },
      stderr: stderr.stream,
      signal: controller.signal,
      providerFactory: () => provider,
    });

    expect(exitCode).toBe(130);
    expect(stderr.text()).toContain("Error [CANCELLED]");
    expect(provider.requests).toHaveLength(0);
  });
});

import { mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
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

class FinalProvider implements ModelProvider {
  readonly requests: ProviderRequest[] = [];

  async *stream(request: ProviderRequest): AsyncIterable<ProviderEvent> {
    this.requests.push(request);
    yield { type: "text_delta", delta: "done" };
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

    const exitCode = await main(
      ["node", "forge", "-p", "update the file", "--output", "transcript"],
      {
        cwd,
        env: { OPENAI_API_KEY: "test", OPENAI_MODEL: "test" },
        stdout: stdout.stream,
        stderr: stderr.stream,
        providerFactory: () => provider,
        sessionsDir: path.join(cwd, ".sessions"),
      },
    );

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

  it("handles slash commands locally without provider credentials", async () => {
    cwd = await mkdtemp(path.join(os.tmpdir(), "forge-cli-"));
    const stdout = outputStream();
    let providerCreated = false;

    const exitCode = await main(["node", "forge", "-p", "/help"], {
      cwd,
      env: {},
      stdout: stdout.stream,
      providerFactory: () => {
        providerCreated = true;
        return new FinalProvider();
      },
      sessionsDir: path.join(cwd, ".sessions"),
    });

    expect(exitCode).toBe(0);
    expect(stdout.text()).toContain("/sessions");
    expect(stdout.text()).not.toContain("/model");
    expect(providerCreated).toBe(false);
  });

  it("rejects json output for local slash commands without polluting stdout", async () => {
    cwd = await mkdtemp(path.join(os.tmpdir(), "forge-cli-"));
    const stdout = outputStream();
    const stderr = outputStream();

    const exitCode = await main(["node", "forge", "-p", "/help", "--output", "json"], {
      cwd,
      env: {},
      sessionsDir: path.join(cwd, ".sessions"),
      stderr: stderr.stream,
      stdout: stdout.stream,
    });

    expect(exitCode).toBe(1);
    expect(stdout.text()).toBe("");
    expect(stderr.text()).toContain("Error [COMMAND_OUTPUT_MODE]");
  });

  it("runs one-shot !! commands without credentials, provider, or session files", async () => {
    cwd = await mkdtemp(path.join(os.tmpdir(), "forge-cli-"));
    const sessionsDir = path.join(cwd, ".sessions");
    const stdout = outputStream();
    let providerCreated = false;

    const exitCode = await main(
      ["node", "forge", "-p", "!!node -e \"process.stdout.write('local-only')\""],
      {
        cwd,
        env: {},
        providerFactory: () => {
          providerCreated = true;
          return new FinalProvider();
        },
        sessionsDir,
        stdout: stdout.stream,
      },
    );

    expect(exitCode).toBe(0);
    expect(stdout.text()).toContain("local-only");
    expect(providerCreated).toBe(false);
    await expect(readdir(sessionsDir)).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("selects the OpenAI-compatible provider from its environment settings", async () => {
    const provider = new FinalProvider();
    let config: unknown;

    const exitCode = await main(
      ["node", "forge", "--provider", "openai-compatible", "-p", "test"],
      {
        env: {
          OPENAI_COMPATIBLE_API_KEY: "test",
          OPENAI_COMPATIBLE_MODEL: "compatible-model",
          OPENAI_COMPATIBLE_BASE_URL: "https://example.test/v1",
        },
        providerFactory: (value) => {
          config = value;
          return provider;
        },
        sessionsDir: path.join(os.tmpdir(), "forge-test-sessions"),
      },
    );

    expect(exitCode).toBe(0);
    expect(config).toEqual({
      apiKey: "test",
      baseURL: "https://example.test/v1",
      model: "compatible-model",
      provider: "openai-compatible",
    });
  });

  it("includes discovered project instructions in the provider system prompt", async () => {
    cwd = await mkdtemp(path.join(os.tmpdir(), "forge-cli-"));
    await writeFile(path.join(cwd, "AGENTS.md"), "Use the project rules.", "utf8");
    const provider = new FinalProvider();

    expect(
      await main(["node", "forge", "-p", "test"], {
        cwd,
        env: { OPENAI_API_KEY: "test", OPENAI_MODEL: "test" },
        providerFactory: () => provider,
        sessionsDir: path.join(cwd, ".sessions"),
      }),
    ).toBe(0);

    expect(provider.requests[0]?.systemPrompt).toContain("Use the project rules.");
    expect(provider.requests[0]?.systemPrompt).toContain(path.join(cwd, "AGENTS.md"));
  });

  it("returns a stable error for an invalid OpenAI-compatible base URL", async () => {
    const stderr = outputStream();

    const exitCode = await main(
      ["node", "forge", "--provider", "openai-compatible", "--base-url", "not-a-url", "-p", "test"],
      {
        env: { OPENAI_COMPATIBLE_API_KEY: "test", OPENAI_COMPATIBLE_MODEL: "compatible-model" },
        stderr: stderr.stream,
      },
    );

    expect(exitCode).toBe(1);
    expect(stderr.text()).toContain("Error [PROVIDER_CONFIG_ERROR]");
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
      sessionsDir: path.join(os.tmpdir(), "forge-test-sessions"),
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
      sessionsDir: path.join(os.tmpdir(), "forge-test-sessions"),
    });

    expect(exitCode).toBe(130);
    expect(stderr.text()).toContain("Error [CANCELLED]");
    expect(provider.requests).toHaveLength(0);
  });

  it("lists and resumes a project session with its message history", async () => {
    cwd = await mkdtemp(path.join(os.tmpdir(), "forge-cli-"));
    const sessionsDir = path.join(cwd, ".sessions");
    const firstProvider = new FinalProvider();
    const firstCode = await main(["node", "forge", "-p", "first"], {
      cwd,
      env: { OPENAI_API_KEY: "test", OPENAI_MODEL: "test" },
      sessionsDir,
      providerFactory: () => firstProvider,
    });
    expect(firstCode).toBe(0);

    const listOutput = outputStream();
    expect(
      await main(["node", "forge", "sessions"], {
        cwd,
        sessionsDir,
        stdout: listOutput.stream,
      }),
    ).toBe(0);
    const sessionId = listOutput.text().split("\t")[0];
    expect(sessionId).toBeTruthy();

    const resumedProvider = new FinalProvider();
    expect(
      await main(["node", "forge", "-p", "second", "--resume", sessionId ?? ""], {
        cwd,
        env: { OPENAI_API_KEY: "test" },
        sessionsDir,
        providerFactory: () => resumedProvider,
      }),
    ).toBe(0);
    expect(resumedProvider.requests[0]?.messages).toMatchObject([
      { role: "user", content: "first" },
      { role: "assistant", content: "done" },
      { role: "user", content: "second" },
    ]);
  });

  it("persists complete user, assistant, and tool messages", async () => {
    cwd = await mkdtemp(path.join(os.tmpdir(), "forge-cli-"));
    const sessionsDir = path.join(cwd, ".sessions");
    await writeFile(path.join(cwd, "input.txt"), "alpha", "utf8");
    await main(["node", "forge", "-p", "update"], {
      cwd,
      env: { OPENAI_API_KEY: "test", OPENAI_MODEL: "test" },
      sessionsDir,
      providerFactory: () => new AcceptanceProvider(),
    });
    const listOutput = outputStream();
    await main(["node", "forge", "sessions"], { cwd, sessionsDir, stdout: listOutput.stream });
    const sessionId = listOutput.text().split("\t")[0];
    const projectDirectories = await readdir(sessionsDir);
    const content = await readFile(
      path.join(sessionsDir, projectDirectories[0] ?? "", `${sessionId}.jsonl`),
      "utf8",
    );
    const roles = content
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line).message?.role)
      .filter(Boolean);
    expect(roles).toEqual([
      "user",
      "assistant",
      "tool",
      "assistant",
      "tool",
      "assistant",
      "tool",
      "assistant",
    ]);
  });

  it("runs multiple interactive prompts while keeping slash commands out of context", async () => {
    cwd = await mkdtemp(path.join(os.tmpdir(), "forge-cli-"));
    const stdin = new PassThrough();
    const stdout = outputStream();
    const provider = new FinalProvider();
    stdin.end("first\n/help\nsecond\n/quit\n");

    const exitCode = await main(["node", "forge"], {
      cwd,
      env: { OPENAI_API_KEY: "test", OPENAI_MODEL: "test" },
      providerFactory: () => provider,
      sessionsDir: path.join(cwd, ".sessions"),
      stdin,
      stdout: stdout.stream,
    });

    expect(exitCode).toBe(0);
    expect(provider.requests).toHaveLength(2);
    expect(provider.requests[1]?.messages).toMatchObject([
      { role: "user", content: "first" },
      { role: "assistant", content: "done" },
      { role: "user", content: "second" },
    ]);
    expect(provider.requests[1]?.messages).not.toContainEqual({ role: "user", content: "/help" });
    expect(stdout.text()).toContain("Available commands:");
  });

  it("starts a fresh interactive session on clear without deleting history", async () => {
    cwd = await mkdtemp(path.join(os.tmpdir(), "forge-cli-"));
    const stdin = new PassThrough();
    const stdout = outputStream();
    const provider = new FinalProvider();
    stdin.end("first\n/clear\nsecond\n/quit\n");

    expect(
      await main(["node", "forge"], {
        cwd,
        env: { OPENAI_API_KEY: "test", OPENAI_MODEL: "test" },
        providerFactory: () => provider,
        sessionsDir: path.join(cwd, ".sessions"),
        stdin,
        stdout: stdout.stream,
      }),
    ).toBe(0);

    expect(provider.requests).toHaveLength(2);
    expect(provider.requests[1]?.messages).toEqual([{ role: "user", content: "second" }]);
    const listOutput = outputStream();
    await main(["node", "forge", "sessions"], {
      cwd,
      sessionsDir: path.join(cwd, ".sessions"),
      stdout: listOutput.stream,
    });
    expect(listOutput.text().trim().split("\n")).toHaveLength(2);
  });

  it("routes ! and !! terminal commands without sending them to the provider", async () => {
    cwd = await mkdtemp(path.join(os.tmpdir(), "forge-cli-"));
    const stdin = new PassThrough();
    const stdout = outputStream();
    const provider = new FinalProvider();
    stdin.end(
      "!node -e \"process.stdout.write('visible')\"\nfirst\n" +
        "!!node -e \"process.stdout.write('hidden')\"\nsecond\n/quit\n",
    );

    expect(
      await main(["node", "forge"], {
        cwd,
        env: { OPENAI_API_KEY: "test", OPENAI_MODEL: "test" },
        providerFactory: () => provider,
        sessionsDir: path.join(cwd, ".sessions"),
        stdin,
        stdout: stdout.stream,
      }),
    ).toBe(0);

    expect(provider.requests).toHaveLength(2);
    expect(provider.requests[0]?.messages[0]).toMatchObject({
      role: "user",
      content: expect.stringContaining("visible"),
    });
    expect(JSON.stringify(provider.requests[1]?.messages)).not.toContain("hidden");
    expect(stdout.text()).toContain("visible");
    expect(stdout.text()).toContain("hidden");
  });

  it("returns exit code 130 when an idle interactive session is cancelled", async () => {
    cwd = await mkdtemp(path.join(os.tmpdir(), "forge-cli-"));
    const stdin = new PassThrough();
    const stdout = outputStream();
    const controller = new AbortController();
    const provider = new FinalProvider();
    controller.abort();

    expect(
      await main(["node", "forge"], {
        cwd,
        env: { OPENAI_API_KEY: "test", OPENAI_MODEL: "test" },
        providerFactory: () => provider,
        sessionsDir: path.join(cwd, ".sessions"),
        signal: controller.signal,
        stdin,
        stdout: stdout.stream,
      }),
    ).toBe(130);
    expect(provider.requests).toHaveLength(0);
  });

  it("closes an active idle readline session when cancelled", async () => {
    cwd = await mkdtemp(path.join(os.tmpdir(), "forge-cli-"));
    const stdin = new PassThrough();
    const stdout = outputStream();
    const controller = new AbortController();
    const provider = new FinalProvider();
    const started = new Promise<void>((resolve) => stdout.stream.once("data", () => resolve()));

    const running = main(["node", "forge"], {
      cwd,
      env: { OPENAI_API_KEY: "test", OPENAI_MODEL: "test" },
      providerFactory: () => provider,
      sessionsDir: path.join(cwd, ".sessions"),
      signal: controller.signal,
      stdin,
      stdout: stdout.stream,
    });
    await started;
    controller.abort();

    await expect(running).resolves.toBe(130);
    expect(provider.requests).toHaveLength(0);
  });
});

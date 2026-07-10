import { mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { ToolExecutionContext } from "../agent/index.js";
import { editFileTool } from "./edit-file.js";
import { readFileTool } from "./read-file.js";
import { shellTool } from "./shell.js";
import { writeFileTool } from "./write-file.js";

describe("coding tools", () => {
  let cwd: string;
  let context: ToolExecutionContext;

  beforeEach(async () => {
    cwd = await mkdtemp(path.join(os.tmpdir(), "forge-tools-"));
    context = { cwd, signal: new AbortController().signal };
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  it("reads text and reports truncation", async () => {
    await writeFile(path.join(cwd, "file.txt"), "abcdefghij", "utf8");

    const result = await readFileTool.execute({ path: "file.txt", maxBytes: 4 }, context);

    expect(result.ok).toBe(true);
    expect(result.content).toBe("abcd\n[truncated at 4 bytes]");
    expect(result.data).toMatchObject({ size: 10, bytesRead: 4, truncated: true });
  });

  it("rejects paths outside the workspace", async () => {
    const result = await readFileTool.execute({ path: "../outside.txt" }, context);

    expect(result).toMatchObject({
      ok: false,
      error: { code: "PATH_OUTSIDE_WORKSPACE" },
    });
  });

  it("rejects paths that escape through a directory link", async () => {
    const outside = await mkdtemp(path.join(os.tmpdir(), "forge-outside-"));
    try {
      await writeFile(path.join(outside, "secret.txt"), "secret", "utf8");
      try {
        await symlink(outside, path.join(cwd, "linked"), "junction");
      } catch (error) {
        if (error instanceof Error && "code" in error && error.code === "EPERM") {
          return;
        }
        throw error;
      }

      const result = await readFileTool.execute({ path: "linked/secret.txt" }, context);
      expect(result).toMatchObject({
        ok: false,
        error: { code: "PATH_OUTSIDE_WORKSPACE" },
      });
    } finally {
      await rm(outside, { recursive: true, force: true });
    }
  });

  it("creates files and requires explicit overwrite", async () => {
    const created = await writeFileTool.execute(
      { path: "nested/file.txt", content: "first" },
      context,
    );
    const refused = await writeFileTool.execute(
      { path: "nested/file.txt", content: "second" },
      context,
    );
    const overwritten = await writeFileTool.execute(
      { path: "nested/file.txt", content: "second", overwrite: true },
      context,
    );

    expect(created.ok).toBe(true);
    expect(refused).toMatchObject({ ok: false, error: { code: "WRITE_FILE_ERROR" } });
    expect(overwritten.ok).toBe(true);
    expect(await readFile(path.join(cwd, "nested/file.txt"), "utf8")).toBe("second");
  });

  it("performs one exact edit and preserves line endings", async () => {
    await writeFile(path.join(cwd, "file.txt"), "one\r\ntwo\r\n", "utf8");

    const result = await editFileTool.execute(
      { path: "file.txt", oldText: "two", newText: "changed" },
      context,
    );

    expect(result.ok).toBe(true);
    expect(await readFile(path.join(cwd, "file.txt"), "utf8")).toBe("one\r\nchanged\r\n");
  });

  it.each([
    ["missing", "EDIT_NOT_FOUND"],
    ["same", "EDIT_NOT_UNIQUE"],
  ])("rejects a %s edit target", async (oldText, code) => {
    await writeFile(path.join(cwd, "file.txt"), "same same", "utf8");

    const result = await editFileTool.execute(
      { path: "file.txt", oldText, newText: "changed" },
      context,
    );

    expect(result).toMatchObject({ ok: false, error: { code } });
  });

  it("runs shell commands and captures output", async () => {
    const result = await shellTool.execute(
      { command: "node -e \"process.stdout.write('ok')\"" },
      context,
    );

    expect(result.ok).toBe(true);
    expect(result.data).toMatchObject({ exitCode: 0, stdout: "ok", timedOut: false });
  });

  it("returns non-zero shell exits as structured failures", async () => {
    const result = await shellTool.execute(
      { command: "node -e \"process.stderr.write('bad'); process.exit(3)\"" },
      context,
    );

    expect(result).toMatchObject({
      ok: false,
      data: { exitCode: 3, stderr: "bad" },
      error: { code: "SHELL_EXIT" },
    });
  });

  it("times out shell commands", async () => {
    const result = await shellTool.execute(
      { command: 'node -e "setTimeout(() => {}, 1000)"', timeoutMs: 50 },
      context,
    );

    expect(result).toMatchObject({
      ok: false,
      data: { timedOut: true },
      error: { code: "SHELL_TIMEOUT" },
    });
  });

  it("truncates shell output", async () => {
    const result = await shellTool.execute(
      { command: "node -e \"process.stdout.write('x'.repeat(100))\"", maxOutputBytes: 10 },
      context,
    );

    expect(result).toMatchObject({
      ok: true,
      data: { truncated: true, stdout: "xxxxxxxxxx\n[output truncated]" },
    });
  });

  it("supports cancellation", async () => {
    const controller = new AbortController();
    const running = shellTool.execute(
      { command: 'node -e "setTimeout(() => {}, 1000)"' },
      { cwd, signal: controller.signal },
    );
    controller.abort();

    await expect(running).rejects.toMatchObject({ name: "AbortError" });
  });
});

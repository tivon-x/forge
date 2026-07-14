import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  discoverProjectContext,
  findProjectRoot,
  PROJECT_INSTRUCTION_MAX_BYTES,
} from "./project-context.js";

describe("project context discovery", () => {
  const directories: string[] = [];

  afterEach(async () => {
    await Promise.all(
      directories.splice(0).map((directory) => rm(directory, { recursive: true, force: true })),
    );
  });

  async function temporaryDirectory(prefix: string): Promise<string> {
    const directory = await mkdtemp(path.join(os.tmpdir(), prefix));
    directories.push(directory);
    return directory;
  }

  it("prefers the nearest Git root over nested language markers", async () => {
    const root = await temporaryDirectory("forge-context-");
    const cwd = path.join(root, "packages", "app", "src");
    await mkdir(path.join(root, ".git"));
    await mkdir(cwd, { recursive: true });
    await writeFile(path.join(root, "packages", "app", "package.json"), "{}", "utf8");

    expect(await findProjectRoot(cwd)).toBe(root);
  });

  it("loads root-to-cwd instructions and cwd .forge instructions in order", async () => {
    const root = await temporaryDirectory("forge-context-");
    const nested = path.join(root, "packages");
    const cwd = path.join(nested, "app");
    await mkdir(path.join(root, ".git"));
    await mkdir(path.join(cwd, ".forge"), { recursive: true });
    await writeFile(path.join(root, "AGENTS.md"), "root", "utf8");
    await writeFile(path.join(nested, "AGENTS.md"), "nested", "utf8");
    await writeFile(path.join(cwd, "AGENTS.md"), "cwd", "utf8");
    await writeFile(path.join(cwd, ".forge", "AGENTS.md"), "forge", "utf8");

    const context = await discoverProjectContext(cwd);

    expect(context.projectRoot).toBe(root);
    expect(context.files.map((file) => file.content)).toEqual(["root", "nested", "cwd", "forge"]);
    expect(context.diagnostics).toEqual([]);
  });

  it("skips oversized and invalid UTF-8 instructions with stable diagnostics", async () => {
    const root = await temporaryDirectory("forge-context-");
    const cwd = path.join(root, "app");
    await mkdir(path.join(root, ".git"));
    await mkdir(cwd);
    await writeFile(path.join(root, "AGENTS.md"), "x".repeat(PROJECT_INSTRUCTION_MAX_BYTES + 1));
    await writeFile(path.join(cwd, "AGENTS.md"), Buffer.from([0xc3, 0x28]));

    const context = await discoverProjectContext(cwd);

    expect(context.files).toEqual([]);
    expect(context.diagnostics.map((item) => item.code)).toEqual([
      "PROJECT_CONTEXT_TOO_LARGE",
      "PROJECT_CONTEXT_INVALID_UTF8",
    ]);
  });

  it("rejects instruction symlinks that resolve outside the project root", async ({ skip }) => {
    const root = await temporaryDirectory("forge-context-");
    const outside = await temporaryDirectory("forge-context-outside-");
    await mkdir(path.join(root, ".git"));
    await mkdir(path.join(root, ".forge"));
    const target = path.join(outside, "AGENTS.md");
    await writeFile(target, "outside", "utf8");
    try {
      await symlink(target, path.join(root, ".forge", "AGENTS.md"), "file");
    } catch (error) {
      if (error instanceof Error && "code" in error && error.code === "EPERM") {
        skip();
        return;
      }
      throw error;
    }

    const context = await discoverProjectContext(root);

    expect(context.files).toEqual([]);
    expect(context.diagnostics).toMatchObject([{ code: "PROJECT_CONTEXT_OUTSIDE_ROOT" }]);
  });

  it("rejects instruction symlinks even when they stay inside the project root", async ({
    skip,
  }) => {
    const root = await temporaryDirectory("forge-context-");
    await mkdir(path.join(root, ".git"));
    await mkdir(path.join(root, ".forge"));
    const target = path.join(root, "instructions.md");
    await writeFile(target, "inside", "utf8");
    try {
      await symlink(target, path.join(root, ".forge", "AGENTS.md"), "file");
    } catch (error) {
      if (error instanceof Error && "code" in error && error.code === "EPERM") {
        skip();
        return;
      }
      throw error;
    }

    const context = await discoverProjectContext(root);

    expect(context.files).toEqual([]);
    expect(context.diagnostics).toMatchObject([{ code: "PROJECT_CONTEXT_OUTSIDE_ROOT" }]);
  });
});

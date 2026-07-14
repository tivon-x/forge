import type { Stats } from "node:fs";
import { lstat, open, realpath } from "node:fs/promises";
import path from "node:path";

const FALLBACK_PROJECT_MARKERS = [
  "package.json",
  "pnpm-workspace.yaml",
  "pyproject.toml",
  "Cargo.toml",
  "go.mod",
] as const;

export const PROJECT_INSTRUCTION_MAX_BYTES = 100_000;
export const PROJECT_INSTRUCTIONS_MAX_TOTAL_BYTES = 300_000;

export interface ProjectInstruction {
  path: string;
  content: string;
}

export interface ProjectContextDiagnostic {
  code:
    | "PROJECT_CONTEXT_INVALID_FILE"
    | "PROJECT_CONTEXT_INVALID_UTF8"
    | "PROJECT_CONTEXT_OUTSIDE_ROOT"
    | "PROJECT_CONTEXT_READ_FAILED"
    | "PROJECT_CONTEXT_TOO_LARGE";
  path: string;
  message: string;
}

export interface ProjectContext {
  projectRoot: string;
  files: readonly ProjectInstruction[];
  diagnostics: readonly ProjectContextDiagnostic[];
}

function isWithin(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return (
    relative === "" ||
    (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative))
  );
}

function ancestorsFrom(start: string): string[] {
  const ancestors = [];
  let current = start;
  while (true) {
    ancestors.push(current);
    const parent = path.dirname(current);
    if (parent === current) return ancestors;
    current = parent;
  }
}

async function exists(candidate: string): Promise<boolean> {
  try {
    await lstat(candidate);
    return true;
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT") return false;
    throw error;
  }
}

export async function findProjectRoot(cwd: string): Promise<string> {
  const resolvedCwd = await realpath(path.resolve(cwd));
  const ancestors = ancestorsFrom(resolvedCwd);

  for (const directory of ancestors) {
    if (await exists(path.join(directory, ".git"))) return directory;
  }
  for (const directory of ancestors) {
    for (const marker of FALLBACK_PROJECT_MARKERS) {
      if (await exists(path.join(directory, marker))) return directory;
    }
  }
  return resolvedCwd;
}

function instructionCandidates(projectRoot: string, cwd: string): string[] {
  const relative = path.relative(projectRoot, cwd);
  if (isWithin(projectRoot, cwd)) {
    const directories = [projectRoot];
    let current = projectRoot;
    for (const part of relative.split(path.sep).filter(Boolean)) {
      current = path.join(current, part);
      directories.push(current);
    }
    return [
      ...directories.map((directory) => path.join(directory, "AGENTS.md")),
      path.join(cwd, ".forge", "AGENTS.md"),
    ];
  }
  return [path.join(cwd, "AGENTS.md"), path.join(cwd, ".forge", "AGENTS.md")];
}

function diagnostic(
  code: ProjectContextDiagnostic["code"],
  filePath: string,
  message: string,
): ProjectContextDiagnostic {
  return { code, path: filePath, message };
}

class InstructionReadError extends Error {
  constructor(
    readonly code: ProjectContextDiagnostic["code"],
    message: string,
  ) {
    super(message);
    this.name = "InstructionReadError";
  }
}

function sameFile(left: Stats, right: Stats): boolean {
  return left.dev === right.dev && left.ino === right.ino;
}

async function readInstructionFile(
  candidate: string,
  projectRoot: string,
  initialInfo: Stats,
  byteLimit: number,
): Promise<{ contents: Buffer; resolvedFile: string }> {
  if (initialInfo.isSymbolicLink()) {
    throw new InstructionReadError(
      "PROJECT_CONTEXT_OUTSIDE_ROOT",
      "instruction file symlinks are not allowed",
    );
  }
  if (!initialInfo.isFile()) {
    throw new InstructionReadError(
      "PROJECT_CONTEXT_INVALID_FILE",
      "instruction path is not a file",
    );
  }
  if (initialInfo.size > byteLimit) {
    throw new InstructionReadError(
      "PROJECT_CONTEXT_TOO_LARGE",
      `instruction limits are ${PROJECT_INSTRUCTION_MAX_BYTES} bytes per file and ${PROJECT_INSTRUCTIONS_MAX_TOTAL_BYTES} bytes total`,
    );
  }

  const resolvedBeforeOpen = await realpath(candidate);
  if (!isWithin(projectRoot, resolvedBeforeOpen)) {
    throw new InstructionReadError(
      "PROJECT_CONTEXT_OUTSIDE_ROOT",
      "instruction file resolves outside the project root",
    );
  }

  const file = await open(candidate, "r");
  try {
    const openedInfo = await file.stat();
    const currentInfo = await lstat(candidate);
    const resolvedAfterOpen = await realpath(candidate);
    if (
      !openedInfo.isFile() ||
      currentInfo.isSymbolicLink() ||
      !currentInfo.isFile() ||
      !sameFile(openedInfo, currentInfo) ||
      resolvedAfterOpen !== resolvedBeforeOpen ||
      !isWithin(projectRoot, resolvedAfterOpen)
    ) {
      throw new InstructionReadError(
        "PROJECT_CONTEXT_OUTSIDE_ROOT",
        "instruction file changed during validation",
      );
    }
    if (openedInfo.size > byteLimit) {
      throw new InstructionReadError(
        "PROJECT_CONTEXT_TOO_LARGE",
        `instruction limits are ${PROJECT_INSTRUCTION_MAX_BYTES} bytes per file and ${PROJECT_INSTRUCTIONS_MAX_TOTAL_BYTES} bytes total`,
      );
    }

    const buffer = Buffer.alloc(byteLimit + 1);
    let offset = 0;
    while (offset < buffer.byteLength) {
      const { bytesRead } = await file.read(buffer, offset, buffer.byteLength - offset, offset);
      if (bytesRead === 0) break;
      offset += bytesRead;
    }
    if (offset > byteLimit) {
      throw new InstructionReadError(
        "PROJECT_CONTEXT_TOO_LARGE",
        `instruction limits are ${PROJECT_INSTRUCTION_MAX_BYTES} bytes per file and ${PROJECT_INSTRUCTIONS_MAX_TOTAL_BYTES} bytes total`,
      );
    }
    return { contents: buffer.subarray(0, offset), resolvedFile: resolvedAfterOpen };
  } finally {
    await file.close();
  }
}

export async function discoverProjectContext(cwd: string): Promise<ProjectContext> {
  const resolvedCwd = await realpath(path.resolve(cwd));
  const projectRoot = await findProjectRoot(resolvedCwd);
  const files: ProjectInstruction[] = [];
  const diagnostics: ProjectContextDiagnostic[] = [];
  const seen = new Set<string>();
  let totalBytes = 0;

  for (const candidate of instructionCandidates(projectRoot, resolvedCwd)) {
    let fileInfo: Stats;
    try {
      fileInfo = await lstat(candidate);
    } catch (error) {
      if (error instanceof Error && "code" in error && error.code === "ENOENT") continue;
      diagnostics.push(
        diagnostic(
          "PROJECT_CONTEXT_READ_FAILED",
          candidate,
          error instanceof Error ? error.message : String(error),
        ),
      );
      continue;
    }

    let contents: Buffer;
    let resolvedFile: string;
    try {
      const loaded = await readInstructionFile(
        candidate,
        projectRoot,
        fileInfo,
        Math.min(PROJECT_INSTRUCTION_MAX_BYTES, PROJECT_INSTRUCTIONS_MAX_TOTAL_BYTES - totalBytes),
      );
      contents = loaded.contents;
      resolvedFile = loaded.resolvedFile;
    } catch (error) {
      diagnostics.push(
        diagnostic(
          error instanceof InstructionReadError ? error.code : "PROJECT_CONTEXT_READ_FAILED",
          candidate,
          error instanceof Error ? error.message : String(error),
        ),
      );
      continue;
    }
    if (seen.has(resolvedFile)) continue;

    let content: string;
    try {
      content = new TextDecoder("utf-8", { fatal: true }).decode(contents);
    } catch {
      diagnostics.push(
        diagnostic(
          "PROJECT_CONTEXT_INVALID_UTF8",
          candidate,
          "instruction file is not valid UTF-8",
        ),
      );
      continue;
    }

    seen.add(resolvedFile);
    totalBytes += contents.byteLength;
    files.push({ path: candidate, content });
  }

  return { projectRoot, files, diagnostics };
}

import type { Stats } from "node:fs";
import { lstat, readFile, realpath } from "node:fs/promises";
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

    if (!fileInfo.isFile() && !fileInfo.isSymbolicLink()) {
      diagnostics.push(
        diagnostic("PROJECT_CONTEXT_INVALID_FILE", candidate, "instruction path is not a file"),
      );
      continue;
    }

    let resolvedFile: string;
    try {
      resolvedFile = await realpath(candidate);
    } catch (error) {
      diagnostics.push(
        diagnostic(
          "PROJECT_CONTEXT_READ_FAILED",
          candidate,
          error instanceof Error ? error.message : String(error),
        ),
      );
      continue;
    }
    if (!isWithin(projectRoot, resolvedFile)) {
      diagnostics.push(
        diagnostic(
          "PROJECT_CONTEXT_OUTSIDE_ROOT",
          candidate,
          "instruction file resolves outside the project root",
        ),
      );
      continue;
    }
    if (seen.has(resolvedFile)) continue;

    let contents: Buffer;
    try {
      contents = await readFile(resolvedFile);
    } catch (error) {
      diagnostics.push(
        diagnostic(
          "PROJECT_CONTEXT_READ_FAILED",
          candidate,
          error instanceof Error ? error.message : String(error),
        ),
      );
      continue;
    }

    if (
      contents.byteLength > PROJECT_INSTRUCTION_MAX_BYTES ||
      totalBytes + contents.byteLength > PROJECT_INSTRUCTIONS_MAX_TOTAL_BYTES
    ) {
      diagnostics.push(
        diagnostic(
          "PROJECT_CONTEXT_TOO_LARGE",
          candidate,
          `instruction limits are ${PROJECT_INSTRUCTION_MAX_BYTES} bytes per file and ${PROJECT_INSTRUCTIONS_MAX_TOTAL_BYTES} bytes total`,
        ),
      );
      continue;
    }

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

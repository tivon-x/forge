import { lstat, realpath } from "node:fs/promises";
import path from "node:path";

import { ToolOperationError } from "./errors.js";

function isWithin(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return (
    relative === "" ||
    (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative))
  );
}

async function nearestExistingPath(candidate: string): Promise<string> {
  let current = candidate;
  while (true) {
    try {
      await lstat(current);
      return current;
    } catch (error) {
      if (!(error instanceof Error) || !("code" in error) || error.code !== "ENOENT") {
        throw error;
      }
    }

    const parent = path.dirname(current);
    if (parent === current) {
      throw new ToolOperationError("PATH_NOT_FOUND", `No existing parent for path: ${candidate}`);
    }
    current = parent;
  }
}

export async function resolveWorkspacePath(
  cwd: string,
  requestedPath: string,
  mustExist: boolean,
): Promise<string> {
  const workspace = await realpath(path.resolve(cwd));
  const candidate = path.resolve(workspace, requestedPath);

  if (!isWithin(workspace, candidate)) {
    throw new ToolOperationError(
      "PATH_OUTSIDE_WORKSPACE",
      `Path is outside the workspace: ${requestedPath}`,
    );
  }

  let resolvedTarget: string;
  try {
    resolvedTarget = await realpath(candidate);
  } catch (error) {
    if (!(error instanceof Error) || !("code" in error) || error.code !== "ENOENT") {
      throw error;
    }
    if (mustExist) {
      throw new ToolOperationError("PATH_NOT_FOUND", `Path does not exist: ${requestedPath}`);
    }
    const existingParent = await nearestExistingPath(path.dirname(candidate));
    resolvedTarget = await realpath(existingParent);
  }

  if (!isWithin(workspace, resolvedTarget)) {
    throw new ToolOperationError(
      "PATH_OUTSIDE_WORKSPACE",
      `Path resolves outside the workspace: ${requestedPath}`,
    );
  }

  return candidate;
}

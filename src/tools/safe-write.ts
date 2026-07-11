import { constants } from "node:fs";
import { lstat, open } from "node:fs/promises";

import { ToolOperationError } from "./errors.js";

export type SafeWriteMode = "create" | "overwrite" | "replace";

const noFollowFlag = (constants as unknown as Record<string, number>).O_NOFOLLOW ?? 0;

function isFileSystemError(error: unknown, code: string): boolean {
  return error instanceof Error && "code" in error && error.code === code;
}

async function rejectFinalSymlink(target: string): Promise<void> {
  try {
    const fileStat = await lstat(target);
    if (fileStat.isSymbolicLink()) {
      throw new ToolOperationError("PATH_SYMLINK", `Refusing to write through symlink: ${target}`);
    }
  } catch (error) {
    if (isFileSystemError(error, "ENOENT")) {
      return;
    }
    throw error;
  }
}

function openFlags(mode: SafeWriteMode): number {
  const base = constants.O_WRONLY | noFollowFlag;
  if (mode === "create") {
    return base | constants.O_CREAT | constants.O_EXCL;
  }
  if (mode === "overwrite") {
    return base | constants.O_CREAT | constants.O_TRUNC;
  }
  return base | constants.O_CREAT | constants.O_TRUNC;
}

export async function safeWriteText(
  target: string,
  content: string,
  mode: SafeWriteMode,
): Promise<void> {
  await rejectFinalSymlink(target);

  let handle: Awaited<ReturnType<typeof open>>;
  try {
    handle = await open(target, openFlags(mode), 0o666);
  } catch (error) {
    if (isFileSystemError(error, "ELOOP")) {
      throw new ToolOperationError("PATH_SYMLINK", `Refusing to write through symlink: ${target}`);
    }
    throw error;
  }

  try {
    await handle.writeFile(content, "utf8");
  } finally {
    await handle.close();
  }
}

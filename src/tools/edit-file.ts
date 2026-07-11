import { readFile } from "node:fs/promises";
import path from "node:path";

import { z } from "zod";

import { defineTool } from "../agent/index.js";
import { ToolOperationError, toolFailure } from "./errors.js";
import { safeWriteText } from "./safe-write.js";
import { resolveWorkspacePath } from "./workspace-path.js";

function countOccurrences(content: string, search: string): number {
  let count = 0;
  let position = 0;
  while (true) {
    const index = content.indexOf(search, position);
    if (index === -1) {
      return count;
    }
    count += 1;
    position = index + search.length;
  }
}

export const editFileTool = defineTool({
  name: "editFile",
  description: "Replace exactly one occurrence of text in an existing UTF-8 workspace file.",
  inputSchema: z.object({
    path: z.string().min(1).describe("Workspace-relative or workspace-contained absolute path"),
    oldText: z.string().min(1).describe("Exact text that must occur exactly once"),
    newText: z.string(),
  }),
  promptGuidelines: [
    "oldText must match the file exactly, including whitespace and line endings.",
    "Include enough surrounding text to make oldText unique.",
  ],
  execute: async ({ path: requestedPath, oldText, newText }, context) => {
    try {
      context.signal.throwIfAborted();
      const target = await resolveWorkspacePath(context.cwd, requestedPath, true);
      const content = await readFile(target, "utf8");
      const occurrences = countOccurrences(content, oldText);

      if (occurrences === 0) {
        throw new ToolOperationError("EDIT_NOT_FOUND", `oldText was not found in ${requestedPath}`);
      }
      if (occurrences !== 1) {
        throw new ToolOperationError(
          "EDIT_NOT_UNIQUE",
          `oldText occurs ${occurrences} times in ${requestedPath}`,
        );
      }

      const updated = content.replace(oldText, newText);
      await safeWriteText(target, updated, "replace");
      return {
        ok: true,
        content: `Edited ${requestedPath}`,
        data: { path: path.relative(context.cwd, target), replacements: 1 },
      };
    } catch (error) {
      if (context.signal.aborted) {
        throw error;
      }
      return toolFailure(error, "EDIT_FILE_ERROR");
    }
  },
});

import { open, stat } from "node:fs/promises";
import path from "node:path";

import { z } from "zod";

import { defineTool } from "../agent/index.js";
import { ToolOperationError, toolFailure } from "./errors.js";
import { resolveWorkspacePath } from "./workspace-path.js";

const DEFAULT_MAX_BYTES = 100_000;

export const readFileTool = defineTool({
  name: "readFile",
  description: "Read a UTF-8 text file inside the current workspace.",
  inputSchema: z.object({
    path: z.string().min(1).describe("Workspace-relative or workspace-contained absolute path"),
    maxBytes: z.int().min(1).max(1_000_000).default(DEFAULT_MAX_BYTES),
  }),
  promptGuidelines: ["Read a file before editing it when its current contents are unknown."],
  execute: async ({ path: requestedPath, maxBytes }, context) => {
    try {
      context.signal.throwIfAborted();
      const target = await resolveWorkspacePath(context.cwd, requestedPath, true);
      const fileStat = await stat(target);
      if (!fileStat.isFile()) {
        throw new ToolOperationError("NOT_A_FILE", `Path is not a file: ${requestedPath}`);
      }

      const bytesToRead = Math.min(fileStat.size, maxBytes);
      const buffer = Buffer.alloc(bytesToRead);
      const handle = await open(target, "r");
      try {
        await handle.read(buffer, 0, bytesToRead, 0);
      } finally {
        await handle.close();
      }

      if (buffer.includes(0)) {
        throw new ToolOperationError("BINARY_FILE", `File appears to be binary: ${requestedPath}`);
      }

      const truncated = fileStat.size > bytesToRead;
      const content = buffer.toString("utf8");
      return {
        ok: true,
        content: truncated ? `${content}\n[truncated at ${bytesToRead} bytes]` : content,
        data: {
          path: path.relative(context.cwd, target),
          size: fileStat.size,
          bytesRead: bytesToRead,
          truncated,
        },
      };
    } catch (error) {
      if (context.signal.aborted) {
        throw error;
      }
      return toolFailure(error, "READ_FILE_ERROR");
    }
  },
});

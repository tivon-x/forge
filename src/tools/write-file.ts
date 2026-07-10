import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { z } from "zod";

import { defineTool } from "../agent/index.js";
import { toolFailure } from "./errors.js";
import { resolveWorkspacePath } from "./workspace-path.js";

export const writeFileTool = defineTool({
  name: "writeFile",
  description: "Create a UTF-8 text file inside the workspace, optionally overwriting it.",
  inputSchema: z.object({
    path: z.string().min(1).describe("Workspace-relative or workspace-contained absolute path"),
    content: z.string(),
    overwrite: z.boolean().default(false),
  }),
  promptGuidelines: [
    "Use editFile for small changes to an existing file.",
    "Set overwrite to true only when replacing the entire file intentionally.",
  ],
  execute: async ({ path: requestedPath, content, overwrite }, context) => {
    try {
      context.signal.throwIfAborted();
      const target = await resolveWorkspacePath(context.cwd, requestedPath, false);
      await mkdir(path.dirname(target), { recursive: true });
      await writeFile(target, content, { encoding: "utf8", flag: overwrite ? "w" : "wx" });
      return {
        ok: true,
        content: `Wrote ${Buffer.byteLength(content, "utf8")} bytes to ${requestedPath}`,
        data: {
          path: path.relative(context.cwd, target),
          bytesWritten: Buffer.byteLength(content, "utf8"),
        },
      };
    } catch (error) {
      if (context.signal.aborted) {
        throw error;
      }
      return toolFailure(error, "WRITE_FILE_ERROR");
    }
  },
});

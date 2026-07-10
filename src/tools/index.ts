import type { ToolDefinition } from "../agent/index.js";
import { editFileTool } from "./edit-file.js";
import { readFileTool } from "./read-file.js";
import { shellTool } from "./shell.js";
import { writeFileTool } from "./write-file.js";

export { editFileTool } from "./edit-file.js";
export { readFileTool } from "./read-file.js";
export { shellTool } from "./shell.js";
export { writeFileTool } from "./write-file.js";

export const CODING_TOOLS: readonly ToolDefinition[] = [
  readFileTool,
  writeFileTool,
  editFileTool,
  shellTool,
];

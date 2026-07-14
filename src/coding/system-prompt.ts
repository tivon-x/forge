import type { ToolDefinition } from "../agent/index.js";
import type { ProjectInstruction } from "./project-context.js";

export interface SystemPromptOptions {
  contextFiles?: readonly ProjectInstruction[];
  cwd: string;
  tools: readonly ToolDefinition[];
}

function collectUnique(values: Iterable<string>): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const value of values) {
    const normalized = value.trim();
    if (normalized.length === 0 || seen.has(normalized)) continue;
    seen.add(normalized);
    result.push(normalized);
  }
  return result;
}

function escapeAttribute(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatProjectContext(files: readonly ProjectInstruction[]): string | undefined {
  if (files.length === 0) return undefined;
  const lines = ["<project_context>", "Project-specific instructions and guidelines:", ""];
  for (const file of files) {
    lines.push(`<project_instructions path="${escapeAttribute(file.path)}">`);
    lines.push(file.content);
    lines.push("</project_instructions>", "");
  }
  lines.push("</project_context>");
  return lines.join("\n");
}

export function buildSystemPrompt(options: SystemPromptOptions): string {
  const snippets = collectUnique(
    options.tools.flatMap((tool) => (tool.promptSnippet === undefined ? [] : [tool.promptSnippet])),
  );
  const guidelines = collectUnique(options.tools.flatMap((tool) => tool.promptGuidelines ?? []));

  const sections = [
    "You are Forge, a coding agent. Work directly on the user's task and report the result concisely.",
    `The workspace is ${options.cwd}. File tools are restricted to this workspace.`,
    "Inspect relevant files before changing them. Prefer precise edits over replacing entire files. Run appropriate checks after changes.",
  ];

  if (snippets.length > 0) sections.push(snippets.join("\n"));
  if (guidelines.length > 0) {
    sections.push(`Tool guidelines:\n${guidelines.map((item) => `- ${item}`).join("\n")}`);
  }
  const projectContext = formatProjectContext(options.contextFiles ?? []);
  if (projectContext !== undefined) sections.push(projectContext);

  return sections.join("\n\n");
}

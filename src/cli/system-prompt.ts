import type { ToolDefinition } from "../agent/index.js";

interface SystemPromptOptions {
  cwd: string;
  tools: readonly ToolDefinition[];
}

export function buildSystemPrompt(options: SystemPromptOptions): string {
  const snippets = new Set<string>();
  const guidelines = new Set<string>();

  for (const tool of options.tools) {
    if (tool.promptSnippet !== undefined) {
      snippets.add(tool.promptSnippet);
    }
    for (const guideline of tool.promptGuidelines ?? []) {
      guidelines.add(guideline);
    }
  }

  const sections = [
    "You are Forge, a coding agent. Work directly on the user's task and report the result concisely.",
    `The workspace is ${options.cwd}. File tools are restricted to this workspace.`,
    "Inspect relevant files before changing them. Prefer precise edits over replacing entire files. Run appropriate checks after changes.",
  ];

  if (snippets.size > 0) {
    sections.push([...snippets].join("\n"));
  }
  if (guidelines.size > 0) {
    sections.push(`Tool guidelines:\n${[...guidelines].map((item) => `- ${item}`).join("\n")}`);
  }

  return sections.join("\n\n");
}

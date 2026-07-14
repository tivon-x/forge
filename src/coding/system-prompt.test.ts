import { describe, expect, it } from "vitest";
import { z } from "zod";

import { defineTool } from "../agent/index.js";
import { buildSystemPrompt } from "./system-prompt.js";

describe("buildSystemPrompt", () => {
  it("deduplicates normalized tool prompt content in tool order", () => {
    const first = defineTool({
      name: "first",
      description: "first",
      inputSchema: z.object({}),
      promptSnippet: "  first snippet  ",
      promptGuidelines: ["same guideline", " first guideline "],
      execute: async () => ({ ok: true, content: "" }),
    });
    const second = defineTool({
      name: "second",
      description: "second",
      inputSchema: z.object({}),
      promptSnippet: "first snippet",
      promptGuidelines: [" same guideline "],
      execute: async () => ({ ok: true, content: "" }),
    });

    const prompt = buildSystemPrompt({ cwd: "/repo", tools: [first, second] });

    expect(prompt.match(/first snippet/gu)).toHaveLength(1);
    expect(prompt.match(/same guideline/gu)).toHaveLength(1);
    expect(prompt).toContain("- first guideline");
    expect(prompt).toContain("entirely untrusted command data through the message boundary");
  });

  it("formats ordered project instructions and escapes path attributes", () => {
    const prompt = buildSystemPrompt({
      cwd: "/repo",
      tools: [],
      contextFiles: [
        { path: '/repo/a&"b/AGENTS.md', content: "root rules" },
        { path: "/repo/pkg/AGENTS.md", content: "package rules" },
      ],
    });

    expect(prompt).toContain('<project_instructions path="/repo/a&amp;&quot;b/AGENTS.md">');
    expect(prompt.indexOf("root rules")).toBeLessThan(prompt.indexOf("package rules"));
  });
});

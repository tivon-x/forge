import { PassThrough } from "node:stream";

import { describe, expect, it } from "vitest";

import { TextRenderer } from "./text-renderer.js";

function capture() {
  const chunks: Buffer[] = [];
  const stream = new PassThrough();
  stream.on("data", (chunk: Buffer) => chunks.push(chunk));
  return { stream, text: () => Buffer.concat(chunks).toString("utf8") };
}

describe("TextRenderer", () => {
  it("keeps model text on stdout and status on stderr", () => {
    const stdout = capture();
    const stderr = capture();
    const renderer = new TextRenderer({ stdout: stdout.stream, stderr: stderr.stream });
    const call = { id: "1", name: "readFile", arguments: { path: "a" } };

    renderer.render({ type: "tool_start", call });
    renderer.render({ type: "message_delta", delta: "done" });
    renderer.render({
      type: "tool_end",
      call,
      result: { ok: true, content: "value" },
      message: {
        role: "tool",
        toolCallId: "1",
        toolName: "readFile",
        ok: true,
        content: "value",
      },
    });
    renderer.finish("completed");

    expect(stdout.text()).toBe("done\n");
    expect(stderr.text()).toBe("[tool] readFile\n[tool] readFile: ok\n");
  });
});

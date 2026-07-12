import { PassThrough } from "node:stream";

import { describe, expect, it } from "vitest";

import { FinalTextRenderer } from "./final-text-renderer.js";
import { JsonEventRenderer } from "./json-renderer.js";

function capture() {
  const chunks: Buffer[] = [];
  const stream = new PassThrough();
  stream.on("data", (chunk: Buffer) => chunks.push(chunk));
  return { stream, text: () => Buffer.concat(chunks).toString("utf8") };
}

describe("output renderers", () => {
  it("text mode prints only the final assistant message", () => {
    const stdout = capture();
    const stderr = capture();
    const renderer = new FinalTextRenderer({ stdout: stdout.stream, stderr: stderr.stream });
    renderer.render({ type: "message_delta", delta: "ignored" });
    renderer.render({
      type: "message_end",
      message: { role: "assistant", content: "final", toolCalls: [] },
    });
    renderer.finish("completed");
    expect(stdout.text()).toBe("final\n");
    expect(stderr.text()).toBe("");
  });

  it("text mode reports errors only on stderr", () => {
    const stdout = capture();
    const stderr = capture();
    const renderer = new FinalTextRenderer({ stdout: stdout.stream, stderr: stderr.stream });
    renderer.render({ type: "error", code: "FAILED", message: "nope" });
    renderer.finish("error");
    expect(stdout.text()).toBe("");
    expect(stderr.text()).toBe("Error [FAILED]: nope\n");
  });

  it("json mode emits one parseable event per line", () => {
    const stdout = capture();
    const renderer = new JsonEventRenderer({ stdout: stdout.stream });
    renderer.render({ type: "agent_start" });
    renderer.render({ type: "message_delta", delta: "hello" });
    renderer.finish("completed");
    expect(
      stdout
        .text()
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line)),
    ).toEqual([{ type: "agent_start" }, { type: "message_delta", delta: "hello" }]);
  });
});

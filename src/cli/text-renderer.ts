import type { Writable } from "node:stream";

import type { AgentEndReason, AgentEvent } from "../agent/index.js";

export interface TextRendererOptions {
  stdout: Writable;
  stderr: Writable;
}

export class TextRenderer {
  readonly #stdout: Writable;
  readonly #stderr: Writable;
  #hasText = false;

  constructor(options: TextRendererOptions) {
    this.#stdout = options.stdout;
    this.#stderr = options.stderr;
  }

  render(event: AgentEvent): void {
    if (event.type === "message_delta") {
      this.#stdout.write(event.delta);
      this.#hasText = true;
    } else if (event.type === "tool_start") {
      this.#stderr.write(`[tool] ${event.call.name}\n`);
    } else if (event.type === "tool_end") {
      const status = event.result.ok ? "ok" : "error";
      this.#stderr.write(`[tool] ${event.call.name}: ${status}\n`);
    } else if (event.type === "error") {
      this.#stderr.write(`Error [${event.code}]: ${event.message}\n`);
    }
  }

  finish(reason: AgentEndReason): void {
    if (this.#hasText && reason === "completed") {
      this.#stdout.write("\n");
    }
  }
}

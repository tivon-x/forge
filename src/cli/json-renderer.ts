import type { Writable } from "node:stream";

import type { AgentEndReason, AgentEvent } from "../agent/index.js";
import type { EventRenderer } from "./event-renderer.js";

export class JsonEventRenderer implements EventRenderer {
  readonly #stdout: Writable;

  constructor(options: { stdout: Writable }) {
    this.#stdout = options.stdout;
  }

  render(event: AgentEvent): void {
    this.#stdout.write(`${JSON.stringify(event)}\n`);
  }

  finish(_reason: AgentEndReason): void {}
}

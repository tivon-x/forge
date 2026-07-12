import type { Writable } from "node:stream";

import type { AgentEndReason, AgentEvent } from "../agent/index.js";
import type { EventRenderer } from "./event-renderer.js";

export class FinalTextRenderer implements EventRenderer {
  readonly #stdout: Writable;
  readonly #stderr: Writable;
  #lastText = "";
  readonly #errors: string[] = [];

  constructor(options: { stdout: Writable; stderr: Writable }) {
    this.#stdout = options.stdout;
    this.#stderr = options.stderr;
  }

  render(event: AgentEvent): void {
    if (event.type === "message_end") this.#lastText = event.message.content;
    else if (event.type === "error") this.#errors.push(`Error [${event.code}]: ${event.message}`);
  }

  finish(reason: AgentEndReason): void {
    if (reason === "completed") {
      if (this.#lastText.length > 0) this.#stdout.write(`${this.#lastText}\n`);
      return;
    }
    for (const error of this.#errors) this.#stderr.write(`${error}\n`);
  }
}

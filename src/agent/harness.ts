import type { AgentEvent, AgentRunResult } from "./events.js";
import { AgentLoop, type AgentLoopOptions } from "./loop.js";
import type { Message } from "./messages.js";

export class AgentHarness {
  readonly #loop: AgentLoop;
  #messages: Message[];
  #running = false;

  constructor(options: AgentLoopOptions, initialMessages: readonly Message[] = []) {
    this.#loop = new AgentLoop(options);
    this.#messages = [...initialMessages];
  }

  get messages(): readonly Message[] {
    return this.#messages;
  }

  appendUserMessage(content: string): void {
    if (this.#running) {
      throw new Error("AgentHarness is already running");
    }
    this.#messages.push({ role: "user", content });
  }

  async *run(
    userInput: string,
    signal?: AbortSignal,
  ): AsyncGenerator<AgentEvent, AgentRunResult, undefined> {
    if (this.#running) {
      throw new Error("AgentHarness is already running");
    }

    this.#running = true;
    try {
      const messages: Message[] = [...this.#messages, { role: "user", content: userInput }];
      const request = signal === undefined ? { messages } : { messages, signal };
      const result = yield* this.#loop.run(request);
      this.#messages = result.messages;
      return result;
    } finally {
      this.#running = false;
    }
  }
}

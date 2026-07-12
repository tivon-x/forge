import type { AgentEndReason, AgentEvent } from "../agent/index.js";

export interface EventRenderer {
  render(event: AgentEvent): void;
  finish(reason: AgentEndReason): void;
}

export type {
  CommandAction,
  CommandContext,
  CommandResult,
  SlashCommand,
} from "./commands.js";
export { CommandRegistry, createDefaultCommandRegistry } from "./commands.js";
export type {
  ProjectContext,
  ProjectContextDiagnostic,
  ProjectInstruction,
} from "./project-context.js";
export {
  discoverProjectContext,
  findProjectRoot,
  PROJECT_INSTRUCTION_MAX_BYTES,
  PROJECT_INSTRUCTIONS_MAX_TOTAL_BYTES,
} from "./project-context.js";
export type {
  OpenCodingSessionOptions,
  TerminalCommandRequest,
  TerminalCommandResult,
  TerminalExecutor,
} from "./session.js";
export {
  CodingSession,
  CodingSessionError,
  executeTerminalCommand,
  parseTerminalCommand,
} from "./session.js";
export type { SystemPromptOptions } from "./system-prompt.js";
export { buildSystemPrompt } from "./system-prompt.js";

import type { ToolDefinition } from "../agent/index.js";
import type { SessionManager } from "../sessions/index.js";
import type { ProjectContext } from "./project-context.js";

export type CommandAction =
  | { type: "clear" }
  | { type: "quit" }
  | { type: "resume"; sessionId: string };

export interface CommandResult {
  handled: boolean;
  action?: CommandAction;
  error?: { code: string; message: string };
  message?: string;
}

export interface CommandContext {
  cwd: string;
  manager: SessionManager;
  projectContext: ProjectContext;
  tools: readonly ToolDefinition[];
}

interface CommandExecutionContext extends CommandContext {
  args: string;
  registry: CommandRegistry;
}

type CommandHandlerResult = Omit<CommandResult, "handled">;
type CommandHandler = (context: CommandExecutionContext) => Promise<CommandHandlerResult>;

export interface SlashCommand {
  name: string;
  description: string;
  usage: string;
  handler: CommandHandler;
}

function normalizeName(name: string): string {
  return name.trim().replace(/^\//u, "").toLowerCase();
}

export class CommandRegistry {
  readonly #commands = new Map<string, SlashCommand>();

  register(command: SlashCommand): void {
    const name = normalizeName(command.name);
    if (!/^[a-z][a-z0-9-]*$/u.test(name)) {
      throw new Error(`Invalid slash command name: ${command.name}`);
    }
    if (this.#commands.has(name)) {
      throw new Error(`Duplicate slash command: /${name}`);
    }
    this.#commands.set(name, { ...command, name });
  }

  list(): readonly SlashCommand[] {
    return [...this.#commands.values()].sort((left, right) => left.name.localeCompare(right.name));
  }

  async execute(context: CommandContext, text: string): Promise<CommandResult> {
    const trimmed = text.trim();
    if (!trimmed.startsWith("/")) return { handled: false };

    const [commandText = "", ...argParts] = trimmed.slice(1).split(/\s+/u);
    const name = normalizeName(commandText);
    if (name.length === 0) {
      return {
        handled: true,
        error: { code: "UNKNOWN_COMMAND", message: "Unknown command: /" },
      };
    }
    const command = this.#commands.get(name);
    if (command === undefined) {
      return {
        handled: true,
        error: { code: "UNKNOWN_COMMAND", message: `Unknown command: /${name}` },
      };
    }
    const result = await command.handler({
      ...context,
      args: argParts.join(" "),
      registry: this,
    });
    return { ...result, handled: true };
  }
}

export function createDefaultCommandRegistry(): CommandRegistry {
  const registry = new CommandRegistry();

  registry.register({
    name: "help",
    usage: "/help",
    description: "List available commands.",
    handler: async (context) => ({
      message: [
        "Available commands:",
        ...context.registry.list().map((command) => `${command.usage}\t${command.description}`),
      ].join("\n"),
    }),
  });
  registry.register({
    name: "sessions",
    usage: "/sessions",
    description: "List sessions for the current workspace.",
    handler: async (context) => {
      const sessions = await context.manager.list(context.cwd);
      return {
        message:
          sessions.length === 0
            ? "No sessions found."
            : sessions
                .map((session) => `${session.id}\t${session.updatedAt}\t${session.model}`)
                .join("\n"),
      };
    },
  });
  registry.register({
    name: "resume",
    usage: "/resume <id>",
    description: "Resume a session from the current workspace.",
    handler: async (context) => {
      if (context.args.length === 0 || context.args.includes(" ")) {
        return {
          error: { code: "COMMAND_USAGE", message: "Usage: /resume <id>" },
        };
      }
      const record = await context.manager.get(context.cwd, context.args);
      return record === undefined
        ? {
            error: { code: "SESSION_NOT_FOUND", message: `Unknown session: ${context.args}` },
          }
        : { action: { type: "resume", sessionId: record.id } };
    },
  });
  registry.register({
    name: "clear",
    usage: "/clear",
    description: "Start a new session without deleting history.",
    handler: async (context) =>
      context.args.length === 0
        ? { action: { type: "clear" } }
        : { error: { code: "COMMAND_USAGE", message: "Usage: /clear" } },
  });
  registry.register({
    name: "tools",
    usage: "/tools",
    description: "List enabled tools.",
    handler: async (context) => ({
      message: [
        "Enabled tools:",
        ...context.tools.map((tool) => `- ${tool.name}: ${tool.description}`),
      ].join("\n"),
    }),
  });
  registry.register({
    name: "context",
    usage: "/context",
    description: "List loaded project instruction files.",
    handler: async (context) => {
      const lines =
        context.projectContext.files.length === 0
          ? ["No project context files loaded."]
          : [
              "Active project context files:",
              ...context.projectContext.files.map((file) => `- ${file.path}`),
            ];
      if (context.projectContext.diagnostics.length > 0) {
        lines.push(
          "Project context diagnostics:",
          ...context.projectContext.diagnostics.map(
            (item) => `- [${item.code}] ${item.path}: ${item.message}`,
          ),
        );
      }
      return { message: lines.join("\n") };
    },
  });
  registry.register({
    name: "quit",
    usage: "/quit",
    description: "Exit the interactive session.",
    handler: async (context) =>
      context.args.length === 0
        ? { action: { type: "quit" } }
        : { error: { code: "COMMAND_USAGE", message: "Usage: /quit" } },
  });

  return registry;
}

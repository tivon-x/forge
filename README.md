# Forge

Forge is a provider-neutral coding agent CLI written in TypeScript.

The Phase 3 implementation provides persistent project sessions, project instruction discovery,
one-shot and line-oriented interactive modes, local slash commands, and workspace tools for
reading, writing, exact editing, and shell execution. The full roadmap is in
[`docs/forge-implementation-plan.md`](docs/forge-implementation-plan.md).

## Requirements

- Node.js 24
- pnpm 11

## Development

```bash
pnpm install
pnpm verify
```

## Usage

Set credentials and a model in the environment:

```powershell
$env:OPENAI_API_KEY = "..."
$env:OPENAI_MODEL = "..."
pnpm build
node dist/index.js
```

Running Forge without `--prompt` starts an interactive session. Use print mode for one task:

```bash
node dist/index.js -p "read package.json and explain this project"
```

The model can also be selected with `--model`:

```bash
node dist/index.js -p "fix the failing test and run it" --model <model>
```

### OpenAI-compatible APIs

Compatible services use the Chat Completions API. Set their separate credentials and base URL,
then select the provider explicitly:

```powershell
$env:OPENAI_COMPATIBLE_API_KEY = "..."
$env:OPENAI_COMPATIBLE_MODEL = "..."
$env:OPENAI_COMPATIBLE_BASE_URL = "https://example.com/v1"
node dist/index.js --provider openai-compatible -p "read package.json"
```

`--base-url` overrides `OPENAI_COMPATIBLE_BASE_URL`; `--model` overrides the model environment
variable. API keys are accepted only through environment variables.

Forge creates project-scoped sessions for agent and interactive runs. List and resume sessions with:

```bash
node dist/index.js sessions
node dist/index.js -p "continue the previous task" --resume <session-id>
node dist/index.js --resume <session-id>
```

### Project instructions

Forge finds the nearest Git project root, falling back to common project markers when no `.git`
entry exists. It loads instructions in this order:

1. `<project-root>/AGENTS.md`
2. each descendant-directory `AGENTS.md` down to the startup directory
3. `<startup-directory>/.forge/AGENTS.md`

Instructions are rebuilt into the system prompt whenever Forge starts or resumes. They are not
stored in session JSONL and do not change the startup-directory boundary used by file tools.
Instruction symlinks are rejected. Each file is limited to 100,000 bytes and all loaded instruction
files together are limited to 300,000 bytes.

### Interactive commands

The line-oriented interactive mode supports:

```text
/help
/sessions
/resume <session-id>
/clear
/tools
/context
/quit
```

Slash commands are handled locally and are never sent to the model or written as conversation
messages. `/clear` creates a new session without deleting existing history.

Terminal shortcuts use the same shell tool as model tool calls:

- `!cmd` executes the command and adds its structured result as explicitly untrusted data to the
  current session context.
- `!!cmd` executes the command without model credentials, provider creation, or session context.

Print mode supports three output protocols:

```bash
node dist/index.js -p "explain this project" --output text
node dist/index.js -p "explain this project" --output json
node dist/index.js -p "explain this project" --output transcript
```

- `text` is the default and prints only the final assistant response.
- `json` writes one agent event as JSON per stdout line for scripts.
- `transcript` streams assistant text to stdout and tool status to stderr.

Exit code `0` means the run completed, `1` means it failed, and `130` means it was cancelled.

File tools reject paths outside the startup directory. Shell commands start in that directory but
are not sandboxed and can access anything allowed by the current operating-system user. Command
approval and destructive-command policies are intentionally deferred to Phase 9.

See [`docs/phase-3-smoke-test.md`](docs/phase-3-smoke-test.md) for the current real API smoke test.

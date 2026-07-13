# Forge

Forge is a provider-neutral coding agent CLI written in TypeScript.

The Phase 2 implementation provides persistent project sessions, session resume, stable output
modes, and workspace tools for reading, writing, exact editing, and shell execution. The full
roadmap is in
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

Forge creates a project-scoped session for each run. List and resume sessions with:

```bash
node dist/index.js sessions
node dist/index.js -p "continue the previous task" --resume <session-id>
```

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

See [`docs/phase-2-smoke-test.md`](docs/phase-2-smoke-test.md) for the current real API smoke test.

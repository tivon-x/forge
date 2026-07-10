# Forge

Forge is a provider-neutral coding agent CLI written in TypeScript.

The Phase 1 implementation provides a one-shot OpenAI coding agent with streaming output and
workspace tools for reading, writing, exact editing, and shell execution. The full roadmap is in
[`docs/forge-implementation-plan.md`](docs/forge-implementation-plan.md).

## Requirements

- Node.js 24
- pnpm 11

## Development

```bash
pnpm install
pnpm typecheck
pnpm test
pnpm lint
pnpm build
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

Exit code `0` means the run completed, `1` means it failed, and `130` means it was cancelled.

File tools reject paths outside the startup directory. Shell commands start in that directory but
are not sandboxed and can access anything allowed by the current operating-system user. Command
approval and destructive-command policies are intentionally deferred to Phase 9.

See [`docs/phase-1-smoke-test.md`](docs/phase-1-smoke-test.md) for the real API smoke test.

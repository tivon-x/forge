# Phase 1 OpenAI Smoke Test

The default test suite uses fake providers and does not make network requests. Run this smoke test
manually with a valid OpenAI API key and a Responses API model available to that account.

## PowerShell

```powershell
$env:OPENAI_API_KEY = "..."
$env:OPENAI_MODEL = "..."
pnpm install --frozen-lockfile
pnpm smoke:openai
$smokeExit = $LASTEXITCODE
git status --short
$smokeExit
```

## POSIX shell

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="..."
pnpm install --frozen-lockfile
pnpm smoke:openai
smoke_exit=$?
git status --short
echo "$smoke_exit"
```

The test passes when Forge:

1. Calls `readFile` for `package.json`.
2. Prints a tool success status on stderr.
3. Prints a final answer containing the package name `forge` on stdout.
4. Does not modify the working tree.
5. Exits with code `0`.

The `git status` output must be empty and the saved smoke-test exit code must be `0`.

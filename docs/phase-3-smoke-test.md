# Phase 3 Project Context and Command Smoke Test

The default test suite uses fake providers and does not make network requests. The model prompts in
this manual test require a valid OpenAI API key and a Responses API model available to that account.
Local `/context`, `/help`, `/tools`, and terminal shortcut behavior is also covered deterministically
by Vitest.

## PowerShell

```powershell
$env:OPENAI_API_KEY = "..."
$env:OPENAI_MODEL = "..."
pnpm install --frozen-lockfile
pnpm verify

Push-Location docs
node ..\dist\index.js -p "/context"
Pop-Location

@'
/tools
!node -e "process.stdout.write('visible-context')"
What exact output did the user-run terminal command produce?
!!node -e "process.stdout.write('hidden-local-only')"
What output from the most recent terminal command is present in your context?
/sessions
/clear
/quit
'@ | node dist\index.js

node dist\index.js -p "Remember the phrase phase-three-resume."
$sessionId = ((node dist\index.js sessions | Select-Object -First 1) -split "`t")[0]
node dist\index.js -p "Repeat the phrase I asked you to remember." --resume $sessionId

$smokeExit = $LASTEXITCODE
git status --short
$smokeExit
```

## POSIX shell

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="..."
pnpm install --frozen-lockfile
pnpm verify

(cd docs && node ../dist/index.js -p "/context")

node dist/index.js <<'EOF'
/tools
!node -e "process.stdout.write('visible-context')"
What exact output did the user-run terminal command produce?
!!node -e "process.stdout.write('hidden-local-only')"
What output from the most recent terminal command is present in your context?
/sessions
/clear
/quit
EOF

node dist/index.js -p "Remember the phrase phase-three-resume."
session_id="$(node dist/index.js sessions | head -n 1 | cut -f 1)"
node dist/index.js -p "Repeat the phrase I asked you to remember." --resume "$session_id"

smoke_exit=$?
git status --short
echo "$smoke_exit"
```

The test passes when:

1. `/context` run from `docs` lists the repository-root `AGENTS.md`.
2. `/tools`, `/sessions`, `/clear`, and `/quit` execute locally without model responses.
3. The model reports `visible-context` after `!cmd`.
4. The model does not receive `hidden-local-only` after `!!cmd`.
5. The resumed one-shot run remembers `phase-three-resume`.
6. The final exit code is `0` and tracked project files remain unchanged.

Shell commands are not sandboxed. This smoke test uses harmless commands, but Forge does not have a
general command approval policy before Phase 9. Session files under `~/.forge/sessions` are expected
runtime data and are not part of the Git working tree.

# Phase 2 Session Smoke Test

The default test suite uses fake providers and does not make network requests. This manual test
requires a valid OpenAI API key and a Responses API model available to that account.

## PowerShell

```powershell
$env:OPENAI_API_KEY = "..."
$env:OPENAI_MODEL = "..."
pnpm install --frozen-lockfile
pnpm verify

node dist/index.js -p "Read package.json and remember the package name." --output transcript
$sessionId = ((node dist/index.js sessions | Select-Object -First 1) -split "`t")[0]
node dist/index.js -p "What package name did I ask you to remember?" --resume $sessionId
node dist/index.js -p "Report the package name without modifying files." --output json |
  ForEach-Object { $_ | ConvertFrom-Json | Out-Null }

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

node dist/index.js -p "Read package.json and remember the package name." --output transcript
session_id="$(node dist/index.js sessions | head -n 1 | cut -f 1)"
node dist/index.js -p "What package name did I ask you to remember?" --resume "$session_id"
node dist/index.js -p "Report the package name without modifying files." --output json |
  while IFS= read -r line; do printf '%s' "$line" | node -e 'JSON.parse(require("fs").readFileSync(0, "utf8"))'; done

smoke_exit=$?
git status --short
echo "$smoke_exit"
```

The test passes when:

1. `forge sessions` lists the newly created session for the current project.
2. The resumed run remembers the prior user and assistant messages.
3. Every stdout line from JSON mode parses as one JSON object.
4. The final exit code is `0`.
5. Forge does not modify tracked project files.

Session files under `~/.forge/sessions` are expected runtime data and are not part of the Git
working tree.

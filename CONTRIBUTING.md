# Contributing to Forge

Thanks for helping improve Forge. Forge is both a usable terminal coding agent and a teaching codebase for understanding how coding agents are built. Contributions should preserve that dual purpose: make the tool better while keeping the architecture small, readable, and easy to learn from.

## Project philosophy

Forge is organized around three layers:

```text
forge_agent   LangChain-native runtime, harness, events, and session primitives
forge_coding  project context, safe tools, providers, sessions, and resources
forge_cli     Typer CLI, renderers, formatting, and Textual TUI
```

The key boundary is:

```text
AgentHarness  = runtime event/concurrency facade
CodingSession = coding-domain session assembly and persistence
forge_cli     = product presentation/composition root
```

Please keep these principles in mind:

- **Small layers beat magic.** Each package should have one clear job.
- **Events are the contract.** The harness emits typed events; UI and renderers consume them.
- **The core stays portable.** `forge_agent` should not depend on the CLI, Textual, Rich, local config paths, or Forge-specific resource loading.
- **Tools are ordinary typed functions.** Prefer explicit schemas and structured results.
- **Sessions are durable and inspectable.** Avoid changes that make history hard to read, resume, or export.
- **Documentation follows implementation.** User-facing behavior and architectural decisions should be documented.

## Local development

Use `uv` for Python commands so they run in the project environment.

```bash
uv sync --dev --extra providers --locked
uv run forge --version
```

Run Forge from the checkout:

```bash
uv run forge
uv run forge -p "explain this repo"
```

## Checks before submitting

Run the relevant focused tests while developing, then run the full checks before opening a pull request when practical:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

There is no checked-in documentation-site build. Keep current user-facing
documentation in the Markdown files in this repository; do not create a
parallel `website/` tree without an explicit architecture decision.

## Where changes belong

Use the layer boundaries to decide where code should live:

- Provider integrations, model adapters, and provider configuration belong in `forge_coding.providers`.
- LangChain runtime behavior, Forge events, harnesses, and portable session primitives belong in `forge_agent`.
- CLI behavior, slash commands, renderers, and TUI integration belong in `forge_cli`.
- Project context, safe coding tools, sessions, resources, and prompt/profile
  assembly belong in `forge_coding`.
- Textual-specific code should stay behind the TUI layer.
- Rich rendering should not leak into the reusable agent harness.

If a change crosses layers, prefer adding a small typed boundary instead of importing app-specific details into core code.

## Adding a provider or model

The built-in provider catalog is data, not code: edit
`src/forge_coding/data/catalog.toml` and open a PR — no Python changes needed.
Each `[[providers]]` table declares the provider's name, kind
(`openai-compatible`, `anthropic`, or `openai-codex`), base URL, models,
default model, context windows, and thinking configuration. Validation happens
at load time, so a typo fails tests with a pointed error message.

For personal or unreleased providers, create `~/.forge/catalog.toml` with the
same schema — it is overlaid on the built-in catalog (your values win, models
are unioned) and needs no PR at all.

## Testing expectations

- Add or update tests for behavior changes.
- Use fake providers and fake tools for deterministic agent-loop tests.
- Keep core tests free of provider-specific assumptions.
- Add regression tests for bugs.
- Prefer focused tests that describe the behavior being protected.

## Documentation expectations

For substantial architectural or phase-oriented work, keep the decision in a
tracked Markdown document and explain:

- what changed
- why it exists
- how it maps to Forge's architecture
- how to test or use it

For user-facing behavior, update `README.md` or the relevant document under
`docs/`. The ignored `docs/dev/` directory contains historical local records;
it is not a current documentation or release source.

## Release process

The distribution name in `pyproject.toml` is `forge-ai`, but this repository is
not currently publishing it to PyPI. There is no checked-in release workflow;
do not infer publication from the distribution name or from an unrelated PyPI
project with the same name.

To prepare a future release, intentionally bump `[project].version` in
`pyproject.toml`, update the release notes, and add or follow an explicitly
approved publishing workflow. Until then, release status is a maintainer
decision, not an automatic CI result.

## Pull request guidelines

Good Forge pull requests are small, focused, and easy to review. Please include:

- the motivation for the change
- a summary of behavior changes
- tests or checks you ran
- screenshots or terminal output for TUI/CLI changes when useful
- notes about compatibility, migrations, config changes, or provider-specific behavior

Avoid unrelated refactors in feature or bug-fix PRs. If a larger design change is needed, open an issue or discussion first.

## Roadmap alignment

Forge is developed incrementally. For larger changes, check the current issues
and discussions in the repository configured as `origin`; do not use a copied
roadmap URL from another Forge project. Plans must identify their own status,
date, and commit baseline.

When in doubt, favor the smallest step that preserves the architecture and teaches the design clearly.

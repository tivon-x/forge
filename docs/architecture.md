# Forge architecture

Forge is one Python distribution with three packages. The split keeps the
runtime facts separate from coding-domain behavior and presentation concerns;
it does not add another runtime, loop, or persistence system.

```text
forge_cli  ->  forge_coding  ->  forge_agent
CLI/renderers/TUI   coding domain       runtime
```

## Package ownership

### `forge_agent`

The runtime owns LangChain message/model/tool integration, `AgentHarness`,
Forge product events, and session primitives. Production execution is the
official `create_agent()` graph streamed with `astream_events(version="v3")`.
The package must not import `forge_coding`, `forge_cli`, UI libraries, provider
SDKs, filesystem helpers, or shell commands.

### `forge_coding`

The coding layer owns project context, workspace-safe coding tools, provider
configuration/runtime construction, session domain behavior, resources, and
JSONL storage. It consumes `forge_agent` and does not import the CLI layer or
presentation libraries.

### `forge_cli`

The presentation layer owns the `forge` Typer entry point, print/transcript
renderers, Textual TUI, terminal-title behavior, and
`forge_cli.formatting`—pure formatting shared by renderers and TUI state. Its
`__init__` is intentionally lightweight and does not import Textual eagerly.
The console entry point is:

```toml
forge = "forge_cli.cli:app"
```

The wheel includes `forge_agent`, `forge_coding`, and `forge_cli`; users still
install and invoke one `forge-ai` distribution.

## Runtime and persistence

Model/tool execution and Forge event projection remain unchanged. The CLI and
TUI consume `AgentEvent` values, while JSONL sessions retain LangChain message
content, tool calls, artifacts, and metadata through the existing codec. Slash
commands remain presentation/session controls and do not enter the model
transcript. No second tool loop or LangGraph checkpointer is introduced.

## Attribution

Pi is the primary architecture and learning source. Tau is the Python product
baseline and attribution source for coding tools, sessions, provider
configuration, and the Textual TUI. Forge is independently maintained; Tau is
not an active package, CLI, or configuration name.

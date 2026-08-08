# Forge LangChain-Native 架构记录

状态：最终架构（迁移已完成，后续审查修复已收口）
基线：`main`（本记录维护时点）
前置决策记录：`plan.md`（历史删除计划，未重写）

本文档只描述 Forge 当前的最终架构。早期阶段记录、遗留协议描述和过时的
测试数字已合并删除；`plan.md` 保留当时的分阶段删除决策作为历史记录。

## 1. 最终架构

```text
AgentHarness queues
       │
       ▼
LangChain before_model middleware ── inject steering HumanMessage
       │
       ▼
create_agent(BaseChatModel, BaseTool[])
       │
       ▼
astream_events(version="v3")
       │
       ├── message/tool/value projections ──▶ AgentEvent ──▶ CLI/TUI
       │
       └── AnyMessage transcript ───────────▶ message_codec ──▶ JSONL
```

- 生产 agent/tool 循环只有 LangChain 官方 `create_agent()` +
  `astream_events(version="v3")`。`AgentHarness` 只负责 transcript、队列、
  取消和 Forge UI 事件 facade；没有第二套 provider/tool loop。
- 生产 Provider 构造位于 `forge_coding.provider_runtime`，唯一输出类型是
  `BaseChatModel`。`forge_ai` 已删除。
- 运行时消息只有 LangChain `AnyMessage`；`message_codec.py` 只做
  JSON 往返与显示文本提取，并负责 `ToolMessage` artifact 的持久化投影。
- Coding tools 是原生 `StructuredTool`（`ForgeStructuredTool`），通过
  `ToolRuntime[ForgeRuntimeContext]` 注入工作区与 shell 前缀，返回
  `(content, artifact)`。`ForgeStructuredTool.execute()` 是 Forge
  direct-execution seam（slash command 与直接工具执行用），生产循环走
  `ainvoke`/`ToolRuntime`。
- Steering 通过官方 `before_model` middleware 在每次模型调用前注入；
  follow-up 仍在 agent 正常结束后开始下一次 invocation。

## 2. 关键取舍

- **Event Streaming v3 仍为 experimental**：由 `uv.lock` 固定版本，并由
  Forge 契约测试（middleware reducer 假设、message/tool/value 投影形状）
  控制。上游契约变化时契约测试先失败，不得回退成 Forge 自建工具循环。
- **不用 LangGraph checkpointer 替代 JSONL Session**：append-only 历史、
  branch tree、label、model/thinking change、compaction、export/replay 都是
  Forge 产品语义；同时使用 JSONL 和 checkpointer 会形成两个权威来源。
- **`langchain-core` 不是 Forge 直接依赖**：由 `langchain` 传递安装
  （`langchain>=1.0,<2.0`）。lockfile、CI `--locked` 与包元数据测试固定该
  假设；上游变化时元数据测试明确失败。
- **`langchain-openai==1.4.1` 精确锁定**：Codex 使用其私有 experimental
  类 `_ChatOpenAICodex`；升级前先运行契约测试。
- **Codex 是 experimental 能力**：私有 import 只在 `provider_runtime`
  的 Codex factory；OAuth 凭据存 `~/.forge/`，不读 `~/.codex/`；
  固定官方 Codex endpoint，不接受调用方 base URL。

## 3. Forge 保留 / 删除的类型

保留（Forge 产品事件与持久化契约）：

- `ToolCall`、`AgentToolResult`、`ToolExecutor`、`ToolCancellationToken`；
- `AgentEvent` 系列（`MessageStart/Delta/End`、`ThinkingDelta`、
  `ToolExecutionStart/Update/End`、`QueueUpdate`、`TurnStart/End`、
  `Retry`、`Error`、`AgentStart/End`）；
- `message_codec.py`（`message_to_json` / `message_from_json` /
  `message_text` / artifact 投影）；
- `ForgeRuntimeContext`、Session entries、Provider catalog 与凭据存储。

已删除（迁移收口）：

- `forge_ai` 包、`forge_agent/{messages,provider,loop,compat}.py`、
  `forge_coding/compat.py`、`run_compat_agent`、`ForgeProviderChatModel`；
- `AgentTool` / `to_agent_tool()`；
- `AgentHarnessConfig.chat_model` fallback、`ClosableModelProvider` /
  `ClosableModel`、`is_langchain_message` / `to_langchain_message`
  identity 转换器、`_call_executor` 的签名反射；
- 旧 role-row JSONL 读取（项目无历史数据，接受删除）；
- `run_langchain_agent` 的 `stream_deltas` / `transcript_adapter` /
  `error_policy` 投影旋钮。

## 4. 事件投影契约

每次模型调用严格形成一组闭合生命周期：

```text
TurnStart -> MessageStart -> deltas -> MessageEnd -> TurnEnd
```

- 文本/reasoning delta 按 message id 记录；最终 `AIMessage` 有正文但没有
  chunk 时补发一次 `MessageDeltaEvent`（非 chunk 模型正文可见）。
- tool call AIMessage、ToolMessage 与最终 AIMessage 不共享未关闭的
  `MessageStartEvent`；取消与异常也闭合已开始的 turn。
- 工具参数 chunk（`AIMessageChunk.tool_call_chunks` / v3
  `tool_call_chunk` block delta）投影为 `ToolExecutionUpdateEvent`
  （`data.arguments_delta` 累积、`tool_name` 粘性）；`ToolExecutionStart`
  只在 finalized tool call 或 `tool-started` 发一次；partial JSON 不触发
  工具执行。
- Steering 在工具批次后的下一次 model call 前可见（middleware drain），
  队列 drain 后立即投影 `QueueUpdateEvent`。

## 5. 持久化与恢复

- Session JSONL 无损保存 LangChain Message（content blocks、tool_calls、
  usage/response metadata）。
- `ToolMessage` artifact 在持久化边界做 JSON-safe 投影：JSON-compatible
  原样保存；任意对象/bytes 替换为稳定占位
  `{"forge_serialization": {"status": "omitted", "python_type": ...}}`，
  不保存 repr()、原始 bytes 或对象字段。
- JSONL 只恢复 torn tail（文件不以换行结尾且最后一行 JSON 不完整）：
  `read_all()` 忽略该尾行，下一次 append 在文件锁内截断后再写；
  中间损坏与完整但 schema 错误的行继续抛 `SessionJsonlError`。
- `/new` 首次持久化先写 session/model/thinking，再写 message/leaf 并完成
  索引；`resume`/`new_session` 通过唯一 `_adopt_replacement()` 原子接管
  replacement 的全部运行状态，被淘汰的 Provider 立即关闭。
- Session `aclose()` 关闭所有创建过的官方 Provider，每个 client 恰好一次
  （`aclose()`、OpenAI `root_async_client.close()`、Anthropic
  `_async_client.close()`、Mistral `async_client.aclose()`），单个失败不
  泄漏其余 Provider。

## 6. 当前验证基线

默认测试离线、确定性；真实 Provider 冒烟为 opt-in，不进默认测试/CI。

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
uv run pytest
uv run forge --help
uv run forge --version
uv build
```

- `tests/test_architecture.py` 固定依赖方向（`forge_agent` 不 import
  `forge_coding`）与删除面扫描（禁止 `chat_model` fallback、
  `ClosableModel*`、identity message 转换器、旧协议名、`forge_ai`）。
- `tests/test_package_metadata.py` 固定 `langchain-core` 不由 Forge 直接
  依赖且由已解析的 `langchain` distribution 提供。
- Codex expiry 契约测试固定毫秒/秒换算；shell prefix sentinel 不得出现在
  artifact、Session、export 或测试快照。
- v3/Codex 的 experimental 警告为预期。

## 7. 参考

- LangChain Event Streaming：
  <https://docs.langchain.com/oss/python/langchain/event-streaming>
- LangChain Custom Middleware：
  <https://docs.langchain.com/oss/python/langchain/middleware/custom>
- 历史删除决策与阶段计划：`plan.md`
- 后续审查修复计划（已实施）：`docs/langchain-native-review-fix-plan.md`

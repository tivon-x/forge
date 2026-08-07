范围已经彻底摸清。关键事实：`AgentToolResult`/`ToolCall`/`ToolExecutor` 是**生产**的 Forge 工具事件契约（`langchain_runtime`/`tui/state`/`tools.py`/`session` 都在用），而 `AgentTool`（旧工具包装类）生产侧**无人构造**、只活在测试里；`forge_ai` 与 `forge_agent/{messages,provider,loop}`、两处 `compat` 构成一个相互引用的闭包，必须原子拆除。

------

# 遗留协议层删除计划（决策完整版）

## Building（做什么）

一次性把迁移计划第 5 阶段"保留"的遗留协议层**真正删除**：`forge_ai` 包的旧 Provider 协议、`forge_agent` 的旧消息/工具/Provider/旧循环、两个 `compat.py` 接缝、`ModelProvider | BaseChatModel` 联合类型面，以及它们专属的测试。生产路径变为**纯 LangChain-native**：`BaseChatModel` + `BaseTool` + `AnyMessage`，不残留任何旧协议。

## Not building（明确不做）

- **不删** `ToolCall` / `AgentToolResult` / `ToolExecutor` / `ToolCancellationToken`——它们是迁移计划 §3.2 明示保留的"Forge 产品事件/工具 artifact"类型，被 `langchain_runtime.py`、`tui/state.py`、`tools.py`、`session.py` 生产使用。删它们=重写整个 TUI 工具事件面，属于另一个计划。
- **不删** `message_codec.py` 本体——`message_text`/`to_langchain_message` 是全生产路径的核心（session/context_window/branch_summary/TUI/export）。只删其中的旧格式分支。
- **不删** `forge_agent/events.py` 的 `AgentEvent` 投影（生产 UI 事件），只把 `MessageEndEvent.message` 从 `AgentMessage | AnyMessage` 收紧为 `AnyMessage`。
- **不删** Tau 归属与许可证（README/NOTICE/LICENSE 的 Tau 引用必须保留）。

## 依赖关系（先看图，避免循环残留）

```
forge_ai (旧Provider) ──imports──▶ forge_agent/{messages,provider,tools,loop}
        ▲                                │
        │                                ▼
forge_coding/compat.py ◀──imports── forge_agent/compat.py ──▶ langchain_runtime (生产)
        │
        └──▶ session / branch_summary / tui/app (LoginRequiredProvider)
```

`forge_ai`、`forge_agent/{messages,provider,loop,compat}`、`forge_coding/compat.py` 构成**互相引用的闭包**——只能一个阶段原子拆除，不能拆成多个连续绿提交。**无环**：拆除后 `langchain_runtime → tools.py(ToolCall/AgentToolResult) → TUI/events` 保持单向。

## 阶段计划（每阶段独立可回滚、独立绿）

### Phase 1：共享基础设施从 `forge_ai` 搬出（纯搬迁，零行为变化）

**目的**：把"被生产真正使用、只是住错了包"的东西先搬走，为 Phase 2 原子拆除腾路。

| 动作                                | 目标                                                         |
| :---------------------------------- | :----------------------------------------------------------- |
| 新建 `forge_coding/http.py`         | 迁入 `forge_ai/http.py`（`create_async_client`、`get_json`、proxy 工具） |
| 新建 `forge_coding/http_errors.py`  | 迁入 `forge_ai/http_errors.py`                               |
| 新建 `forge_coding/provider_env.py` | 迁入 `forge_ai/env.py`（`AnthropicConfig`、`OpenAICompatibleConfig`、`DEFAULT_*` 常量） |
| 更新生产 import                     | `oauth.py`、`update_check.py`、`provider_config.py`、`cli.py` 改指新模块；`forge_ai` 原文件暂留（Phase 2 同批删除） |

**验证**：`ruff` + `mypy` + `pytest` 全绿；`grep -rn "forge_ai" src/forge_coding/oauth.py src/forge_coding/update_check.py src/forge_coding/provider_config.py src/forge_coding/cli.py` 无命中。
**回滚**：单 commit revert，无数据影响。

### Phase 2：原子拆除遗留协议闭包（核心大阶段）

**目的**：生产路径彻底无旧协议。**同时更新 AGENTS.md**（见下方矛盾说明）。

**删除文件（10 个 src + 2 个测试文件）**：

| 文件                                                         | 内容                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| `src/forge_ai/anthropic.py` `google.py` `mistral.py` `openai_compatible.py` `openai_codex.py` `fake.py` `events.py` `provider.py` `retry.py` `__init__.py` `py.typed` | 旧 Provider 协议实现 + ProviderEvent + FakeProvider + re-export |
| `src/forge_agent/provider.py` `loop.py` `messages.py`        | `ModelProvider`/`CancellationToken` 协议、`run_agent_loop`、旧消息模型 |
| `src/forge_agent/compat.py`                                  | `run_compat_agent` + `ForgeProviderChatModel` + 适配器       |
| `src/forge_coding/compat.py`                                 | `stream_legacy_provider_*` + `LoginRequiredProvider`         |
| `tests/test_forge_ai.py`（1824 行）`tests/test_agent_loop.py`（633 行） | 旧 Provider / 旧循环的专属测试                               |

**生产代码修改（13 处）**：

1. `harness.py`：删 `isinstance(provider, BaseChatModel)` 的 else 兼容路由（303–318），`provider` 收紧为 `BaseChatModel | None`，删 `langchain_tool` 兜底转换。
2. `session.py`：`provider` 收紧为 `BaseChatModel`；删 1584/1662 两处 `stream_legacy_provider_text` 分支（命名/压缩摘要），`UserMessage(...)` → `HumanMessage(...)`；删 `cast(ModelProvider, ...)`（1027/1104）。
3. `branch_summary.py`：`provider` 收紧为 `BaseChatModel`，删 `stream_legacy_provider_final` 分支。
4. `cli.py`：`run_print_mode` 的 `provider` 收紧为 `BaseChatModel`，删 `ModelProvider` import。
5. `provider_runtime.py`：新增原生 `LoginRequiredChatModel(BaseChatModel)`（`_generate` 抛 `RuntimeError(startup_message)`），替换 TUI 登录占位；`tui/app.py:4328` 改用它，删 `forge_coding.compat` import。
6. `tui/app.py:2346`：`isinstance(message, UserMessage | HumanMessage)` → `HumanMessage`；删 `forge_agent.messages` import。
7. `events.py`：`MessageEndEvent.message: AgentMessage | AnyMessage` → `AnyMessage`。
8. `message_codec.py`：删 `UserMessage/AssistantMessage/ToolResultMessage/ToolCall` 旧格式分支，`message_from_json` 只接受原生 LangChain 消息 dict。
9. `forge_agent/__init__.py`：删 `AgentMessage/AgentTool/...` 遗留导出。
10. `pyproject.toml`：`packages` 去掉 `"src/forge_ai"`。
11. `AGENTS.md`：**同 commit 更新**——把"old protocol modules are compatibility-only seams"改写为"遗留协议已删除，生产路径仅 LangChain-native"（迁移计划 §14 曾要求文档与实现同 commit，此处同理）。

**测试改写（8 个文件）**：`test_agent_harness.py`（删 FakeProvider/ModelProvider 路径）、`test_langchain_runtime.py`（删 `run_compat_agent` 2 个用例，native 用例保留）、`test_cli.py`、`test_coding_session.py`、`test_http.py`（改测 `forge_coding/http.py`）、`test_agent_types.py`、`test_migration_regressions.py`（删 `run_compat_agent`/`ForgeProviderChatModel` 断言，架构测试改为断言仓库无 `forge_ai`/`forge_agent.messages`/`loop`/`compat` 残留）。

**验证**：`grep -rn "forge_ai\|ModelProvider\|AgentMessage\|run_agent_loop\|run_compat_agent\|compat" src/forge_agent src/forge_coding --include="*.py"` 除 `ForgeRuntimeContext` 等白名单外零命中；全部门禁绿。
**回滚**：git revert 本 commit（代码层）。**副作用提示**：此 commit 起旧格式 JSONL 不可读（用户已确认无旧数据，接受）。

### Phase 3：`AgentTool` 遗留面收口（独立、可选并入 Phase 2）

**目的**：`AgentTool` 生产无人构造（仅 `to_agent_tool` 声明 + 测试构造），从 `BaseTool | AgentTool` 联合收紧为 `BaseTool`。`ToolCall`/`AgentToolResult`/`ToolExecutor` 按 Not building **保留**。

| 文件                                                         | 修改                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| `forge_agent/tools.py`                                       | 删 `AgentTool` 类（保留 `ToolCall`/`AgentToolResult`/`ToolExecutor`/`ToolCancellationToken`） |
| `forge_coding/tools.py`                                      | 删 `to_agent_tool` + `AgentTool` import（148–149）           |
| `commands.py` `system_prompt.py` `context_window.py` `tui/widgets.py` `session.py` `harness.py` | `BaseTool | AgentTool` → `BaseTool`，`context_window.py:165` 删 `isinstance(tool, AgentTool)` 分支 |
| 测试                                                         | `test_system_prompt.py`、`test_agent_types.py` 的 `AgentTool(...)` → `StructuredTool(...)` |

**验证**：`grep -rn "AgentTool\b" src --include="*.py"` 零命中；门禁绿。
**回滚**：git revert。

### Phase 4：文档收口与最终门禁

1. `docs/langchain-native-migration-plan.md`：追加"遗留边界已删除"小节（状态更新、`forge_ai` 移除说明），更新 §5.1/§12/§16 相关描述。
2. `README.md`：架构描述去掉"兼容层"表述（Tau 归属行保留）。
3. 最终验证：`uv run ruff check src tests`、`uv run ruff format --check src tests`、`uv run mypy`、`uv run pytest`、`uv run forge --help`、`uv run forge --version`、`uv build` + 隔离 venv 装 wheel 冒烟。

## 关键决策（3 条）

1. **原子拆除而非渐进**：闭包依赖决定了 `forge_ai` + `forge_agent/{messages,provider,loop}` + 两处 `compat` 必须同 commit 删除，否则中间态 mypy 必然红。Phase 1 的搬迁把这个原子块变小，但块本身不可再拆。
2. **`LoginRequiredProvider` 用原生替代**：不删除 TUI 登录前占位能力，改为 `provider_runtime.py` 里的 `LoginRequiredChatModel(BaseChatModel)`，占位能力保持、旧 Provider 协议消失。
3. **`AgentToolResult`/`ToolCall` 保留**：它们是 Forge 产品事件契约（迁移计划 §3.2/§8 明示保留），删除会波及 `langchain_runtime`/TUI 全套事件面，收益不成比例。

## 被否决的替代方案

- **整体保留**（现状）：你已明确"无生产、无旧数据"，纯维护面保留没有收益，用户困惑（"为什么还有 tau 代码"）会持续。
- **就地内联**（把 http/env 代码塞进 oauth/update_check/provider_config 而不是新建模块）：会放大各自文件、且 provider_env 被 4 处共享，独立模块更干净。

## 风险与前提

- **最脆弱假设**：无外部消费者 import `forge_ai`/`forge_agent.messages`（`pyproject` 当前把它们打包进 wheel，属公开 API）。**若此假设不成立**（有下游依赖 0.1.5 的 `forge_ai`），删除即 breaking change。你已确认"没有上生产"→ 接受。若想稳一点，可在 Phase 2 前先发一版仅含 Phase 1 的 release 通知。
- **旧 JSONL**：Phase 2 后旧格式 Session 不可读（无数据，接受）。
- **改动面**：**远超过 8 个文件**（Phase 2 约 23 个 src + 10 个测试文件），已明确列出，无隐藏项。

## 验证命令（每阶段通用 + 最终）

```
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
uv run pytest
uv run forge --help && uv run forge --version
# 仅 Phase 4：uv build && 隔离 venv 安装 wheel 冒烟
```

## 矛盾点说明（按要求明示）

- **AGENTS.md §forge_agent** 现有规则"old protocol modules are compatibility-only seams for historical tests and JSONL reads"与 Phase 2 删除冲突 → 解决方式：Phase 2 同一 commit 内改写该规则（迁移计划 §14 的先例）。
- **迁移计划 §12 完成条件"旧 Session 可读取"** 与删除 `message_codec` 旧格式分支冲突 → 解决方式：Phase 4 更新文档，标注该条件因"无历史数据"而退役，并在文档中记录决策。

------

计划已决策完整、无占位符。**批准后**我会按 Phase 1 → 4 逐阶段实施，每阶段独立 commit、独立跑门禁。
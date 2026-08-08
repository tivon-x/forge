# Forge LangChain-native 审查修复计划

状态：已完成（四个 Phase 均已实施、独立提交并通过门禁）

基线：`main` / `2ab283a`

来源：2026-08-08 LangChain-native 迁移完成后独立代码审查

## 1. 结论

Forge 的结构迁移已经完成：生产路径只有 LangChain `create_agent()`、
`astream_events(version="v3")`、`BaseChatModel`、`BaseTool` 和 LangChain Message。
本计划不重新设计 runtime，而是修复审查发现的正确性、安全、持久化和事件语义问题，
并删除迁移完成后仍留在源码、测试组织和现行文档中的过渡脚手架。

完成本计划后，项目才能把 `plan.md` 和
`docs/langchain-native-migration-plan.md` 标记的迁移目标视为真正收口。

## 2. 已确认决策

### Building

- 修复 Codex OAuth expiry 毫秒/秒错误。
- 禁止把 `shell_command_prefix` 原文写入 ToolMessage artifact、Session、导出或 trace。
- 修复 `/new`、`resume` 的 Session 状态接管和 Provider 所有权。
- 让官方 Provider 的 HTTP client 能被确定性关闭。
- 让任意合法 LangChain artifact 不再中断 Session 持久化。
- 恢复被截断的 JSONL 尾行，同时继续拒绝中间损坏。
- 使用 LangChain 官方 middleware 在下一次模型调用前注入 steering。
- 修复非 chunk 模型正文丢失、消息生命周期错配和工具参数流缺失。
- 删除剩余迁移脚手架，并把测试从“迁移回归集合”归位到对应模块。
- 重写现行迁移文档，使其只描述最终架构；保留 `plan.md` 作为历史决策记录。

### Not building

- 不引入自定义 `StateGraph`、checkpointer、第二套 provider/tool loop 或第二语言 runtime。
- 不恢复旧 Forge message/provider/tool 协议，不恢复旧 JSONL 格式读取。
- 不删除 `ToolCall`、`AgentToolResult`、`ToolExecutor`、`ToolCancellationToken`、
  `AgentEvent` 或 `message_codec.py`；它们仍是 Forge 产品事件和持久化契约。
- 不删除 `ForgeStructuredTool.execute()`；slash command 和直接工具执行仍需要它，
  但要把“compatibility helper”改成准确的 Forge direct-execution seam 描述。
- 不把 `langchain-core` 加入直接依赖。Forge 通过 `langchain` 获得它，
  这是项目明确选择，不按本地 skill 的通用建议执行。
- 不在本计划内发布版本、推送分支或运行真实 Provider 冒烟。

## 3. 最终边界

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

`AgentHarness` 仍只负责 transcript、队列、取消和 Forge UI 事件；
工具循环和下一步模型调度仍全部由 `create_agent()` 执行。

官方依据：

- LangChain Event Streaming：<https://docs.langchain.com/oss/python/langchain/event-streaming>
- LangChain Custom Middleware：<https://docs.langchain.com/oss/python/langchain/middleware/custom>

## 4. Phase 1：凭据与敏感信息修复

目标：先消除会导致实验 Codex 完全不可用和凭据落盘的两个 P1。

### 4.1 Codex expiry

修改 `src/forge_coding/provider_runtime.py`：

1. 将 Codex token provider 从 factory 内部类提取为模块私有类，允许直接契约测试。
2. `OAuthCredential.expires` 保持毫秒存储；构造 `_ChatGPTToken.expires_at` 时除以
   `1000` 后再传给 `datetime.fromtimestamp(..., tz=UTC)`。
3. 不改变 `oauth.py` 的毫秒比较、刷新阈值和 credential JSON 格式。

回归测试放入 `tests/test_provider_runtime.py`：

- 未来一小时的毫秒 expiry 能通过同步和异步 token 获取路径；
- 生成的 datetime 与原毫秒时间相差不超过 1 ms；
- 过期凭据仍触发 refresh；
- 不读取真实 credential store，不访问网络。

### 4.2 shell prefix 脱敏

修改 `src/forge_coding/tools.py`：

1. `_apply_runtime_context()` 不再复制 `context.shell_command_prefix`。
2. artifact 只记录 `shell_command_prefix_applied: bool`；该布尔值只对 bash 有意义，
   read/write/edit 不增加无关字段。
3. workspace root、session id 只保留确有 UI/诊断消费者的字段；没有消费者的字段一并删除，
   避免把 `ToolRuntime.context` 当作通用转储对象。

测试覆盖：

- 使用 `export FORGE_SENTINEL=not-a-real-secret` 作为假值；
- 工具 artifact、Session JSONL、JSONL export 和 HTML export 均不得包含 sentinel；
- bash 结果仍能判断前缀是否已应用；
- 不读取或构造真实 API key。

### Phase 1 验收

```powershell
uv run pytest tests/test_provider_runtime.py tests/test_coding_tools.py tests/test_coding_session.py tests/test_session_export.py
uv run ruff check src tests
uv run mypy
```

建议提交：`fix(security): prevent credential leakage in native runtime`

## 5. Phase 2：Session、artifact 与 Provider 生命周期

目标：修复会话树损坏、Provider 泄漏和 append-only Session 的恢复缺口。

### 5.1 原子接管 replacement

修改 `src/forge_coding/session.py`：

1. 新增唯一的异步 `_adopt_replacement(replacement)`，替代 `resume()` 和
   `new_session()` 中两段逐字段复制。
2. 必须接管：config、state、harness、last parent、pending initial entries、资源、
   command registry、provider settings、runtime provider config、resource paths、
   compaction/thinking 状态、credential store、diagnostics 和 owned providers。
3. 接管成功后关闭不再使用的旧 Provider；若 replacement 构造或校验失败，原 Session
   保持原状。
4. `/new` 首次持久化必须先写 `session/model/thinking`，再写 message/leaf 并完成索引。

回归测试放入 `tests/test_coding_session.py`：

- 从已 flush 的 Session `/new`；
- 从仍有 pending metadata 的 Session `/new`；
- resume 到相同/不同 Provider；
- 每个 MessageEntry 的 parent 均存在；
- 新 Session 首次 prompt 后能从 `SessionManager` 恢复；
- replacement 失败时旧 Session 和 Provider 仍可用。

### 5.2 Provider 关闭

修改 `src/forge_coding/provider_runtime.py` 和 Session 生命周期：

1. `aclose_model()` 明确覆盖当前支持的官方对象：
   `aclose()`、OpenAI `root_async_client.close()`、Anthropic `_async_client.close()`、
   Mistral `async_client.aclose()`；同一个 client 最多关闭一次。
2. 缺少关闭接口时安全 no-op；关闭一个 Provider 失败时继续关闭其余 Provider，
   最后汇总或记录异常。
3. `resume/new_session` 转移 owned provider；被 replacement 淘汰的 Provider 立即关闭。
4. 暂不把同步的 model/provider 选择 API 全部改成 async；该范围单独决策。
   当前阶段保证 Session `aclose()` 最终能关闭所有创建过的 Provider。

测试使用假 client，不访问 Provider 网络，并逐个覆盖 OpenAI、Anthropic、Mistral、
通用 `aclose()` 和重复关闭。

### 5.3 artifact 持久化降级

修改 `src/forge_agent/message_codec.py`：

1. 原生 LangChain Message 仍是内存事实，不限制第三方 BaseTool 的 artifact 类型。
2. 持久化前只对 ToolMessage artifact 做 JSON-safe 投影。
3. JSON-compatible artifact 原样保存；无法序列化的对象或 bytes 替换为稳定占位：
   `{"forge_serialization": {"status": "omitted", "python_type": "..."}}`。
4. 占位不保存 `repr()`、原始 bytes 或对象字段，避免再次泄漏敏感信息。
5. content、tool_call_id、name、status、response metadata 和 usage metadata 必须保留。

测试覆盖普通 dict、Pydantic 数据、任意对象、UTF-8 bytes、非 UTF-8 bytes，以及重载后
tool-call 配对不变。

### 5.4 JSONL 尾行恢复

修改 `src/forge_agent/session/storage.py` 与 `jsonl.py`：

1. 只把“文件不以换行结尾且最后一行 JSON 不完整”视为可恢复的 torn tail。
2. `read_all()` 忽略该尾行并返回之前的完整记录；中间损坏、完整但 schema 错误的末行
   继续抛 `SessionJsonlError`。
3. 下一次 append 前在文件锁内截断 torn tail，再写入完整新行，避免损坏永久扩散。
4. 不自动改写其他历史记录，不吞掉中间错误。

测试覆盖空文件、正常尾换行、半行 JSON、最后一行完整但非法、中间损坏和恢复后继续追加。

### Phase 2 验收

```powershell
uv run pytest tests/test_coding_session.py tests/test_session.py tests/test_session_export.py tests/test_provider_runtime.py
uv run ruff check src tests
uv run mypy
```

建议提交：`fix(session): make native state adoption durable`

## 6. Phase 3：steering 与 v3 事件语义

目标：保留唯一 `create_agent` runtime，同时恢复 Forge 交互契约。

### 6.1 SteeringMiddleware

在 `forge_agent` 内新增一个小型 LangChain middleware，不创建自定义 graph：

1. middleware 使用官方 `before_model` hook，在每次模型调用前检查并 drain
   `AgentHarness` steering queue。
2. drain 出来的 `HumanMessage` 通过 middleware state update 追加到 LangChain
   `messages`；它同时进入 Harness transcript 和 Session JSONL。
3. steering 在当前正在执行的 model/tool call 中不抢占；工具批次完成后的下一次
   model call 必须看到它。
4. follow-up 不进入 middleware，仍只在 agent 正常结束后开始下一次 invocation。
5. queue drain 后立即投影 `QueueUpdateEvent`，TUI 不得一直显示已消费消息。
6. middleware 与 `ModelCallLimitMiddleware` 同时启用；steering 不额外消耗或重置
   已完成的 model-call 计数。

最脆弱假设：当前 LangChain `before_model` 返回的 `messages` state update 会使用消息 reducer
稳定追加并出现在 v3 values 投影中。若升级后契约改变，middleware 契约测试必须先失败；
不得回退成 Forge 自建工具循环。

### 6.2 每条模型消息独立生命周期

重构 `src/forge_agent/langchain_runtime.py` 的投影状态：

1. 不再用全局 `streamed_ids` 判断正文是否已经输出；按 message id 记录“实际产生过的
   text/reasoning/tool-call delta”。
2. 最终 `AIMessage` 有正文但没有 chunk 时，补发一次 `MessageDeltaEvent`。
3. 每个 model call 必须严格形成一组
   `TurnStart -> MessageStart -> deltas -> MessageEnd -> TurnEnd`。
4. tool call AIMessage、ToolMessage 和最终 AIMessage 不得共享一个未关闭的
   `MessageStartEvent`；取消和异常也要闭合已开始的生命周期。
5. 累计 values 仍按稳定 message id 去重；测试同时固定 LangChain reducer 会给缺失 id
   的消息补 UUID 的假设。

### 6.3 工具参数流

1. 消费 `AIMessageChunk.tool_call_chunks` / v3 `message.tool_calls`，累积部分参数。
2. 使用现有 `ToolExecutionUpdateEvent` 投影参数增量，不新增第二套工具事件类型；
   `data` 固定包含 `arguments_delta` 和已知时的 `tool_name`。
3. `ToolExecutionStartEvent` 只在 finalized tool call 或 `tool-started` 时发一次；
   partial JSON 不提前执行工具。
4. CLI/TUI 对参数流可选择展示，但必须容忍 Provider 不提供 chunk。

回归测试放入 `tests/test_langchain_runtime.py`、`test_agent_harness.py` 和
`test_coding_session.py`：

- 合法非 chunk BaseChatModel 的正文可见；
- 正常流式模型不重复正文；
- tool call + final answer 产生两套完整 model-call 生命周期；
- blocking tool 期间 steering，第二次模型调用已经包含 steering；
- 多工具并行批次只 drain 一次；
- follow-up 仍在最终回答后执行；
- one-at-a-time/all queue mode 保持；
- 工具参数 chunk、无 chunk、无 id 和 malformed partial JSON 均不崩溃。

### Phase 3 验收

```powershell
uv run pytest tests/test_langchain_runtime.py tests/test_agent_harness.py tests/test_coding_session.py tests/test_tui_adapter.py tests/test_tui_app.py
uv run ruff check src tests
uv run mypy
```

建议提交：`fix(agent): restore native steering and event boundaries`

## 7. Phase 4：删除迁移脚手架并重写文档

目标：不是把旧描述挪到文档末尾，而是让生产源码、测试组织和现行架构文档只有一套事实。

### 7.1 删除源码残余

删除或收紧以下内容：

| 位置 | 动作 |
|---|---|
| `AgentHarnessConfig.chat_model` | 删除，只保留必填 `provider: BaseChatModel` |
| `ClosableModelProvider` | 删除，所有权集合直接使用 `BaseChatModel` |
| `is_langchain_message()` | 删除，当前未使用且类型已经唯一 |
| `to_langchain_message()` | 删除，runtime/helper 直接使用 `AnyMessage` |
| `_call_executor()` 的签名反射 | 删除；所有 `ToolExecutor` 统一接收 `context` |
| “legacy-style / compatibility helper / provider-neutral messages” | 改成当前 LangChain-native 或 Forge direct-execution 语义 |

`ForgeStructuredTool.execute()`、`message_text()`、`message_to_json()`、
`message_from_json()` 保留。

### 7.2 测试归位

1. 将 `tests/test_migration_regressions.py` 中的用例移动到对应模块测试；
   只把依赖方向和删除面扫描放入 `tests/test_architecture.py`。
2. `tests/fake_native.py` 改名为 `tests/fake_models.py`，删除对已移除 `forge_ai`
   的说明；fake 只描述它模拟的 LangChain model 行为。
3. 架构测试禁止生产源码再次出现：`chat_model` fallback、`ClosableModelProvider`、
   identity message converter、旧 provider/tool/message/loop/compat 名称。

### 7.3 文档重写

1. 重写 `docs/langchain-native-migration-plan.md`：
   - 删除所有把旧协议、compat、旧 JSONL 描述成当前状态的段落；
   - 合并重复的阶段完成记录和过时测试数字；
   - 保留最终架构、关键取舍、已删除内容和当前验证基线；
   - 明确 v3 event streaming 仍为 experimental，并由 lockfile 与契约测试控制。
2. `plan.md` 保留为历史删除决策，在末尾追加“后续审查修复”链接，不重写当时的决策过程。
3. 检查 README、AGENTS.md、CLI help，只允许 Tau/Pi 归属和最终 LangChain-native 架构。

### 7.4 依赖决策

用户已明确覆盖通用 skill 建议：

- `langchain-core` 不写入 `pyproject.toml`，只能由 `langchain` 传递安装；
- 现行文档不得再声称它是 Forge 的直接核心依赖；
- 将 `langchain` 约束收紧为 `>=1.0,<2.0`，继续使用 `uv.lock` 和 CI `--locked`；
- 保持 `langchain-openai==1.4.1`，因为 Codex 使用其私有 experimental 类；
- 增加包元数据测试，确认当前已解析的 `langchain` distribution 声明依赖
  `langchain-core`，让该项目假设在上游变化时明确失败。

这项决定与项目内 `langchain-dependencies` skill 的“Python 显式安装
langchain-core”建议冲突；本计划以用户的项目级决定为准，并用 `<2.0`、lockfile、CI 和
元数据测试承担风险。

### Phase 4 验收

```powershell
uv lock --check
uv run pytest tests/test_architecture.py tests/test_package_metadata.py
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
uv run pytest
uv run forge --help
uv run forge --version
uv build
```

额外检查：

```powershell
rg -n "chat_model|ClosableModelProvider|to_langchain_message|is_langchain_message|legacy-style" src tests
rg -n "旧 .*保留|compatibility-only|native_messages" docs README.md AGENTS.md
```

第一条必须零命中。第二条只允许命中 `plan.md` 的历史决策和迁移文档中明确标注的
“已删除”事实，不能描述为当前兼容能力。

建议提交：`refactor(agent): remove migration scaffolding`

文档可同提交更新；若代码与文档改动过大，则紧随一个独立
`docs: finalize langchain-native architecture record` 提交。

## 8. 完整验收标准

以下条件必须同时满足：

1. 四个 Phase 各自独立绿提交；任何 Phase 停止后项目仍可运行。
2. Codex token provider 毫秒 expiry 契约测试通过。
3. shell prefix sentinel 不出现在 artifact、Session、export 或测试快照。
4. `/new` 与 `resume` 的所有 entry parent 存在，新 Session 可被恢复。
5. 所有创建过的官方 Provider 在 Session 关闭时恰好关闭一次。
6. 非 JSON artifact 和 torn JSONL tail 不再使完整 Session 丢失。
7. steering 在工具批次后的下一次 model call 前可见，follow-up 语义不变。
8. 每次 model call 具有完整、成对的 Forge turn/message 生命周期。
9. 非 chunk、chunk、reasoning、tool-call chunks 均有离线回归测试。
10. 生产源码没有第二套 loop、旧协议或迁移脚手架。
11. 文档不存在“旧兼容层仍保留”的当前时态描述。
12. `langchain-core` 不出现在 Forge 直接依赖中，但 lockfile 和元数据测试证明它由
    `langchain` 提供。
13. Ruff、format、mypy、pytest、CLI、build、隔离 wheel 安装全部通过。
14. 最终再执行一次独立只读 code review，并修复所有可处理的 P1/P2。

## 9. 预计影响范围

本计划预计修改 12～18 个文件，超过 8 个文件，主要集中在：

- `src/forge_agent/{harness,langchain_runtime,message_codec,tools}.py`
- `src/forge_coding/{provider_runtime,session,tools}.py`
- `src/forge_agent/session/{storage,jsonl}.py`
- CLI/TUI adapter 的事件消费代码
- 对应测试、`pyproject.toml`、`uv.lock`、README/AGENTS/两份计划文档

不新增服务、数据库、运行时或外部账号依赖。

## 10. 风险与回滚

- **Steering middleware 风险最高**：v3 和 middleware state update 都依赖 LangChain
  当前契约。必须用真实 `create_agent` + 离线 fake model 做契约测试，不能只 mock
  `run_langchain_agent`。
- **artifact 降级是有意的信息损失**：只丢弃无法安全 JSON 化的 artifact，工具正文和
  配对字段不丢；不能使用 `repr()` 兜底。
- **JSONL 恢复只处理 torn tail**：中间损坏仍应失败，避免把真实数据错误静默吞掉。
- **源码脚手架删除可能影响外部 import**：项目尚未发布生产版本，接受 0.x 阶段的
  breaking cleanup，不保留 deprecated shim。
- 每个 Phase 可通过 revert 自身提交回滚；不需要 Session 数据迁移。Phase 2 的尾行恢复
  逻辑首次修复文件时会丢弃不可解析的末尾片段，但保留此前所有完整记录。

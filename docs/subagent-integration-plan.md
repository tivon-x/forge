# Forge Subagent 集成计划

状态：已实现
范围：后端运行时、Coding 组装、CLI/TUI 展示、测试与文档
交付方式：一个原子里程碑；不拆成“后端先合并、TUI 以后补”的半成品

## 1. 结论

Forge 的第一版 subagent 采用一个原生 LangChain `task(agent, instruction)` 工具：

- 父代理仍由现有 `langchain.agents.create_agent()` 创建，并继续通过 `astream_events(version="v3")` 运行。
- 每次 `task` 调用临时创建一个无状态子代理；子代理完成后，只把最终结果返回给父代理。
- 不引入 Deep Agents、不编写自定义 `StateGraph`、不增加 checkpointer，也不建立第二套模型/工具循环。
- 第一版只有同步、进程内、串行执行；没有后台任务、并行 fan-out、链式工作流、递归委派和独立子会话。
- 内置 `scout`、`worker`、`reviewer` 三个角色，定义保持在 Python 中，不做配置发现或 Markdown agent DSL。
- TUI 在现有对话流中显示可折叠的 subagent 任务块，不增加侧栏、任务中心或新的焦点模式。

这是最小但完整的教学实现：它展示 LangChain 的 agent-as-tool、上下文隔离、`BaseTool`、`ToolRuntime`、middleware、artifact 和 v3 嵌套事件，又不把 Forge 变成编排框架。

## 2. 要解决的问题

父代理擅长维持用户对话和做最终决策，但代码侦察、独立实现和复核会快速挤占主上下文。Subagent 的核心价值不是“多开几个模型”，而是把一个有明确边界的任务放进新的上下文窗口，只把结论带回主会话。

第一版必须同时满足：

1. 父会话是唯一的用户会话和持久化事实源。
2. 子代理能复用当前 provider、model、项目上下文和安全工具。
3. 子代理的中间消息不能污染父 transcript。
4. 用户能在 TUI 中看到任务是否排队、正在做什么、是否完成。
5. 取消、失败、恢复历史会话时，父消息与 `ToolMessage` 配对仍然正确。

## 3. 明确不做

- 不依赖 `deepagents`。它自带 planning、filesystem、memory 和 subagents，会与 Forge 已有 harness、工具和会话模型重叠。
- 不把父循环改成自定义 LangGraph。
- 不支持后台运行、任务列表、恢复子代理、跨进程执行或 worktree 隔离。
- 不支持一个 `task` 调用提交多个任务，也不承诺并行执行。
- 不允许子代理再次调用 `task`。
- 不保存完整子代理 transcript、thinking 或独立 JSONL。
- 不增加 `/subagent` 命令；用户自然描述任务，由父模型决定是否调用工具。
- 不在第一版开放项目级/用户级自定义角色配置。
- 不把 `scout`、`reviewer` 的提示词约束描述成安全沙箱；`bash` 本身仍可能修改工作区。

## 4. 参考来源与取舍

- [LangChain Subagents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents)：采用 supervisor 把 subagent 当工具调用、每次调用无状态、父代理保存会话记忆的基本模式。
- [LangChain Middleware](https://docs.langchain.com/oss/python/langchain/middleware/overview)：继续使用 `create_agent` 内建 middleware 限制子代理模型调用次数，不增加外层循环。
- [LangChain Multi-agent](https://docs.langchain.com/oss/python/langchain/multi-agent)：把 Deep Agents 视为完整 harness 方案；Forge 不采用，因为现有产品层职责已经完整。
- [pi-subagents](https://github.com/nicobailon/pi-subagents)：吸收自然语言委派、聚焦角色、紧凑进度、递归默认关闭、结果截断这些设计；不复制其后台任务、并行链、任务舰队、session sharing、worktree 和工作流系统。

## 5. 总体架构

```text
User
  │
  ▼
ForgeTuiApp / CLI renderer
  │ consumes AgentEvent
  ▼
CodingSession ─────────────── JSONL parent session
  │ composes                       ▲
  ▼                                │ only parent messages
AgentHarness                       │ and task ToolMessage
  │                                │
  ▼                                │
parent create_agent() ──calls──> task(agent, instruction)
  │                                │
  │ v3 root events                 ▼
  │                         SubagentRunner + lock
  │                                │
  │                                ▼
  │                         child create_agent()
  │                         fresh messages=[instruction]
  │                                │
  └──── v3 nested events <─────────┘
           │
           ├─ child messages/values: suppress
           └─ lifecycle/tool activity: project to parent task updates
```

包依赖方向保持不变：

```text
forge_cli  →  forge_coding  →  forge_agent  →  LangChain
```

`forge_agent` 只提供通用 subagent 运行原语；角色、项目提示词和 coding 工具组合属于 `forge_coding`；展示属于 `forge_cli`。

## 6. 后端设计

### 6.1 对模型暴露的工具

工具名固定为 `task`，输入严格校验：

| 字段 | 类型 | 规则 |
| --- | --- | --- |
| `agent` | `scout \| worker \| reviewer` | 必填，未知角色在执行前拒绝 |
| `instruction` | `str` | 必填，去除首尾空白后不能为空 |

工具描述直接列出三种角色的用途，并明确一次调用只能完成一个有边界的任务。父模型不需要学习新的命令或协议。

### 6.2 内置角色

| 角色 | 用途 | 可见工具 | 行为约束 |
| --- | --- | --- | --- |
| `scout` | 定位代码、理解调用链、收集证据 | `read`, `bash` | 只调查，不写文件；输出文件和符号位置、结论与不确定项 |
| `worker` | 完成边界明确的实现 | `read`, `write`, `edit`, `bash` | 只修改任务范围内文件；运行针对性验证；报告修改与风险 |
| `reviewer` | 独立检查现有改动或方案 | `read`, `bash` | 不写文件；先给可执行问题，再给剩余风险 |

所有子代理都看不到 `task`，因此不能递归委派。`scout` 和 `reviewer` 不拥有 `write`/`edit`，但它们的 `bash` 约束只是能力收窄，不是安全边界；这一点必须在架构文档和代码注释中如实保留。

当调用方显式传入 `CodingSessionConfig.tools` 时，该列表仍是可用能力的上限：`worker` 获得其中除 `task` 外的全部工具，`scout`/`reviewer` 只获得其中名为 `read` 或 `bash` 的工具。Forge 不悄悄补回调用方没有提供的 coding tool。

### 6.3 通用运行原语

在 `forge_agent.subagents` 增加以下小型类型：

- `SubagentSpec`：`name`、`description`、`system_prompt`、`tools`、`max_model_calls`、`max_result_bytes`。
- `SubagentRuntime`：当前 `BaseChatModel`、model 名、`ForgeRuntimeContext`。
- `SubagentRunResult`：稳定的 v1 结果结构。
- `SubagentRunner`：角色注册表、当前运行时读取函数和一个 session 级 `asyncio.Lock`。

`SubagentRunner.run()` 的固定流程：

1. 校验角色和 instruction。
2. 在等待 lock 时保持调用处于 `queued`；拿到 lock 后进入 `running`。
3. 调用运行时读取函数，获取此刻父 session 的 provider、model 和 context，避免 `/model`、`/provider` 后继续使用旧对象。
4. 使用角色自己的 system prompt 和工具子集调用 `create_agent(name=spec.name, ...)`。
5. 输入只包含一个新的 `HumanMessage(instruction)`；不复制父 transcript。
6. 使用 `ModelCallLimitMiddleware(run_limit=8, exit_behavior="error")` 限制单次子代理。
7. `await child.ainvoke(...)`，并把同一个 `ForgeRuntimeContext` 传给 LangChain，使现有 coding 工具继续按 workspace/session 规则工作。
8. 从返回 messages 中读取最后一条非空 `AIMessage`，统计模型调用数和 `ToolMessage` 数。
9. 最终文本按 UTF-8 字节限制截断，默认上限 `50 * 1024`；结果标记是否截断。
10. 返回 `(content, artifact)`，由 LangChain 写成父会话的 `ToolMessage`。

父模型等待 `task` 返回，因此同一时刻不会与子模型并发调用 provider。即使模型在一个 tool batch 中发出多个 `task`，session 级 lock 也会把它们串行化，避免多个 `worker` 同时修改工作区。

### 6.4 稳定的结果 artifact

`task` 的 `AgentToolResult.data` 保存：

```json
{
  "kind": "subagent_run",
  "version": 1,
  "agent": "scout",
  "status": "completed",
  "instruction": "Inspect the authentication flow",
  "final_output": "...",
  "model_calls": 4,
  "tool_calls": 7,
  "queued_ms": 0,
  "duration_ms": 12400,
  "truncated": false,
  "error": null
}
```

约束：

- `status` 只取 `completed` 或 `failed`；运行态不持久化到 artifact。
- 子模型失败或达到模型调用上限时，`task` 本身仍完成协议交付，artifact 使用 `failed`，content 返回可供父模型处理的简短错误说明。
- 非法角色/参数属于工具输入错误，沿用现有 `ToolException` 路径。
- `asyncio.CancelledError` 不转换成普通结果，必须继续抛出，让现有 harness 完成整轮取消和缺失 ToolMessage 修复。
- artifact 必须经过 Forge 现有 JSON 投影约束，不保存异常对象、provider 响应或完整子 transcript。

这里把“subagent 执行失败”和“task 工具协议损坏”分开：前者是父模型可以读取并调整策略的业务结果，后者才是 LangChain tool error。

### 6.5 CodingSession 组装

`CodingSession.load()` 调整为以下顺序：

1. 创建现有基础 coding tools。
2. 创建可变的 `AgentHarnessConfig`，先放入基础工具。
3. 使用当前 resources 为三个角色分别构建 system prompt；默认 prompt 复用 `build_system_prompt()`，但传入各角色自己的工具子集，再追加简短角色契约。
4. 创建引用该 config 的 `SubagentRunner`。运行时读取函数每次从 config 读取 provider、model 和 runtime context。
5. 创建 `task` 工具并加入父工具列表；若已有同名自定义工具，立即报明确配置错误。
6. 使用包含 `task` 的最终工具列表构建父 system prompt。
7. 最后构造 `AgentHarness`。

这样不需要在 model/provider 切换时重建 task 工具。现有 `set_model()`、`set_provider()` 和 `_refresh_runtime_provider()` 更新同一个 harness config，下一次子调用自然读取新值。

`reload()` 在默认 system prompt 由 Forge 管理时，同时重建三个角色的 prompt/spec，使项目说明、skills 和 context files 与父 session 一起刷新。用户显式提供 `CodingSessionConfig.system` 时，子 prompt 以该 system 为基础追加角色契约，不自行改写用户内容。

第一版默认启用 subagents。为测试或嵌入场景增加 `CodingSessionConfig.enable_subagents: bool = True`；关闭时保持当前工具集合和行为完全不变。

### 6.6 v3 事件投影

当前 `run_langchain_agent()` 会处理所有 v3 `messages`、`tools` 和 `values`。子 `create_agent().ainvoke()` 的事件带非空 `params.namespace`；如果不处理，子代理的 Human/AI/Tool messages 会被误写进父 transcript。

新增一个小型 `_NestedTaskProjection`，规则如下：

1. `namespace == []`：沿用现有根事件逻辑。
2. 非空 namespace：永远不交给父 `_ProjectionState`，不产生 `MessageStart/Delta/End` 或 `ThinkingDeltaEvent`。
3. 根级 `ToolExecutionStartEvent(name="task")` 到达时，记录 task call id、角色和 instruction。
4. namespace 中第一个 `tools:<call-id>` 段用于关联父 task；lifecycle `started` 的 `cause.tool_call_id` 作为补充校验。
5. 子 lifecycle started 产生一条现有 `ToolExecutionUpdateEvent`，表示从 `queued` 进入 `running`。
6. 子工具 started/finished 产生紧凑 update，`tool_call_id` 固定使用父 task call id，不暴露子 tool call id。
7. 子 lifecycle completed 不伪造结束；权威结束仍是根级 task `ToolExecutionEndEvent`，它包含持久化 artifact。
8. namespace 无法识别时安全丢弃嵌套事件，绝不能退回根 transcript 投影。

Update 的 `data` 约定：

```json
{
  "kind": "subagent_activity",
  "agent": "scout",
  "status": "running",
  "activity": {
    "phase": "started",
    "tool": "read",
    "summary": "Reading src/forge_coding/session.py"
  }
}
```

不新增 `AgentEvent` 类型，避免 CLI、JSON renderer 和外部消费者被迫同步一组只服务于 UI 的事件。非 TUI transcript renderer 继续把 update 当普通工具进度显示；JSON renderer 自动得到结构化 data。

`namespace` 形状是本方案中最脆弱的框架接缝。必须用当前锁定的 LangChain 版本编写回归测试：若未来格式变化，正确降级是“仍显示 task 开始/结束，但不显示子活动”，而不是污染父 transcript。

### 6.7 取消、失败与恢复

- TUI `Esc` 继续调用现有 session/harness cancel；当前 prompt task 被取消时，取消会自然传入正在 await 的 child。
- `SubagentRunner` 对 `CancelledError` 只做 lock/finally 清理并重新抛出。
- harness 继续负责为被中断的父 task 调用补齐 ToolMessage，保证下一次 provider 请求合法。
- 普通子失败不会结束父 session；父模型收到 failed artifact 后可解释、重试或换角色。
- `max_model_calls=8` 是每次子调用的固定限制，父代理自己的 `max_turns` 不与它共享计数。
- 不设置 child checkpointer；每次调用都是 fresh context。
- JSONL 只保存父 AI 的 task call 和对应 ToolMessage artifact。旧 session 不需要迁移。

## 7. TUI 设计

### 7.1 展示形态

Subagent 是对话中的一种工具活动，使用内联任务块，而不是独立页面：

```text
◌ scout  Inspect authentication flow
  reading src/forge_coding/session.py
```

完成后：

```text
✓ scout  Inspect authentication flow
  7 tool calls · 12.4s · Ctrl+O show result
```

失败或取消：

```text
× reviewer  Review the persistence changes
  failed · Ctrl+O show result

– worker  Implement the approved change
  cancelled
```

状态必须同时使用符号和文字，不能只靠颜色。运行超过 200ms 后才显示 spinner，避免快速任务闪烁。

### 7.2 TUI 状态

`ChatItemRole` 增加 `subagent`，并为 `ChatItem` 增加可选的 `SubagentDisplay`：

- `tool_call_id`
- `agent`
- `instruction`
- `status`: `queued | running | completed | failed | cancelled`
- `activity`
- `tool_calls`
- `queued_ms`
- `duration_ms`
- `final_output`
- `truncated`

`TuiState` 增加三个明确操作：

- `add_subagent_task(tool_call)`
- `update_subagent_activity(event)`
- `finish_subagent_task(result)`

普通工具仍走现有 `add_tool_call()` / `record_tool_result()`。`task` 只根据工具名和 v1 artifact 特化展示；无法识别的旧版/未来 artifact 回退为普通 tool item。

### 7.3 事件适配与增量更新

- `ToolExecutionStartEvent(name="task")`：创建 queued 任务块。
- `ToolExecutionUpdateEvent(data.kind="subagent_activity")`：原位更新状态和最后一条活动，不追加一行日志。
- `ToolExecutionEndEvent(name="task")`：用 artifact 完成任务块。
- 子代理文本和 thinking 不进入 TUI state。
- `ForgeTuiApp._apply_streaming_transcript_event()` 对 subagent block 做 widget 原位更新，避免每个子工具事件触发整份 transcript 重绘。

增加专用 `SubagentTranscriptWidget`，只负责两行折叠态和展开后的 final output。它不拥有执行状态，不直接订阅后端，也不创建 timer 之外的后台任务。

### 7.4 键盘与响应式规则

- 继续复用 `Ctrl+O`：全局切换普通工具结果和 subagent final output。
- 继续复用 `Esc`：取消整个当前 prompt，包括正在执行的子代理。
- 不增加 Tab focus、modal 或新快捷键。
- 80x24 下折叠态最多两行，instruction 和 activity 按可用宽度省略；final output 展开后正常换行。
- 120 列以上仍保持现有居中对话宽度，不横向扩展成面板。
- dark、light、high-contrast 三套主题增加 `subagent`、`subagent-running`、`subagent-success`、`subagent-error` 语义样式；组件不硬编码颜色。
- 单色环境依靠 `◌ / ✓ / × / –` 和状态文字保持可读。

### 7.5 历史恢复

`TuiState.load_messages()` 在读取历史时：

1. 看到父 AI 的 `task` tool call，创建 subagent item。
2. 看到匹配的 ToolMessage 且 artifact 为 `kind=subagent_run, version=1`，恢复最终状态、计数、耗时和 output。
3. 看到 harness 生成的 interrupted tool result，恢复为 `cancelled`。
4. artifact 缺失、损坏或版本未知时，作为普通 tool result 展示，不能让 session 打不开。

不会恢复实时 activity，因为它不是产品事实，也没有必要写入 JSONL。

## 8. 文件级改动

本功能预计触及 18 个文件，原因是它跨越明确的三包边界和对应测试；不做跨层捷径。

| 文件 | 修改 |
| --- | --- |
| `src/forge_agent/subagents.py` | 新增通用 spec、runner、结果限制和 fresh child `create_agent` 调用 |
| `src/forge_agent/langchain_runtime.py` | 区分 root/nested namespace，投影 task activity，阻止子消息污染父 transcript |
| `src/forge_agent/__init__.py` | 导出必要的通用 subagent 类型 |
| `src/forge_coding/subagents.py` | 定义三个 coding 角色、工具子集、role prompt 和 `task` ToolDefinition |
| `src/forge_coding/session.py` | 按新顺序组装 base tools、runner、task、system、harness；reload 同步 specs |
| `src/forge_cli/tui/state.py` | 增加 `SubagentDisplay` 及创建、更新、结束、恢复逻辑 |
| `src/forge_cli/tui/adapter.py` | 把 task start/update/end 映射到专用 state 操作 |
| `src/forge_cli/tui/widgets.py` | 增加内联 `SubagentTranscriptWidget` |
| `src/forge_cli/tui/app.py` | 对 task activity 做原位增量更新，复用取消和展开行为 |
| `src/forge_cli/tui/config.py` | 为三套主题增加语义样式 |
| `tests/test_subagents.py` | runner、角色、限制、串行、失败与取消单元测试 |
| `tests/test_langchain_runtime.py` | nested v3 namespace、活动投影、父 transcript 隔离回归测试 |
| `tests/test_coding_session.py` | 组装、开关、model/provider 切换、reload、持久化和恢复测试 |
| `tests/test_tui_adapter.py` | task 事件到 display state 的映射测试 |
| `tests/test_tui_app.py` | 增量更新、Ctrl+O、Esc、窄终端和恢复交互测试 |
| `tests/test_tui_config.py` | 三套主题的 subagent 语义样式测试 |
| `docs/architecture.md` | 记录 agent-as-tool、无第二循环、无子会话和事件隔离边界 |
| `README.md` | 增加自然语言委派示例、三个角色、限制和 TUI 行为 |

如实现中无需修改某个现有文件，应删除对应改动，而不是为了匹配计划制造空改动。

## 9. 实现顺序

整个功能作为一个原子里程碑交付，内部按以下顺序推进：

1. 写 `SubagentRunner` 和 fake-model 单元测试，确认 fresh context、工具子集、结果 artifact、限制、失败和取消。
2. 写 nested v3 事件回归测试，再修改 `langchain_runtime`；先证明子消息不会进入父 transcript。
3. 在 `forge_coding` 定义角色和 task 工具，调整 session 组装、切换与 reload。
4. 验证父 AI task call 与 ToolMessage 的持久化、恢复和中断修复。
5. 增加 TUI state/adapter，再实现专用 widget 和 app 原位更新。
6. 补主题、80x24 行为、README 和 architecture 文档。
7. 执行全量静态检查、测试、CLI/build 检查和人工 TUI 验收。

后端和 TUI 不能分别宣称完成：没有事件隔离的后端会污染 transcript；没有 TUI 消费路径的后端会让用户面对不透明的长任务。

## 10. 测试矩阵

### 后端

- `scout` 只收到 `read`、`bash`，`reviewer` 同样只读提示，`worker` 收到四个 coding tools；所有角色都没有 `task`。
- 自定义 `CodingSessionConfig.tools` 保持能力上限，角色只能从该列表取子集。
- 子输入只有当前 instruction，不包含父历史消息。
- 子最终文本成为父 task ToolMessage；完整子 transcript 不持久化。
- 空 instruction、未知 agent 和重复 `task` 工具名被明确拒绝。
- 子模型正常完成、工具失败、模型失败、达到 8 次模型调用、空 final message 均得到确定结果。
- 超过 50 KiB 的 final output 按 UTF-8 安全截断并记录 metadata。
- 两个同批 task 调用按 lock 串行，第二个先显示 queued。
- 等待 lock 和运行中的 task 都能被取消，lock 不泄漏。
- `/model`、provider 切换后，下一次 child 使用新 provider/model。
- reload 后 child 使用新的项目说明/resources。
- nested `messages`、`values`、thinking 和 child ToolMessage 不进入父 transcript。
- nested child tool activity 正确关联父 task id；未知 namespace 被丢弃。
- LangChain namespace 形状改变时，至少保留父 task start/end，且不污染 transcript。
- 中断后下一次 prompt 前补齐 task ToolMessage，provider 消息配对合法。
- 老 JSONL 无迁移可加载；新 artifact 可往返序列化。

### TUI

- queued、running、completed、failed、cancelled 五种状态都有文字和符号。
- 多条 child tool update 只更新一个 block，不向 transcript 追加噪音行。
- `Ctrl+O` 展开/折叠 final output；不显示 child thinking 或完整 transcript。
- `Esc` 取消后 block 收敛为 cancelled，composer 恢复可用。
- 历史 session 能重建 completed、failed、cancelled block。
- artifact 缺失/损坏/未知版本回退为普通 tool item。
- 80x24 不横向溢出，长 instruction/activity 省略，窗口 resize 不崩溃。
- dark、light、high-contrast 下可读；去掉颜色后仍能辨认状态。

## 11. 验证命令

实现完成后执行：

```powershell
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
uv run pytest
uv run forge --help
uv run forge --version
uv build
git diff --check
```

真实 provider smoke test 不进入默认测试，也不读取或输出本地凭据。若用户主动允许，可额外在 TUI 中完成一次 `scout` 和一次 `worker` 调用。

人工 TUI 验收尺寸：80x24、120x30、200x60；主题：dark、light、high-contrast；至少在 Windows Terminal 验证 resize、Ctrl+O 和 Esc。

## 12. 验收标准

- 父代理能自然调用 `task` 完成 scout、worker、reviewer 三类任务。
- 生产路径中父、子都使用 LangChain `create_agent`；不存在第二套 provider/tool loop。
- 子消息、thinking 和工具结果不会混入父 transcript；父会话只新增一次 task call 和配对 ToolMessage。
- 当前 provider/model、workspace context、项目资源和工具安全边界在子调用中生效。
- 多 task 调用串行，取消无泄漏，失败可由父代理继续处理。
- TUI 在原对话流中实时显示紧凑状态，完成后可用 Ctrl+O 查看结果。
- JSONL 可重放新 task block，旧 session 无需迁移。
- 全量 lint、type check、tests、CLI 和 build 验证通过。

## 13. 风险与回退

| 风险 | 缓解 |
| --- | --- |
| LangChain v3 nested namespace 变化 | 针对锁定版本做回归测试；未知 nested 事件一律丢弃，保留根 task start/end |
| 子结果过长重新挤占父上下文 | 50 KiB UTF-8 上限和 truncation metadata |
| 多 worker 同时修改文件 | session 级 lock 强制串行；第一版不提供 parallel |
| 角色 prompt 与项目 reload 脱节 | reload 与父 system 同时重建 role specs |
| model/provider 切换后 runner 捕获旧对象 | 每次 run 从共享 harness config 延迟读取 |
| `bash` 绕过只读角色意图 | 文档明确不是安全边界；真正写入安全仍由工作区规则和用户授权承担 |
| 新 TUI 状态无法读取旧/坏 artifact | 版本校验失败后回退普通 tool item |

回退不需要数据迁移：关闭 `enable_subagents` 或移除 task 工具即可恢复当前运行行为；已经写入 JSONL 的 task call/ToolMessage 对旧展示层仍是普通 LangChain 工具消息，可被通用 renderer 读取。

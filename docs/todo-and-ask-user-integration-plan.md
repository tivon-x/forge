# Forge 集成 Todo 和 Ask User Question 工具

直接采用 LangChain 官方能力——`TodoListMiddleware` 提供 Todo，`HumanInTheLoopMiddleware + respond` 提供用户提问。Forge 只负责事件投影、JSONL 补充快照和 Pi 风格 TUI，不复制 rpiv 的完整扩展框架。

这最符合 “small, readable, fully modular and a working example”：读代码的人能清楚看到 LangChain 的 tool、middleware、state、interrupt、checkpointer、streaming 和 `Command(resume=...)` 是怎么组合起来的。

## 设计目标

构建：

- `write_todos`：模型维护多步骤任务，TUI 在输入框上方实时显示进度。
- `ask_user_question`：模型提出 1–4 个结构化问题，TUI 打开 Pi 风格问卷，回答后恢复同一次 LangChain 执行。
- Todo 可以随 Forge JSONL 会话恢复；问题答案作为 `ToolMessage` 自然进入 transcript。
- Textual TUI 尽量复刻 rpiv 的布局和键盘操作。

不构建：

- 不移植 rpiv 的 TypeScript、扩展加载器、事件总线、RPC/ACP fallback。
- Todo 不实现 ID、依赖图、owner、metadata、墓碑等高级能力。
- 不做数据库级 LangGraph 持久化、跨进程恢复未回答问卷。
- 不做国际化、外部编辑器和可安装第三方扩展系统。

## 为什么这样选

### Todo：使用官方 `TodoListMiddleware`

LangChain 已提供 `TodoListMiddleware`，自动注入 `write_todos` 工具和使用说明，状态只有：

- `content`
- `status: pending | in_progress | completed`

当前 Forge 锁定的 LangChain 是 `1.3.14`，本地探测确认 `write_todos` 会通过 v3 `values` 事件输出完整 `todos` 快照。[uv.lock](../uv.lock)

这正好覆盖 Forge 的教学目标。rpiv 的核心经验只借用两点：

- 每次更新携带完整快照，通过最后一个有效快照恢复。
- TUI 始终显示当前计划，并限制高度、隐藏较旧的完成项。

rpiv 自己也是用完整 post-mutation snapshot 做 replay，但它的任务 ID、依赖和墓碑状态机明显超出 Forge 当前需求。[rpiv Todo README](https://github.com/juicesharp/rpiv-mono/tree/main/packages/rpiv-todo)、[Todo tool schema](https://github.com/juicesharp/rpiv-mono/blob/main/packages/rpiv-todo/docs/tool-schema.md)、[LangChain TodoListMiddleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in#to-do-list)

### Ask user：使用官方 HITL `respond`

LangChain HITL 现在有专门面向 ask-user 工具的 `respond` 决策：

- 模型生成 `ask_user_question` tool call。
- `HumanInTheLoopMiddleware` 在工具执行前中断。
- TUI 收集答案。
- Forge 使用 `Command(resume=...)` 和 `respond` 返回答案。
- LangChain 合成成功的 `ToolMessage`，占位工具本身不会执行。

这是比“工具内部等待一个 asyncio Future”更好的方案，因为它真正展示了 LangChain 的 interrupt/resume，而不是 Forge 自建暂停机制。[LangChain HITL](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)、[rpiv Ask User schema](https://github.com/juicesharp/rpiv-mono/blob/main/packages/rpiv-ask-user-question/docs/tool-schema.md)

## 总体数据流

```text
                     create_agent()
                           │
          ┌────────────────┴────────────────┐
          │                                 │
 TodoListMiddleware              HumanInTheLoopMiddleware
          │                                 │
 write_todos state                    interrupt + respond
          │                                 │
          └──────── astream_events(v3) ─────┘
                           │
                    forge_agent events
                           │
                 ┌─────────┴──────────┐
                 │                    │
          CodingSession          Textual TUI
          JSONL snapshots       Todo / Questionnaire
                 │                    │
                 └────── answer ──────┘
                           │
                    Command(resume)
```

没有第二套 agent/tool loop，`create_agent()` 仍然是唯一生产循环。

## 后端方案

### 1. Middleware 注入边界

扩展 `AgentHarnessConfig`，允许上层传入一组 LangChain `AgentMiddleware`。

`forge_agent` 只负责：

- 合并 Forge 自带的 steering、model-call-limit middleware。
- 调用 `create_agent()`。
- 投影 v3 事件和中断。
- 保存当前未完成的 graph/config，供恢复使用。

`forge_coding` 负责选择产品能力：

- 默认加入 `TodoListMiddleware`。
- TUI 模式加入 `HumanInTheLoopMiddleware`。
- 非交互模式不注册 `ask_user_question`，避免模型调用一个永远无法回答的工具。

这样不破坏 `forge_cli → forge_coding → forge_agent → LangChain`。

### 2. Todo 事件和持久化

新增 `TodoUpdateEvent`：

- 字段为完整、已验证的 todo tuple。
- 只在 v3 `values.todos` 相比上次发生变化时发出。
- 不把 todos 塞进通用 `ToolExecutionUpdateEvent.data`。

`CodingSession` 收到事件后：

1. 先持久化对应的 AI tool call 和 `ToolMessage`。
2. 再追加 `CustomEntry`：
   - `custom_type = "forge.todo.v1"`
   - `data = {"todos": [...]}`。
3. 持久化成功后才把事件交给 TUI。

恢复、切换分支、compaction 后：

- 从当前 branch 找最后一个合法的 `forge.todo.v1`。
- 损坏、未知版本的快照忽略。
- compaction 完成后把当前 Todo 快照重新挂到新 branch，避免它只存在于被压缩历史中。

模型仍能从 `write_todos` 的历史 `ToolMessage` 看到上次列表；CustomEntry 只负责产品恢复和 TUI，不成为第二份模型 transcript。

### 3. `ask_user_question` schema

在 `forge_coding` 定义 Pydantic schema：

- 每次调用 1–4 个问题。
- `header` 最多 16 字符。
- 每题 2–4 个选项。
- 选项包含 `label`、`description`、可选 `preview`。
- 支持 `multi_select`。
- `Other`、`Type something.`、`Next` 为保留标签。
- 描述明确要求：需要答案才能继续时才调用；多个问题合并为一次调用；该工具应单独占一个 tool-call turn。

占位工具如果真的进入执行函数，返回受控错误；正常路径永远由 HITL `respond` 截获。

回答结果序列化为稳定 JSON，包含：

- `question_index`
- `question`
- `kind: option | custom | multi`
- `answer`
- `selected`
- `notes`
- `cancelled`

取消问卷也使用 `respond`，返回 `cancelled: true`；不能用 `reject`，因为用户不是在拒绝一个副作用操作。

### 4. 暂停和恢复

新增通用事件 `HumanInputRequestedEvent`，由 `forge_agent` 发出：

- interrupt id
- tool call id
- tool name
- JSON-safe arguments
- allowed decisions

`AgentHarness` 增加：

- `is_waiting_for_input`
- `respond_to_human_input(message)`
- `cancel_pending_input()`

执行流程：

1. 为这一次 prompt 创建 `InMemorySaver` 和唯一 thread id。
2. 正常消费 `astream_events(version="v3")`。
3. 流结束后调用 stream 的 `interrupted()` / `interrupts()`。
4. 有中断时保存 graph、config 和 interrupt，发出 `HumanInputRequestedEvent`。
5. TUI 返回答案后，用同一个 graph/config 执行 `Command(resume={"decisions": [...]})`。
6. 恢复完成后立即丢弃 graph 和内存 checkpoint。

若模型意外生成多个 ask tool call，按 LangChain 的 multiple-decisions 协议顺序展示多个问卷，最后一次性恢复，不静默丢弃调用。

### 5. 与现有 JSONL 原则的冲突

当前架构明确写着“不引入 LangGraph checkpointer”。[architecture.md](architecture.md)

Phase 2 需要有意识地修改这句话为：

> JSONL 是唯一持久化会话事实源；允许为当前未完成的 HITL 调用使用一次性、进程内 checkpointer。完成、取消或退出后立即销毁，不能用于 session replay、branch、compaction 或模型记忆。

这不是两个持久化权威来源：checkpoint 只保存尚未提交完的执行现场，JSONL 仍然负责所有可恢复产品状态。

## TUI 方案

### Todo 面板

在 `TranscriptView` 与 `#queued-messages` 之间插入 `TodoPanel`，位置与 Pi 的 “above editor” 一致：

```text
  ● Todos (2/5)
  ├─ ✓ Inspect current runtime
  ├─ ◐ Design HITL flow · designing questionnaire
  ├─ ○ Add backend events
  └─ ○ Add TUI tests

  ┌──────────────────────────────────────────┐
  │ Ask Forge…                               │
  └──────────────────────────────────────────┘
```

行为：

- `◐` 当前进行中、`○` 待处理、`✓` 已完成。
- 默认最多 12 行。
- 超限先隐藏完成项，再截断尾部，显示 `+N more`。
- 完成项保留到下一次用户 turn，再从面板隐藏。
- 空列表自动消失。
- `Ctrl+Shift+T` 折叠/展开。
- `/todos` 使用现有 `CommandOutputScreen` 显示完整列表。
- 非 TUI renderer 仍把 `write_todos` 当普通工具输出。

这基本保留 rpiv 的可见计划、行预算和折叠交互，不照搬它的复杂 task graph。[rpiv Todo overlay](https://github.com/juicesharp/rpiv-mono/blob/main/packages/rpiv-todo/todo-overlay.ts)

### Questionnaire

新增独立的 `AskUserQuestionScreen`，不要继续膨胀 `app.py`。

布局尽量与 Pi 一致：

- 底部全宽 overlay，保留后方 transcript。
- 1 个问题直接显示；多个问题使用 tabs，并追加 Submit tab。
- 单选是纵向 option list。
- 多选用 checkbox，`Next` 提交当前题。
- 每题自动追加 `Type something.`。
- 有 preview 且宽度 ≥100 列时左右分栏；低于 100 列时上下堆叠。
- 内容过高时仅 body 滚动，标题和快捷键提示固定。
- `Ctrl+]` 折叠为一行，允许查看 transcript，再次按键恢复且保留草稿。

键盘与 Pi 对齐：

- `↑/↓`：移动。
- `Enter`：选择、确认文字、进入下一步。
- `Space`：切换多选。
- `Tab/Shift+Tab`：切换问题。
- `Shift+Enter`：输入换行。
- `n`：回答备注。
- `Esc`：取消整个问卷。
- `Ctrl+]`：折叠/恢复。

参考：[rpiv questionnaire keyboard/layout](https://github.com/juicesharp/rpiv-mono/blob/main/packages/rpiv-ask-user-question/docs/keyboard.md)。

## 实施阶段

涉及约 15–20 个文件，超过 8 个，但可拆成两个独立、可合并阶段。

### Phase 1：Todo

主要文件：

- [pyproject.toml](../pyproject.toml)：把 LangChain 下限提高到已验证的 `1.3.14`。
- [events.py](../src/forge_agent/events.py)：增加 `TodoUpdateEvent`。
- [langchain_runtime.py](../src/forge_agent/langchain_runtime.py)：注入 middleware，投影 `values.todos`。
- [harness.py](../src/forge_agent/harness.py)：接受 middleware。
- [session.py](../src/forge_coding/session.py)：持久化和恢复快照。
- 新增 `forge_coding/planning.py`。
- 新增 `forge_cli/tui/todos.py`。
- [state.py](../src/forge_cli/tui/state.py)、[adapter.py](../src/forge_cli/tui/adapter.py)、[app.py](../src/forge_cli/tui/app.py)：接入面板。
- [config.py](../src/forge_cli/tui/config.py)：增加 Todo 折叠键。
- [commands.py](../src/forge_coding/commands.py)：增加 `/todos`。
- README、architecture、对应单元测试。

Phase 1 单独合并后就是完整可用的 Todo 功能。

### Phase 2：Ask user + HITL

主要文件：

- 新增 `forge_coding/human_input.py`：schema、占位工具、结果序列化、middleware 工厂。
- [events.py](../src/forge_agent/events.py)：增加中断事件。
- [langchain_runtime.py](../src/forge_agent/langchain_runtime.py)：保存 pending graph、检查 stream interrupts、执行 resume。
- [harness.py](../src/forge_agent/harness.py)：暴露回答和取消接口。
- [session.py](../src/forge_coding/session.py)：恢复执行并持久化生成的 `ToolMessage`。
- 新增 `forge_cli/tui/questionnaire.py`。
- [app.py](../src/forge_cli/tui/app.py)：收到中断后打开问卷并启动 resume worker。
- README、architecture、runtime/session/TUI 测试。

Phase 2 不依赖 Todo 的 UI，但共享 Phase 1 已建立的通用 middleware 注入接口。

## 验收标准

自动测试至少覆盖：

- Todo 创建、替换、清空和非法状态。
- 同一 model turn 多次 `write_todos` 被官方 middleware 拒绝。
- Todo 事件去重、JSONL replay、branch 隔离、compaction 后恢复。
- Todo 面板空状态、折叠、行预算、完成项延迟隐藏。
- Ask schema 上下限、重复问题、重复/保留标签、preview 限制。
- 单选、多选、自定义答案、备注、部分提交、取消。
- interrupt 产生后占位工具没有执行。
- `respond` 生成配对的 `ToolMessage`，恢复后模型继续输出。
- 多个 interrupt decisions 保持原顺序。
- 取消进程、切换 session、关闭 TUI 时不留下不可恢复的 dangling tool call。
- 非交互模式不向模型暴露 ask 工具。
- 80×24、120×40、≥100 列 preview、窄屏堆叠和 resize。

完成后运行：

```text
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
uv run pytest
uv run forge --help
uv run forge --version
```

并在 Windows Terminal 手工验证 Todo、问卷、折叠、恢复、窄屏。

## 主要风险

最脆弱的假设是：未回答问卷只需要进程内恢复，暂时不考虑 Forge 重启后继续同一个问卷。

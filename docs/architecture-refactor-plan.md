# Forge 架构重构执行方案

状态：已实施（2026-08-18）
基线：`main` at `7fd28c9`，并包含当前工作区尚未提交的工具模块拆分
适用版本：Python 3.12，LangChain 1.3.14，LangChain Core 1.5.3，LangGraph 1.2.10
参考实现：本机 `@earendil-works/pi-coding-agent` 0.84.2

## 1. 结论

Forge 值得继续重构，但范围应收缩为“工具产品契约、保守执行语义、文件安全和展示复用”，
不应倒置现有三包依赖，也不应复制 Pi 的 parallel/sequential 混合调度。

最终依赖方向保持不变：

```text
forge_cli  ───────▶  forge_coding  ───────▶  forge_agent  ───────▶  LangChain
presentation        coding product          runtime facade          runtime facts
```

目标工具数据流为：

```text
forge_cli ToolViewRegistry
        ▲ consumes AgentEvent only
        │
forge_coding CodingSession ── owns ToolSet ──▶ tuple[BaseTool, ...]
        │                         │                    │
        │                         └── prompt metadata  ▼
        └─────────────────────────────────────▶ create_agent()
                                                      │
                                                      ▼
                                    SequentialToolCallMiddleware
                                    model order + fail-fast pairing
```

`BaseTool`、Pydantic schema、`ToolRuntime` 和 `ToolMessage` 仍是执行事实；
`ToolDefinition`/`ToolSet` 只保存 Forge 产品层需要的目录和提示元数据，不能成为第二套工具运行时。

## 2. 为什么修订上一版方案

### 2.1 不倒置 `forge_cli` 与 `forge_coding`

上一版建议把 `forge_cli` 变成类似 `pi-tui` 的通用基础库，再让
`forge_coding` 依赖它。这个类比不成立：

- Pi 的 `pi-tui` 是独立、通用的终端组件包，`pi-coding-agent` 是产品组装层；
- Forge 的 `forge_cli` 是当前唯一产品的 presentation/composition root，负责 Typer、
  print renderer 和 Textual App；
- 当前 `forge_cli` 25 个 Python 文件中有 10 个直接使用 coding domain，迁移意味着移动
  CLI 入口、完整 TUI App、session 启动、配置屏幕和命令适配，不会减少产品耦合，只会换目录；
- 当前单分发、单产品没有复用通用 TUI 包的实际需求。

因此保留 `forge_cli → forge_coding → forge_agent`。只有出现第二个真实前端或第二个产品
复用同一套终端组件时，才重新评估独立 `forge_tui` 包；本方案不预先建设它。

### 2.2 采用 Forge 严格串行，而不是 Pi 混合并发

Pi 0.84.2 的 agent loop 在拿到完整 tool-call batch 后统一决策：批次包含任一
`sequential` 工具时整批串行，否则先逐个 prepare，再并行 execute，最后按模型调用顺序
生成结果。

Forge 当前使用的 LangChain 1.3.14 会在 `ToolNode._afunc()` 中为整批调用创建协程并执行
`asyncio.gather()`。这保证返回列表保持输入顺序，但默认不保证副作用完成顺序。仅给
`write`/`edit` 加同路径锁仍有三个缺口：

- `read` 可能与 `write`/`edit` 同时访问同一文件，看到旧内容或中间状态；
- `bash` 可能与任意文件工具同时修改工作区，无法可靠推断 resource key；
- 两个不同路径的副作用也可能存在业务依赖，按路径加锁不能表达模型意图。

Forge 不需要 Pi 的“默认并行、按工具切换整批模式”。安全保守的默认值更符合项目定位：

1. 同一 model tool-call batch 中的所有工具按 AIMessage 中的顺序逐个执行；
2. 第一个 error ToolMessage 或执行异常出现后，剩余调用不再进入 handler；
3. 每个被跳过的调用都生成同 tool_call_id 的 error ToolMessage，保持消息配对；
4. 取消立即停止当前批次，沿用现有 cancellation settlement，不吞掉 `CancelledError`；
5. root agent 和所有 subagent 使用同一语义，不提供关闭开关；
6. `ToolDefinition` 不增加 `execution_mode`，串行是 Forge runtime invariant，不是单工具偏好。

实现使用 LangChain 公开的 `awrap_tool_call`：一个 middleware 实例持有一个公平
`asyncio.Lock`，并从公开的 `ToolCallRequest.state` 读取最新 AIMessage 的 tool-call 顺序。
middleware 必须处于最外层，因此 Goal/HITL/validation 和实际工具调用都在同一个顺序门内。
这不是新的 tool loop，也不拆分 LangChain 的 validation/preflight/execute。

已用当前锁定依赖做最小实测：`create_agent()` 同一 AIMessage 发出两个 async tool call 时，
最外层 `awrap_tool_call` 加公平 lock 后，实际事件严格为
`first start → first end → second start → second end`，ToolMessage 仍按原 ID 顺序返回。

为避免把当前 LangChain 调度顺序当成无条件事实，middleware 在进入 handler 前校验当前
tool_call_id 是否等于批次的下一个预期 ID。若上游升级导致到达顺序变化、ID 缺失或重复，
Forge fail closed：不执行工具，返回配对错误结果。契约测试固定当前版本的顺序假设。

串行只能让既定调用的副作用确定化，不能让后一个调用动态获得前一个调用的返回值，因为
整批参数在执行前已由模型生成。system prompt 必须明确：若后一个工具的参数依赖前一个结果，
应等待下一轮再调用，而不是放在同一批次。

这套保证的边界是单个 Forge graph invocation。独立 OS 进程、用户在外部终端的操作，以及
`bash` 显式启动并脱离等待的后台进程不受 asyncio middleware 保护；shell 仍然不是沙箱。

### 2.3 保留 `ToolDefinition`，但删除重复事实

当前 `ToolDefinition` 同时保存 name、description、JSON schema、Pydantic schema、executor、
prompt metadata，并负责转换为 `ForgeStructuredTool`。这会让 LangChain 原生 tool 与 Forge
definition 各保存一份执行和 schema 信息。

保留名称是合理的，但新契约应让原生 tool 成为唯一执行事实：

```text
ToolDefinition
├── tool: BaseTool                  # 唯一 schema/description/execution 事实
├── label: str                      # 产品显示名称
├── prompt_snippet: str | None      # Available tools 一行说明
└── prompt_guidelines: tuple[str]   # 激活工具时的提示规则
```

不加入 `renderer`：渲染是 `forge_cli` 的职责。
不加入 `execution_mode`：Forge 全局采用严格串行，不需要单工具策略。
不保存 `input_schema`、`args_schema`、`description`、`executor` 的副本：这些从 `tool` 派生。

## 3. 目标与成功标准

### 3.1 目标

- 完成当前 `tools.py` 到 `tools/` 包的 LangChain-native 模块拆分；
- 让 schema、description 和执行逻辑只由原生 `BaseTool` 持有；
- 用 `ToolDefinition`/`ToolSet` 统一工具顺序、选择、prompt metadata 与查找；
- 用统一 middleware 保证 root/subagent 工具按模型顺序执行并在失败后停止；
- 用 process-local、跨 event loop 的文件操作队列保护同一文件的 `read`/`write`/`edit`；
- 让 plain renderer 与 Textual TUI 复用同一套纯工具格式化注册表；
- 保持 JSONL、AgentEvent、CLI、配置、Provider 与 agent loop 行为兼容。

### 3.2 完成标准

- `forge_agent` 不知道 `ToolDefinition`、`ToolSet` 或任何 coding tool；
- `forge_coding` 不导入 `forge_cli`、Textual、Rich 或 Typer；
- `create_agent()` 只收到 `BaseTool` 序列；
- provider-visible schema 只来自 `tool.tool_call_schema`，其中不出现 `runtime`；
- `ToolDefinition` 不存储第二份 schema、description 或 executor；
- 内置工具顺序仍为 `read, write, edit, bash`，`task` 的插入与 profile allowlist 行为不变；
- 同批工具实际执行顺序与模型顺序一致，任一失败后的工具只生成配对 skip result；
- 同一路径文件操作严格串行，不同路径可并发，取消/异常后队列不死锁且空 key 被回收；
- 恢复历史 JSONL 时的 tool call/result 展示与实时展示使用相同 formatter；
- 默认离线检查、构建和隔离 wheel CLI 验证全部通过。

## 4. 明确不做

- 不改变三包依赖方向，不移动 `forge` console entry point；
- 不增加第四个 Python 包、第二个分发、第二种语言或新服务；
- 不创建自定义 Agent loop、`StateGraph` 或长期 checkpointer；
- 不用 `ToolSet` 替代 LangChain tools，不让它进入 JSONL；
- 不在 `ToolDefinition` 中保存 Textual/Rich component 或 renderer；
- 不实现 Pi 混合并发、resource-key 推断或自定义 `ToolNode`；
- 不增加 `grep/find/ls` 等新工具，不改工具权限边界；
- 不修改 Provider、Goal、Todo、HITL、subagent trace 或 session schema；
- 不增加环境变量、凭据、外部 API 或运行时依赖。

## 5. 稳定接口

### 5.1 `ToolDefinition`

实现位置：`src/forge_coding/tools/definition.py`

存储字段：

| 字段 | 类型 | 规则 |
|---|---|---|
| `tool` | `BaseTool` | 必填，唯一执行/schema/description 事实 |
| `label` | `str` | 必填，非空；内置工具默认等于 tool name |
| `prompt_snippet` | `str | None` | 缺省不进入 Available tools |
| `prompt_guidelines` | `tuple[str, ...]` | 默认空元组，保持声明顺序 |

兼容只读属性/方法：

| 接口 | 行为 |
|---|---|
| `.name` | 返回 `tool.name` |
| `.description` | 返回 `tool.description` |
| `.args_schema` | 返回 `tool.args_schema` |
| `.input_schema` | 从 `tool.tool_call_schema.model_json_schema()` 即时派生 |
| `.executor` | 仅 `ForgeStructuredTool` 可用，返回其 direct executor |
| `.to_langchain_tool()` | 返回同一个 `tool` 对象，不再转换或复制 |

兼容接口保留一个发布周期；本次不删除现有顶层导出和
`create_{read,write,edit,bash}_tool_definition()`。

### 5.2 `ForgeStructuredTool`

继续继承 `StructuredTool`，只增加 Forge direct-execution seam：

- 生产调用走 LangChain `ainvoke()` 和 `ToolRuntime[ForgeRuntimeContext]`；
- `execute(arguments, signal)` 仅供 slash/direct execution 和测试；
- 内部只保存 direct executor，不保存整个 `ToolDefinition`；
- 删除 `forge_input_schema`、`prompt_snippet`、`prompt_guidelines` 和 `_definition`；
- `response_format="content_and_artifact"`、`handle_tool_error=True` 与
  `AgentToolResult` artifact 形状保持不变。

### 5.3 `ToolSet`

实现位置：`src/forge_coding/tools/tool_set.py`

公开行为：

- 构造参数是有序 `tuple[ToolDefinition, ...]`；
- 构造时拒绝重复或空 tool name；
- `.tools` 返回同序 `tuple[BaseTool, ...]`；
- `.by_name` 返回只读 name-to-definition mapping；
- `.select(names)` 按原 ToolSet 顺序返回子集，并拒绝重复请求和未知名称；
- `.with_tools(...)` 追加 plain `BaseTool` 时生成 label=name、无 prompt contribution 的
  definition；
- 不持有运行状态、锁、session、renderer 或持久化数据。

新增 `create_coding_tool_set()`。现有 `create_coding_tools()` 保留，并精确返回
`list(create_coding_tool_set(...).tools)`，避免调用方迁移。

### 5.4 展示注册表

实现位置：`src/forge_cli/tool_rendering.py`

- 注册表只包含纯 call/result formatter，不返回 Textual widget；
- 内置 `read/write/edit/bash/task` formatter 以 tool name 注册；
- 未注册工具使用当前 generic fallback；
- `forge_cli.formatting` 保留现有函数名，内部委托注册表；
- transcript renderer、TUI live state 和 JSONL restore path 都调用相同入口；
- 注册表不进入 `forge_coding`，也不反向改变 AgentEvent。

## 6. 分阶段实施

每个阶段都可独立合并。任一阶段停止后，Forge 仍可运行、测试和发布。

### Phase 1：收口当前工具模块拆分

目的：先把当前工作区已经完成的行为保持型拆分独立落地，避免与后续契约重构混在一起。

文件范围：

- `src/forge_coding/tools.py` 删除；
- `src/forge_coding/tools/{__init__,base,registry,common,truncation,edit_diff,shell}.py`；
- `src/forge_coding/tools/{read,write,edit,bash}.py`；
- `src/forge_agent/{langchain_runtime,subagents}.py`；
- `src/forge_coding/context_window.py`；
- `tests/test_coding_tools.py`。

实施要求：

1. 保留当前顶层 import surface 和四个工厂的行为；
2. 每个内置工具使用显式 Pydantic input model；
3. root/subagent `create_agent()` 都声明 `context_schema=ForgeRuntimeContext`；
4. token 估算使用 model-visible `tool_call_schema`；
5. 不引入本方案 Phase 2 的新契约；
6. 独立提交建议：`refactor(tools): modularize LangChain-native coding tools`。

验收：

- 既有工具 success/error/cancellation/timeout/truncation/path/symlink 测试不变；
- runtime 不出现在 provider-visible schema；
- `create_coding_tools()` 顺序不变；
- 完整门槛通过后工作区只剩后续文档或下一阶段改动。

回滚：单独 revert Phase 1 commit 即恢复单文件工具实现；无数据迁移。

### Phase 2：建立单一工具事实与 `ToolSet`

目的：删除当前 adapter 型 `ToolDefinition` 的重复 schema/executor 状态，同时不破坏公开工厂。

新增文件：

- `src/forge_coding/tools/definition.py`；
- `src/forge_coding/tools/tool_set.py`；
- `tests/test_tool_set.py`。

修改文件：

- `src/forge_coding/tools/base.py`；
- `src/forge_coding/tools/registry.py`；
- `src/forge_coding/tools/{read,write,edit,bash}.py`；
- `src/forge_coding/tools/__init__.py`；
- `src/forge_coding/__init__.py`；
- `src/forge_coding/system_prompt.py`；
- `src/forge_coding/session.py`；
- `src/forge_coding/subagents.py`；
- `tests/{test_coding_tools,test_system_prompt,test_coding_session,test_subagents}.py`；
- `docs/architecture.md` 与 `docs/dev/langchain-native-migration-plan.md`。

实施顺序：

1. 新增 `ToolDefinition` 和 `ToolSet`，先覆盖唯一性、顺序、选择与兼容属性测试；
2. 让 `ForgeStructuredTool` 只持有 direct executor；
3. 四个内置工具先创建 native tool，再组装 definition；
4. 新增 `create_coding_tool_set()`，让兼容 `create_coding_tools()` 委托它；
5. `CodingSession` 内部把 built-in、custom、Goal/HITL/Todo 和 `task` 工具归一为一个
   ToolSet；传给 `AgentHarness`/`create_agent()` 的仍是 `.tools`；
6. system prompt 从 definitions 读取 snippet/guidelines，不再 `getattr(tool, ...)`；
7. subagent profile 用 `ToolSet.select()` 做 allowlist，仍禁止递归 `task`；
8. 删除动态 JSON-schema-to-Pydantic 转换、`forge_input_schema` 和 tool 上的 prompt attrs；
9. 更新架构文档，明确 ToolSet 是 product catalog，不是 runtime 或持久化事实；
10. 独立提交建议：`refactor(tools): introduce product tool catalog`。

验收：

- custom `BaseTool` 仍能直接传入 `CodingSessionConfig`；
- built-in 和 task 的 provider schema、description、artifact 与错误状态不变；
- prompt 中工具顺序和 guideline 去重行为不变；
- subagent role tool intersection 与未知工具错误不变；
- 兼容属性由 native tool 派生，不保存副本；
- `tests/test_architecture.py` 增加断言：`forge_agent` 不出现 `ToolDefinition`/`ToolSet`，
  `forge_coding` 不导入 `forge_cli`。

回滚：单独 revert Phase 2 commit；Phase 1 的模块化工具仍可独立运行，无 session 数据变化。

### Phase 3：加入严格顺序与失败即停的工具 middleware

目的：让同一 agent 批次中的工具副作用、错误和结果配对具有确定语义，默认用正确性换并发性能。

新增文件：

- `src/forge_agent/tool_execution.py`；
- `tests/test_tool_execution_middleware.py`。

修改文件：

- `src/forge_agent/langchain_runtime.py`；
- `src/forge_agent/subagents.py`；
- `src/forge_agent/__init__.py`；
- `src/forge_coding/system_prompt.py`；
- `tests/{test_langchain_runtime,test_agent_harness,test_subagents,test_system_prompt}.py`；
- `AGENTS.md`；
- `docs/architecture.md` 与 `docs/dev/langchain-native-migration-plan.md`。

实施要求：

1. `SequentialToolCallMiddleware` 只使用公开 `AgentMiddleware.awrap_tool_call`、
   `ToolCallRequest.state`、`ToolMessage` 和 handler；
2. `_agent_middleware()` 始终把它放在 middleware 列表首位，使其成为最外层 tool wrapper；
3. `SubagentRunner` 创建 child graph 时同样把新 middleware 放在首位；
4. middleware 从最新 AIMessage 派生有序、非空、无重复的 tool_call_id tuple；
5. 公平 `asyncio.Lock` 一次只允许一个 handler 执行，并检查进入者是否为下一个预期 ID；
6. handler 返回 error ToolMessage 或抛出普通 Exception 时，将批次标记为失败；异常转换为
   bounded error ToolMessage，不暴露 traceback、绝对路径或原始参数；
7. 失败后的调用直接返回同 ID/name 的 bounded error ToolMessage，不调用下层 middleware/tool；
8. `CancelledError` 原样抛出；`finally` 必须释放 lock，现有 cancellation settlement 负责
   持久化层的未完成调用配对；
9. batch key 变化时重置常量大小状态；不维护跨批次 map、barrier 或任务 registry；
10. system prompt 增加规则：依赖前序返回值的工具必须下一轮调用；
11. 独立提交建议：`fix(agent): execute tool batches sequentially`。

验收：

- 两个 sleep tool 的实际 start/end 严格为 `first start/end, second start/end`；
- ToolMessage 顺序和 tool_call_id 与 AIMessage 顺序一致；
- 第一个工具返回 error 或抛错时，后续 handler 调用次数为零且每个调用都有 error ToolMessage；
- duplicate/missing/unknown/out-of-order tool_call_id 全部 fail closed，不产生副作用；
- 当前工具取消时 lock 释放，graph/Session 取消结算不死锁；
- Goal validation、Todo、HITL interrupt/resume 与 steering 组合不改变顺序且不重复执行；
- root 与 child agent 使用同一 middleware，多个 `task` 调用不再并发；
- 无工具和单工具批次行为不变。

回滚：revert Phase 3 后恢复 LangChain 默认并行；Phase 1/2 仍可独立运行，无数据迁移。

### Phase 4：用文件操作队列做第二道防线

目的：覆盖 direct execution、同进程多 session 等绕过单个 agent middleware 的文件访问，
并修复当前模块级 lock map 不回收的问题。

新增文件：

- `src/forge_coding/tools/file_operation_queue.py`；
- `tests/test_file_operation_queue.py`。

修改文件：

- `src/forge_coding/tools/base.py`；
- `src/forge_coding/tools/{read,write,edit}.py`；
- `tests/test_coding_tools.py`。

实施要求：

1. 使用 process-local、跨 event loop 的共享 `FileOperationQueue`；
2. key 使用完成 workspace boundary 检查后的 resolved path；
3. `read`、`write`、`edit` 对同一路径都走同一个独占 FIFO 队列；
4. 不同路径不共享锁，保持可并发；
5. 底层文件 await 完成或失败前不得提前释放；
6. 取消、异常和正常完成都结算 waiter，并回收无 owner/无 waiter 的 path state；
7. direct execution 与 LangChain execution 使用同一个共享队列；
8. 删除模块级 `_file_locks`；
9. 不尝试把 bash command 映射为文件 key；
10. 独立提交建议：`fix(tools): serialize same-file operations`。

验收：

- 同文件 read/write/edit 的执行区间不重叠且 FIFO；
- 不同文件的执行区间可重叠；
- 第一个调用取消或抛错后第二个调用继续；
- 所有调用结束后 path state 数为零；
- 多个 event loop 不复用彼此的 asyncio primitive；
- symlink escape、final write symlink 和 exact edit 规则继续 fail closed。

回滚：revert Phase 4 后仍有 Phase 3 的 agent 内严格串行；只失去 direct/multi-session 防线。

### Phase 5：统一 CLI/TUI 工具展示入口

目的：学习 Pi 的“每类工具可定义展示”，但保持 Python 分层与 Forge Event 边界。

新增文件：

- `src/forge_cli/tool_rendering.py`；
- `tests/test_tool_rendering.py`。

修改文件：

- `src/forge_cli/formatting.py`；
- `src/forge_cli/rendering/transcript.py`；
- `src/forge_cli/tui/state.py`；
- 仅在仍存在重复分支时修改 `src/forge_cli/tui/widgets.py`；
- `tests/{test_rendering,test_tui_adapter,test_tui_app}.py`；
- `docs/architecture.md`。

实施要求：

1. 把现有 read/write/edit/bash 特判迁入纯 formatter registry；
2. `format_tool_call_block()`、`format_tool_call_invocation()`、
   `format_tool_result_block()` 保持公开签名；
3. live event 与 restored `ToolMessage` 使用同一 formatter；
4. `task` 只显示 role/name 等已 allowlist 的 product projection，不显示 child prompt、
   原始 tool args/result 或内部 ID；
5. 未注册工具保持通用 name + bounded args/result preview；
6. model-provided 文本继续禁用 markup，不让内容解释为 Rich/Textual markup；
7. 独立提交建议：`refactor(cli): centralize tool presentation`。

验收：

- 同一 tool call/result 的 plain、TUI live、JSONL restore 文本语义一致；
- edit patch、bash timeout、read line range 和 unknown tool fallback 保持现有快照；
- 大结果仍受行数与字符数预算限制；
- 不新增 raw tool args/results、绝对路径、child trace、凭据或 reasoning 暴露。

回滚：revert Phase 5；ToolSet、顺序 middleware 和文件队列不依赖展示注册表。

## 7. 验证矩阵

每个阶段先跑对应 targeted tests，再跑完整门槛。

### 7.1 Targeted tests

```bash
uv run pytest tests/test_coding_tools.py tests/test_context_window.py
uv run pytest tests/test_tool_set.py tests/test_system_prompt.py tests/test_subagents.py
uv run pytest tests/test_tool_execution_middleware.py tests/test_langchain_runtime.py tests/test_agent_harness.py
uv run pytest tests/test_file_operation_queue.py tests/test_coding_tools.py tests/test_coding_session.py
uv run pytest tests/test_tool_rendering.py tests/test_rendering.py tests/test_tui_adapter.py
uv run pytest tests/test_architecture.py tests/test_package_metadata.py
```

只运行已经存在于当前阶段的测试文件；新测试文件随对应阶段一起加入。

### 7.2 完整自动门槛

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
uv run pytest
uv run forge --help
uv run forge --version
uv lock --check
uv build
git diff --check
```

### 7.3 隔离安装

从 `uv build` 生成的 wheel 在新的临时虚拟环境中安装并验证：

```text
forge --help
forge --version
python -c "import forge_agent, forge_coding, forge_cli"
```

临时目录使用系统临时目录创建；清理只针对本次明确创建并已解析确认的临时目录。

### 7.4 手工验收

- 启动空 session，read/write/edit/bash 各完成一次；
- 在同一模型消息中依次触发 read、edit、bash，确认实际执行和结果都按 tool-call 顺序；
- 让第一个工具失败，确认后续工具未执行且每个调用都有配对 error ToolMessage；
- 触发 direct 同文件 read/write/edit 并发 harness，确认文件队列 FIFO；
- 恢复包含 read/edit/bash/task 的历史 JSONL，确认展示不变；
- 在 TUI 中检查长输出折叠、edit patch、取消和 subagent task block；
- print/json 模式各运行一次，确认事件和退出码不变。

真实 Provider smoke 仍为 opt-in；没有用户本地凭据授权时不运行，也不把未运行描述为通过。

## 8. 风险与防线

| 风险 | 防线 |
|---|---|
| ToolSet 变成第二套 runtime | 只持有 definition；执行永远委托 `BaseTool`，架构测试禁止进入 `forge_agent` |
| 兼容 API 被无意删除 | 保留现有工厂、属性和 `create_coding_tools()`，新增兼容回归 |
| prompt 与 provider tool 列表漂移 | ToolSet 单一顺序；测试 active definitions 与 `.tools` name 集合一致 |
| 上游调度顺序变化 | middleware 在副作用前校验预期 ID；不一致时 fail closed，契约测试固定当前版本 |
| 前序失败后仍执行危险后续调用 | 批次标记失败，后续 handler 短路但生成配对 error ToolMessage |
| middleware 取消后死锁 | 公平 lock + `async with`/`finally`，取消与 HITL resume 回归 |
| 文件队列取消后死锁 | `finally` 结算、等待者接力、空 path state 回收测试 |
| UI registry 泄露内部数据 | 只消费 AgentEvent/JSON-safe result projection，维持 preview budget 与 allowlist |
| LangChain 升级改变 ToolNode 行为 | 固定当前依赖范围；升级时重新检查 ToolNode 并跑顺序/失败配对契约 |
| 外部进程与 bash 后台任务竞争 | 明确不在 asyncio 保证内；shell 不是沙箱，工具文档不得宣称跨进程安全 |
| 重构范围过大难以 review | 五个独立提交，每个阶段完整可运行，不跨阶段混合移动和行为变化 |

最脆弱的假设是 Forge 在可预见阶段仍是一个产品、一个 CLI/TUI。如果出现第二个真实产品
需要复用终端组件，当前 `forge_cli` 边界会阻碍复用；届时应基于两个真实消费者提取
`forge_tui`，而不是现在先倒置依赖。

## 9. 发布与回滚

- 无 JSONL/schema/config migration，不需要数据备份或迁移命令；
- 无新依赖、环境变量、API key、MCP server 或第三方 CLI；
- 每阶段使用独立 Conventional Commit，禁止 squash 掩盖阶段边界；
- 每阶段合并前检查工作区，避免混入用户当前未提交的其他改动；
- 发布说明只在行为可见时记录：Phase 1/2 属内部结构，Phase 3/4 属工具安全语义，Phase 5
  仅在输出确有变化时记录；
- 回滚按 Phase 5 → Phase 4 → Phase 3 → Phase 2 → Phase 1 逆序 revert；任意中间状态都可运行；
- 不重写历史、不自动 push、不自动创建分支或 PR。

## 10. 被拒绝的替代方案

### 方案 A：`forge_coding → forge_cli`，把 coding 变成产品组装层

拒绝。它需要迁移超过 10 个与 coding domain 紧耦合的 CLI/TUI 文件、改变 entry point 和
架构测试，却没有第二个 TUI 消费者，也不会减少耦合。它是在复制 Pi 包拓扑，不是在复用
Pi 的设计机制。

### 方案 B：保留 LangChain 默认并行，只锁推断出的冲突资源

拒绝。它无法可靠覆盖 read/write、bash/file tool、不同路径但存在业务依赖的副作用，也无法
证明模型把同一批调用都视为独立操作。Forge 宁可损失并行收益，也不接受不确定副作用顺序。

### 方案 C：实现 Pi 的 parallel/sequential 批次 scheduler

拒绝。当前官方 hook 是 per-call，不提供 prepare/execute 的 batch 扩展点；完整复刻需要自定义
ToolNode 或复杂 barrier 状态。Forge 的全局严格串行只需一个最外层公开 middleware 和常量大小
批次状态，更容易审计和回滚。

### 最小方案：只提交当前 tools 模块拆分

可行，且对应 Phase 1。如果后续不继续，Forge 已能获得主要可读性收益；但
`ToolDefinition` 的重复 schema/executor 和全局文件锁仍会保留，因此不是本方案推荐的最终状态。

## 11. 决策摘要

- **Building**：LangChain-native 工具目录、严格顺序且失败即停的 tool middleware、同文件操作队列、
  CLI 纯展示注册表。
- **Not building**：包依赖倒置、工具并行模式、自定义 ToolNode、第二 Agent loop、第四个包。
- **Approach**：保留当前三层；原生 `BaseTool` 是唯一执行事实；产品 metadata 与 UI formatter
  分别留在 `forge_coding` 和 `forge_cli`；`forge_agent` 统一执行安全语义。
- **Key decisions**：`ToolDefinition` 保留名称但收缩字段；新增 `ToolSet`；不加入 renderer 与
  execution mode；所有工具串行；前序失败后剩余调用短路并保持 ToolMessage 配对；现有公开工厂
  和 JSONL/Event 契约保持兼容。
- **Unknowns**：无阻塞未知项。跨 OS 进程和 bash 脱离后台任务明确不在保证范围内。

## 12. 依据

- LangChain Tools：<https://docs.langchain.com/oss/python/langchain/tools>
- LangChain Runtime：<https://docs.langchain.com/oss/python/langchain/runtime>
- LangChain Custom Middleware：<https://docs.langchain.com/oss/python/langchain/middleware/custom>
- LangGraph ToolNode reference：
  <https://reference.langchain.com/python/langgraph.prebuilt/tool_node/ToolNode>
- Forge 当前架构：`docs/architecture.md`
- Forge LangChain-native 架构记录：`docs/dev/langchain-native-migration-plan.md`
- Pi 0.84.2 本机实现：`pi-agent-core/dist/agent-loop.js`、
  `pi-coding-agent/dist/core/extensions/types.d.ts`、
  `pi-coding-agent/dist/core/tools/file-mutation-queue.js`

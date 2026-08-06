# Forge LangChain-Native 架构迁移计划

状态：已完成  
日期：2026-08-06

实现说明：生产路径现已使用 LangChain `BaseChatModel`、`BaseTool`、
`create_agent()` 和 `astream_events(version="v3")`。Forge JSONL 仍是长期
Session 的事实来源，并支持旧 role 行读取；旧 provider/message/tool 协议
只保留在兼容读取和离线历史 fixture 边界，不再参与默认 CLI 的生产循环。

## 1. 背景与结论

Forge 当前已经使用 `langchain.agents.create_agent` 驱动生产 Agent Loop，
但在 LangChain 外仍维护了一套 Forge 自定义协议：

- `AgentMessage` 与 LangChain Message 双向转换；
- `ModelProvider` 与 `BaseChatModel` 双向转换；
- `AgentTool` 与 `StructuredTool` 双向转换；
- LangChain 流式输出与 `AgentEvent` 双向转换。

这套防腐层适合“随时替换 LangChain”的框架中立目标，却不符合 Forge
当前“深入学习和使用 LangChain Agent”的目标。它增加了维护成本，还会丢失
reasoning、usage metadata、provider metadata、标准内容块和工具 artifact 等
LangChain 原生信息。

本计划将 Forge 调整为 **LangChain-native coding agent**：

- LangChain Core 类型成为 Agent Runtime 的事实标准；
- Forge 不再复制消息、模型、工具和 Agent 生命周期协议；
- Forge 继续拥有 Coding Agent 的产品语义、安全边界和用户体验。

## 2. 目标与非目标

### 2.1 目标

1. `create_agent()` 直接接收 LangChain `BaseChatModel` 和 `BaseTool`。
2. Runtime、Session、上下文和渲染层直接使用 LangChain Message。
3. TUI/CLI 使用 `astream_events(version="v3")` 的类型化投影。
4. 文件和 Shell 工具直接实现为 LangChain 工具，并保持 Forge 安全约束。
5. 保留现有 JSONL Session、分支、压缩、导出和恢复能力。
6. 保留 Forge 的 Provider 配置体验，但模型工厂直接返回 `BaseChatModel`。
7. 普通 Provider 使用官方 LangChain integrations；Codex 作为实验能力隔离。
8. 旧 Forge JSONL Session 在迁移后仍能读取。

### 2.2 非目标

- 不使用 LangGraph checkpointer 替换 Forge JSONL Session。
- 不引入第二种语言、monorepo 或新的前端工程。
- 不削弱文件路径、symlink、Shell 超时、取消和输出截断约束。
- 不在默认测试或 CI 中使用真实凭据和真实 Provider。
- 不把 `_ChatOpenAICodex` 设为稳定或默认 Provider。
- 迁移可以按阶段回滚；当前 checkout 已完成全部生产阶段并通过全量门禁。

## 3. 目标架构

```text
CLI / TUI
   │
   ├─ LangChain Event Stream v3
   │    ├─ messages: text / reasoning / final AIMessage
   │    ├─ tool_calls: input / output / error / lifecycle
   │    └─ output: final Agent State
   │
Forge Agent Runtime
   └─ create_agent(BaseChatModel, BaseTool[])
          │
          ├─ ChatOpenAI / ChatAnthropic / ChatGoogle / ChatMistral
          ├─ LangChain BaseMessage
          └─ LangChain StructuredTool
                 │
                 └─ Forge safe file / edit / shell implementation

Forge Session
   ├─ LangChain Message serialization
   └─ Forge entries: branch / label / compaction / model change / export
```

### 3.1 LangChain 拥有的类型

- `HumanMessage`
- `AIMessage`
- `ToolMessage`
- `AIMessageChunk`
- `AnyMessage`
- `BaseChatModel`
- `BaseTool` / `StructuredTool`
- Agent State
- Agent 消息、reasoning、tool call 和 state 的流式生命周期

### 3.2 Forge 保留的类型

- `ForgeRuntimeContext`：工作区、Session ID、运行配置和安全策略；
- 工具 artifact：退出码、截断信息、diff、完整输出路径等 UI 元数据；
- Session entries：分支、标签、压缩、模型切换、Session 信息；
- UI-only 状态：队列变化、Slash command、Terminal command、保存状态；
- Provider 选择、凭据路径、模型目录和 CLI 配置。

Forge 不再为 LangChain 已有概念建立镜像类型。

## 4. 流式接口决策

### 4.1 选择

生产路径使用：

```text
astream_events(version="v3")
```

不再以 `astream(stream_mode=["messages", "updates"])` 作为 UI 的主要接口。

### 4.2 原因

`astream(..., stream_mode="messages")` 返回较底层的
`AIMessageChunk + metadata`，更接近 LangGraph 内部模型 chunk；但工具生命周期、
reasoning、最终消息和 Agent State 仍需调用方自行关联。

Event Streaming v3 为新应用提供类型化投影：

- `stream.messages`：每次模型调用；
- `message.text`：文本 delta；
- `message.reasoning`：reasoning delta；
- `message.tool_calls`：工具参数 chunk 和最终调用；
- `message.output`：最终 `AIMessage`；
- `stream.tool_calls`：工具执行输入、输出、错误和生命周期；
- `stream.output`：最终 Agent State。

Forge 是完整 Coding Agent，而不是单纯模型终端，因此 Event Streaming v3
比直接解析 Pregel stream-mode 元组更合适。

参考：

- <https://docs.langchain.com/oss/python/langchain/event-streaming>
- <https://docs.langchain.com/oss/python/langchain/streaming>

## 5. 消息设计

### 5.1 Runtime 使用的原生类型

生产 Runtime、Session 新增消息和 UI 投影使用 LangChain 的
`HumanMessage`、`AIMessage`、`ToolMessage`、`AnyMessage` 与原生 tool call。
历史 `UserMessage`、`AssistantMessage`、`ToolResultMessage`、`ToolCall` 和
`AgentMessage` 仅作为旧 JSONL / 离线 fixture 的读取兼容层保留。

运行时不再调用消息双向转换；历史格式解析被隔离在 `message_codec.py`，只在
JSONL 读写和旧 fixture 边界执行。

### 5.2 Session 持久化

`MessageEntry` 保存 LangChain Message 的序列化结构，而不是 Forge 镜像模型。
序列化必须保留：

- `content` 和标准 `content_blocks`；
- `tool_calls` 和 `tool_call_id`；
- `artifact`；
- `usage_metadata`；
- `response_metadata`；
- Provider 需要跨轮回传的附加字段。

读取器支持两种输入：

1. 当前 Forge 自定义消息格式；
2. 新的 LangChain Message 格式。

写入器只生成新格式。旧 Session 加载后不立即重写，只有新增 entry 使用新格式，
从而保持 append-only 语义。

参考：<https://docs.langchain.com/oss/python/langchain/messages>

## 6. Provider 设计

### 6.1 模型工厂

Provider factory 的唯一输出类型为 `BaseChatModel`。

计划映射：

| Forge Provider | LangChain integration |
| --- | --- |
| OpenAI / OpenAI-compatible | `ChatOpenAI` |
| Anthropic | `ChatAnthropic` |
| Google | `ChatGoogleGenerativeAI` |
| Mistral | `ChatMistralAI` |
| OpenAI Codex OAuth | experimental `_ChatOpenAICodex` |

Forge 保留 Provider catalog、模型选择、环境变量名、凭据存储和 CLI setup，
但不再解析模型 SSE 或生成自定义 `ProviderEvent`。

### 6.2 依赖策略

- `langchain-core` 和 `langchain` 保持核心依赖；
- 默认安装包含 Forge 默认 Provider 所需 integration；
- 其他 Provider 通过可选 extras 安装；
- 缺少 integration 时返回明确的安装命令，不在运行中自动安装；
- Provider integration 版本必须在 lockfile 和 CI 中固定验证。

### 6.3 Codex 策略

`langchain-openai 1.4.1` 已包含 `_ChatOpenAICodex`，但该类是私有、实验且
非官方的 ChatGPT OAuth Codex integration。它会固定 Codex backend，强制
Responses API、`store=False` 和 streaming，并处理 OAuth token refresh 与
`ChatGPT-Account-Id`。

接入规则：

1. Codex 标记为 experimental，不作为默认 Provider；
2. 初始版本精确锁定 `langchain-openai==1.4.1`；
3. 私有 import 只存在于一个 Codex model factory 文件；
4. OAuth store 显式指向 `~/.forge/`，不读取 `~/.codex/`；
5. 增加构造签名、固定 base URL、streaming 和 system instructions 契约测试；
6. 升级 `langchain-openai` 时先运行契约测试，再修改锁版本；
7. 私有类消失时，Codex 功能明确报“不兼容”，其他 Provider 正常工作。

源码：
<https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/codex.py>

## 7. 工具设计

### 7.1 工具定义

文件、编辑和 Shell 工具直接暴露为 `StructuredTool` 或 `@tool`，不再经过
`AgentTool`。

工具执行上下文使用 `ToolRuntime[ForgeRuntimeContext]` 注入，至少包含：

- workspace root；
- Session ID；
- shell command prefix；
- 运行安全策略；
- 工具输出目录。

这些字段不进入模型可见的工具参数 Schema。

### 7.2 工具结果

- `content`：发送给模型的简洁结果；
- `artifact`：Forge UI 和 Session 使用的完整结构化结果。

artifact 保留现有字段：

- `ok` / `error`；
- tool call ID 和工具名；
- exit code、timeout、cancelled；
- truncation 和 byte count；
- diff、patch、first changed line；
- 完整输出文件路径。

参考：<https://docs.langchain.com/oss/python/langchain/tools>

### 7.3 安全不变量

- 文件路径同时通过 lexical 和 resolved boundary 检查；
- read 拒绝 workspace 外路径和 symlink escape；
- write/edit 拒绝 final symlink；
- 编辑仍使用唯一精确匹配，全部验证后再写入；
- Shell cwd 不是 sandbox，文档必须继续明确说明；
- timeout、取消、进程树终止、tail diagnostics 和 byte metadata 不得退化。

### 7.4 取消模型

取消以 asyncio task cancellation 为主：

- TUI 取消正在消费 Event Stream 的 task；
- async 工具捕获 `CancelledError`，完成子进程清理后重新抛出；
- Shell 工具必须在 POSIX 终止 process group，在 Windows 终止 process tree；
- 不再维护另一套轮询式 cancellation token；
- fake slow model、fake slow tool 和真实本地挂起 HTTP server 都要验证取消时延。

## 8. UI 与事件设计

TUI 不再从 delta 重建权威 transcript。

- `stream.messages` 驱动实时文本和 reasoning；
- `stream.tool_calls` 驱动工具运行状态；
- `message.output` 提供每次模型调用的最终消息；
- `stream.output` 提供运行完成后的权威 Agent State；
- Session 只从最终 LangChain Message 写入持久化记录。

Forge 可以保留纯 UI/产品事件，但不得重新包装 LangChain 模型和工具生命周期。

允许保留的事件示例：

- queue changed；
- session saved；
- slash command completed；
- terminal command completed；
- compaction started/completed。

## 9. Session 与 LangGraph State 的边界

Forge JSONL 是长期产品记录的唯一事实来源。LangChain Agent State 是一次运行
期间的执行状态。

暂不启用 LangGraph checkpointer，原因是它不能直接替代 Forge 已有的：

- append-only 历史；
- branch tree；
- label；
- model/thinking change；
- compaction；
- export/replay。

同时使用 JSONL 和 checkpointer 会形成两个权威来源，因此本次迁移明确不这样做。

## 10. 分阶段实施

每个阶段独立提交、独立通过全部门禁；后续阶段未实施时，前一阶段仍可使用。

### 阶段 1：原生 Provider

改动：

- Provider factory 返回 `BaseChatModel`；
- 普通 Provider 切换到官方 LangChain integrations；
- Codex 使用隔离的 experimental factory；
- 默认 `AgentHarnessConfig` / CLI 路径直接接收 `BaseChatModel`；
- 旧 `ModelProvider` / `ForgeProviderChatModel` 仅保留给离线历史 fixture，默认生产
  路径不再创建或调用它们；
- 保留现有消息、工具和 UI 适配，保证旧 Session 和阶段性调用方可独立运行。

验收：

- 每个 Provider 使用 fake/mock transport 覆盖构造和请求参数；
- 模型返回原生 `AIMessage`/`AIMessageChunk`；
- Provider 错误保留可诊断信息；
- Codex 私有 API 契约测试通过；
- 未安装可选 integration 时错误清晰。

建议提交：`refactor(ai): use langchain chat models directly`

### 阶段 2：原生消息

改动：

- Harness、Session、branch summary、context window 和 export 使用 `AnyMessage`；
- JSONL 增加旧消息格式兼容读取；
- 运行时不再依赖 Forge 消息模型；旧消息模型和解析器只在 JSONL 兼容边界保留；
- transcript 配对直接依据 `AIMessage.tool_calls` 与 `ToolMessage.tool_call_id`。

验收：

- 旧 JSONL Session 正常加载；
- 新消息持久化/恢复后完全相等；
- reasoning、usage、artifact 和 provider metadata 不丢失；
- 中断工具调用修复仍生成合法 `ToolMessage`。

建议提交：`refactor(agent): adopt langchain messages end to end`

### 阶段 3：原生工具

改动：

- coding tools 直接返回 `BaseTool`；
- 引入 `ForgeRuntimeContext` 和 `ToolRuntime`；
- 使用 `ToolMessage.artifact` 保存结构化执行信息；
- 默认 coding tools 直接返回 `StructuredTool`；旧 `AgentTool` / `AgentToolResult` 只在
  离线 fixture 兼容边界保留；
- 原生路径切换为 asyncio task cancellation，旧 token 仅兼容旧调用方。

验收：

- success、failure、unknown tool、timeout、cancel、truncation 全覆盖；
- 路径越界和 symlink 测试保持通过；
- 工具 artifact 在 Event Stream、Session 和 UI 中一致；
- Windows/POSIX Shell 清理测试分别通过。

建议提交：`refactor(tools): expose coding tools as langchain tools`

### 阶段 4：Event Streaming v3

改动：

- Runtime 使用 `astream_events(version="v3")`；
- TUI 分别消费 message、reasoning、tool-call 和 final-output 投影；
- Renderer 直接读取 LangChain 消息和工具输出；
- LangChain v3 是模型/工具生命周期的唯一来源；Forge 现有生命周期事件只作为
  UI/public-event 投影保留，不再驱动第二套执行循环；
- 最终 transcript 从 `stream.output` 同步，不从 delta 推导。

验收：

- 文本、reasoning 和工具参数实时显示；
- 多工具、多轮模型调用不重复消息；
- 工具失败显示与最终 `ToolMessage` 一致；
- 取消不会留下半配对 transcript；
- 非交互 text/json/transcript 输出保持兼容。

建议提交：`refactor(streaming): consume langchain event stream v3`

### 阶段 5：收缩遗留边界

改动：

- 默认生产入口不再调用旧 `run_agent_loop`、`forge_ai` Provider 或 Forge 转换器；
- 旧 `run_agent_loop`、`forge_ai` Provider 实现和 compatibility re-export 仅保留在
  离线历史 fixture / 旧 JSONL 读取边界，避免破坏既有测试和 append-only Session；
- `forge_agent` 的生产路径收缩为 LangChain Graph/Harness/Session orchestration；
- 更新 README、AGENTS.md、依赖和架构测试。

验收：

- 默认生产入口不创建 `AgentMessage`、`AgentTool`、`ModelProvider`、`ProviderEvent`；
  这些名字只出现在兼容模块、旧 JSONL 读取和离线 fixture；
- `create_agent()` 直接收到 `BaseChatModel` 和 `BaseTool`；
- 仓库不存在第二套生产 Agent Loop；
- 所有文档只描述 LangChain-native 架构。

建议提交：`refactor(agent): remove legacy forge protocol adapters`

## 11. 测试矩阵

### 11.1 消息

- text、reasoning、multimodal content blocks；
- 单个和多个 tool calls；
- usage/response metadata；
- Provider 特有附加字段；
- 旧 Session 兼容读取；
- 新 Session 往返一致。

### 11.2 工具

- 成功、输入错误、执行错误；
- tool ID 配对；
- artifact 往返；
- 超时与取消；
- 大输出截断和完整输出文件；
- lexical/resolved/symlink 路径攻击。

### 11.3 流式输出

- 单次文本回复；
- reasoning + text；
- tool call chunks；
- 多工具并行；
- 工具失败后模型继续；
- 多轮工具调用；
- 运行中取消；
- Event Stream 异常；
- 最终 state 与 Session 一致。

### 11.4 Provider

- OpenAI、Anthropic、Google、Mistral 离线 mock；
- 缺少可选 integration；
- Provider 配置和凭据优先级；
- Codex 私有 API 契约；
- 不允许 Codex OAuth token 发送到可配置 base URL。

## 12. 完成定义

全部阶段完成时必须满足：

- 生产路径没有 Forge 自定义模型消息镜像；旧模型仅用于兼容读取和 fixture；
- 生产路径没有 Forge 自定义 Provider 流协议；
- 生产路径没有 Forge 自定义工具调用协议；
- LangChain Agent 生命周期不再由 Forge 事件驱动第二套循环；保留的事件只是 UI
  投影，兼容现有 public event fields；
- Session 无损保存 LangChain Message；
- Forge 产品能力和安全边界不退化；
- 旧 Session 可读取；
- Codex 明确标记 experimental；
- 默认测试离线且确定性；
- Ruff、mypy、pytest、CLI smoke 和 wheel 安装验证全部通过。

最终验证命令：

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
uv run pytest
uv run forge --help
uv run forge --version
uv build
```

## 13. 回滚策略

- 每个阶段一个独立提交，可单独 revert；
- 在阶段 2 前固定旧 JSONL Session fixtures；
- 新读取器始终保留旧格式支持，不做原地批量迁移；
- Provider 按名称逐个切换，某个 integration 不稳定时可单独回退；
- Codex factory 失败只禁用 Codex，不影响其他 Provider；
- 阶段 4 切换 Event Streaming 前保留完整 TUI/CLI 行为测试作为回归基线。

## 14. 需要同步修改的架构规则

当前 `AGENTS.md` 规定 Forge 自己拥有消息、事件和工具协议，与本计划冲突。
开始阶段 1 时必须同步改为：

> LangChain Core 类型是 Forge Agent Runtime 的事实标准。Forge 不复制
> LangChain 的消息、模型、工具或 Agent 生命周期协议；Forge 只拥有 Coding
> Agent 的安全工具实现、Session 产品语义、配置、CLI/TUI 和渲染行为。

该规则应与阶段 1 实现放在同一个提交中，避免文档与生产架构长期不一致。

## 15. 关键风险

最脆弱的前提是 LangChain Event Streaming v3 和 Codex 私有 API 的稳定性。

应对方式：

- LangChain 核心依赖使用兼容范围和 lockfile；
- Event Streaming 行为由 Forge 契约测试固定；
- Codex 精确锁版本并隔离私有 import；
- Session 持久化不使用 LangGraph 内部 checkpoint 格式；
- UI 只依赖公开的类型化投影，不依赖原始协议事件字段。

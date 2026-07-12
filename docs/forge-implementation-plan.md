# Forge 实现计划书

## 1. 目标

Forge 是一个 TypeScript 编写的终端 coding agent。目标不是逐行复刻 [Tau](https://github.com/huggingface/tau.git)，而是学习它的分层方式和功能边界，做出功能等价、实现风格独立的版本。

最终 Forge 应具备 Tau 已有的主要能力：

- 终端交互模式和一次性 print 模式
- provider 中立的 agent loop
- 流式模型输出
- 工具调用循环
- 文件读写、精确编辑、shell 执行
- JSONL 持久会话
- session 恢复、导出、分支
- slash commands
- provider 配置、模型切换、凭证管理
- 项目指令发现，例如 `AGENTS.md`
- skills 和 prompt templates
- context 估算、手动压缩、自动压缩
- 可替换渲染层：text、json、transcript、TUI
- 诊断、测试、发布文档

## 2. 非目标

Forge 不追求和 Tau 的代码结构一一对应。

不做：

- Python 兼容层
- Tau 配置文件兼容
- Tau session 文件格式兼容
- 复用 Tau 的 prompt 文案、工具描述、命令名称细节
- 第一阶段就做完整 TUI 和多 provider

这些不是价值核心，照搬反而会让 Forge 缺少自己的边界。

## 3. 总体架构

采用单 package、分层目录结构。Forge 最终只发布一个 CLI，在核心 API 稳定前拆成 monorepo 只会增加构建、版本和依赖管理成本。

```text
src/
  agent/           -> provider-neutral agent brain
  providers/       -> model provider adapters
  tools/           -> filesystem, shell and external tools
  sessions/        -> session state and JSONL persistence
  config/          -> configuration and credential lookup
  cli/             -> commands and non-interactive rendering
  tui/             -> interactive terminal UI
  index.ts          -> executable entry point
```

数据流：

```text
User input
  -> CLI / TUI
  -> CodingSession
  -> AgentHarness
  -> AgentLoop
  -> ModelProvider
  -> Agent events
  -> Renderer / Session JSONL / UI
```

核心原则：

- `agent` 不知道文件系统、终端 UI、配置路径和第三方 SDK 类型
- `providers`、`tools`、`sessions`、`config` 依赖 `agent` 提供的协议，不能反向被 `agent` 依赖
- `cli` 和 `tui` 负责用户交互与渲染，只消费 agent events
- 工具是普通 async 函数，带 schema 和结构化结果
- 所有 UI 都消费事件，不直接插进 agent loop
- 只有需要独立发布 `@forge/core` 或 plugin SDK 时，才把单 package 拆成 pnpm workspace

## 4. 技术栈

推荐：

- Runtime：Node.js 24 LTS
- Language：TypeScript 7、strict mode、ESM only
- Package manager：pnpm
- Build：TypeScript 7 自带的 `tsc`，负责类型检查和输出 ESM JavaScript
- Development runner：`tsx`
- Test：Vitest
- Lint / format：Biome
- Schema：Zod v4
- Config：JSONC，使用 `jsonc-parser` 解析、Zod 校验
- CLI：Commander
- TUI：React + Ink
- Streaming protocol：`AsyncIterable` / `AsyncGenerator`
- Cancellation：`AbortController` / `AbortSignal`
- Shell：Node `child_process.spawn`，不要一开始引入复杂任务框架
- Storage：append-only JSONL
- Diagnostic logging：Phase 2 后引入 Pino，日志只写诊断通道
- IDs：Node `crypto.randomUUID()`
- Provider：使用官方 SDK 编写 Forge adapter；先接 OpenAI Responses API，再扩展 Anthropic 和 OpenAI-compatible
- MCP：后期接官方 TypeScript SDK

TypeScript 7 编译约束：

- `tsconfig.json` 显式设置 `module` 和 `moduleResolution` 为 `NodeNext`
- 显式设置 `rootDir: "src"`、`outDir: "dist"` 和 `types: ["node"]`，不依赖 TypeScript 7 的新默认值
- `strict`、`noUncheckedSideEffectImports` 和 `verbatimModuleSyntax` 显式开启
- 第一阶段不生成 `.d.ts`，因为 Forge 是应用，不是供其他项目导入的库
- Forge 和构建链不得导入 TypeScript 编程 API；TypeScript 7.0 暂时只保证 CLI 和语言服务能力
- 以后发布 `@forge/core` 或 plugin SDK 时，再评估支持 TypeScript 7 的 tsdown 稳定版，并启用 `isolatedDeclarations`

依据：

- TypeScript 7 已改为 Go 原生实现，标准 `typescript` 包继续提供 `tsc`，但 7.0 暂时没有编程 API。
- Forge 是 Node CLI 应用，不需要 bundler；直接使用 `tsc` 可以减少构建层和 TypeScript 7 迁移风险。
- OpenAI 官方 Node/TypeScript SDK 当前主入口是 Responses API，且支持流式响应。
- MCP 官方 TypeScript SDK 目前 v2 仍在 beta，生产能力应优先按稳定 v1.x 设计适配层，避免锁死在 beta API。

参考：

- TypeScript 7 announcement: https://devblogs.microsoft.com/typescript/announcing-typescript-7-0/
- tsdown TypeScript 7 support tracking: https://github.com/rolldown/tsdown/issues/1010
- OpenAI streaming docs: https://developers.openai.com/api/docs/guides/streaming-responses
- OpenAI Node SDK: https://github.com/openai/openai-node
- MCP TypeScript SDK: https://github.com/modelcontextprotocol/typescript-sdk

## 5. 核心数据模型

Forge 需要先固定这几类协议，后续所有功能围绕它们展开。

### Message

- `user`
- `assistant`
- `tool`

assistant message 包含：

- text content
- tool calls
- provider metadata

tool message 包含：

- tool call id
- tool name
- ok / error
- textual content
- structured data

### Event

事件是系统边界。至少包括：

- agent start / end
- turn start / end
- message start / delta / end
- thinking delta
- tool start / update / end
- retry
- queue update
- error

### Tool

工具定义包含：

- name
- description
- input schema
- executor
- prompt snippet
- prompt guidelines

executor 接收：

- arguments
- abort signal
- execution context

返回：

- ok
- content
- data
- error

### Provider

provider 只做一件事：把第三方模型流转换成 Forge 的 provider event。

provider 和 agent loop 都使用 `AsyncIterable` 输出事件，工具和 provider 统一接收 `AbortSignal`。不引入 RxJS，也不使用 EventEmitter 承担核心控制流。

它不应该知道：

- CLI
- session 文件
- TUI
- 具体项目路径

## 6. 功能地图：Tau 到 Forge

| Tau 能力 | Forge 对应能力 | 实现位置 |
|---|---|---|
| `tau_ai` provider 层 | `src/providers` | Phase 1 起步，Phase 4 完整 |
| `tau_agent` loop/harness | `src/agent` | Phase 1 |
| `tau_coding` session | `src/sessions` | Phase 2 |
| `read/write/edit/bash` | `src/tools/fs` 和 `src/tools/shell` | Phase 1 |
| Textual TUI | React + Ink TUI | Phase 6 |
| print mode | CLI one-shot mode | Phase 1 |
| JSONL sessions | append-only session log | Phase 2 |
| resume / branching | session tree | Phase 7 |
| slash commands | command registry | Phase 3 |
| provider catalog | provider config registry | Phase 4 |
| login / credentials | credential store | Phase 4 |
| AGENTS.md discovery | project context discovery | Phase 3 |
| skills | skill loader | Phase 5 |
| prompt templates | prompt template loader | Phase 5 |
| context accounting | token estimator | Phase 5 |
| compaction | summarization session entry | Phase 5 |
| export | session export | Phase 7 |
| renderers | event renderers | Phase 2 |
| update check / release notes | product polish | Phase 9 |

## 7. 阶段路线

每个阶段都必须能独立合并。Phase N 完成后，即使后续阶段不做，Forge 也应该处于可用状态。

### Phase 1：最小可用 coding agent

目标：跑通一次完整 coding agent 闭环。

包含：

- Node.js 24、TypeScript 7、pnpm 单 package 项目初始化
- `tsc` build / typecheck、`tsx` dev、Vitest、Biome 基础配置
- ESM 与 TypeScript 7 显式 `tsconfig.json` 配置
- `agent` message / event / tool / provider 协议
- agent loop
- agent harness
- OpenAI Responses provider
- 基础 system prompt
- `readFile`
- `writeFile`
- `editFile`
- `shell`
- one-shot CLI：`forge -p "..."` 或 `forge "..."` 
- 基础 text renderer
- 基础测试：无工具、单工具、多轮工具、未知工具、工具错误、取消

验收：

- 能让模型读取一个文件
- 能修改一个文件
- 能运行测试命令
- 能把工具结果继续发回模型并得到最终答复

不包含：

- 会话恢复
- TUI
- 多 provider
- 自动压缩
- monorepo 和独立 package 发布
- `.d.ts` 生成

### Phase 2：持久会话和渲染协议

目标：Forge 不再只是一次性工具，而是能记录、恢复基本会话，并通过稳定输出协议供人和脚本消费。文件导出保留到 Phase 7。

包含：

- append-only JSONL session storage
- session entry 类型
- message entry
- model change entry
- session info entry
- in-memory storage for tests
- session manager
- `forge sessions`
- `forge --resume <id>`
- text renderer：成功后只输出最终 assistant 文本
- json renderer：每个 agent event 输出一行 JSON
- transcript renderer：流式文本输出和工具状态
- CLI print mode 输出失败时返回非零退出码

验收：

- 所有 user / assistant / tool message 都写入 JSONL
- 能恢复上一轮上下文继续对话
- 能输出 JSON 事件流，方便脚本集成

### Phase 3：项目上下文和命令系统

目标：Forge 开始像 coding agent，而不是普通 chat CLI。

包含：

- project root 发现
- 从当前目录向上读取 `AGENTS.md`
- 支持 `.forge/AGENTS.md`
- command registry
- slash commands：
  - `/help`
  - `/model`
  - `/sessions`
  - `/resume`
  - `/compact`
  - `/export`
  - `/clear`
  - `/tools`
- terminal command shortcut：
  - `!cmd` 执行并加入上下文
  - `!!cmd` 只执行不加入上下文
- system prompt builder
- tool prompt guidelines 合并去重

验收：

- 在项目目录运行 Forge 时，模型能看到项目指令
- slash commands 不进入模型上下文
- terminal command 可选择是否进入上下文

### Phase 4：provider catalog、凭证和模型切换

目标：支持真实使用中的多模型、多平台。

包含：

- provider config schema
- user config 路径：`~/.forge/config.jsonc`
- provider catalog：`~/.forge/catalog.jsonc`
- `jsonc-parser` 负责读取带注释配置，Zod 负责结构和默认值校验
- credential store
- env API key 读取
- `/login`
- `/providers`
- `/model`
- 默认 provider / model 保存
- OpenAI provider
- Anthropic provider
- OpenAI-compatible provider
- OpenRouter / Hugging Face / local model 走 OpenAI-compatible
- provider retry envelope
- provider-specific thinking/reasoning 配置

验收：

- 用户能在不改代码的情况下添加兼容 OpenAI API 的 provider
- session 中切换模型后，后续请求使用新模型
- provider 错误能转成统一 Forge error event

### Phase 5：上下文管理、skills、prompt templates

目标：长会话可持续，项目可扩展。

包含：

- rough token estimator
- context usage command
- manual compaction
- auto compaction threshold
- compaction entry
- compaction summary prompt
- previous summary update
- skills discovery
- skills invocation rule
- prompt templates discovery
- `/context`
- `/skills`
- `/prompts`
- `/reload`

验收：

- 长会话超过阈值时能压缩旧上下文
- 压缩后保留最近消息，旧消息变成 summary
- 新增 skill 文件后可 reload 生效

### Phase 6：交互式 TUI

目标：提供接近日常使用的终端体验。

包含：

- full-screen TUI
- input bar
- streaming message view
- tool call 展开/折叠
- thinking 展示控制
- session list modal
- model picker
- command autocomplete
- keyboard shortcuts
- cancellation
- queued steering message
- queued follow-up message
- theme support

验收：

- 用户可以在 TUI 中持续对话
- 工具执行时能看到状态
- 长任务可中断
- agent 运行中输入新消息不会破坏当前 turn

### Phase 7：session tree、branching、export

目标：会话从线性日志升级为可探索的工作树。

包含：

- parent id session tree
- leaf pointer
- branch from prior message
- branch summary
- session tree picker
- auto session title
- export HTML
- export Markdown
- export JSONL
- transcript renderer polish

验收：

- 用户可以回到旧消息处开新分支
- 原分支不会丢
- session 可导出成人能读的文档

### Phase 8：扩展工具和 MCP

目标：Forge 从内置工具扩展到外部工具生态。

包含：

- tool registry
- custom tool loading
- MCP client
- stdio MCP transport
- streamable HTTP MCP transport
- MCP tools 映射到 Forge tools
- MCP resources 注入上下文
- MCP prompts 映射为 prompt templates
- per-tool timeout
- per-tool permission policy

验收：

- 能连接一个本地 MCP server
- MCP tool 能被模型调用
- MCP tool 错误不会打断整个 agent loop

### Phase 9：安全、诊断、发布质量

目标：从能用变成可靠。

包含：

- command approval policy
- workspace boundary
- destructive command detection
- file write scope policy
- diagnostic logs
- provider request redaction
- structured error classification
- update check
- release notes notice
- shell profile / command prefix
- Windows/macOS/Linux 行为差异测试
- package publish pipeline
- docs site

验收：

- 删除、覆盖、越界写入等高风险操作有策略控制
- 诊断日志不泄露 API key
- Windows 下路径、编码、shell 行为可用

### Phase 10：产品完整度和生态

目标：Forge 有自己的方向，而不是 Tau 的 TS 影子。

包含：

- Forge plugin API
- built-in review mode
- git workflow helpers
- issue/PR integration
- browser tool
- visual artifact support
- agent profile
- reusable workflows
- benchmark/eval harness
- public docs：concepts、architecture、tools、commands、providers、sessions

验收：

- 用户能扩展 Forge，而不是只能改源码
- Forge 有自己的默认工作流和产品判断
- 文档足够支撑别人贡献代码

## 8. 关键决策

### 8.1 先写自己的 loop，不直接用 Vercel AI SDK

Vercel AI SDK 很适合快速做应用，但 Forge 的学习目标是理解 coding agent 内核。第一版自己写 loop，后续可以参考 AI SDK 的 provider 包装方式，但不把核心控制权交出去。

### 8.2 用 Responses API 作为 OpenAI 默认路径

OpenAI 官方 Node SDK 已把 Responses API 放在主路径。Forge 应以 Responses API 为优先实现，再兼容 Chat Completions 和 OpenAI-compatible。

### 8.3 session 用 append-only JSONL

这是最适合 agent 的格式：

- 易调试
- 易恢复
- 易导出
- 易做分支
- 不需要数据库迁移

缺点是大型 session 查询不如数据库快，但这不是早期瓶颈。

### 8.4 edit 工具必须保守

`editFile` 只做精确替换，并要求 `oldText` 唯一。不要第一版做 AST edit 或 fuzzy patch。coding agent 写坏文件的主要风险来自“不确定地改对了”，不是能力不够强。

### 8.5 TUI 晚一点做

TUI 很花时间，但不决定 agent 是否成立。先把 event stream、session、tools 打牢，TUI 只是事件消费者。

### 8.6 第一阶段使用 TypeScript 7 自带的 tsc

Forge 是 CLI 应用，发布时可以直接携带 `dist` 和运行时依赖，不需要先把所有代码打成单个 bundle。`tsc` 同时承担类型检查和 ESM 输出，`tsx` 只用于开发运行。这样不会依赖 TypeScript 7.0 暂时缺失的编程 API，也避免为了构建引入额外抽象。

### 8.7 先保持单 package

`agent`、`providers`、`tools`、`sessions`、`cli` 和 `tui` 先通过目录保持边界。只有核心协议稳定且确实需要独立发布 `@forge/core` 或 plugin SDK 时，才升级为 pnpm workspace。拆分前禁止跨层反向依赖，确保未来拆包只是移动文件和调整 exports，而不是重写架构。

## 9. 风险和处理

### 风险 1：provider API 变化

处理：

- provider 层独立
- `agent` 不依赖 OpenAI SDK 类型
- 所有 provider 输出先转 Forge event

### 风险 2：MCP v2 仍不稳定

处理：

- MCP 放到 Phase 8
- 抽象 Forge tool registry
- MCP 只是 tool source，不是 `agent` dependency

### 风险 3：shell 和文件写入风险

处理：

- Phase 1 先限制在 cwd
- Phase 9 加 approval policy 和 workspace boundary
- 所有工具返回结构化结果，失败也进入上下文

### 风险 4：上下文压缩丢关键信息

处理：

- 压缩是 session entry，不改写历史
- 保留最近消息
- summary prompt 固定结构
- 支持用户手动 export 原始 JSONL

### 风险 5：功能面太大

处理：

- 每阶段独立可用
- 不做“Phase 0 调研”
- 每阶段只合并能验证的功能

### 风险 6：TypeScript 7 工具生态仍在迁移

处理：

- 构建和类型检查只调用 `tsc` CLI，不导入 TypeScript 编程 API
- 第一阶段不生成 `.d.ts`，不依赖声明文件 bundler
- 使用 Biome，避免 typescript-eslint 对 TypeScript 6 API 的兼容层
- CI 固定 TypeScript 7 的精确版本，升级前先跑完整 typecheck 和 test
- 第三方工具如果必须调用 TypeScript API，推迟到 TypeScript 7.1 或该工具发布明确兼容版本后接入

## 10. 测试策略

基础工程门禁：

- `pnpm typecheck`：TypeScript 7 `tsc --noEmit`
- `pnpm test`：Vitest
- `pnpm lint`：Biome
- `pnpm build`：TypeScript 7 `tsc`
- 在 Node.js 24 上运行构建后的 `dist` CLI smoke test

### agent

- provider 流式文本
- provider thinking delta
- tool call 循环
- 多工具调用
- 未知工具
- 工具异常
- cancellation
- max turns
- queued steering / follow-up

### runtime

- JSONL 读写
- session resume
- session tree path
- compaction entry replay
- context token estimate
- provider retry
- credential lookup
- project context discovery

### coding tools

- read text
- read large file truncation
- write new file
- overwrite file
- exact edit
- duplicate oldText failure
- missing oldText failure
- line ending preserve
- shell success
- shell failure
- shell timeout
- output truncation

### cli / tui

- print mode exit code
- slash command routing
- terminal command syntax
- event renderer output
- TUI cancellation
- autocomplete
- model picker

## 11. 交付顺序

推荐顺序：

1. Phase 1：最小闭环
2. Phase 2：持久会话
3. Phase 3：项目上下文和命令
4. Phase 4：多 provider
5. Phase 5：上下文和 skills
6. Phase 6：TUI
7. Phase 7：分支和导出
8. Phase 8：MCP
9. Phase 9：安全和诊断
10. Phase 10：生态和差异化

原因：先让 agent 会干活，再让它记住，再让它适合项目，再让它好用。

## 12. 第一阶段完成后的使用形态

第一阶段结束时，Forge 应该已经能这样用：

```bash
forge -p "explain this repo"
forge -p "read package.json and suggest next steps"
forge -p "fix the failing test and run it"
```

这时它还不漂亮，但已经是一个真正的 coding agent。

## 13. 最脆弱假设

这个计划假设你想做的是“学习 Tau 并拥有自己的 agent 内核”，不是最快做出一个可用 AI CLI。

如果这个假设不成立，推荐路线会变成：

- 直接用 Vercel AI SDK 或 OpenAI Agents SDK 做产品
- Forge 只保留 CLI、工具、session、TUI 这些产品层
- 不自己维护 agent loop

但按你当前目标，自己写 agent core 是对的。

## 14. 后续讨论顺序

后续不要从 TUI 或 MCP 开始讨论。建议按这个顺序细化：

1. Phase 1 的目录结构和 package 配置
2. agent 协议和类型设计
3. OpenAI Responses provider
4. 文件工具语义
5. CLI 交互
6. 测试用例

这些定下来，Forge 的根就稳了。

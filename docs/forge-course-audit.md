# Forge 课程事实审计与基线设计

> 建设阶段 A 交付物。审计日期：2026-08-04。

## 结论先行

- 当前 `main` 的 `HEAD` 是 `8b92a99`（`docs: document phase 3 workflows`），实际可运行能力覆盖实现计划的 Phase 1、Phase 2 和 Phase 3；Phase 4 只有一个提前实现的 OpenAI-compatible adapter，其余内容尚未实现。
- 当前仓库只有 `main` 分支，没有课程标签、starter 分支或课程验收命令。`git status --short` 只有一个已有的未跟踪文件 `docs/forge-course-plan.md`；本次审计不修改、不覆盖它。
- 当前生产代码和测试适合作为参考实现与行为 oracle（课程约定不修改），但不能直接当作学习 starter：核心协议、loop、tools、provider、session 和 CLI 都已经完成。学习者要重做这些能力，必须先建立从历史提交派生的、带明确 TODO 的 starter 工作区。
- `pnpm verify` 已通过：14 个测试文件，79 个测试通过，3 个测试跳过；`node dist/index.js --help` 也通过。真实 OpenAI smoke test 没有执行，因为它需要用户凭证；没有读取 `.env`。
- 本文的 Git 参考标签、starter、检查点均是方案；在课程文档提交、课程基础设施和 starter 方案获批前，不执行任何 Git 分支、标签、worktree 或历史写操作。

## 1. 审计范围、事实来源和当前状态

事实来源按仓库约定取值：当前源码和同目录测试优先，其次是 `package.json`、`tsconfig.json`、CI，再是产品实现计划、README 和阶段 smoke test。审阅过的范围包括：

- `src/agent/`、`src/tools/`、`src/providers/`、`src/sessions/`、`src/coding/`、`src/cli/` 的全部 TypeScript 源码及其 `*.test.ts`。
- `package.json`、`tsconfig.json`、`tsconfig.build.json`、`biome.json`、`.github/workflows/ci.yml`。
- `docs/forge-implementation-plan.md`、`docs/forge-course-plan.md`、`docs/phase-1-smoke-test.md`、`docs/phase-2-smoke-test.md`、`docs/phase-3-smoke-test.md`、`README.md`。
- Git 线性历史（从 `727e925` 到 `8b92a99`）、当前分支和标签列表。

审计开始时的 Git 状态（保留的既有未跟踪文件）：

```text
## main
?? docs/forge-course-plan.md
```

本交付新增本文件后，状态预期为上面的既有文件加上 `?? docs/forge-course-audit.md`；生产代码、测试和配置没有变更。

历史上没有远程分支显示在当前 checkout，也没有 tag；不能把某个现存 tag 当作课程基线。

验证记录：

```text
pnpm verify                         # typecheck、Vitest、Biome、build 全部通过
node dist/index.js --help          # 退出码 0
```

`pnpm verify` 输出为 `14 passed`、`79 passed | 3 skipped`。跳过项是平台相关的 symlink 测试，不是失败。`pnpm smoke:openai`、三个 `docs/phase-*-smoke-test.md` 中的真实模型流程均未执行；它们需要外部 API key，不能作为本次审计的已验证事实。

## 2. 当前实际能力清单

以下只记录代码已经提供并有测试或文档证据的能力；“有类型”不等于“运行时已经发出”。

### 2.1 公共协议、loop 和 harness

| 能力 | 当前行为 | 代码证据 | 测试证据 |
| --- | --- | --- | --- |
| Message 协议 | `user`、`assistant`、`tool` 三种消息；assistant 有文本、tool calls 和可选 provider metadata；tool 有结构化成功/错误结果 | `src/agent/messages.ts` | `src/agent/loop.test.ts` 的无工具、工具闭环和错误用例 |
| AgentEvent 协议 | 定义 agent/turn/message/tool/thinking/error 事件，并预留 `retry`、`queue_update`、`tool_update` | `src/agent/events.ts` | `src/agent/loop.test.ts`、`src/cli/output-renderers.test.ts` |
| ToolDefinition | Zod input schema、`ToolExecutionContext`（cwd + `AbortSignal`）、结构化 `ToolResult`，`defineTool` 负责解析输入 | `src/agent/tools.ts` | `src/agent/loop.test.ts`、`src/tools/tools.test.ts` |
| Provider 中立边界 | provider 只接收 Forge message、tools、system prompt 和 signal，返回 `AsyncIterable<ProviderEvent>` | `src/agent/provider.ts` | `src/providers/openai-responses.test.ts`、`src/providers/openai-compatible.test.ts` |
| 多轮 loop | 收集 text/thinking delta、tool calls、metadata；要求 provider 发出 `response_end`；无 tool call 时完成，有 tool call 时继续下一轮 | `src/agent/loop.ts` (`AgentLoop.run`) | `src/agent/loop.test.ts`：工具闭环、多工具、未知工具、tool exception、max turns、缺少 `response_end` |
| tool 错误闭环 | 未知工具返回 `UNKNOWN_TOOL`；executor 抛错转为 `TOOL_ERROR`；结果作为 tool message 发回模型 | `src/agent/loop.ts` | `src/agent/loop.test.ts` 对应两项错误测试 |
| 取消和配对 | 取消时返回 `cancelled`；执行中的/尚未执行的 tool call 都产生对应 `CANCELLED` tool result | `src/agent/loop.ts` | `src/agent/loop.test.ts` 的两个取消测试 |
| max turns | 默认 20 轮，可注入正整数上限；超限发 `MAX_TURNS` error 和 `agent_end` | `src/agent/loop.ts` | `src/agent/loop.test.ts` 的 max-turns 测试 |
| prompt 并发互斥 | `AgentHarness` 保留消息 transcript，运行期间拒绝第二个 prompt 和 append | `src/agent/harness.ts` | `src/agent/loop.test.ts` 的并发测试 |

注意：`retry`、`queue_update`、`tool_update` 只是公共 union 成员，当前 loop 和 provider 没有产生这些事件；queued steering/follow-up 也没有实现。课程不能把它们当成现成能力。

### 2.2 文件和 shell 工具

| 能力 | 当前行为 | 代码证据 | 测试证据 |
| --- | --- | --- | --- |
| workspace path 边界 | 先做 lexical path 检查，再做 realpath 检查；拒绝 `..` 和解析到 workspace 外的链接 | `src/tools/workspace-path.ts` | `src/tools/tools.test.ts` 的越界和目录链接测试 |
| UTF-8 read | 只读文本文件，支持 1–1,000,000 byte 上限，检测 NUL，返回 size/bytesRead/truncated | `src/tools/read-file.ts` | `src/tools/tools.test.ts` 的截断测试 |
| 安全写入 | `safeWriteText` 拒绝最终 symlink，`O_NOFOLLOW` 可用时使用；create/overwrite/replace 模式 | `src/tools/safe-write.ts` | `src/tools/tools.test.ts` 的显式 overwrite 和 dangling symlink 测试 |
| writeFile | 自动创建父目录；默认只新建，覆盖需要 `overwrite: true` | `src/tools/write-file.ts` | `src/tools/tools.test.ts` 的 create/overwrite 测试 |
| 精确 edit | `oldText` 必须出现且仅出现一次；保留原始换行；失败返回 `EDIT_NOT_FOUND`/`EDIT_NOT_UNIQUE` | `src/tools/edit-file.ts` | `src/tools/tools.test.ts` 的 line-ending 和目标唯一性测试 |
| shell | `child_process.spawn`、shell 模式、workspace cwd；支持超时、AbortSignal、进程树终止、stdout/stderr | `src/tools/shell.ts` | `src/tools/tools.test.ts` 的成功、非零退出、timeout、取消测试 |
| 输出诊断 | 保留输出尾部；记录 stdout/stderr 总字节数、截断和 timeout 信息 | `src/tools/shell.ts` | `src/tools/tools.test.ts` 的 tail truncation 测试 |

这些边界是当前已验证的工具能力，不是完整安全产品。shell 仍然不是 sandbox，也没有 command approval、destructive-command 检测或 write-scope policy；README 和 `docs/phase-3-smoke-test.md` 均明确把它们留到 Phase 9。

### 2.3 Provider adapter

| 能力 | 当前行为 | 代码证据 | 测试证据 |
| --- | --- | --- | --- |
| OpenAI Responses | Forge message/tool schema 映射到 Responses API；流式 text/refusal/reasoning；从 completed output 解析 function call；保存 response metadata | `src/providers/openai-responses.ts` | `src/providers/openai-responses.test.ts` 的 mapping、stream、并行 call 顺序、metadata、坏 JSON、失败响应测试 |
| OpenAI-compatible | Chat Completions 请求映射；增量拼接 tool arguments；在 `stop`/`tool_calls` 完成时只发一次 `response_end`；拒绝未完成/非成功 finish reason | `src/providers/openai-compatible.ts` | `src/providers/openai-compatible.test.ts` 的 mapping、增量 tool call、坏 JSON、unfinished、finish reason 测试 |
| 第三方类型隔离 | OpenAI SDK 类型只出现在 `src/providers/`；agent 公共协议只使用 Forge 类型 | `src/agent/provider.ts`、`src/providers/*.ts` | `pnpm typecheck` 与 provider tests |

当前没有 Anthropic adapter、配置 catalog、retry envelope 或 provider-specific thinking 配置。OpenAI-compatible adapter 是已存在的能力，但不代表实现计划中 Phase 4 的完整 provider catalog 已完成。

### 2.4 Session、coding context 和本地命令

| 能力 | 当前行为 | 代码证据 | 测试证据 |
| --- | --- | --- | --- |
| JSONL entry | strict schema version 1，包含 message、model_change、session_info | `src/sessions/entries.ts` | `src/sessions/storage.test.ts`、`src/sessions/manager.test.ts` |
| JSONL storage | `JsonlSessionStorage` append/read；`MemorySessionStorage` 用于测试；损坏行报告行号 | `src/sessions/storage.ts`、`src/sessions/jsonl.ts` | `src/sessions/storage.test.ts` |
| 线性 replay | 按 append 顺序恢复 messages 和最新 model/session metadata | `src/sessions/state.ts` | `src/sessions/manager.test.ts` |
| 项目 session index | 按 cwd hash 隔离项目；create/list/get/touch；index 采用临时文件再 rename | `src/sessions/manager.ts` | `src/sessions/manager.test.ts` |
| CodingSession | 打开/恢复 session，追加 user/assistant/tool message，单 prompt/terminal shortcut 互斥，模型改变追加 model entry | `src/coding/session.ts` | `src/coding/session.test.ts`、`src/cli/main.test.ts` |
| 项目根和指令 | 优先 Git root，回退到常见 marker；按 root 到 cwd 加载 `AGENTS.md`，再加载 cwd `.forge/AGENTS.md`；拒绝 symlink、超限和非法 UTF-8 | `src/coding/project-context.ts` | `src/coding/project-context.test.ts` |
| system prompt | 注入 workspace、工具 snippets/guidelines、项目指令和 untrusted terminal result 边界；内容去重、路径属性转义 | `src/coding/system-prompt.ts` | `src/coding/system-prompt.test.ts` |
| slash commands | `/help`、`/sessions`、`/resume`、`/clear`、`/tools`、`/context`、`/quit` 本地处理 | `src/coding/commands.ts` | `src/coding/commands.test.ts`、`src/cli/main.test.ts` |
| terminal shortcuts | `!cmd` 以明确 untrusted user message 加入 context；`!!cmd` 只执行，不建 session、不进入 context | `src/coding/session.ts`、`src/cli/interactive.ts`、`src/cli/main.ts` | `src/coding/session.test.ts`、`src/cli/main.test.ts` |

当前 session 是线性的：没有 parent/leaf、分支、导出、compaction 或 context token accounting。JSONL 日志是 append-only，但项目 session index 是单独的 JSON 文件并通过临时文件替换；课程应分别讲这两个事实。

### 2.5 CLI、renderer 和工程门禁

| 能力 | 当前行为 | 代码证据 | 测试/配置证据 |
| --- | --- | --- | --- |
| executable entry | `SIGINT` 转 `AbortController`，设置 `process.exitCode` | `src/index.ts` | `src/cli/main.test.ts` 的取消测试 |
| 参数和 provider 选择 | Commander 支持 prompt/provider/base-url/model/resume/output；API key/model 仅从指定环境变量或参数读取 | `src/cli/main.ts` | `src/cli/main.test.ts` 的 credential/model/provider config 测试 |
| one-shot/interactive | `-p` 一次运行；无 prompt 时启动 readline 行式会话 | `src/cli/main.ts`、`src/cli/interactive.ts` | `src/cli/main.test.ts` |
| output protocols | `FinalTextRenderer`（默认最终文本）、`JsonEventRenderer`（一行一个 AgentEvent）、`TextRenderer`/`TranscriptRenderer`（文本流 + stderr 工具状态） | `src/cli/final-text-renderer.ts`、`src/cli/json-renderer.ts`、`src/cli/text-renderer.ts` | `src/cli/output-renderers.test.ts`、`src/cli/text-renderer.test.ts` |
| exit semantics | completed=0、failed=1、cancelled=130；stdout/stderr 分离 | `src/cli/main.ts`、各 renderer | `src/cli/main.test.ts` |
| build gate | Node >=24、pnpm 11、TypeScript 7 strict ESM、Vitest、Biome、tsc build | `package.json`、`tsconfig.json`、`tsconfig.build.json`、`biome.json` | `.github/workflows/ci.yml`、本次 `pnpm verify` |

当前不存在 `pnpm lesson:list`、`pnpm lesson:check` 或 lesson manifest；课程计划中的命令只是目标形态。

## 3. 当前能力到课程模块和产品 Phase 的映射

状态含义：**已实现**表示现有实现和测试覆盖了该模块的主要目标；**部分**表示有真实代码但仍缺关键子能力或独立验收；**未实现**表示只有计划/类型占位，没有可运行能力；**计划冲突**表示文档把“类型/计划”误读成了运行时能力，必须在课程正文中纠正。

### 3.1 课程模块映射

| 课程模块 | 当前状态 | 当前证据和缺口 | 对应产品 Phase |
| --- | --- | --- | --- |
| 00 开始之前 | 部分 | 仓库地图、事实来源、环境和 smoke 文档存在（`AGENTS.md`、`README.md`、`docs/phase-*-smoke-test.md`）；没有课程网站、进度、Git 学习基线和诊断页 | 全部，入口模块 |
| 01 TypeScript 与异步基础 | 已实现（作为生产代码基础） | strict NodeNext、ESM、Zod、AsyncIterable、AbortSignal、Vitest/fake 都在源码和测试中；没有独立练习或学习验收 | Phase 1 |
| 02 Agent 公共协议 | 部分 | message/event/tool/provider 协议已在 `src/agent/`；没有单独 contract-test 套件，且 `retry`/`queue_update`/`tool_update` 只有类型定义 | Phase 1 |
| 03 Agent Loop 与 Harness | 部分 | 文本、单/多工具、错误、max turns、取消、并发都有实现和 fake tests；queued steering/follow-up 未实现；类型中的预留事件不能算实现 | Phase 1 |
| 04 Coding Tools 与文件安全 | 部分 | cwd/realpath、symlink、read/write/edit、shell timeout/cancel/truncation 已覆盖；approval policy、sandbox、destructive detection、write scope 尚无 | Phase 1，完整安全延伸到 Phase 9 |
| 05 真实 Provider Adapter | 部分 | Responses 和 OpenAI-compatible adapter 及 mapping tests 已有；真实 API smoke 未验证，无 Anthropic/retry/config catalog | Phase 1/2，完整 provider 在 Phase 4 |
| 06 Session 与 CodingSession | 部分 | JSONL、replay、index、resume、project instructions、terminal shortcut 已有；无 compaction、tree、branch/export | Phase 2/3，扩展到 Phase 5/7 |
| 07 CLI、Renderer 与交互协议 | 已实现（Phase 3 范围） | one-shot、JSON/text/transcript、line interactive、slash commands、`!`/`!!`、clear/resume 已有；无 TUI | Phase 1/2/3 |
| 08 配置、多 Provider 与模型切换 | 未实现（adapter 提前存在） | 只有 CLI 环境变量和 `OpenAICompatibleProvider`；无 JSONC 配置、catalog、credential store、login、model persistence、Anthropic/retry | Phase 4 |
| 09 上下文、Skills 与 Prompt Templates | 未实现 | 没有 token estimator、compaction、skills、prompt templates、reload | Phase 5 |
| 10 交互体验与高级 Session | 未实现 | 没有 React/Ink TUI、queued steering/follow-up、session tree、branch、export | Phase 6/7 |
| 11 安全、MCP、Eval 与发布 | 部分（仅早期安全基础） | 现有工具边界、错误码、CI 和 smoke 文档可作前置；没有 approval、MCP、diagnostic redaction、eval、跨平台门禁、发布流水线 | Phase 8/9/10 |

### 3.2 产品 Phase 映射

| 产品 Phase | 状态 | 已有能力 | 尚缺/冲突 |
| --- | --- | --- | --- |
| Phase 1 最小 coding agent | 已实现 | `src/agent/` loop/harness，Responses provider，`src/tools/` 四类工具，system prompt，one-shot CLI，text renderer 和 fake tests | 真正 OpenAI smoke 仅有手工文档，未在本次运行；这不影响 fake/本地门禁结论 |
| Phase 2 持久会话和渲染协议 | 已实现 | OpenAI-compatible adapter、JSONL storage/index、resume、sessions 命令、text/json/transcript renderer | 计划中的文件导出明确留到 Phase 7；当前没有原子锁/损坏日志恢复之外的高级存储能力 |
| Phase 3 项目上下文和命令系统 | 已实现 | project root、`AGENTS.md`、system prompt、line interactive、slash commands、`!`/`!!`、clear/resume | smoke test 的真实模型部分未跑；本地 deterministic 测试已跑通 |
| Phase 4 provider catalog、凭证和模型切换 | 未实现（部分提前能力） | 环境变量、`--provider`、`--model`、OpenAI-compatible adapter | JSONC/catalog/credential store/login/Anthropic/retry/thinking config 都缺；adapter 不应被误标为 Phase 4 完成 |
| Phase 5 context/skills | 未实现 | 无 | 全部能力缺失 |
| Phase 6 TUI | 未实现 | 无 | 全部能力缺失 |
| Phase 7 tree/branch/export | 未实现 | 无 | 全部能力缺失 |
| Phase 8 MCP | 未实现 | 工具 protocol 可作为未来边界 | 无 registry、MCP transport/tools/resources/prompts |
| Phase 9 safety/diagnostics/release | 部分 | lexical+realpath、symlink rejection、timeout/cancel/output diagnostics、CI | README 已明确 shell 非 sandbox，approval/destructive policy/diagnostic redaction/跨平台/发布未完成 |
| Phase 10 product/ecosystem | 未实现 | 无 | plugin、review、git helpers、browser、visual、eval/public docs 均无 |

## 4. 参考实现和学习者必须重做的能力

### 4.1 适合作为参考实现和行为 oracle 的部分

- `src/agent/loop.ts`、`src/agent/harness.ts`：可用来观察消息生命周期、`response_end` 不变量、tool result 配对、取消收尾和并发互斥。
- `src/tools/workspace-path.ts`、`src/tools/safe-write.ts` 及 `src/tools/{read-file,write-file,edit-file,shell}.ts`：可用来对照 lexical/realpath、symlink、精确编辑和尾部截断边界。
- `src/providers/openai-responses.ts`、`src/providers/openai-compatible.ts`：可用来对照第三方流事件到 Forge event 的映射，不应复制 SDK 类型到 `agent`。
- `src/sessions/`、`src/coding/session.ts`：可用来对照 append-only JSONL、replay、模型 entry 和 session mutex。
- `src/coding/project-context.ts`、`src/coding/system-prompt.ts`、`src/coding/commands.ts`：可用来对照项目指令安全读取、本地命令消息边界和 slash command 路由。
- `src/cli/` 和 `src/index.ts`：可用来对照 stdout/stderr、退出码、SIGINT 和 renderer 消费 AgentEvent。
- 同目录测试是行为 oracle；尤其是 `src/agent/loop.test.ts`、`src/tools/tools.test.ts`、`src/providers/*.test.ts`、`src/sessions/*.test.ts`、`src/coding/*.test.ts`、`src/cli/main.test.ts`。

参考实现只用于阅读、比较和故障定位。课程页面不能把完整答案放进默认正文，也不能把“测试通过”当作已经理解。

### 4.2 学习者必须重做的核心能力

学习者完成简历项目所必需、且应通过设计说明和答辩验证的生产能力是：

1. provider-neutral message/event/tool/provider 协议和依赖方向。
2. `AsyncIterable` agent loop、tool 调用闭环、max turns、失败、取消和消息一致性。
3. cwd lexical/realpath 边界、最终 symlink 拒绝、保守 write/edit 和 shell 输出语义。
4. provider adapter 的请求/流事件映射、增量 tool arguments、`response_end` 不变量。
5. append-only session、replay、CodingSession 互斥和项目指令安全发现。
6. CLI one-shot/interactive、slash command、terminal shortcut、renderer 和退出码。

课程允许 Codex 直接维护课程站点、题目、starter、验收工具和基础设施缺陷修复；未经学习者明确要求，不直接代写上述 Forge 生产实现。若确实代写，必须在学习记录中标记为“参考实现，不计独立完成”。

## 5. 隔离练习和直接修改 Forge 的任务划分

| 范围 | 建议工作区 | 任务 | 边界 |
| --- | --- | --- | --- |
| 00.01–00.06 | 参考 checkout（约定不修改生产代码） | 根本问题、CLI/agent/coding agent 区别、仓库地图、环境和 Git 恢复练习 | 不改 `src/`；只提交学习记录或课程进度 |
| 01.01–01.09、01.M | 隔离小练习目录（建议未来 `course/exercises/`） | strict 类型、union、泛型/DI、ESM、Promise、AsyncGenerator、AbortController、Zod、Vitest fake；实现小型可取消流 | 不依赖真实 provider，不修改 Forge 生产代码；当前仓库尚无该目录，需后续建设阶段提供 |
| 02.01–02.M | Forge starter 的 `src/agent/` | 从协议和 contract tests 开始重做 message/event/tool/provider；先用 fake provider | 只改授权的 agent 文件和同目录测试；不得导入 OpenAI、文件系统、shell、CLI |
| 03.01–03.M | Forge starter 的 `src/agent/` | loop/harness 的文本、tool、多轮、错误、max turns、取消和互斥 | 直接改 Forge；每个不变量必须有确定性测试；queued steering/follow-up 另列后续专题 |
| 04.01–04.M | Forge starter 的 `src/tools/` | 路径、realpath、read/write/edit、shell timeout/cancel/truncation | 直接改 Forge；不把 shell 宣称为 sandbox；approval policy 留 Phase 9 |
| 05.01–05.M | Forge starter 的 `src/providers/` 及 adapter tests | 请求映射、流事件、metadata、tool args、失败和显式 smoke | 直接改 provider 边界；fake tests 不访问网络；真实 OpenAI 只在 `pnpm smoke:openai` 中显式运行 |
| 06.01–06.M | Forge starter 的 `src/sessions/`、`src/coding/` | JSONL/replay/index/CodingSession/project context/local commands | 直接改 Forge；保留历史，不做数据库/树结构提前扩展 |
| 07.01–07.M | Forge starter 的 `src/cli/` | 参数、renderer、interactive、slash、`!`/`!!`、resume/clear | 直接改 Forge；只消费 AgentEvent，不把命令插入 agent loop |
| 08–11 | 后续独立分支或选修专题 | 配置、context、TUI、MCP、安全和 capstone | 当前不作为第一批任务；每项需先补事实、前置和验收 |

## 6. 可恢复的 Git 参考版本、starter 和检查点方案（只提方案）

本节明确区分两类来源：`course/ref/*` 参考标签指向已经完成的产品快照，提交/tag 是不可变事实锚点；学习 starter 不从这些完整实现反向删除代码，而从初始化工程提交派生并逐步加入课程骨架。当前阶段只记录方案，在课程文档提交、课程基础设施和 starter 方案获批前不执行任何 Git 写操作。

### 6.1 已审计的参考提交候选

当前历史是线性的，只有 `main`：

```text
727e925 docs: add forge implementation plan
302de18 chore: initialize typescript cli project
8d44b13 feat(agent): add core protocols and agent loop
41011b5 feat(tools): add filesystem and shell tools
b23dbcd feat(openai): add responses provider
0345964 feat(cli): add one-shot command and text renderer
2575a2a test: complete phase one acceptance coverage
... phase-one hardening fixes ...
8068b8c / 20ed28f / e89af75 / 9115121   # Phase 2 session/renderer
9f2b3f8 feat(providers): add OpenAI-compatible API support
1bc56df / bf196f5 / 5adf9ec / 31afb54   # Phase 3 coding/interactive
8b92a99 docs: document phase 3 workflows
```

后续获得用户确认后，可把以下提交作为**候选**参考标签目标（当前不创建标签）：

| 候选名 | 提交 | 用途 |
| --- | --- | --- |
| `course/ref/phase-1` | `78ca05e` | Phase 1 核心和 hardening 后、session 之前的完整参考；包含 phase-1 smoke 文档所需代码 |
| `course/ref/phase-2` | `9f2b3f8` | JSONL、session manager、resume、renderers 和 OpenAI-compatible adapter 完成后的参考 |
| `course/ref/phase-3` | `8b92a99` | 当前 README 和 Phase 3 smoke 文档对应的完整参考 |

候选标签必须在创建前再次运行 `pnpm verify`、`node dist/index.js --help`，并记录目标 commit；标签创建不是本阶段动作。

### 6.2 参考标签与学习 starter 的来源

参考标签和学习 starter 必须分离：

- `course/ref/phase-1`、`course/ref/phase-2`、`course/ref/phase-3` 指向已经完成的产品快照，供阅读、对照和故障定位使用；提交/tag 是不可变事实锚点，不能从这些提交删除实现来制造 starter。这样既会暴露答案，也可能留下与删减实现不匹配的依赖和编译问题。
- 学习主线基础候选是初始化提交 `302de18`，而不是任何完成的 Phase 参考提交。该提交实际包含：`.gitattributes`、`.gitignore`、`README.md`、`biome.json`、`package.json`、`pnpm-lock.yaml`、`pnpm-workspace.yaml`、`tsconfig.json`、`tsconfig.build.json`、`src/index.ts`、`src/version.ts` 和 `src/version.test.ts`，并继承了 `docs/forge-implementation-plan.md`。
- `302de18` 已有 Node 24/pnpm 11 元数据、TypeScript 7 strict NodeNext 配置、Biome 配置、Commander/OpenAI/Zod/Vitest 等依赖和基础 build/dev/test/typecheck/lint/start script；`src/index.ts` 只有空 executable skeleton，测试只有版本匹配。它没有 `src/agent/`、`src/tools/`、`src/providers/`、`src/sessions/`、`src/coding/`、`src/cli/` 的生产实现，也没有 `verify`/`smoke:openai` script、课程网站、lesson manifest、starter 骨架或公开课程测试。
- 后续经批准后，从 `302de18` 派生一条独立的课程 starter 分支/提交，在其中加入课程基础设施、协议骨架、明确的 TODO 边界和公开测试；骨架必须保持可编译，并让每个任务逐步增加真实能力，而不是复制或删减最终实现。

### 6.3 学习主线、参考 worktree 和检查点

1. `main` 保持当前产品历史，不删除实现、不回写、不 squash；完整参考提交放在独立参考 worktree 中，按协作约定不修改。普通 worktree 没有文件系统只读保护，真正不可变的事实锚点是参考 commit/tag。
2. 学习者使用一条长期累积主线（建议 `course/learner-main`），从批准的课程 starter 开始，按模块依次实现；如果并行协作确有需要，最多按模块拆分分支，不为每节课创建 learner 分支。
3. 普通 lesson 用现有 Conventional Commit 类型和 lesson manifest 状态记录（例如 `feat(agent): complete lesson 03.04 tool-call pairing`）；测试和文档分别使用 `test:`、`docs:`，lesson ID 主要记录在 manifest、提交正文或主题中。不为每节 lesson 创建 tag，只有模块里程碑创建少量 `course/checkpoint/module-01`、`course/checkpoint/module-02` 等候选标签，并在批准后执行。
4. 进入下一模块前，先执行该任务真实存在的检查命令，再保存学习者的设计说明和复盘；不存在的 `pnpm lesson:*` 不能写进课程验收。
5. 只读命令包括 `git show`、`git diff`、`git log` 和 `git reflog`。`git worktree add` 会修改 Git worktree 元数据并创建目录，不是只读命令；如后续获批使用，应把它归类为非破坏、可恢复的工作区操作，记录路径和来源提交，避免覆盖现有工作。不要把 `git reset --hard`、`git clean -fd`、强制 push 或历史改写作为教学步骤。
6. starter 的课程基础设施、协议骨架和公开测试应在 manifest 中记录“来源 commit、新增文件、允许修改范围、验证命令、恢复方式”；不执行从完整实现反向删代码的操作。

每个模块 checkpoint 至少保存：来源 commit、学习主线分支、工作区路径（如使用 worktree）、修改文件清单、`pnpm verify` 输出摘要、人工答辩状态和未完成项。课程网站只引用这些稳定模块 ID，不引用易漂移的源码行号。参考代码和学习者代码可以通过不同 worktree 同时打开，但参考 worktree 仅按协作约定不修改，普通 worktree 没有文件系统只读保护；worktree 创建属于后续经批准的非破坏写操作。

## 7. 当前计划与行为的冲突或容易误读处

1. **课程验收命令尚不存在。** `docs/forge-course-plan.md` 的 `pnpm lesson:list`/`pnpm lesson:check` 是目标用法；`package.json` 没有这些 script。第一批任务不能声称已有自动课程验收。
2. **AgentEvent 类型超前于运行时。** `retry`、`queue_update`、`tool_update` 已声明，但 `AgentLoop`、两个 provider 和 CLI 没有相应产生/处理路径；课程模块 03.08/03.11 应标成未实现或类型先行。
3. **OpenAI-compatible adapter 的阶段位置不同。** 当前 `src/providers/openai-compatible.ts` 在 Phase 3 分支上已经存在；实现计划把“完整 provider catalog、Anthropic、凭证和 retry”留到 Phase 4。课程应把 adapter 显示为 Phase 2 的提前增量，把 catalog 等显示为 Phase 4 未完成，而不是整体标记 Phase 4 已完成。
4. **线性 session 不是 session tree。** 当前 `src/sessions/state.ts` 只按 append 顺序 replay，`SessionManager` 只有 project index；不能把 Phase 7 的 branch/export 预告写成现有功能。
5. **shell 的边界不是安全策略。** `src/tools/shell.ts` 使用 `spawn(..., { shell: true })`，README 和 Phase 3 smoke test 都说明没有 approval/sandbox；课程 04 可以教当前边界，课程 11 再讲完整策略。
6. **真实模型验证仍是手工 smoke。** 默认 Vitest fake provider 不访问网络；`docs/phase-1-smoke-test.md`、`docs/phase-2-smoke-test.md`、`docs/phase-3-smoke-test.md` 的 OpenAI 步骤需要凭证，本次没有执行，不能伪造“真实 provider 已验收”。
7. **课程站点和 starter 还没有文件事实。** `course/`、VitePress 配置、lesson manifest、进度组件和隔离练习目录均不存在；应在阶段 B–D 建设，不应在正文中引用为当前可用路径。
8. **工作区不干净但不是生产改动。** `docs/forge-course-plan.md` 是已有未跟踪课程计划；任何 baseline 创建流程都要先决定它是否纳入课程文档提交，不能自动 `git clean` 或覆盖。

## 8. 第一批课程任务的准确前置状态

### 8.1 当前可以立即做的只读诊断

`00.01`–`00.06` 可以针对当前 `main@8b92a99` 做阅读、架构数据流绘制和环境诊断，前置状态为：

- `package.json`、`pnpm-lock.yaml` 和 Node 24/pnpm 11 版本约束存在；全新环境的 `pnpm install --frozen-lockfile` 尚未在本次审计中验证，不能把安装成功写成已验证前置。
- `pnpm verify` 和 `node dist/index.js --help` 通过。
- 当前工作区已知有 `docs/forge-course-plan.md`，不能要求学习者把它当成生产代码改动或强行清理。
- 不需要 API key；不得运行真实 provider。

### 8.2 隔离基础练习的真实前置

`01.01`–`01.M` 不能直接引用尚不存在的 `course/exercises/`。在阶段 B/C/D 创建练习骨架和验收器之前，它们只能作为任务设计，不能宣称“可执行”。练习应使用标准库、Zod 和 fake/stub，不读 `.env`，不改 `src/`。

### 8.3 第一条 Forge 主线的真实前置

`02.01`–`03.M` 若要求学习者重做 agent core，应从初始化工程候选 `302de18` 派生、经批准后形成课程 starter，再在一条累积学习主线上实现；不能从当前 `main@8b92a99` 或完整参考 `course/ref/phase-1@78ca05e` 直接删代码。进入这条主线前必须有：

1. 课程文档提交、课程基础设施和 starter 方案已经获批；在此之前不执行任何 Git 分支、标签、worktree 或历史写操作。
2. starter commit 已从 `302de18` 建立，新增的课程基础设施、协议骨架、公开 contract tests、允许修改的文件范围和恢复方式已写入 manifest；参考实现 worktree 保持独立，按协作约定不修改，事实锚点仍是参考 commit/tag。
3. fake provider 测试可以在无网络/无 key 环境运行，且课程 starter 的骨架仍可编译。
4. 每一步使用当前真实命令（至少 `pnpm typecheck`、`pnpm test`，模块结束 `pnpm verify`）；不能提前依赖 `pnpm lesson:check`。

后续 `04`–`07` 依次以前一模块通过的累计学习主线状态为前置：tools 依赖协议，provider 依赖协议和 tools，session/coding 依赖 loop/storage，CLI 依赖上述结构。只有模块里程碑需要 checkpoint tag；普通 lesson 通过提交和 manifest 状态记录。没有这些模块状态时，不应把整条主线合并成一个“从空白实现 Forge”的任务。

## 9. 风险、未核实项和下一阶段输入

### 风险

- 课程正文、starter 和验收器可能随着协议变更漂移；所有任务必须绑定稳定 lesson ID、来源 commit 和实际命令。
- 直接在当前 `main` 或完整参考上重做会覆盖用户已有实现、暴露答案，或因删减产生依赖/编译问题；必须从 `302de18` 派生获批的课程 starter，并在一条累积学习主线上推进。
- 课程文档提交、课程基础设施和 starter 方案获批前，不创建分支、标签或 worktree，也不改写 Git 历史；获批后的 `git worktree add` 仅作为可记录、可恢复的非破坏工作区操作。
- fake provider 测试通过不等于真实 OpenAI API、Windows shell 或模型输出稳定；真实 smoke 只能显式、凭证隔离地运行。
- shell 当前允许操作系统用户可访问的能力，且没有 approval policy；任何课程文案都不能称其为 sandbox。
- 未来阶段的 TUI、MCP、context/compaction 和 provider catalog 规模较大，不能在第一批课程中用占位页伪装完成。

### 未核实项

- 未运行 `pnpm smoke:openai` 及三个阶段文档的真实 API 流程。
- `pnpm-lock.yaml` 存在，但未执行全新环境的 `pnpm install --frozen-lockfile`；本次 `pnpm verify` 使用现有安装完成，安装可重建性仍未核实。
- 未做真实 Windows/macOS/Linux 全平台矩阵；当前 CI 只有 Ubuntu Node 24。
- symlink 相关测试在当前平台有 3 项跳过；代码路径通过静态阅读和可运行的其余测试审阅。
- 没有课程网站、manifest、starter、lesson checker 或学习者 dry run，因此不能给出课程页面可用性结论。

### 下一阶段输入

阶段 B 开始前需要用户确认：

1. 是否接受 `78ca05e`、`9f2b3f8`、`8b92a99` 作为 Phase 1/2/3 参考提交候选（commit/tag 作为不可变事实锚点），并接受 `302de18` 作为学习主线基础候选。
2. 是否批准从 `302de18` 派生课程 starter，加入课程基础设施、协议骨架和公开测试，再让学习者在一条累积主线上完成模块 01–07；完整参考实现放在独立参考 worktree 中并按协作约定不修改，commit/tag 作为不可变事实锚点。
3. 隔离练习目录是否采用 `course/exercises/`，以及是否允许课程基础设施新增 `course/`/VitePress 文件。
4. 是否授权后续阶段创建一条长期 learner 分支和少量模块 checkpoint tag；普通 lesson 只用 commit/manifest，不创建 lesson 分支或 tag。本阶段不执行任何 Git 写操作。
5. 第一批发布范围是否固定为模块 00–03；在用户确认前不批量生成后续模块正文。

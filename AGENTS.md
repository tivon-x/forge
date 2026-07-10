# Forge Agent Instructions

## 沟通与范围

- 对用户使用中文，先给结论，再说明必要原因。
- 代码标识符、代码注释和错误码使用英文；编辑文档时保持文件现有语言。
- 直接完成用户已明确要求的改动。需求不清时先从仓库和文档寻找答案，只有高风险歧义才提问。
- 只实现当前请求或已批准阶段。不要顺手引入路线图后续阶段的功能。
- 每个阶段必须可独立合并、运行和回滚，不能依赖下一阶段才能成立。

## 项目目标与事实来源

Forge 是 Node.js 24、TypeScript 7 编写的 provider-neutral coding agent CLI。

按以下顺序确认事实：

1. 当前代码和测试
2. `package.json`、`tsconfig.json`、CI 配置
3. `docs/forge-implementation-plan.md`
4. `README.md` 和阶段性文档

代码或配置与计划冲突时，不要静默选择一方。指出具体冲突，并优先保持当前已验证行为，除非任务明确要求改变它。

## 项目地图

- `src/agent/`：Forge 自有 message、event、tool、provider 协议，以及 agent loop/harness。
- `src/providers/`：第三方模型 SDK adapter；第三方类型只能停留在该边界。
- `src/tools/`：文件系统和 shell 工具，以及 cwd 边界检查。
- `src/cli/`：命令解析、system prompt、renderer 和进程退出语义。
- `src/index.ts`：CLI executable entry point。
- `docs/`：路线图、验收和人工 smoke test。
- `.github/workflows/ci.yml`：Node 24 持续集成门禁。
- `pnpm-lock.yaml`：pnpm 生成的依赖锁文件；只通过 pnpm 更新，不手工编辑。

## 架构边界

- `agent` 不得导入 OpenAI SDK、文件系统、shell、Commander、renderer 或具体配置路径。
- `providers`、`tools` 和 `cli` 依赖 `agent` 协议；禁止 `agent` 反向依赖它们。
- provider 只负责 Forge message 与第三方请求/流事件之间的转换，不执行工具、不管理会话、不渲染 UI。
- 工具保持普通 async function：Zod input schema、结构化 `ToolResult`、`AbortSignal` 和显式 execution context。
- CLI 和未来 UI 只消费 `AgentEvent`。不要把输出、终端状态或 UI callback 插入 agent loop。
- 核心流式控制继续使用 `AsyncIterable` / `AsyncGenerator`，取消继续使用 `AbortSignal`。不要换成 EventEmitter 或 RxJS。
- provider metadata 可以保存 provider-specific opaque data，但 Forge 公共协议不得暴露第三方 SDK 类型。
- 修改 message、event、tool 或 provider 协议时，必须同时检查 agent、provider、CLI 和相关测试的契约影响。

## 长期技术约束

- 保持单 package、ESM only、strict TypeScript；相对导入在源码中使用 `.js` 扩展名。
- 构建和类型检查只调用 TypeScript 7 `tsc` CLI，不导入 TypeScript 编程 API。
- Forge 仍是应用时不生成 `.d.ts`，不引入 bundler。
- 未经明确批准，不新增语言、运行时、monorepo、独立 package 或重量级任务框架。
- 只有确实需要独立发布 Forge core package 或 plugin SDK 时才考虑拆包。
- session 功能采用 append-only JSONL；不要提前引入数据库或 Tau 格式兼容。
- 精确编辑保持保守：`oldText` 必须存在且唯一，不做 fuzzy replace 或隐式多处替换。

## 安全边界

- 不读取、打印、记录或提交 API key、token、`.env`、credentials 或 secrets。
- OpenAI key 只从 `OPENAI_API_KEY` 获取；模型从 `--model` 或 `OPENAI_MODEL` 获取，不硬编码当前模型名。
- 文件工具必须保持 lexical path 和 realpath 两层 cwd 检查，不能通过 `..`、绝对路径或目录链接越界。
- shell 只保证启动 cwd，不是 sandbox。不得把它描述成隔离环境，也不得静默放宽文件边界。
- 未经用户明确要求，不执行破坏性命令、覆盖用户改动或访问工作区外数据。
- Phase 9 前没有完整 approval policy。新增高风险能力时必须明确当前缺少的保护，而不是伪造安全保证。

## 实现方式

- 先读取相关实现、测试和 `git status`，再修改代码。
- 优先复用现有协议和 helper；只有能消除真实重复或复杂度时才增加抽象。
- 解析结构化数据时使用结构化 API 或 schema，不用脆弱的字符串拼接。
- 错误必须可供模型和 CLI 判断：使用稳定错误码、可读消息和结构化数据。
- 在每个异步边界前后考虑取消竞态；资源、timer、listener 和子进程必须可靠清理。
- 不改无关文件，不做顺手格式化或大规模重构，不撤销不是自己产生的改动。
- 注释只解释不明显的约束或原因，不重复代码表面行为。

## 测试与验证

默认总门禁：

```bash
pnpm verify
```

它必须依次通过 typecheck、Vitest、Biome 和 build。修改 CLI 或构建配置时，再运行：

```bash
node dist/index.js --help
```

测试规则：

- 测试与源码同目录，命名为 `*.test.ts`。
- agent/provider 默认使用 fake provider，测试必须确定性且不得访问网络。
- provider adapter 测试请求映射、流事件映射、工具调用、provider metadata 和失败事件。
- 工具测试覆盖成功、错误、取消、超时、输出截断和 cwd/链接越界。
- CLI 测试保持 stdout、stderr 和退出码分离，并至少覆盖完整工具闭环。
- 真实 OpenAI 只运行 `pnpm smoke:openai`，需要用户本地凭证；不得加入默认测试或 CI。
- 修复 bug 时先增加能复现问题的测试。测试范围与改动风险匹配，不为通过测试缩窄真实需求。

## Git 工作流

- 开始前检查 `git status --short`，尊重已有未提交修改。
- 多步骤任务按可验证里程碑提交，不必等整个 phase 完成；每个提交必须可理解、可回滚并保持门禁通过。
- 使用现有 Conventional Commit 风格：`feat(scope): ...`、`fix(scope): ...`、`test: ...`、`docs: ...`、`chore: ...`。
- 每个提交只包含本任务相关文件。不要提交 `dist`、coverage、缓存、日志、凭证或 `.env`。
- 不使用 `git reset --hard`、`git checkout --` 或其他会丢失用户工作的命令。
- 未经明确要求，不创建远程仓库、不 push、不改写或 squash 历史。
- 结束时报告验证结果、提交和任何未验证事项；预期工作区应干净。

## 文档维护

- 使用方式、环境变量或退出码改变时更新 `README.md`。
- 人工验收方式改变时更新 `docs/phase-1-smoke-test.md` 或对应阶段文档。
- 只有路线、阶段边界或长期决策改变时才更新实现计划，不把临时调试记录写入长期文档。
- 文档引用必须指向仓库中真实存在的文件和命令。

## 停止条件

- 连续两个检查点没有新增证据或进展时，停止重复尝试并重新判断原因。
- 同一错误、断言或堆栈连续出现三次时，视为当前假设错误，不继续盲目重试。
- 遇到缺失凭证、不可达网络、目标分支冲突或无法安全处理的用户改动时，保留现场并报告具体阻塞。
- 不以时间紧、任务困难或测试暂时失败为由降低完成标准。

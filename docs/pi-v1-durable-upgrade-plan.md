# Forge 对齐 Pi v1.0 / durable 的优化升级方案

- 状态：D1 的三个独立审查问题已修复，并于 2026-10-04 通过重新验收；D2–D6 与 V1 尚未实施。最新事实见 D1 修复记录。
- 日期：2026-10-02；D1 实施：2026-10-03；独立审查修复与重新验收：2026-10-04。
- Forge 基线：`f016e658de0aec038749d72e4bdaa3a243ef14f6`，版本 `0.3.2`。
- Pi CLI 基线：`v1.0.0`，提交 `a13d35a742c6ef8462812a28fbe1d8c8b7431c32`。本地 pnpm 安装已核实为 `@earendil-works/pi-coding-agent@1.0.0`。
- Pi durable 基线：`4812cb268efc3f0846337cdfb8f00ce51b7b8d5a`。与上一轮阅读的 `c10bfb0d79dbbc998539a0a3e6c6a736a4e6db06` 比较，新增提交没有修改 `packages/durable/`。
- 框架基线：本地安装及 `uv.lock` 为 LangChain `1.4.3`、LangChain Core `1.6.6`、LangGraph `1.2.12`、langgraph-checkpoint `4.1.1`。
- 当前实施范围：仅 D1 原子 journal、历史兼容及其必要前置修复、验收与文档；不提前实施后续阶段，不创建分支、提交、推送或调用真实 provider。

## 1. 决策

采用 Pi durable 的完整核心执行契约：原子提交、持久化任务阶段、工具执行前记录、提交去重、持久化输入队列、任务所有权、跨重启恢复、已提交进度视图与后台压缩。

这些机制按 Pi 源码实现，不因其初次发布而改成临时状态、整段任务重跑或只有完成结果才保存的简化版本。Python 与 LangChain 的调整只改变实现载体，不改变已选机制的可观察行为。

模型与工具执行继续由 `create_agent()` / `astream_events(version="v3")` 承担。通过官方 middleware、原生 ToolNode 与 `BaseCheckpointSaver` 承接 Pi 的机制，不移植 Pi 的 provider/tool loop，也不添加另一套模型循环。

本方案不是再次实施 0.99 的 MCP、工具发现或 Codemode。它直接复用现有工具目录、授权、嵌套调用、后台子智能体、Fleet、工作流、Goal、JSONL 消息编解码和用量账本。

可验证分为两件事：执行行为能否用提交记录和故障实验核对；任务产物是否满足用户要求。本方案交付前者，保留现有结构化任务报告和审查工作流。上一轮提出的强制完成验收门禁不是 Pi durable 的现成契约，不纳入本次移植。

### 1.1 对齐标准

每个移植项必须有固定提交的源码与测试依据；移植对应的成功、失败、取消、重启行为，而不只移植 API 名字。

Python 本土化允许：TypeScript 类型改为 Pydantic / dataclass，Promise 与 AbortSignal 改为 asyncio 与现有取消令牌，工具注册改为原生 BaseTool metadata，框架内部检查点使用 LangGraph 官方接口。

以下改变不属于本土化：默认把读取工具设为可重放、关闭时把全部任务终结、失败后重跑整个工作流、只持久化最终结果、让 UI 直接消费未提交进度、把业务阻塞当执行失败、根据模型摘要判断恢复位置。

Pi 的核心设计作为本轮目标实现。其 README 的 experimental 标记与单进程所有权描述保留为来源事实，但不作为削减交付范围的理由，也不据开发者背景宣称已经经过大规模生产验证。

### 1.2 与现有项目规则的明确变更

`AGENTS.md` 当前规定检查点仅用于待回答问卷，并禁止持久恢复；同时要求 child 没有独立持久会话。本方案的 D4 / D5 明确替换这两条限制：

1. 允许官方 LangGraph 检查点用于 root / child 执行恢复；检查点存入与产品记录相同的 JSONL 提交存储，没有独立 SQLite / PostgreSQL 状态源。
2. 允许 task 拥有持久 child conversation；child 消息仍留在其会话内，不灌入 root transcript，继续禁止递归委派。
3. 区分 `close` 与 `abort`：关闭当前执行宿主停止在途 invocation 并保留未完成记录；显式取消持久化取消意图并完成清理。
4. 将“renderers 只消费进程内 AgentEvent”收敛为“renderers 消费已提交状态派生的 AgentEvent / snapshot”，保留公共事件字段。

这些调整必须与对应阶段一起修改 `AGENTS.md` 和 `docs/architecture.md`。不能在旧规则仍生效时偷偷启用持久检查点；本次只记录拟议调整，不修改规则。

保留的边界：一个 Python 分发包、既有包依赖方向、LangChain 原生消息与工具、JSONL 唯一持久事实源、既有文件安全与 MCP 授权、没有新的常驻服务或第二语言运行时。

## 2. 固定来源与优先级

下表的 Pi 链接均固定到 durable 基线提交。实现时先读源码和测试；设计文档存在遗留的未实施措辞时，以同一提交的实现和测试为准。

| 编号 | 依据 | 用途 |
| --- | --- | --- |
| P1 | [durable README](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/README.md) | 总体行为、运行与会话、配置、存储、观察接口 |
| P2 | [Session](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/src/session/session.ts)、[Transaction](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/src/session/transaction.ts) | 原子提交、存储成功后采用与发布状态 |
| P3 | [JSONL storage](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/src/storage/jsonl/storage.ts)、[存储测试](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/test/jsonl-storage.test.ts) | main marker、sidecar、尾部恢复、poison、回收 |
| P4 | [Task types](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/src/types.ts)、[Scheduler](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/src/harness/scheduler.ts) | 阶段、waiting、completing、ownership、版本与取消 |
| P5 | [ToolTask](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/src/harness/tool.ts)、[工具恢复测试](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/test/harness-tools-recovery.test.ts) | 执行前 intent、safe / unsafe、部分输出与结果配对 |
| P6 | [GenerationTask](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/src/harness/generation.ts)、[生成恢复测试](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/test/harness-generation-recovery.test.ts) | 请求绑定、重试时间、部分回答、工具轮次 |
| P7 | [Submissions](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/src/harness/submissions.ts)、[Inbox](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/src/harness/inbox.ts) | requestId、输入准入、steer / follow-up / write |
| P8 | [Observation](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/src/session/observation.ts)、[View](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/src/harness/view.ts)、[Events](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/src/harness/events.ts) | 已提交视图、晚加入、慢消费者、事件投影 |
| P9 | [前台子智能体](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/test/examples/22-subagent-foreground.ts)、[后台子智能体](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/test/examples/23-subagent-background.ts)、[Child tasks](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/test/examples/24-child-tasks.ts) | 持久 child、报告去重、等待与取消树 |
| P10 | [Compaction](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/src/harness/compaction.ts)、[压缩测试](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/test/harness-compaction.test.ts) | 后台摘要、边界放置、stale 与 overflow |
| P11 | [任务恢复测试](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/test/harness-tasks-recovery.test.ts)、[任务所有权测试](https://github.com/earendil-works/pi/blob/4812cb268efc3f0846337cdfb8f00ce51b7b8d5a/packages/durable/test/harness-ownership.test.ts) | 不同崩溃位置、版本迁移、缺失定义、取消竞态 |
| P12 | [Pi CLI v1.0 changelog](https://github.com/earendil-works/pi/blob/a13d35a742c6ef8462812a28fbe1d8c8b7431c32/packages/coding-agent/CHANGELOG.md)、[Codemode 包](https://github.com/earendil-works/pi/tree/a13d35a742c6ef8462812a28fbe1d8c8b7431c32/packages/codemode) | 提示收缩与可操作错误 |
| L1 | [LangGraph checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)、[LangChain custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom) | 官方检查点与 middleware 接口；以本地安装签名复核 |

## 3. 当前差距

| 当前实现 | 已有能力 | 本次需要改变的边界 |
| --- | --- | --- |
| `src/forge_agent/session/storage.py` | 单条 JSONL 追加、进程内路径锁、撕裂尾部恢复 | 从逐条写入升级到同一操作的原子提交；存储结果不确定后停止采用与发布状态 |
| `src/forge_coding/sessions/session.py` | 消息完成时保存、分支、Goal/Todo、用量、生命周期 | 保存执行阶段与 live progress；不等待 root 空闲才保存后台任务终态 |
| `src/forge_agent/langchain_runtime.py` | create_agent、v3 投影、临时问卷 InMemorySaver | 使用同一 journal 的官方持久检查点，并保留中断尝试 |
| `src/forge_agent/tool_execution.py` | 批次防护、直接与嵌套共享执行入口、原生 ToolNode | 在共享入口加入 intent / outcome，恢复不能绕过授权与结果配对 |
| `src/forge_coding/features/subagent_runs.py` | 队列、后台回执、取消、Fleet、终态保存 | 当前 restore 将 queued/running 改为 interrupted；升级为恢复原任务与 child conversation |
| `src/forge_coding/features/workflows.py` | workflow_id、step_key、同一 scope 中的步骤复用 | 当前 scope 内字典不是跨重启恢复位置；复用 Pi 的持久任务与 memo，不保存 JS 栈 |
| `src/forge_coding/features/goals.py` | Goal 快照、自动续跑、明确结束工具 | 连接持久执行所有权；Goal 仍是业务目标，不能充当工具恢复日志 |
| `src/forge_coding/codemode/runtime.py` | 一次性 QuickJS worker、原生工具调用、API 搜索与描述 | v1.0 的短提示与可操作错误，保留既有隔离与宿主调用边界 |

当前未跟踪的 `docs/forge-project-guide.md`、`docs/pi-autoresearch-integration-plan.md`、`docs/reliability-security-remediation-plan.md` 保持原状。旧方案只提供历史决策背景，不能覆盖当前源码。

## 4. 目标架构与唯一事实源

```text
CLI / Textual
  ├─ submit / wait / abort / close
  └─ 已提交 snapshot + 派生 AgentEvent
                    │
CodingSession：会话、资源、授权与唯一提交入口
  ├─ 输入准入 / inbox / requestId
  ├─ 持久任务协调器：阶段、所有权、waiting、completing
  ├─ root / child create_agent + 原生 LangGraph checkpointer
  │     └─ 共享 execute_tool_call：intent → 原生 ToolNode → outcome
  └─ JSONL commit journal：entries + tasks + documents + checkpoint writes
        ├─ main.jsonl：唯一提交标记与不可变记录
        └─ task/doc sidecars：仅由 main 标记确认的提交内容
```

所有写入经同一条提交线。这里的“同一”是每个打开的 Session，不是全项目全局大锁。外部工具和模型请求不占用存储事务。

职责按事实拆分：LangChain 的消息、graph checkpoint 和 pending writes 是原生执行事实；Forge TaskRecord 保存操作所属、阶段、恢复规则与结算结果；Goal/Todo/Fleet 是产品投影。不得再实现一份 Forge 版的模型消息、ToolNode 队列或 provider 状态机。

native checkpoint 与产品状态共用一套 journal，并通过原生 checkpoint ID、task ID、call ID 与 commit sequence 关联。可重建的索引、TUI state、消息列表不单独写成另一套权威数据库。

### 4.1 提交与 JSONL 布局

采用 P2 / P3 的提交协议，不用“多次 append 外面包一个锁”冒充原子事务。

- 一次 commit 可同时追加消息与用量、更新任务阶段、更新 live / inbox、创建 child、结算 submission。
- JSONL 跟随 Pi 的 main marker 与 task/doc sidecar：先写 sidecar，再写确认它们的 main marker；内存采用与观察者发布发生在 marker 成功之后。
- 成功确认的记录不可变，commit sequence 严格递增，ID 不重复使用。正文排序与恢复不依赖 UUID 的字典序。
- 未确认 sidecar 尾部不得成为恢复事实；确认内容缺失、完整但损坏的记录、中间损坏必须报错，不能当撕裂尾部吞掉。
- 不确定的 append / flush 失败令当前存储与 Session 失效，停止新派发；重新打开后才能依据 main marker 决定哪些事实存在。
- 沿用 Pi 的 latest / rewindable 文档历史与回收顺序。不能删掉 fork 或恢复仍需要的历史。
- fsync 选项与 Pi JSONL 的默认值和刷新顺序对齐。进程崩溃恢复与主机断电保证分别说明，不把完成一次文件写入表述为抵抗断电。

存储 I/O 属于 `forge_coding`；`forge_agent` 持有记录类型、协议和注入回调。升级时把受影响的旧存储实现整理到正确层，不顺带重构无关工具。

### 4.2 记录契约

| 记录 | 必须保存的语义 |
| --- | --- |
| Submission | conversation、可选 request_id、input/write 类型、queued/placed/done/unanswered、entry/answer 或 reason |
| TaskRecord | kind/version、conversation、owner、background、input、checkpoint、pending/running/waiting/completing/terminal、abort_requested、memo |
| Tool intent | 原 assistant/call ID、最终有效参数、执行阶段、当时的 safe/unsafe 策略 |
| Tool outcome | 原生 ToolMessage 或原生 Command 的受支持结果、任务结算、部分输出、details、用量、结果 entry 引用 |
| Generation request | graph checkpoint 关联、请求身份、历史 cutoff、模型/思考等级/请求选项、attempt、retry deadline |
| Live document | 正在生成的部分回答、工具输出与状态、压缩任务进度；不直接成为下一轮模型历史 |
| Agent document | conversation 的模型/角色/工具选择/目录等声明配置，不保存 provider client、授权令牌或工具闭包 |
| Native graph records | checkpoint namespace/ID、parent、channel versions/values、metadata、pending writes 及其 task key |

Task outcome 使用 Pi 的 completed/failed/aborted/faulted/orphaned 区分，产品展示仍可将其映射为既有 completed/failed/cancelled/interrupted 字段。业务报告中的 blocked 与这些执行结果保持分离。

memo 按 Pi 使用 task-local、first-writer-wins 的小型 JSON 值；completing/terminal 不再保留 memo。需要成为历史事实的 child/submission/result 引用，在结算 commit 中转入结果与不可变 entry，不能依赖终态 memo 找回历史。

只保存恢复所需的可序列化事实。凭据、OAuth token、provider client、MCP client、运行时授权对象、JS VM 与 Python 栈不进入 journal。恢复从现有配置构造客户端并重新校验授权。

## 5. D1：原子 journal 与历史兼容

### 交付

落实第 4.1 节，先把现有消息、用量、Goal/Todo、后台回执的相关写入收敛为一次提交。旧业务循环继续正常运行；D1 不自动重启任何任务。

新增 coding-owned journal；`SessionStorage.append()` 保留为单记录 commit 的兼容入口。现有 replay/tree API 继续获得原来的 SessionEntry 视图，内部不把 TaskRecord 塞进模型消息。

### 迁移

- 旧单文件 JSONL 保持可读，不原地改写、不覆盖。
- 首次升级写入前创建新的 durable session 目录，将已验证历史导入一组已提交 entry；旧叶子、entry ID、消息内容、Goal/Todo 与用量引用保持一致。成功完成导入后再原子更新会话索引指向新位置。
- 导入失败或中断不切换索引；重试依据导入身份复用结果，不重复创建历史。
- 旧历史没有执行 checkpoint，不能凭历史猜测并恢复当时的工作；从新的提交开始具备 durable 保证。
- 旧版导出只输出 legacy entries，并标明不包含执行恢复信息。完整复制/备份必须包含 main 与全部已确认 sidecar。
- 降级采用升级前原件或明确导出的 legacy 副本；旧版本不得写入新的 durable 目录。新执行记录不转换成旧版可恢复任务。

### 验收

根据 P3 先定义失败矩阵，再实施：sidecar 写前/写后、marker 写前/写后、flush 失败、未确认完整尾部、UTF-8 撕裂尾部、确认 sidecar 丢失、错误序号、回收中断、关闭期间写入、并发提交。

新旧会话端到端读取、分支、导出/导入和复制均保留原消息与公开字段。任何 UI 已显示的成功提交在重新打开后必须能找到。

### D1 失败合同（2026-10-02，开发前定义）

开发基线：`f016e658de0aec038749d72e4bdaa3a243ef14f6`；Pi 来源为 P2/P3 固定提交。以下合同先于实现；原定义时状态为待验收，最终结果见下方实施记录。

| 故障点 | 当前宿主 | 新进程重新打开 |
| --- | --- | --- |
| 候选序列化、ID 或文档变更校验失败 | 没有 I/O，不失效，可再提交 | 保持原提交 |
| sidecar 追加前、短写、追加后报错 | 存储与会话失效，停止新派发，不发布成功 | 丢弃未确认尾部，不采用新状态 |
| sidecar flush 前/后报错 | 存储与会话失效，不写 marker | 保持原提交 |
| marker 追加前或短写 | 存储与会话失效 | 新提交不存在；严格按字节去除无换行尾部 |
| marker 完整追加后报错 | 存储与会话失效，不猜测结果 | 找回已确认整笔提交 |
| 完整但无换行的 main / UTF-8 撕裂尾部 | 只能重新打开恢复 | 截断到最后换行；不复用已确认序号 |
| 必需的确认 sidecar 丢失、完整行损坏、序号倒退、未确认记录后又有确认记录 | 拒绝打开 | 不静默修复或吞掉损坏 |
| latest base 回收写、flush、rename / retire、terminal remove 中断 | marker 已确认，提交仍成功 | 删除残留 reclaim，再按 marker 恢复并重试回收 |
| rewindable base/retire | 保留历史 | 历史读取与分支仍成立 |
| 同时提交、取消等待、关闭期间写入 | 单提交线；等待在途 I/O 落定；关闭拒绝新写入 | 只有完整确认的原子组可见 |
| legacy 导入写前/写后、索引切换前中断 | 不覆盖旧文件，未成功不切换索引 | 按导入身份复用确认历史，重试不重复 |

D1 只验证存储重开与历史视图，不将模型/工具调用或任务的自动恢复计作已交付。进程崩溃实验不等于主机断电保证；默认 `fsync=false`，启用时先刷新受影响 sidecar，main 的回收授权刷新沿用 P3。

### D1 实施与验收记录（2026-10-03，历史验收）

2026-10-04 独立审查撤回下述历史完成结论；三个遗漏边界与修复验收见后续记录。

- 开发开始：2026-10-02；验收日期：2026-10-03。开发基线与最终 HEAD 均为 `f016e658de0aec038749d72e4bdaa3a243ef14f6`，未创建分支、提交或推送。保留原有 WIP；`docs/subagent-optimization-plan-v3.md` 只作历史记录，未由本阶段修改。
- 来源：严格使用 P2/P3 的 `4812cb268efc3f0846337cdfb8f00ce51b7b8d5a`，实际阅读 JSONL storage、memory storage、Session 事务实现及 JSONL 测试。安装依赖与 `uv.lock` 保持原版本。
- 实际交付：`forge_coding.sessions.journal` 实现 main 确认 marker、全提交 ordinal、递增 seq、task/document sidecar、候选状态校验、当前文档回收与 rewindable 历史、未确认尾部恢复。候选准备失败无 I/O；追加/sidecar flush 的不确定结果使存储及 CodingSession 失效，旧实例禁止继续读取或派发，只能重新打开。main marker 已追加后的报错在重开后采用完整提交；回收失败不撤销已确认提交。
- `SessionStorage` 在 agent 层只保留协议；具体持久 I/O 归 coding 层。`append()` 兼容为单记录 commit，现有 SessionEntry/tree/replay 保留。消息与用量、Goal/Todo 原生工具结果与快照、后台接纳回执及终态用量按相关组提交，提交成功后才采用业务投影。已接纳提交在取消等待或关闭时先结算，关闭拒绝新写入；读取/采用失败同样使会话失效。
- 旧 JSONL 无覆盖导入；验证全部历史、稳定导入身份和已确认前缀，重试复用原目标，确认成功后才原子切换索引。entry ID、父分支、消息和业务引用保留。完整存储复制包含 main 与所有确认 sidecar；legacy 导出只包含 entries，CLI/TUI 明确标明没有恢复信息。分支复制继续保留原产品语义，完整备份使用独立存储复制入口。
- 本阶段必要的前置修复：v3 的 root tool-finished 回调早于原生 ToolMessage；现改为等原生 values 后才发布 ToolEnd，避免工具结果与 Goal/Todo 快照拆开及取消时伪造结果。Goal 完成通过公开 `before_model` / `abefore_model` hook 停止下一模型轮次。唯一 `create_agent()` / `astream_events(version="v3")` 保持不变，不使用私有 API。AGENTS.md 同步上述 D1 边界，没有启用 D4/D5 的持久检查点或 child 恢复规则。
- Windows 实测 `rb` 文件句柄不能完成本契约所需 fsync，采用 `r+b` 刷新。默认仍为 `fsync=false`；启用时保持 sidecar 先刷、marker 后写、main 刷新授权回收。进程内兼容适配器按路径串行提交，只在 main 时间戳/大小变化时重开，避免每次读取全量重放；不支持跨进程同时写同一 storage。

验收入口为 `tests/test_durable_recovery_e2e.py` 与 `tests/durable_d1_driver.py`：先定义失败矩阵、写失败场景，再实现；覆盖部分写入、UTF-8 撕裂、flush 前后失败、确认 sidecar 丢失、错误序号与数据损坏、回收 write/flush/rename/remove 中断、并发/关闭/已接纳等待取消、迁移中断与索引重试、新旧分支/导入/导出/复制和原子业务投影。恢复场景使用真实子进程 `os._exit(17)` 后启动新 reader。受控工具效果场景记录 writer 模型 2 次/工具 1 次，reader 模型 0 次/工具 0 次，原生调用/结果 ID 配对且效果文件没有重复；这是历史重开验证，不是执行恢复。

| 命令 | 最终结果 |
| --- | --- |
| `uv sync --dev --extra providers --locked` | exit 0；uv.lock 未变化 |
| `uv run pytest --junitxml=.artifacts/d1/acceptance/full-tests.xml` | 1489 passed、10 skipped、0 failed；其中 D1 50 项 |
| `uv run ruff check .` | exit 0 |
| `uv run ruff format --check .` | exit 0 |
| `uv run mypy` | exit 0 |
| `uv run forge --help` | exit 0 |
| `uv run forge --version` | exit 0；0.3.2 |

工件目录：`.artifacts/d1/acceptance/`（忽略，不提交）。根 `manifest.json` / `report.json` 记录基线、源码校验值、命令与实际场景；`commit-log.jsonl` 记录受控存储的 marker、序号、entry ID、业务类型与配对；`baseline-log.txt` 保留已有 Git 提交日志；`final/cases/` 含每例 manifest/report、进程 PID、故障点、reader/writer 结果、main/sidecar 与效果文件及 SHA-256。`pre-final/` 和 `final-before-cancellation/` 保留修正前完整检查；未通过结果不计作最终验收。复验运行 `uv run python .artifacts/d1/acceptance/run_checks.py`；单阶段运行 `uv run pytest tests/test_durable_recovery_e2e.py`，可用 `FORGE_D1_ARTIFACT_DIR` 指定保存目录。

未验证：Ubuntu CI、主机断电、真实 provider（未授权且未调用）；Windows 中 POSIX shell/process-group、symlink 不可用和显式 opt-in integration 的跳过项见 JUnit/日志。一次完整检查的 TUI 单行滚动断言失败，原例独立复跑及最终全套均通过，未为 D1 改动滚动实现。

下一阶段：D1 存储前置条件具备，D2 必须重新核对其任务生命周期与输入准入，不能把 D1 的 task/submission 存储类型当作已交付调度。固定 Pi 存储允许 terminal task ID 再写 pending，但 terminal 回收后旧 sidecar 已移除，重开会拒绝必需记录缺失；D2 应保持终态不可复活的生命周期规则，不复用 terminal task ID。D2–D6 与 V1 未实施；本阶段不自动重启任务，不提供模型/工具执行恢复。

### D1 独立审查修复与重新验收（2026-10-04）

2026-10-04 使用一个与主智能体相同模型的独立 reviewer，复现两项 P1 与一项 P2，因此撤回 2026-10-03 的 D1 完成及 D2 可启动结论。旧验收和 review 工件保留；以下结果为修复后的重新验收，不以历史绿灯替代新增边界验证。

- 修复基线与最终 HEAD：`f016e658de0aec038749d72e4bdaa3a243ef14f6`。仍仅实施 D1，保留既有 WIP、uv.lock 和依赖，不创建分支、提交或推送。
- 先定义三类失败矩阵并新增 11 项端到端回归。修改源码前实跑得到 9 failed、2 passed；覆盖 task 最后状态与 fsync 开关、document 两种顺序及合法外部 marker、legacy/CLI/Session/TUI 导出保护。复现与验证均使用受控临时 workspace/session、离线 fake 模型和真实 reader 子进程。
- task 回收按每个 ID 的最后 write 判定：同一提交 terminal→pending 保留已确认 sidecar；最后 terminal 仍回收。与上一记录的“跨提交终态回收后复活”上游限制区分，未以该限制绕过本次缺陷。
- document 校验返回合并 action，候选采用和恢复共用 action；同一新文档 create/retire 两种顺序均得到退休文档，current-only 正文正常回收。瞬时退休的新 incarnation 可以与既有 live incarnation 同地址。移除独立顺序采用的备用路径，避免恢复与写入再次分叉。
- 全部用户导出入口在写入前共用源存储保护：拒绝源 journal 目录内目标、迁移保留原件、解析后路径别名及目录外硬链接；legacy、CLI 和 Session/TUI 均接入。拒绝时源 main/sidecar/原件字节保持不变，新 reader 正常重开；目录外正常导出保持可用，不派发模型或工具。
- AGENTS.md、README 和 architecture 同步上述 D1 约束，没有扩大后续阶段权限。同模型 reviewer 只读复核三项均 fixed；独立 5 个新 reader 成功、6 次覆盖尝试拒绝，未发现同机制遗漏。挂载 TUI 的新回归也实际验证错误提示且没有导出成功提示。

| 验收命令 | 实际结果 |
| --- | --- |
| `uv run pytest tests/test_durable_recovery_e2e.py -k test_review_`（修复前） | 9 failed、2 passed，验证原缺陷可触发 |
| 同一新回归集合（修复后） | 11 passed |
| document/新 review 相关 focused 集合 | 13 passed |
| `uv sync --dev --extra providers --locked` | exit 0；uv.lock 未改变 |
| `uv run pytest --junitxml=.artifacts/d1/repair-2026-10-04/full-tests.xml` | 1500 passed、10 skipped、0 failed；D1 61 项全部通过 |
| `uv run ruff check .` | exit 0 |
| `uv run ruff format --check .` | exit 0 |
| `uv run mypy` | exit 0；128 source files |
| `uv run forge --help` / `uv run forge --version` | 均 exit 0；0.3.2 |

最新工件：`.artifacts/d1/repair-2026-10-04/manifest.json`、`report.json`、`commit-log.jsonl`、`final/cases/`。`failure-matrix.json`、`red.log/xml` 保留修复前失败；`green-final.log/xml` 与 `focused.log/xml` 保留开发验证；`review/report.json`、probe 和效果文件为同模型独立复核。根 manifest 保存固定 Pi/源码/检查日志/JUnit 校验值、Git 基线日志和命令；每例 report/evidence 保存实际 PID、调用次数、消息配对、提交序号、保护前后的源文件校验值及重开结果。全部工件被忽略，不提交；原 `.artifacts/d1/acceptance/` 与 `.artifacts/d1/review-2026-10-04/` 保留原样且属于修复前证据。

复验：`uv run python .artifacts/d1/repair-2026-10-04/run_checks.py`；三项问题单独复验使用 `uv run pytest tests/test_durable_recovery_e2e.py -k test_review_`，可设置 `FORGE_D1_ARTIFACT_DIR` 保留每例工件。

未验证项：Ubuntu CI、主机断电、真实 provider、多进程同时写同一 storage；Windows 本轮平台/显式 opt-in 的 10 个跳过项见 JUnit。D1 不交付任何模型/工具/任务执行恢复。D1 重新验收通过，D2 的存储前置条件恢复具备；D2 仍须独立核对其任务准入与生命周期，不复用已终结 task ID。

## 6. D2：持久任务、输入准入与所有权

### 交付

在现有后台协调器基础上加入 P4 的阶段调度。只派发已提交任务。任务每个阶段必须提交新的 checkpoint、waiting 或终态；阶段返回但没有持久进度属于 faulted，不能无条件再跑。

状态和行为对齐 Pi：

- open 将 surviving running 归一为 pending，保留 checkpoint；不调用模型或工具。resume 启用派发；submit/wait 按 Pi 接口启动调度。
- waiting 保存 on 与 allSettled/failFast。等待期间不持有 Python handler，不调用模型轮询任务状态。
- 父任务决定结束时，若普通 owned work 尚未结束，记录 completing 与 held outcome，等待子工作真正结算才成为 terminal。
- 取消先保存 abort mark，再沿所有权自下而上处理。failed/aborted 的 owner 先取消其普通 owned work，再完成自己的 abort handler。
- background 是所有权遍历边界：普通 conversation abort 与 idle 不跨越它；显式包含后台工作的 abort 才跨越。
- close 结束当前 invocation、停止派发并关闭资源，保留未完成 checkpoint；不等同于 abort。不能允许关闭后的回调继续提交。
- 定义缺失、存储版本较新或缺少迁移时保留 pending，并在 inspect 中报告具体原因。版本迁移在 reservation 时发生，失败不改写原记录。被取消且定义缺失的任务按 Pi orphaned 处理。
- invocation 结束后持有的 runtime handle 不再能写入；拒绝过期回调和已被替换实现的迟到提交。

任务定义先只服务于 Forge 内部模型运行、工具调用、child/report 与 compaction，不附带第三方插件系统。

### 输入与等待接口

增加 SDK 层 submit/submission/wait/resume/inspect 接口，submit 使用 Pi 的 when_busy=follow_up/steer/reject，默认 follow_up；write 仅追加 entry，不启动模型。

保留既有 prompt 的并发默认拒绝行为作为兼容适配器，它显式传 reject；TUI 使用明确的提交方式，不在迁移中改变已输入消息的用途。

request_id 的去重范围是 conversation。同一 ID、同一 submission 类型返回原 submission；Pi 不对不同内容额外做一致性校验，因此不能把 Forge workflow 的 assignment 冲突规则误移到 submit。同一 ID 的类型不同则报错。

queued inputs 与 inbox 必须和 submission 一起提交。steer 在当前工具轮结束边界放置，follow-up 在当前 run 回答边界放置，保持 Pi 的放置顺序。run 失败后队列仍保留，不丢弃也不偷偷作为当前请求重送。

取消 submission.wait/task wait 只取消观察者等待，不取消工作；显式 abort 才改变任务状态。撤回 queued input 与取消已放置工作分开。

### 验收

对齐 P7 / P11，覆盖重复 request ID、队列顺序、失败后队列保留、重复唤醒、等待者超时、父子失败、completing、后台取消边界、版本迁移和每个取消阶段的重启。

D2 可独立交付：输入与协调任务具备持久状态；模型运行仍按已有中断策略处理，尚不宣称模型/工具阶段恢复完成。

## 7. D3：已提交进度与观察视图

### 交付

按照 P8 将 TUI、print/JSON、Fleet 与导出的 live activity 从已提交视图派生。raw v3 事件先进入会话持久化入口，不能直接绕到 renderer。

- 部分模型回答和工具输出按 Pi 的 100 ms trailing-write 节奏保存；同时最多一个 progress commit 在途。
- 终态保存前停止并排空进度写入，防止终态之后出现旧 attempt 的部分内容。
- 显示的内容必须已经提交；崩溃时可能损失尚未提交、也尚未展示的进度，不损失此前已展示的状态。
- 晚加入或重新连接先取得当前 snapshot，再接收后续提交；不要求从启动开始完整重放内存事件。
- 慢消费者最多积压 100 个提交 frame，超出后合并为最新完整 snapshot，回调顺序和快照切换按 Pi 保持。
- task graph 只显示 live tasks，并报告 owner、background、waiting/completing。缺失 definition 等阻塞原因由 inspect 提供，不混成 task graph 新状态。

Python 不复制 Chord 的代理对象实现。typed documents 使用受控 JSON 值与显式变更操作；append/trim 等进度变化保存增量，必要时保存 complete base，遵守 Pi 的文档历史和 snapshot 契约。若未来提供 Chord-compatible wire API，应作为单独项目，不能声称本轮已兼容其字节协议。

### 验收

mounted Textual 测试与实际子进程重启：在部分回答、后台输出和任务结算中途停止；重新挂载时显示内容与最后确认 snapshot 一致。慢消费者能获得新快照，后台进度不再等父任务空闲才保存。

D3 独立改善当前可观察性，不依赖恢复模型调用才能成立。

## 8. D4：LangChain 原生恢复与工具 replay

### 8.1 官方检查点适配

用 coding-owned `BaseCheckpointSaver` 将原生 graph checkpoints 与 pending writes 存入同一 journal。`forge_agent` 只接收注入的 saver、执行上下文及提交回调。

本地已核对的公开异步契约为 aget_tuple/alist/aput/aput_writes/adelete_thread；同时正确处理 channel versions、parent config、namespace、写入去重与 delta channel history。复制和裁剪接口只有在完整保留 fork/恢复引用时才使用。不得读取或修改 Pregel / ToolNode 私有字段。

恢复使用同一 thread/checkpoint 与重建的 create_agent；没有新输入时传 None，问卷仍通过官方 HITL respond 与 Command(resume=...)。采用 durability="sync"，确认 native checkpoint 保存后才进入后续 super-step；v3 的公共 kwargs 可以传该选项，不覆写 v3 管理的 stream_mode/subgraphs。

sync checkpoint 不能代替逐工具 intent，也不能代替 UI 的 commit-before-publish。因此 D3 与下节仍是必要边界。

serializer 使用官方协议并关闭 pickle fallback，限制可重建类型；运行时 client/context 不进 channel values。不能用 pickle 保存 provider、工具闭包或任意用户对象。不能重复注入恢复 checkpoint 已包含的 root/child 消息。

### 8.2 请求阶段

原生 middleware 与图状态中保存最小的请求身份引用，Pi 的 prepare/request/retry/tools 阶段映射到原生执行，不另写模型循环。

请求身份固定为 conversation ID、run ID 与 native graph state 中的 request index。index 只在原响应已经提交后通过官方 after_model 状态更新前进；崩溃恢复到旧 checkpoint 时仍命中原请求。attempt 属于同一请求的重试计数，不用新 run ID 掩盖重试；tool call ID 再按所属请求限定作用域。正常下一轮与恢复旧轮次必须据此区分，不能凭消息文本、当前时间或模型自报 ID 去重。

- prepare 绑定该请求的模型、思考等级、请求选项、历史 cutoff 与工具/提示声明。配置变化对下一次 prepare 生效。
- request 恢复沿用上述绑定；beforeRequest 对应的 middleware 仍按 Pi 在请求时运行，不声称任意动态 hook 的输出已经被冻结。需要稳定的决定使用该 task 的持久 memo。
- 响应提交同时保存 assistant、用量、请求 outcome 与 live 清理。若框架 checkpoint 晚于结果提交，middleware 根据原请求身份返回已提交原生响应，不再次请求 provider。
- interrupted partial 记录为 aborted attempt，只作为可查看历史，排除出以后 provider 的有效上下文，然后重发相同 cutoff 的请求。不能伪装成完整回答，也不从半个 token 继续。
- retry attempt 与绝对 until 时间保存下来，重启不能清零计数或重复等待完整间隔。持久化包装现有 retry，不叠加另一层独立重试。
- Pi 的 deferred handle/poll 阶段仅在 provider 暴露等价公开能力时启用；普通 LangChain 流式请求不能捏造可续传句柄。这是 provider 能力差异，不改变正常请求的恢复规则。

### 8.3 工具阶段

改造既有共享 execute_tool_call，直接、nested、root、child 路径一体生效：

1. 解析当前可用 native tool，校验参数与当前授权，运行必要的参数修正/执行前拦截，再次校验最终参数。
2. 原子提交 execute intent 与 running live slot，随后才进入 ToolNode handler。
3. handler 结束后原子保存原生结果、output/details/usage、结算与 live 清理，再返回原生 ToolMessage/Command。
4. 同一 call 已有 outcome 时返回已提交结果，不再调用工具。
5. intent 存在但 outcome 不存在时，只在存储与当前定义的 replay 都为 safe 时重跑；否则结算 interrupted，保留已提交部分输出，并让原生 agent 处理错误结果。

`forge.replay` 使用 safe/unsafe，缺省 unsafe。Pi 内置 read/write/edit/bash 没有声明 safe，所以 Forge 也不因“read 看起来无副作用”而自动豁免。safe 首先用于 P9 那类通过 owner/request ID 找回原工作、具备幂等实现的组合工具。

安全重跑前清理上次 attempt 的 live progress，使用已保存最终参数与当前工具实现，按当前 conversation 环境重新构造执行上下文。Pi 并不保证重跑时 cwd 永远等于初次调用，所以不能把它改成隐式历史环境回滚。

保留原生 ToolMessage 配对、普通错误不停止兄弟调用、任一 sequential 工具令整轮顺序执行等现有规则。嵌套调用 outcome 仍留在父任务作用域，不另追加为 root transcript。

Codemode/任意 JS workflow 自身保持 unsafe。JS 栈、Promise 和 VM 不恢复；不能由于内部存在安全工具，就将整个脚本自动标成 safe。已完成 nested effect 在所属任务中可核对，不意味着能安全重启任意父脚本。

### 8.4 生命周期

区分用户 abort、Harness close、崩溃与 HITL interrupt。close / crash 保留待恢复工作；abort 保存取消意图并使相应工作真正终结。问卷可以从原 checkpoint 恢复，非交互模式依旧不注册 ask_user_question。

历史分支导航不等于重放外部效果；恢复针对已记录的 live task / thread。fork 继承截止位置的历史与 agent 配置，创建新的执行身份，不复制父任务的活 invocation。只有明确提交才开始新工作。

### 验收

逐项移植 P5 / P6 的恢复场景，另用真实进程强制退出验证：模型调用前、partial 已提交、response 已提交但 graph checkpoint 未提交、工具 intent 前后、effect 后但 outcome 前、outcome 后、顺序/并发工具轮、nested call、问卷等待。

验证 unsafe 的原调用没有自动重跑、safe/safe 会重跑、safe/unsafe 与 unsafe/safe 均不重跑；结果已提交的调用复用原结果。注意恢复后模型可能自主发出新的工具调用，这与系统重放旧 call 是两件事，验收按 call ID 区分。

D4 后，root 模型/工具 run 才具备本方案承诺的跨进程阶段恢复。

## 9. D5：持久子会话、后台报告与任务组合

### 交付

复用 SubagentRuns、SubagentRunner、角色资源和 Fleet，将 in-memory invocation 升级为 P9 的 task-owned conversation。保存角色/模型/工具选择的声明配置，每次启动重建客户端；执行中使用既有原生 child graph，不放开递归控制工具。

前台组合工具可声明 safe 的必要条件是：创建 child 的 commit 中查询当前 owner task 的已有 child；重复执行复用该 child；child input 通过稳定 request ID 提交；恢复继续等待原 submission。不能每次新建 conversation 后声称安全重放。

后台使用持久 background anchor 拥有 child，以及 reporter task 拥有等待与报告工作。普通 root abort 不取消 anchor；reporter 使用稳定 request ID 将结算结果作为 follow-up 提交父 conversation，恢复不能重复发消息或报告。

这会替代当前“完成后提示、只在模型下一轮告知、用户手动 collect 才启动父轮”的唯一通路。Fleet 仍支持查看与手动 collect；自动报告是明确的后台提交行为，不把每一条 child 消息灌入 root。

TaskRun 的历史导航身份与 owner 不混为一谈。owner 决定取消/idle 范围，不提供工具权限；child 工具目录仍由当前授权与角色能力限制决定。

### 工作流

复用稳定 workflow_id/step_key 与现有 assignment 冲突检查。阶段调度与等待采用 P4 的 TaskDefinition/checkpoint/memo：完成阶段保留，等待阶段恢复原 on 集合，不重启已完成 child。

普通 saved JS workflow 继续是一段受限工具执行。其已提交步骤报告可查看，但跨进程恢复不能依赖当前 scope 的字典或重跑 JS 来猜断点。需要跨重启的组合改为具名持久阶段，并以 task memo 记录已有 child/submission；参考 P9 的示例组织，不添加新的流程 DSL 或 VM 快照系统。

保留四个并发、十六个 queued task、现有工作流 launch/时间/报告限制。这些是 Forge 已有产品策略，不冒称 Pi 的统一默认值；重启不清零已消耗额度。

角色 fallback 与 Goal 的自动续跑限额同样保留为 Forge 已有策略，不宣称来自 durable。其已发生尝试与用量进入统一提交记录，恢复不重置；本轮不另外设计模型路由或 Goal 完成判据。

### 验收

spawn 后、child submit 前后、child 工具中途、child 结算后 reporter 提交前后分别退出重启。验证 owner 不变、child 不重复、submission 不重复、report 只提交一次，前台取消树与后台边界一致。

现有工作流 review/fix、并行审查继续工作；业务 blocked 不转换为执行 failed。已完成步骤不因后续失败重跑。

## 10. D6：后台压缩与配置阶段边界

### 交付

移植 P10 的独立 CompactionTask。摘要生成时 conversation 可继续工作；摘要在 idle 或下一个合法 turn boundary 放置，不能在工具结果尚未配对时更换上下文。

- 摘要保存 cutoff/head 与 submission 身份，历史 entry 不删除；上下文压缩不是存储清理。
- 多个摘要同时完成时，已经超出有效范围的结果按 Pi 结算 stale，不能让较旧摘要倒退有效上下文。
- 背景阈值启动摘要，必要阈值等待摘要；provider overflow 触发压缩并只重试一次，不叠加无限重试。
- 压缩消耗进入同一用量账本，partial/retry/终态进入 live view；关闭后可恢复阶段，用户 abort 按任务所有权取消。
- 保留 Goal/Todo、授权状态、tool loadout、Codemode store 和 child 所有权，不从摘要文本重新构造这些状态。

配置语义对齐 P1 / P4：请求在 prepare 绑定，工具与 hook 在各自阶段解析；执行中已经取得的实现继续运行，下一阶段采用新注册定义。资源缺失/版本不匹配保留 pending 与 inspect 原因，不无条件落回另一个角色或工具。

不要求移植 Pi 全部扩展加载生态；先让当前 Forge 角色、MCP 目录与任务定义按相同阶段边界生效。凭据与授权始终由现有运行时重建。

### 验收

压缩期间继续输入、工具执行中等待放置、多个摘要乱序完成、过期摘要、取消/close/restart、overflow 一次恢复、用量计入和有效上下文保持。长期会话不能因为 native checkpoint 或 sidecar 回收而丢掉分支所需历史。

## 11. V1：Codemode 提示与错误增量

此项可独立实施，与 durable 各阶段没有运行依赖。

跟随 P12：工具声明只保留必要调用形式，详细 schema/API 通过已有 searchTools/describeTool 获取，避免 system prompt 与工具 description 重复整份说明。MCP 目录变化不反复改写 Codemode 的稳定主体说明。

未知成员按 Pi 提供近似名称与可用发现接口；参数错误给期望形状；store 超限说明用途和边界。复用 worker/runtime 的现有错误回传，不新增自动重试、不改变失败与取消分类，不将隐藏/未授权工具放进建议。

验收使用相同离线工具目录比较提示文本及 token 化结果，保存前后工件。工具错误后模型能通过现有入口修正调用；任何节省值必须来自 Forge 的实测，不能引用 Pi 的比例作为 Forge 成果。

v1.0 的 fullscreen 默认、图像生成、Radius 登录、MCP OAuth 增量不属于 durable 核心。本方案不顺带实施；它们可另行按 Pi 源码单独对齐。当前 MCP 权限和凭据隔离仍参与所有恢复验收。

## 12. 文件责任与实施顺序

整个升级涉及超过八个源文件与对应验证面，属于明确的执行/存储边界升级。按 D1 → D2 → D3 → D4 → D5 → D6 推进，V1 可独立交付。每阶段都有独立收益与验收，不以完成后一阶段作为前一阶段可用的条件。

| 模块 | 责任 |
| --- | --- |
| `src/forge_agent/session/entries.py`、`jsonl.py`、`tree.py` | 保留历史消息与分支编解码，支持提交后的 legacy entry 投影 |
| `src/forge_agent/session/transactions.py`（新增） | typed commit / task / submission / document 协议，不执行文件 I/O |
| `src/forge_coding/sessions/journal.py`（新增） | Pi JSONL 提交、恢复、回收与兼容导入，唯一磁盘事实源 |
| `src/forge_coding/features/tasks.py`（新增） | Pi 阶段调度、ownership、waiting/completing、memo、inspect |
| `src/forge_coding/sessions/checkpoints.py`（新增） | 官方 BaseCheckpointSaver 的 journal 适配，不运行模型/工具 |
| `src/forge_coding/sessions/session.py` | 准入、提交、迁移、资源生命周期、root 恢复与公开 SDK 适配 |
| `src/forge_agent/langchain_runtime.py`、`harness.py`、`retry.py` | 原生 graph 恢复、请求身份、已提交事件投影与重试阶段 |
| `src/forge_agent/tool_execution.py` | direct/nested 共用 intent/outcome/replay 边界 |
| `src/forge_agent/subagents.py`、`src/forge_coding/features/subagent_runs.py` | child 原生执行、持久 child 配置、anchor/reporter，删除被新协调器替代的重复状态 |
| `src/forge_coding/features/workflows.py`、`goals.py` | 复用持久任务；保留现有业务规则与预算 |
| `src/forge_coding/sessions/compaction.py`、`manager.py`、`copying.py`、`importing.py`、`export.py` | 持久压缩、会话格式、完整复制与 legacy 导出 |
| `src/forge_cli/tui/app.py`、`state.py` 及既有 Fleet 展示模块 | 提交后视图、task graph、迟加入快照与恢复提示 |
| `src/forge_coding/codemode/runtime.py`、worker | V1 短提示与可操作错误 |
| `README.md`、`docs/architecture.md`、`AGENTS.md`、`NOTICE`、release notes | 随已交付阶段同步使用方式、边界、来源与版本事实 |

新增四个源码文件承接确实新增的职责，优先复用旧模块，不先搭完整 durable 包或插件框架。实施过程中若文件拆分增加，必须由实际模块长度与跨层责任解释，不能为未来扩展提前铺设。

## 13. 端到端验收与工件

先按 Pi 的失败边界写验收场景，再实现。新增验证优先集中于真实独立进程，不在实现后补一套逐函数单元测试。现有 focused tests 继续作为回归面。

新增 `tests/test_durable_recovery_e2e.py` 作为统一入口；离线 fake LangChain model、受控工具效果与专用临时 session 目录，不能访问实际用户会话或凭据。测试驱动启动子进程，在指定持久阶段终止进程，再启动新进程执行恢复。

| 场景 | 必须核对 |
| --- | --- |
| input 已准入但调用者没拿到回执 | 同 request ID 找回原 submission，不重复输入 |
| 存储部分写入、结果不确定 | 失效 Session 不继续发布；重开仅采用已确认提交 |
| model partial 后退出 | partial 作为 aborted 历史；原 cutoff 重发，不能作为完整结果 |
| response commit 后 graph checkpoint 前退出 | 复用已提交原生响应，不重复 provider 请求 |
| unsafe intent 后 effect 发生、outcome 未保存 | 旧 call 不重跑，保留部分输出并结算 interrupted |
| safe/safe、safe/unsafe、unsafe/safe | 仅第一种重跑；当前权限不满足时不能重跑 |
| 多工具批次与 nested calls | 配对完整；完成效果不重复；sequential 顺序不改变 |
| close、用户 abort、取消 wait | 三者语义不同，等待者结束不使工作结束 |
| child/report 前后退出 | child、input、report 不重复，owner/background 正确 |
| waiting/completing 与 failFast | 不耗模型轮询，子树结算前 parent 不 terminal |
| definition 缺失/升级/迁移失败 | pending/inspect 可见，无强制新实现重跑旧副作用 |
| HITL、branch/fork、compaction | 同 graph 合法恢复，不泄露子会话或重放历史效果 |
| late join / slow consumer | 从已提交 snapshot 恢复，最多 100 frames 后合并 |
| legacy 导入/导出/复制 | 消息、分支、Goal/Todo、用量一致；不伪造旧恢复能力 |

每阶段生成一份可复验工件目录：`manifest.json`、`report.json`、脱敏提交日志、可计数的工具效果文件及产物校验值。manifest 记录 Forge/Pi/依赖基线、场景、执行命令、故障点、随机种子；report 记录实际调用次数、配对检查、提交序号、恢复结果和断言。

测试直接终止子进程时只能操作由该场景创建并记录 PID 的进程。输出保存在测试指定临时目录，既可保留核查，也不能提交进仓库。所有文件与工具操作使用专用测试 workspace。

阶段开发只跑有关的 focused cases；结束时运行对应端到端集合与项目规定的完整检查，不在开发期间反复跑全套端到端测试：

```bash
uv sync --dev --extra providers --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run forge --help
uv run forge --version
```

文档本身不承诺测试已经通过。实现交付必须附实际命令、结果、工件路径与未验证项。真实 provider smoke 仍是明确 opt-in，不需要新增账号、服务、API key 或外部测试费用。

## 14. 持续跟进 Pi 的方式

保持固定基线与来源索引，不在实现中追随浮动 main。每次更新选择一个新提交，比较 P2 至 P11 涉及的源码和测试，形成“行为变化 → Forge 对应模块 → 回归场景”的差异表。

优先同步恢复、取消、存储和工具结果契约的修正；UI、模型目录等独立更新单独交付。Pi 的新增测试优先转成 Forge 等价的端到端失败场景，随后修改对应实现，更新 NOTICE 与方案基线。

不只复制上游新函数，也不根据名称相似跳过行为核查。若 LangChain 公共接口变化，保留 Pi 的可观察契约并调整官方适配器；不借更新之名加入私有 API、第二循环或新的事实源。

## 15. 后续独立方向

完成本方案后，另行设计带可执行 verifier 的 Goal 完成证据，以及脱离交互界面的常驻宿主。它们不混入 Pi 移植项，不影响本次 durable 核心的完整交付。

本轮默认一个进程拥有一个 storage，与 Pi 一致，不引入跨进程抢任务或分布式 worker。长时间持续执行由同一宿主运行；关闭后的工作保留并在重新打开/resume 后继续，不声称关闭程序后仍有进程在执行。

第一项实施工作是 D1：以 Pi JSONL 的故障矩阵为验收合同，落地原子 journal 和旧会话兼容。不要先增加“恢复”按钮，也不要先给现有后台任务套无限重试。

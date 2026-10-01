# Forge subagent V3 升级路线

状态：R1 已实现，2026-10-02 完成离线验收；R2–R3 尚未实施。
日期：2026-10-01。代码基线：`ed15006bf74afb95782802fd3a189a55fcb45d7f`。
Pi 参考基线：`d6726aeaa3cc849b91140c6d87fc76774f9585a7`。
已核对本机安装版与锁文件：LangChain 1.4.3、LangGraph 1.2.12。

本文件替换此前偏保守的“prompt 示例 + 结构化报告”计划。它们仍然有用，但应是升级中的
基础能力，不应占据整个版本。small/readable 限制的是实现方式，不是产品能力的上限。

## 1. 推荐交付范围

V3 从一次性委派升级为用户可以观察、组合和控制的多代理工作台，推荐连续完成 R1–R3：

| 里程碑 | 交付 | 用户能做什么 |
| --- | --- | --- |
| R1 | 会话内后台任务 + Fleet + 状态/等待/取消 | 子代理审查时继续与 parent 讨论，随时看进度、收结果或停止 |
| R2 | 角色模型配置 + 可选 fallback + 父历史继承 | scout 用一个模型，reviewer 用另一个；fork child 自动理解前文 |
| R3 | 结构化结果 + Codemode 工作流 + 可保存模板 | 一次执行定位、实现、并行审查、按结果修正的可重复流程 |

估计：R1 5–8 个工程日，R2 3–5 日，R3 3–5 日，跨阶段最终验收再预留 2–3 日，合计
约 13–21 个工程日。这里的工程日是含实现、端到端验收和返工的排期单位，不是实测耗时，
也不是 AI 墙钟时间预测。开始编码后的首个里程碑应据实际复杂度更新估计。

后续能力也进入路线，而不是被提前排除：

| 后续里程碑 | 交付目标 | 粗估工作量 | 为什么单独交付 |
| --- | --- | --- | --- |
| R4 | steer / follow-up / 同一 child 连续工作 | 3–6 日 | 需要运行中的消息队列、消费确认和安全轮次边界 |
| R5 | 跨进程 detached background + 退出后继续 + 重连 | 7–12 日 | 需要进程所有权、通信、日志恢复和 Windows/Linux 双平台验证 |
| R6 | 多 writer worktree 隔离 + 变更汇总 | 4–7 日 | 需要独立 Git 生命周期、合并冲突和用户改动保护 |

R1–R3 是推荐实施范围；R4–R6 是功能与研究边界明确的后续路线，尚不是可直接编码的详细
规格。选择执行其中某项时，再按当时源码写对应接口、持久化协议和文件级实施计划。
不把这些研究工作伪装成已经解决的架构细节，也不把它们作为 R1–R3 可用的前提。

## 2. 当前基线与需要纠正的历史描述

已读取 Codex 聊天“设计 Forge subagent 方案”以及 `docs/dev/` 中 V1/V2 历史计划。
当前代码比那些方案更靠前，以下事实以源码为准：

- 父、子都使用 create_agent；没有第二个 provider/tool loop。
- SubagentRunner 已无全局串行锁，同一模型批次的 task 能并发。
- 已有用户/项目 AGENT.md、角色覆盖、工具能力限制、原子 reload，注册表最多 16 个角色。
- child 仍是 fresh context，继承调用时 parent 模型，无独立持久会话，不能递归调用 task。
- 后端保存有界 child trace；当前 TUI 的 SubagentDisplay 主动丢弃 child 内容，只留角色、
  状态和统计。旧 V2“Ctrl+O 展开结果与 trace”的描述不能当作现行展示边界。
- 当前 README 的 trace 展示描述、architecture 的 task 串行描述与代码不一致，R1 同步修正。
- Forge 已有 Codemode：QuickJS 一次性 worker 可以异步调用 Python 侧原生工具。已有
  tools 调用、Promise 组合、结果回传、超时、取消和输出预算，工作流不必另建 JS 运行时。
- 已有 provider catalog、provider construction、模型错误分类、retry 和 usage ledger。
  新增角色模型和 fallback 应复用这些能力，而不是从头实现模型管理。

## 3. Pi 与 LangChain 的职责

从 Pi 学产品机制：后台任务、Fleet、上下文 fork、可组合工作流、输入消息确认、运行工件。
从 LangChain 学执行机制：create_agent、native tools、ToolRuntime、middleware、structured
output、异步事件流。Python/Codemode 负责任务生命周期与组合，不能再造模型推理循环。

需要纠正一个参考事实：当前 Pi 每次 launch 只解析一个模型，provider 错误包括 429 不会
自动换模型。它的 retained child 恢复也不是模型 fallback。Forge 的 fallback 是结合
LangChain 做的新增实验，不能声称照搬了 Pi 的现成机制。

```text
Parent create_agent
  ├─ task / task_control ──> 会话任务协调器 ──> child create_agent
  └─ workflow/Codemode ──> 受限 runs host API ──> 同一个协调器
                                   │
                        有界运行事件与交接结果
                                   │
                     CodingSession（唯一持久化写入者）
                        ├─ 父 JSONL CustomEntry
                        └─ 任务事件 ──> Fleet / inspector
```

协调器拥有活任务，JSONL 记录已提交的运行事实，Fleet 只是展示投影；不增加同等权威的
status.json、SQLite 或第二套业务 ledger。R5 若引入独立进程，这个所有权协议必须重新设计。

## 4. R1：会话内后台任务与 Fleet

### 4.1 公开接口与生命周期

保留 task，增加可选 mode=foreground/background，默认 foreground。background 提交后
立即返回 run_id 和 accepted；accepted 只代表已接受任务，不能冒充完成。

增加一个独立管理工具 task_control，提供 list/status/wait/cancel 四个动作。精确 run_id
关联任务，限制在当前 session 和可见分支；不按任务描述匹配。wait 为有界等待，超时只返回
pending，不取消 child。completed/failed/cancelled 状态以及结果可重复读取。

状态为 queued → running → completed/failed/cancelled；未运行的 queued 也可取消。
默认后台同时运行上限 4，排队上限 16，单个任务超时 15 分钟；支持 session 配置收紧或扩大
上限，服务端统一校验，工具调用不能临时突破上限。排队期间冻结任务配置和上下文快照。

CodingSession 在自己的 scope 内拥有 asyncio.Task 和有界事件队列；泛用 child 执行原语仍在
forge_agent，不导入文件、UI 或 provider SDK。后台 child 使用独立 astream_events(v3) 驱动，
明确设置 run 标识并隔离 parent callbacks/config，不能让已结束的 task tool call 继续收到
伪造的 tool_end，不能将 child values 投影为父消息。

只在角色明确配置 mode=background 或用户明确要求后台运行时启动；普通任务不自动转后台。
首版非交互 CLI 拒绝 background mode，保留 foreground；不会 CLI 退出后静默丢失工作。

### 4.2 消息配对、持久化与结果交付

后台提交的父 task ToolMessage 立即配对，内容为 receipt。稍后的 child 完成不能补写第二个
同 call id 的 ToolMessage，也不能改写已经提交的 receipt。

使用版本化 forge.subagent_run.v1 CustomEntry 保存：run_id、origin 分支、generation、角色、
状态、模型、统计和有界结果。start 必须在原始父 AI call 已持久化后提交；父 receipt 写入
之前收到的 child 完成事件先缓冲，receipt 提交后再写 terminal entry，保证顺序明确。
后台事件仅由 session 的串行持久化入口写入，不能 child coroutine 并发操作 active leaf。

completion 通过 Fleet 状态与一条可点击的产品通知显示。parent 在下一轮请求中收到有界
“任务已完成/失败 + run_id”状态摘要，完整结果通过 task_control 读取；通知去重且与分支
绑定，不在当前 token 流中插入消息，不自动额外发起模型调用。等待结果的用户可以直接
选择收取并让 parent 分析。完整结果不复制进每轮系统提示词。

close、new/adopt、branch switch：先取消并等待当前后台任务收尾，再关闭 provider/MCP。
generation 不匹配的晚到事件丢弃，不写入新会话。parent 的 Esc 取消当前前台 prompt，已接受
的后台任务继续；Fleet cancel 才取消指定后台任务。提交尚未成功时，Esc 取消 provisional child。
用户退出时取消所有后台任务，下一次加载时将没有 terminal 的旧记录显示为 interrupted，
不自动恢复、不自动重跑可能有副作用的任务。这是第一版能力边界，不是长期放弃 detached。

### 4.3 工作区和资源所有权

后台读取会与 parent 修改工作区重叠，不能宣称读取的是稳定快照。审查结果标注其启动时
Git baseline；工作区变化后 parent 应核实受影响的结论，不把 baseline 等同于文件快照。
后台 worker 必须有明确文件责任，parent 不编辑该责任范围；严格的并发写入隔离由 R6 解决。
bash、MCP 和外部进程仍不能靠 same-file queue 获得全工作区隔离。

后台任务冻结 role/spec、tools 和模型引用。provider/MCP 热更新不得提前关闭它们引用的
资源；R1 先拒绝存在活后台任务时的 provider/MCP/catalog reload，并明确提示等待或取消。
这种忙态限制以后可用资源 lease 精细化，但不能忽略资源生命周期以实现表面热切换。

### 4.4 Fleet 与 inspector

composer 上方保留一行状态，例如“2 running · 1 queued · 1 failed”。没有任务时隐藏。
/fleet 打开任务列表，显示角色、模型、状态、耗时、工具数和 token；状态不只依赖颜色。
上下键选中，Enter 进入详情，Esc 逐级返回，取消按钮针对选中的精确 run_id。
80x24 使用全屏详情，宽屏可左右分栏；不自动夺走 composer 焦点、不新增键位冲突。

主 transcript 的 task block 继续遵守现有 allowlist。独立 inspector 按用户选择加载后端
已有的安全 trace 与最终结果，内容不回灌主 transcript/TuiState；只显示经过校验的展示
投影，不保存 thinking/raw tool args/raw tool results。这样增加 inspect 能力而不撤销
当前主对话内容过滤边界。markup 禁用，窄屏换行，复制只含 inspector 当前可见内容。

验收：任务后台运行时 parent 能完成第二次对话；完成后 Fleet 更新并能读取结果；取消单项
不影响 sibling；关闭、换会话、branch 和 provider 资源释放不产生泄漏或串线。

### 4.5 R1 实现与验收记录

代码仍基于文件顶部的 commit，尚未提交。实现位于 `forge_agent.subagents`、
`forge_coding.features.subagent_runs`、`CodingSession` 和 `forge_cli.tui.fleet`。
未增加依赖、独立 provider loop 或持久 LangGraph checkpointer。

后台流程端到端验收覆盖提交、继续对话、单项取消、排队、超时、退出、分支/会话切换、
receipt 落盘失败、切换失败后的继续使用，以及恢复时不重跑。挂载的 Fleet 验收覆盖
80×24 布局、键盘操作、结果字面显示、复制、收取和通知去重。
可重复运行以下命令，生成 synthetic JSONL、summary.json 和 Fleet SVG：

```powershell
uv run pytest tests/test_background_subagents_e2e.py tests/test_tui_fleet.py -q --basetemp .pytest_cache/r1-acceptance
```

真实 provider 未验收；会话恢复将未完成任务标记 interrupted，后台进程脱离会话不在 R1 内。

## 5. R2：模型专门化、fallback 与父历史继承

### 5.1 角色模型与 fallback

AGENT.md 增加 model、fallback-models。model 默认 inherit；显式引用使用已有 catalog 中
可解析的 provider/model，fallback-models 默认空且最多 2 个，不引入 fast/smart 模型 tier。
不让模型临时提供任意 provider URL 或凭据。/agents 展示模型契约和 fallback 链。
profile 只选择当前已授权、已配置的 provider；声明未配置引用时 launch 前失败。

每次 launch 在 coding 层解析并构建 BaseChatModel，由 session 管理所有权；task 专用模型
不会切换 parent 当前模型。请求开始后 primary/fallback 列表固定。

安装版提供官方 ModelFallbackMiddleware，可作为原理和通用行为参考，但它捕获广泛的
Exception，不能直接满足 Forge 的错误分类要求。推荐通过官方 awrap_model_call 小型
middleware 实现受限 fallback，复用 classify_model_error 和现有 provider construction：

- 仅 transient（网络/服务异常、非 quota 的 429）允许切换配置的下一模型。
- auth、quota、overflow、invalid_request、unknown、结构化输出校验错误不自动切换。
- cancellation、LangGraph 控制异常原样传播；工具异常不进入模型 fallback。
- 只重试当前模型请求，保留已有 messages/ToolMessage，不重跑整个 child 或已经执行的工具。
- child 首版关闭 SDK 和 Forge 的额外 transient retry：一个逻辑请求最多 primary + 2 个
  fallback 共 3 次尝试，原有最多 8 次逻辑请求仍有效；最坏 24 次模型尝试，usage 记录实际模型。
- 记录尝试次数和切换原因；原始 provider 错误、凭据和请求内容不进入 Fleet。

跨 provider 输入必须经现有标准消息编码，移除不兼容的 provider metadata、signed reasoning
和专用 model settings；不通过私有 LangChain helper 绕过三包边界。输出工具调用仍由
LangChain 绑定和验证。不同模型窗口先做容量检查；太小的候选明确失败，不静默丢上下文。

### 5.2 fresh / fork / summary

AGENT.md 增加 context=fresh/fork/summary，默认 fresh；task 可选 context 参数覆盖本次。
父历史从 ToolRuntime.state 获取，不暴露 history 参数让模型伪造真实父上下文。

- fresh：保持只有独立 instruction。
- fork：自动复制当前 active branch 的可携带模型历史，追加 instruction。它是输入快照，
  不等于建立 child session，也不自动继承 parent tools、system prompt 或 UI 状态。
- summary：复用现有摘要调用与 usage accounting 生成任务相关简报，再追加 instruction；
  摘要失败明确返回失败，不能装作拿到了同等上下文。不读全磁盘历史或 abandoned branch。

fork 的截点在当前触发委派的 AI 批次之前；整批正在执行的 calls 都排除，防止继承半配对
的 transcript。仅使用已配对 messages；保留完整 AI/ToolMessage 组，不能逐条截断导致
孤立 ToolMessage。移除 artifact、thinking、provider metadata 和 CustomEntry；保留可移植
的用户文本、assistant 文本以及有效工具调用/结果，文本工具结果仍视为不可信输入。
文本 instruction 与 profile prompt 不能从历史获得新的工具权限。

容量按 primary 和 fallback 中最小窗口预留输出预算。可裁剪时只删除最旧完整消息组，
记录 context_truncated；最近用户任务和新 instruction 不能删，仍超限则失败并建议 summary。
本阶段不自动切换 fresh/fork/summary，也不自动改变模型链。Fleet 显示实际 context 模式、
继承消息数与截断标记。

验收：两个角色使用不同 fake 模型；fresh 不见父事实，fork 能复述父事实且所有 calls 配对；
429 切换但 quota/auth 不切换；fallback 不重复文件写入；启动后 parent 的新消息不改变快照。

## 6. R3：结构化结果与可保存 Codemode 工作流

### 6.1 结果协议

AGENT.md 增加 result-format=text/task-report，默认 text。只提供一个固定 Pydantic
TaskReport：outcome=done/blocked，summary、findings、evidence、question。
限制列表和字段大小，blocked 必须说明问题，done 不代表验收通过，空 findings 不代表正确。

通过官方 ToolStrategy 在同一个 create_agent 中生成 structured_response；格式纠错占用
原模型次数，不增加外层模型循环。必须读取 structured_response，缺失不能用最后一条散文
冒充结构化成功；虚拟输出工具不计入文件工具次数，混合同批工具需要实际集成验证。

task-report 以有效 JSON 作为最终交接文本；返回 artifact 可增加经过校验的 report 投影供
Codemode 使用，整包仍受角色结果预算约束，旧 v2 字段继续可读。JSON 不作字节截断，超限
明确失败。blocked 的执行状态可 completed，业务 outcome 单独表达，不能把两种状态混用。

### 6.2 工作流语言与执行方式

采用 Pi 的脚本式组合思路，复用 Forge 已有 QuickJS/Codemode，不另写 chain/parallel JSON
解释器、不加 Node。提供文档化的工具组合和三个完整示例，而不是先造一套插件框架：

1. scout → worker → reviewer 顺序交接。
2. Promise.allSettled 并行两名 reviewer，合并结构化发现。
3. 固定最多 3 轮的 review/fix；根据 report 判定下一步，失败或 blocked 停止。

当前 task 在 tool_execution.py 中被硬性列为 model-only，Codemode 的 catalog 和 nested
executor 都拒绝它。因此不能把 tools.task 写成现成可调用能力，也不能解除 task 的 model-only
限制来接通工作流。扩展已有 worker IPC，增加 workflow scope 专用的 runs.run 和 runs.control
host API，Python 侧调用同一个任务协调器；普通 Codemode 默认没有这个授权 scope。
child 工具集仍不含 task/workflow/Codemode 控制接口，继续禁止递归委派。

runs.run 接受稳定 step key、agent、instruction 与已定义的模式参数，返回统一 run receipt
或结果；runs.control 只允许管理当前 workflow 拥有的 run。普通 await 表达依赖，Promise
组合表达并发，if/for 表达条件与有界重试。返回值是有界运行报告，禁止靠解析自然语言 verdict。
task 的失败 artifact 必须让脚本区分“执行失败”和“合法 blocked”，不能仅凭 ToolMessage
传输 success 状态认定 child 成功。

增加项目 `.forge/workflows/<name>.js` 与 parent model-only 的 workflow(name, args) 工具。
资源通过现有安全读取、工作区边界和原子 reload 加载；workflow 工具在正常父工具调用的
runtime scope 内运行已有 Codemode host，不另开 provider loop。/workflows 只列出已加载
工作流，用户以自然语言选择执行，避免新增脱离 LangChain ToolRuntime 的斜杠执行循环。
args 是 JSON 数据，注入只读 workflowArgs，禁止直接拼接成 JavaScript 源码。
脚本权限仍是当前会话授权工具；不能增加文件系统、fetch、require 或任意进程接口。

Codemode 现有总时限 300 秒不够跑多轮 agent。工作流用同一 worker host，但提供独立
有界运行配置：默认 15 分钟，最多 30 分钟，活 child 上限复用 4，单 workflow 累计 launch
上限 16，report 输出上限 64 KiB。提交计数由 Python 协调器执行，JS 忘写预算不能绕过。

工作流创建 workflow_id；子任务 run 记录携带它和稳定 step key，Fleet 显示分组与步骤。
已成功步骤的结果保留；某一步失败，不重跑有副作用的前一步。取消/超时取消并等待该 workflow
拥有的活 child，保留已完成步骤的 partial report；不会把 Promise.race 的剩余任务遗弃。
工作流内后台提交需要显式 wait 到 settled；脚本退出时仍活的 workflow-owned child 取消。

验收：真实 Codemode worker + fake 模型跑完顺序、并行、分支、review loop；超过预算可阻止
继续 launch；取消无残留 worker/child；父 task 和 nested results 不串线。

## 7. 后续路线如何继续扩大

### R4：steer、follow-up 与连续 child

学习 Pi 区分消息接受、消费和模型遵从。task_control 增加 send，消息带唯一 request_id，
队列长度和单条大小受限；返回 queued/consumed/expired，不承诺模型一定遵从。
首版在安全模型轮次边界注入 HumanMessage，不在 bash/write 执行到一半时重跑工具。
取消仍是立即取消整个指定 child；steer 不隐式取消再重启。

已完成 child 的 follow-up 可以重建 create_agent，传入明确保留的 messages + 新指令；
不必用 checkpoint 存对话。若需要保留可恢复完整 child history，必须另定义它与现有有界
展示 trace 的区别、持久化预算和 parent JSONL 所有权，不能把 UI trace 当模型历史使用。
这会扩大 V1/V2 的“一次性 child”目标，需要版本化更新文档，但不必改变唯一模型循环。

### R5：真正 detached background

学习 Pi 的独立 runner、精确 run id、控制收件箱和 completion receipt。Forge 保持 Python，
独立 worker 可用当前解释器运行 create_agent；主进程退出后任务仍运行，重启可查询和取消。
这里的重点是进程所有权和事件恢复，不是给 asyncio.create_task 换个名字。

开始实施前必须具体决定：worker 如何成为运行日志的唯一写入者，parent 如何幂等导入完成
receipt，Windows/Linux 的启动/停止/孤儿进程回收，MCP 是否独立启动，provider 配置如何
正常解析，以及消息、日志的保留与清理。不能让两个进程同时追加同一 parent active leaf。
不通过复制密钥或序列化 provider 对象实现跨进程。已有安全规则仍适用。

若方案需要 child checkpointer、SQLite 或长期恢复数据库，它与 AGENTS.md 对 checkpointer
仅用于待回答问卷的规定冲突；必须单独作明确架构决策，不能用“LangChain 推荐”静默覆盖。
本路线不预先承诺用 checkpoint，R1–R3 不需要这项例外。

### R6：worktree 与并发 writer

每个 writer 在独立 worktree 完成文件改动，reviewer 检查明确 Git baseline 和 diff，parent
集中审阅/合并。必须覆盖用户 WIP、同文件冲突、取消后保留产物、清理前保护未合并变化。
读取队列和路径 allowlist 不能替代 Git 工作区隔离。worktree 的创建、合并与清理由 Forge
明确持有，不能交给模型随意递归删除。

R4–R6 不是“永远不做”，而是在前一阶段产物可用后逐个开展完整设计与交付。

## 8. 文件职责和工作量控制

| 范围 | 当前落点 / 建议新模块 |
| --- | --- |
| 泛用 child 执行、events、context 输入、response schema | src/forge_agent/subagents.py、langchain_runtime.py |
| session 任务所有权与生命周期 | src/forge_coding/sessions/session.py；新增 features/subagent_runs.py |
| profiles、task 和 control 工具 | resources/subagent_profiles.py、features/subagents.py |
| model 解析和客户端所有权 | providers/runtime.py、sessions/model_selection.py |
| Fleet 与 inspector | forge_cli/tui 现有 adapter/state/app/screens；新增 fleet.py |
| workflow 加载、原生工具与列表命令 | resources/prompt_templates.py 的读取模式、features/workflows.py（新增）、commands/default_registry.py |
| script 执行、超时、任务归属 | codemode/runtime.py、worker.py；复用，不新增脚本引擎 |
| 使用说明与架构 | README.md、docs/architecture.md |

R1–R3 预计超过 8 个文件、可能涉及 15–25 个实现/验证/文档文件，这是三个真实功能穿过
session/runtime/UI 边界的成本；不以文件数量为完成目标。只新增有明确职责的少量模块，
不建通用 EventBus、Scheduler interface 或 repository/service 抽象层。

每个里程碑独立可用：R1 不依赖不同模型或 workflow；R2 不依赖 R3 的报告；R3 调用已有
协调器和 Codemode。不做需要所有阶段落地才能跑通的“大重构后再接线”。

## 9. 验收、风险和回退

先写失败契约，再修改代码，优先在现有 pytest session/CLI/mounted TUI 验证面运行真实
create_agent 和真实 Codemode worker，模型使用离线 fake；不在写完代码后补一套 DTO 单测。
开发过程中不跑全套端到端，阶段完成再执行项目门禁。

R1 失败面：queued/accept race、completion 在 receipt 前返回、close/switch/branch 晚到事件、
单项取消、队列满、超时、跨会话 run id、持久化失败、provider/MCP 过早关闭、主 TUI 泄漏。
R2 失败面：auth/quota 误 fallback、真实工具重复执行、模型窗口过小、partial batch 继承、
thinking/provider metadata 混入快照、usage 漏算、模型热切换造成运行契约改变。
R3 失败面：缺少 structured_response、非法 JSON、超预算、无限 launch、Promise 遗弃、
JS 参数注入、symlink 路径逃逸、取消后 worker/child 残留、步骤失败却报告整体成功。

每阶段端到端生成临时工作区中的 summary.json、fake 父 JSONL 和 Fleet 截图/文本快照：
记录 case、pass/fail、run_id/step key、配对数量、终态、残留进程数和相对工件路径。
材料不包含真实用户 session、配置或凭据；不提交生成工件。真实 provider smoke 另外授权。

阶段结束执行：uv sync --dev --extra providers --locked；uv run pytest；uv run ruff check .；
uv run ruff format --check .；uv run mypy；uv run forge --help；uv run forge --version。
Fleet 验收覆盖 80x24、宽屏、各主题、导航、取消、复制和会话切换。

规模增加首先遇到 provider 限流、共享工作区和 JSONL 写入频度。活任务/队列/累计 launch
有硬上限，progress 只保留最后快照并合并更新，不逐 token 持久化；原始完整 child transcript
不因此成为默认日志。错误分类、权限和消息配对不能为缩小代码量而删掉。

回退：关闭后台 mode/workflow 入口后 foreground 仍可用；旧版本忽略新的 CustomEntry；
清除新 profile 字段即可恢复旧 loader；回退前先取消并等待后台任务结束。R1–R3 不需 JSONL
迁移。R5 的跨进程持久化回退必须在它自己的方案中另行定义。

## 10. 研究依据与本轮验证

- [Pi observability / Fleet / background](https://github.com/nicobailon/pi-subagents/blob/d6726aeaa3cc849b91140c6d87fc76774f9585a7/docs/observability.md)
- [Pi workflows](https://github.com/nicobailon/pi-subagents/blob/d6726aeaa3cc849b91140c6d87fc76774f9585a7/docs/workflows.md)
- [Pi tool reference / steer / resume](https://github.com/nicobailon/pi-subagents/blob/d6726aeaa3cc849b91140c6d87fc76774f9585a7/docs/tool-reference.md)
- [Pi models：当前不执行模型 fallback](https://github.com/nicobailon/pi-subagents/blob/d6726aeaa3cc849b91140c6d87fc76774f9585a7/docs/models.md)
- [Pi fork-context 源码](https://github.com/nicobailon/pi-subagents/blob/d6726aeaa3cc849b91140c6d87fc76774f9585a7/src/shared/fork-context.ts)
- [Pi scripted-workflow 源码](https://github.com/nicobailon/pi-subagents/blob/d6726aeaa3cc849b91140c6d87fc76774f9585a7/src/workflows/scripted-workflow.ts)
- [LangChain subagents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents)
- [LangChain structured output](https://docs.langchain.com/oss/python/langchain/structured-output)
- [LangChain middleware：fallback / limits / summarization](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
- [LangChain custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)

本轮核对了当前 runner/session/Codemode/provider 源码、安装版 ModelFallbackMiddleware 和
官方文档；只更新本计划，未修改运行代码、未执行代码测试、未运行真实 provider、未提交或推送。
实施时必须重新核对源码与安装版本，并将 R1 的生命周期协议展开成首个实施切片。

# Forge Subagent V2 集成计划

状态：待批准

本计划建立在已完成的 [第一版计划](subagent-integration-plan.md) 上。V1 已经证明了最小的
supervisor-as-tool 路径：父代理通过 `task(agent, instruction)` 调用 fresh child
`create_agent()`，child 不继承父 transcript、不持久化独立 session、不能再次委派，执行结果以
有界 artifact 回到父会话，TUI 在主对话流中显示紧凑任务块。

V2 不改变这条主干。它增加三个互相支撑的小切片：

1. 用户可以用项目或用户级 Markdown 文件定义自己的 subagent；
2. Forge 保存经过裁剪的 child 展示记录，TUI 可在原任务块内查看；
3. Forge 从原生 `AIMessage.usage_metadata` 汇总可选 token 指标，让用户看见委派成本。

目标不是把 Forge 变成生产级多代理平台，而是继续用一份短、可读、可拆卸的实现展示
LangChain 的 agent-as-tool、context engineering、动态工具描述、middleware、原生消息、
usage metadata、artifact、v3 nested events 和 runtime context。

## 1. 结论

推荐实现“声明式角色 + 展示级 trace”，继续保持同步、串行、fresh context。

不推荐在 V2 引入 background jobs、parallel fan-out、recursive delegation、child
checkpointer、worktree、workflow DSL 或跨进程 intercom。这些能力不是自定义角色和可检查历史的
必要条件，而且会要求新的任务生命周期、并发所有权、恢复协议和持久化事实源。

这份方案最脆弱的前提是：当前锁定的 LangChain v3 流在 child lifecycle 结束前至少提供一份
nested `values` 消息快照。如果这个前提不成立，Forge 仍必须保留父 `task` start/end 和最终结果；
本地 child trace 降级为 unavailable，不能改用 root projector 或污染父 transcript。

## 2. V1 基线与 V2 要解决的问题

V1 已有以下正确边界：

- 父、子都使用 `langchain.agents.create_agent`，没有第二个 provider/tool loop；
- child 输入只有当前 instruction，默认无 checkpointer；
- `SubagentRunner` 统一处理注册、串行锁、模型调用上限、失败、取消和 UTF-8 预算；
- `CodingSession` 延迟读取当前 provider/model/runtime context，`/reload` 刷新 role specs；
- nested child messages、thinking 和 tool ids 不进入父 transcript；
- JSONL 只保存父 AI 的 task call 和配对 ToolMessage；
- TUI 只展示 queued/running/completed/failed/cancelled 和最终结果。

V2 解决两个真实缺口：

- 角色固定在 Python 代码里。用户不能为具体项目定义 `oracle`、`test-writer`、
  `migration-auditor` 等小型专家；
- child 的工作过程在结束后消失。用户只能看到最终结果，无法在 TUI 中检查 child 说过什么、
  调了哪些工具、为何得出结论。

第二个缺口不是要求“恢复 child 继续聊天”。V2 保存的是只读、展示级、严格有界的执行记录，
不是新的模型上下文、child session 或 LangGraph checkpoint。

## 3. 参考方案与吸收范围

### 3.1 LangChain

- [LangChain Subagents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents)：
  继续使用单一 dispatcher tool、fresh child、父代理集中控制和上下文隔离。角色名称和描述是路由
  的主要提示；小型 registry 直接枚举到 tool description/system prompt，比新增 `list_agents`
  工具更简单。
- [Context engineering](https://docs.langchain.com/oss/python/langchain/context-engineering)：
  明确区分进入 child model 的 context、工具可见 context 和只供 UI/持久化的 lifecycle context。
  V2 trace 永远不回灌父模型。
- [Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)：
  继续让 `ModelCallLimitMiddleware` 负责 child 模型预算；V2 不为记录 trace 再造一个 agent loop。
- [Tools and ToolRuntime](https://docs.langchain.com/oss/python/langchain/tools)：
  继续通过 runtime context 传 workspace/session 能力，不把 UI 或存储对象塞入 tool args。
- [Subgraph persistence](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)：
  per-thread child memory、interrupt resume 和 nested state inspection 都需要 checkpointer 语义；V2
  不需要，因此不引入。

### 3.2 pi-subagents

- [Agents](https://github.com/nicobailon/pi-subagents/blob/c386b258f33cbb217dc9b4a96421b809be88988d/docs/agents.md)：
  吸收“角色是可读文件”“description 驱动路由”“工具是严格 allowlist”“项目定义覆盖用户定义”这
  四点；不复制 aliases、package discovery、eject/reset、skills inheritance、memory、refinement、
  model fallback 等完整 DSL。
- [Observability](https://github.com/nicobailon/pi-subagents/blob/c386b258f33cbb217dc9b4a96421b809be88988d/docs/observability.md)：
  吸收机器可读、版本化、大小受限的执行记录和 TUI progressive disclosure；不复制 FleetView、
  async artifacts、completion replay 或后台 watcher。
- [Workflows](https://github.com/nicobailon/pi-subagents/blob/c386b258f33cbb217dc9b4a96421b809be88988d/docs/workflows.md)：
  保留 fresh reviewer、单写者和 child 默认不能递归委派；不复制 workflowScript、parallel chain、
  worktree、mission 或 intercom。

pi-subagents 是能力上限参考，不是实现体量参考。其当前 background、workflow 和 agent discovery
已经分别形成大型子系统；Forge V2 只取其中能直接落在现有三层边界上的机制。

## 4. 目标与非目标

### 4.1 目标

- 用户可在项目或用户目录新增一个 Markdown 文件定义同步 subagent；
- 自定义角色与内置 `scout`、`worker`、`reviewer` 进入同一个 `SubagentRunner` registry；
- 项目配置覆盖用户配置，同名自定义角色可以有意覆盖内置角色，并产生可见 diagnostic；
- 自定义角色只能从当前 `CodingSessionConfig.tools` 中选择工具，永远不能获得 `task`；
- `/reload` 原子刷新未来调用使用的角色、task description 和父/子 system prompt；
- nested child 最终消息快照被投影成有界的展示记录，不成为父 LangChain messages；
- 展示记录跟随父 session 的 active branch 写入现有 JSONL `CustomEntry`；
- TUI 的现有 task block 在 `Ctrl+O` 展开时显示最终结果和 child trace；
- 若 provider 提供 usage metadata，task block 显示 input/output/total tokens；缺失时正常降级；
- V1 JSONL、v1 artifact、损坏或未知 trace 都可加载。

### 4.2 非目标

- 不支持后台、detached、steer、follow-up、poll、resume 或 task fleet；
- 不支持 parallel child 或多个 writer；session 级 lock 继续串行所有 task；
- 不支持递归委派；自定义 `tools: task` 必须报错；
- 不支持 child checkpointer、持久对话、fork parent context 或 time travel；
- 不支持每角色 provider/model/fallback/thinking 配置；所有角色继续使用调用时 session runtime；
- 不支持 Python/JS 插件式 agent 定义、package discovery 或远程 registry；
- 不支持任意 output schema、host acceptance gate、workflow chain 或自动 review loop；
- 不保存 system prompt、thinking、provider response、raw metadata、tool artifact 或完整 tool output；
- 不新增 TUI 侧栏、FleetView、任务中心、modal 或新的焦点导航模式。

## 5. 总体架构

```text
user/project AGENT.md
        │
        ▼
forge_coding profile loader ──merge──> builtins
        │                                │
        └──────── compile ───────────────┘
                         │
                         ▼
                   SubagentSpec[]
                         │
parent create_agent ── task(agent, instruction)
                         │
                         ▼
               SubagentRunner + one lock
                         │
                         ▼
                  child create_agent
                         │
               nested v3 values/lifecycle
                         │
                         ▼
         ToolExecutionUpdateEvent(kind=subagent_trace)
                  │                       │
                  ▼                       ▼
       CodingSession CustomEntry      TuiState live block
                  │                       │
                  └──── session load ─────┘
```

依赖方向不变：

```text
forge_cli  →  forge_coding  →  forge_agent  →  LangChain
```

`forge_agent` 只认识通用 spec、native messages、trace projection 和 product events；
`forge_coding` 负责文件发现、工具能力、prompt 组装和 session 持久化；`forge_cli` 只渲染。

## 6. Phase 1：声明式自定义 subagent

Phase 1 可独立合并。完成后用户能定义角色并由父模型调用，即使 Phase 2 永不实现也仍然有用。

### 6.1 资源布局与优先级

每个角色是一个目录，目录名是 canonical agent name，入口固定为 `AGENT.md`：

```text
~/.forge/agents/<name>/AGENT.md
<project>/.forge/agents/<name>/AGENT.md
```

优先级从低到高：

1. Forge 内置角色；
2. `~/.forge/agents`；
3. `<project>/.forge/agents`。

高优先级同名角色完整替换低优先级角色，并产生 `ResourceDiagnostic(kind="subagent")`。不做字段级
merge，避免 prompt、tools 和预算来自不同来源后难以解释。

只有通过完整校验的高优先级角色才参与替换。无效 project profile 被跳过并报告 diagnostic，
低优先级 user/builtin profile 继续生效；配置错误不能通过“遮住”一个已知安全角色改变 registry。

只扫描一层 `<name>/AGENT.md`，不递归，不读取 bare `*.md`，不发现 Python 文件、packages 或网络
资源。这里的 `<project>` 固定是 `CodingSession.cwd`，与现有 project-local skills 的定位一致，不向上
搜索 marker root。user/project 两个 scope 都分别执行 lexical/resolved agents-root 边界；目录或最终
AGENT.md 是 symlink、解析后逃出对应 agents root 时直接拒绝，不能借 user symlink 读取 `.env`、
credentials 或任意 home 文件。

### 6.2 最小文件格式

沿用 `parse_markdown_resource()` 的 dependency-free frontmatter 语法，但新增只供 agent profile 使用
的 strict parser。它在构造 dict 前检测重复 key；现有 skills/prompts parser 的宽松兼容行为不改变：

```markdown
---
description: Challenge an implementation plan and expose hidden assumptions
tools: read, bash
max-model-calls: 6
max-result-bytes: 32768
---

You are an architecture oracle. Do not edit files. Read the relevant code and
return the strongest counterargument, the evidence that would change your
mind, and one recommended decision.
```

规则固定如下：

| 字段 | 必填 | 规则 |
| --- | --- | --- |
| `description` | 是 | 单行非空，最大 300 UTF-8 bytes；直接用于父模型路由 |
| `tools` | 否 | 逗号分隔 allowlist；省略或空值表示无工具 |
| `max-model-calls` | 否 | `1..8`，默认 8；只能收紧 V1 上限 |
| `max-result-bytes` | 否 | `1024..51200`，默认 51200；只能收紧 V1 上限 |

目录名必须匹配 `^[a-z][a-z0-9_-]{0,31}$`。Markdown body 是非空 role contract，追加到 Forge
正常 coding system prompt 后；V2 不提供 `replace` 模式，因此项目说明、工具说明和安全约束不会被
自定义 prompt 静默移除。

整个 AGENT.md 最大 20 KiB UTF-8，body 最大 16 KiB UTF-8。读取时先用文件 metadata/有界 byte
read 拒绝超限文件，再做 UTF-8 decode；不能先把任意大文件完整读入内存。

未知 frontmatter 字段、无效数字、重复工具、未知工具、`task`、空 body、越界路径或同目录重复名
都使该角色被跳过，并产生 error diagnostic。Forge 不静默删掉无效工具后继续运行。

profile strict parser 的允许 key 固定为上述四个；duplicate key 在原始 frontmatter 扫描阶段报错，
因此不会被 dict 的 last-write-wins 吞掉。

`tools` 是能力 allowlist，不是安全声明。`bash` 仍然可以读写工作区外路径或发起网络请求；
`AGENT.md` 的“只读”文字也不是 sandbox。README 和 `/agents` 输出必须保留这一事实。

### 6.3 内部模型

在 `forge_coding` 增加不可变 `CodingSubagentProfile`：

- `name`
- `description`
- `prompt`
- `tool_names`
- `max_model_calls`
- `max_result_bytes`
- `source`

`source` 只存 `builtin`、`user` 或 `project` 这三个 safe token，不存 `Path`。loader 使用的真实
Path 是局部实现细节，不进入 profile、prompt、artifact、event 或 provider 请求。

内置三角色也先表示为 profile，再由一个编译函数统一转成 `forge_agent.SubagentSpec`。这样工具
能力交集、base prompt 拼装、预算和 diagnostic 只实现一次。

编译规则：

1. 用当前 session tools 建立 `name -> BaseTool` 映射；
2. 对 profile 的 allowlist 做严格解析，拒绝 `task` 和未知名字；
3. 使用已有 `build_system_prompt()` 构造 base prompt；
4. 追加 `<subagent_role source="...">...</subagent_role>`；
5. 创建 `SubagentSpec`，不改变 `forge_agent` 的包边界。

`CodingSessionConfig.tools` 继续是绝对能力上限。自定义角色不能通过名称重新创建调用方没有提供的
tool，也不能绕过 MCP deny/availability、workspace context 或 shell prefix。

### 6.4 动态 task registry

固定 `Literal["scout", "worker", "reviewer"]` 改为严格非空 `str`，实际名称由
`SubagentRunner` 在执行前校验。`task` 的 description 和父 system prompt 每次组装时枚举当前
角色：

```text
Available subagents:
- oracle: Challenge an implementation plan and expose hidden assumptions
- reviewer: Independently review existing code without editing files
- scout: Investigate code and collect evidence without editing files
- worker: Implement one bounded change and verify it
```

Forge 面向的是少量本地专家，不增加 `list_agents` 模型工具。若用户配置很多角色，prompt 会线性
增长；V2 在 `/agents` 中提示保持十个左右，而不是加入 registry search 子系统。

### 6.5 Reload 与调用快照

`/reload` 在 session idle 时完成以下原子替换：

1. 重新加载 profiles 与 diagnostics；
2. 重新编译 specs；
3. `SubagentRunner.replace_specs()`；
4. 重建 task tool description 和父 system prompt；
5. 更新 reload summary 的 `Subagents: added/removed/changed`。

`CodingSession.reload()` 若 `_run_active` 为 true，立即抛出明确的 busy error；不排队、不部分刷新。
reload 是同步方法，先在局部变量中完成加载、校验、spec/tool/system 构造，再在无 `await` 的提交段
替换 runner specs、harness config tools/system 和 session resources。任一步失败都保留旧 registry。
因此 reload 不会与正在运行或等待 runner lock 的 task 交错。直接使用通用 `SubagentRunner` 的调用
仍保持 V1 的 call-time spec snapshot 语义。

新增 `/agents` 只读 slash command，列出 name、description、tool names、source 和 diagnostics。
它不直接启动 child，不进入模型 transcript。自然语言委派仍是唯一 model-facing 使用方式。

source/path 的展示固定为 `builtin`、`~/.forge/agents/...` 或相对 cwd 的
`.forge/agents/...`。新增 subagent resource-path formatter；`/agents`、`/reload` 和
`kind="subagent"` diagnostics 不调用当前会输出绝对路径的 `ResourceDiagnostic.format()` 直出
profile 路径。现有 skill/context diagnostic 的展示不在本计划中顺带重构。

## 7. Phase 2：可回放 child trace 与 TUI 检查

Phase 2 是一个原子里程碑：backend trace、session persistence 和 TUI consumer 一起交付。只做
backend 会留下不可发现的数据；只做 TUI 会让历史恢复失效。

### 7.1 展示记录，不是 raw transcript

增加通用、JSON-safe 的 `SubagentTraceItem`：

| kind | 保存内容 | 明确不保存 |
| --- | --- | --- |
| `human` | instruction 文本 | message metadata |
| `assistant` | 可见文本 | thinking、additional kwargs、provider response |
| `tool_call` | tool name、固定 summary、call 序号 | raw args、path、command、child call id |
| `tool_result` | tool name、`ok/error` 状态 | raw content、artifact、stdout、文件内容 |
| `omitted` | 被裁剪的 item 数量 | 原始被裁剪内容 |

这是一份“child 如何推进任务”的展示记录，不是可重新喂给模型的 LangChain message list。
`forge_agent` 不解析 coding path 或 command：已知与未知工具都只保存 `Calling <tool>`，bash 保存
`Running bash`，tool result 只保存 `ok/error`。trace 不复用当前可能包含参数的 live activity summary，
因此 generic runtime 不需要 filesystem helper/workspace root，也不会跨越包边界。

`SubagentTraceItem` 是 frozen dataclass，字段固定为：`kind`、`text: str | None`、
`tool: str | None`、`status: Literal["ok", "error"] | None`、`omitted: int`。每种 kind 只允许对应
字段，序列化时省略 `None`/0；反序列化拒绝未知 kind、未知字段、负 omitted 和类型不匹配。item 顺序
直接来自最后一份 native message snapshot；AI tool calls 按消息内顺序展开，ToolMessage 用
`tool_call_id` 只在内存中匹配 tool name，id 本身不序列化。重复 snapshot 由 parent task id 的
exactly-once drain 去重，而不是按文本去重。

预算固定为：

- 每个文本 item 最多 8 KiB UTF-8；
- 每个 task 最多 64 items；
- 序列化后的整份 trace 最多 64 KiB UTF-8；
- 超限时保留首条 human、最后一条非空 assistant 和尽可能多的尾部活动；中间用一个
  `omitted` item 记录数量；
- `truncated=true` 同时进入 live event 和 CustomEntry；task artifact 不复制 trace metadata。

不提供 `full` 或 unlimited 模式。需要原始 provider/tool trace 的开发者可显式启用 LangSmith；
Forge 的本地 session 仍只保存安全、可读的小投影。

### 7.2 复用 v3 nested projection

不新增 AgentEvent 类型。扩展 `_NestedTaskProjection`：

1. root task start 仍建立 parent task call id 与 nested namespace 的关联；
2. nested `messages` 继续不进入 root `_ProjectionState`；
3. nested `values` 只缓存该 namespace 最新的 native message snapshot；
4. child lifecycle terminal 时，把 snapshot 投影为一次
   `ToolExecutionUpdateEvent(data.kind="subagent_trace")`；
5. 若 lifecycle terminal 缺失，root task end 在清理 namespace 前最多补发一次缓存 trace；drain
   必须发生在 yield root `ToolExecutionEndEvent` 之前；当 end 来自 root `values` 时，还必须发生在
   `state.messages.append(root_tool_message)` 之前；
6. child tool activity 继续走现有 `data.kind="subagent_activity"`；
7. 未知 namespace 继续直接丢弃，不能回退到 root projector。

事件 data v1：

```json
{
  "kind": "subagent_trace",
  "version": 1,
  "agent": "oracle",
  "items": [
    {"kind": "human", "text": "Review this plan"},
    {"kind": "tool_call", "tool": "read", "text": "Calling read"},
    {"kind": "tool_result", "tool": "read", "status": "ok"},
    {"kind": "assistant", "text": "The load-bearing assumption is ..."}
  ],
  "truncated": false,
  "input_tokens": 4200,
  "output_tokens": 730,
  "total_tokens": 4930
}
```

token 字段允许 `null`。只从 `AIMessage.usage_metadata` 的非负整数求和，不解析 provider-specific
response metadata，不计算价格。fake provider 和不报告 usage 的 provider 正常显示 `tokens n/a`。

trace projector 和 usage aggregation helper 放在 `forge_agent.subagents`。nested runtime 使用 trace
projector；`SubagentRunner` 与 nested runtime 复用 usage helper，避免 token metrics 各写一套解析。

### 7.3 task artifact v2

`subagent_run` artifact 升为 version 2，保留所有 v1 字段并新增：

```json
{
  "kind": "subagent_run",
  "version": 2,
  "agent": "oracle",
  "status": "completed",
  "instruction": "Review this plan",
  "final_output": "...",
  "model_calls": 4,
  "tool_calls": 3,
  "queued_ms": 0,
  "duration_ms": 8200,
  "truncated": false,
  "error": null,
  "input_tokens": 4200,
  "output_tokens": 730,
  "total_tokens": 4930
}
```

artifact 不包含 trace items，避免把 child history 复制进父 ToolMessage、父 model context 和 JSONL
message entry；也不保存 `trace_items`、`trace_truncated` 或 availability，避免 nested event 缺失或
storage failure 后 artifact 与 CustomEntry 自相矛盾。trace 是否可用只由 active CustomEntry 决定。
现有整体 UTF-8 budget 继续覆盖 artifact。TUI 同时接受 version 1 和 2；v1 继续显示原有结果，只有
匹配的 CustomEntry 存在时才显示 trace。

### 7.4 Parent JSONL 持久化

`CodingSession.prompt()` 和 `continue_()` 在 yield `subagent_trace` update 前追加：

```json
{
  "type": "custom",
  "namespace": "forge.subagent_trace",
  "data": {
    "version": 1,
    "tool_call_id": "call_123",
    "agent": "oracle",
    "items": [],
    "truncated": false,
    "input_tokens": 4200,
    "output_tokens": 730,
    "total_tokens": 4930
  }
}
```

持久化规则：

- 每个 parent `tool_call_id` 最多一条 trace，重复 terminal/replay event 去重；
- 收到 trace update 时，session 先确认 harness 内尚无匹配的 root ToolMessage；若已经存在，说明
  runtime 顺序契约被破坏，记录 bounded diagnostic 并丢弃 trace，不能把它写到 result 后；
- 顺序有效时，session 调用 `_persist_messages_since(persisted_count)`，强制把包含该
  `tool_call_id` 的父 AI MessageEntry 写入 JSONL，并更新 `persisted_count`；
- 若持久化后 active messages 中仍找不到该 parent task call，视为 malformed nested ordering：记录
  bounded diagnostic，丢弃 trace update，不写 CustomEntry、不交给 TUI；root task start/end 继续；
- 找到 parent task call 后，CustomEntry 以当前 `_last_parent_id` 为 parent，然后自身成为新的
  lineage 候选，并追加指向它的 leaf；两个 append 都成功后才更新 `_last_parent_id`。若 leaf append
  失败，已写入的 CustomEntry 是不在 active leaf path 上的 orphan，resume 时忽略，后续 synthetic
  或正常 ToolMessage 仍挂在旧 `_last_parent_id` 下，与 orphan 同级；只有两个 append 都成功时，
  后续 task ToolMessage 才成为 trace entry 的 child；同一 AI turn 有多个 task 时按实际 terminal
  event 顺序形成一条线性 lineage，关联仍使用各自 `tool_call_id`；
- 这样 trace 位于 task call 和 task result 之间的 active branch，但 `SessionState.messages` 永远忽略
  CustomEntry；
- branch、resume、adopt、export import 和 compaction 都只看到当前 active path 的 trace；
- failed 保存已缓存的 partial trace；cancelled 只有在 LangChain 先发出 nested terminal/cache 时才
  best-effort 保存 partial trace，runner 仍重抛，synthetic ToolMessage 仍无 subagent artifact；进程或
  stream 在 terminal event 前结束时允许没有 trace；
- 旧 JSONL 无迁移；未知 namespace/version/data shape 忽略；
- session 维护本轮 `persisted_trace_tool_call_ids` 集合并同时检查已加载 CustomEntry，保证 live、
  retry 和 resume 去重；
- CustomEntry 与 leaf 都成功 append 后才 yield trace update 给 TUI。任一 storage append 失败沿用
  现有 run persistence 错误路径，事件不进入 TUI；可能存在的 torn tail 由现有 JSONL repair 处理。

Parent JSONL 仍是唯一产品事实源。V2 没有 child session file、sidecar directory、SQLite、
checkpointer 或第二份 run ledger。

### 7.5 TUI 展示

`SubagentDisplay` 增加：

- `trace_items`
- `trace_truncated`
- `input_tokens`
- `output_tokens`
- `total_tokens`

live adapter 在 `data.kind="subagent_trace"` 时原位附加 trace，不新增 transcript row。
`CodingSession.subagent_traces` 从 active `SessionState.custom_entries` 返回经 v1 校验的
`tool_call_id -> trace` mapping；TUI app 在首次 mount、resume、new/adopt 和 branch switch 后调用
`state.load_messages(session.messages, subagent_traces=session.subagent_traces)`，再按父 messages 恢复
task block。TuiState 不直接读取 storage 或 SessionEntry。

折叠态保持两行：

```text
✓ oracle  Review the V2 plan
  completed · 3 tools · 4.9k tokens · 8.2s · Ctrl+O inspect
```

`Ctrl+O` 展开现有 task block，不增加 modal 或新快捷键：

```text
✓ oracle  Review the V2 plan

Result
The load-bearing assumption is ...

Trace · 4 items · 4.9k tokens
user       Review the V2 plan
tool       read · Calling read
tool       read · ok
assistant  The load-bearing assumption is ...
```

规则：

- final result 始终在 trace 前，保持快速阅读；
- tool call/result 使用 dim semantic style，assistant 使用普通正文；
- `omitted` 显示 `… N earlier trace items omitted`；
- 80x24 正常换行，不水平滚动；
- selection/copy 只包含当前可见内容；
- v1、缺失、损坏 trace 显示原 V1 结果，不报错、不加空面板；
- `Esc` 仍取消整个 parent prompt，`Ctrl+O` 仍是唯一展开键；
- dark/light/high-contrast 复用已有 subagent 语义色，不新增一套 palette。

plain/transcript renderer 继续只显示紧凑 activity 和 root task result；JSON renderer 会自然输出
有界 `subagent_trace` update，供调试使用。

## 8. 展示的 LangChain 特性

| LangChain 特性 | Forge V2 中的用途 |
| --- | --- |
| `create_agent()` | 父、子唯一生产执行图 |
| native `BaseTool` / `StructuredTool` | 动态角色仍通过一个 task dispatcher 暴露 |
| `ToolRuntime` context | child 继续继承当前 workspace/session runtime，而不是把内部对象放入 schema |
| `ModelCallLimitMiddleware` | 每角色可收紧、不可放大的模型调用预算 |
| context engineering | profile 控制 system prompt 与 tools；父 history 不自动进入 child |
| native `HumanMessage` / `AIMessage` / `ToolMessage` | 构建 trace 的事实输入 |
| `AIMessage.usage_metadata` | provider-neutral、可选的 token observability |
| `(content, artifact)` tool result | 父模型只得到 compact result，UI 得到稳定 metadata |
| `astream_events(version="v3")` | nested activity 与 trace 关联 parent task id |
| LangSmith tracing | 需要 raw trace 时的 opt-in 外部观测路径，不成为默认依赖 |

这个组合的教学重点是：同一份 child execution 可以同时产生三种不同 context——给 child model 的
执行 context、给 parent model 的 compact result、给用户的 bounded trace——三者不能混在一起。

## 9. 文件级改动

预计触及 23 个文件。数量超过 8 是因为变更横跨明确的 runtime、coding、CLI 三层和对应测试，
不是因为新增服务或第二套框架。

### Phase 1

| 文件 | 修改 |
| --- | --- |
| `src/forge_coding/subagent_profiles.py` | 新增 profile、AGENT.md loader、优先级、校验和 diagnostics |
| `src/forge_coding/resources.py` | 增加 strict frontmatter parser、safe display path 和 agents dirs |
| `src/forge_coding/paths.py` | 增加 user/project `.forge/agents` 路径 helper |
| `src/forge_coding/subagents.py` | 内置 profile、统一 spec compiler、动态 task description/schema |
| `src/forge_coding/session.py` | 加载 profiles、组装 specs、reload 原子替换和公开 agents 属性 |
| `src/forge_coding/reload.py` | reload summary 增加 subagents category |
| `src/forge_coding/commands.py` | 增加 `/agents`，扩展 `/resources`、`/reload` 输出 |
| `tests/test_subagent_profiles.py` | discovery、precedence、schema、边界、diagnostics |
| `tests/test_coding_session.py` | 动态 registry、能力上限、reload、provider/model snapshot |
| `tests/test_commands.py` | `/agents`、resources 和 reload 输出 |

### Phase 2

| 文件 | 修改 |
| --- | --- |
| `src/forge_agent/subagents.py` | trace DTO/projector、usage aggregation、artifact v2 |
| `src/forge_agent/langchain_runtime.py` | nested values cache、terminal trace update、exactly-once drain |
| `src/forge_agent/__init__.py` | 导出必要的通用 trace 类型 |
| `src/forge_coding/session.py` | trace CustomEntry 持久化、去重、active lineage 和恢复 index |
| `src/forge_cli/tui/state.py` | trace/usage state、live attach、历史恢复和损坏降级 |
| `src/forge_cli/tui/adapter.py` | 消费 `subagent_trace` update |
| `src/forge_cli/tui/app.py` | mount/resume/new/adopt/branch 后传入 session trace index |
| `src/forge_cli/tui/widgets.py` | task block 的 result + trace 展开渲染和 selection text |
| `tests/test_subagents.py` | trace sanitization、预算、usage 和 artifact v1/v2 |
| `tests/test_langchain_runtime.py` | nested snapshot、terminal/drain、namespace 退化和去重 |
| `tests/test_tui_adapter.py` / `tests/test_tui_app.py` | live trace、Ctrl+O、copy、窄屏、历史恢复 |

两个 Phase 都更新 `README.md` 和 `docs/architecture.md`。若实现发现某个文件无需修改，应删除该
改动项，而不是制造空改动。

## 10. 实现顺序

### Phase 1：Custom profiles

1. 为 profile loader 写 red tests：格式、用户/项目目录、覆盖、symlink、未知字段和工具能力上限；
2. 实现 resource paths 和 `CodingSubagentProfile`；
3. 把三内置角色改成 profiles，通过统一 compiler 生成现有 specs；
4. 将 task agent schema 改成动态 string validation，并生成当前 registry description；
5. 接入 CodingSession load/reload/adopt/new/resume；
6. 增加 `/agents`、reload/resources 输出和文档；
7. 跑全量验证，Phase 1 可独立提交。

### Phase 2：Trace and inspect

1. 为安全 trace projector 写 red tests，先锁定不保存 thinking/raw args/tool output/artifact；
2. 在 `forge_agent.subagents` 实现 projector、预算算法和 usage aggregation；
3. 扩展 nested projection，验证 terminal 与 root drain exactly once；
4. 在 CodingSession 追加 active-branch CustomEntry，覆盖取消、失败、branch、resume、compaction；
5. 更新 artifact v2，并保持 v1 loader；
6. 接入 TuiState/adapter/widget，复用 Ctrl+O 和现有样式；
7. 更新 README/architecture，跑全量验证和人工 TUI 验收；
8. Phase 2 独立提交。

## 11. 测试矩阵

### 11.1 Custom profiles

- 无 custom 文件时 registry 与 V1 三角色完全一致；
- project `.forge/agents` 按名字覆盖 user `.forge/agents` 和内置角色；
- 同名有效覆盖产生 source 明确的 diagnostic；无效高优先 profile 保留低层角色；
- 缺 description、空 body、非法 name、未知字段和非法整数被跳过；
- 超过 20 KiB 的文件和 16 KiB 的 body 在完整读取前被拒绝；
- `tools` 省略/空值产生 tool-less child；
- tools 顺序稳定，重复、未知工具和 `task` 被拒绝；
- custom profile 不能突破 `CodingSessionConfig.tools` 上限；
- user/project agent 目录、AGENT.md symlink 和 resolved-root escape 都被拒绝；
- `/reload` 添加、修改、删除角色并刷新 task description；忙态直接拒绝且不部分更新；
- reload 构造失败保留旧 profiles/specs/task tool/system prompt；
- `/model`、provider switch 后 custom child 仍读取最新 runtime；
- `enable_subagents=False` 时不发现/编译角色，不保留 task 名限制；
- `/agents`、`/reload` 的 subagent rows 和 subagent diagnostics 不输出完整 prompt、绝对 home path
  或敏感 runtime 数据。

### 11.2 Trace backend

- nested values 只生成 `subagent_trace` update，不产生 root Message events；
- trace 包含 human/assistant/tool timeline，不包含 thinking、raw tool args/results/artifacts；
- 所有 tool summary 都不包含 raw args、command 或 path；
- UTF-8 8 KiB item、64 items、64 KiB aggregate 三层预算都准确；
- 超限保留 instruction、final assistant 和 omitted count；
- lifecycle terminal 和 root task end 只 flush 一次；
- normal、failed、model limit、empty final 都生成确定 trace/artifact；
- cancelled 继续重抛 `CancelledError`、由 harness 补 synthetic ToolMessage，无 `subagent_run`
  artifact；nested projection 只在收到 terminal/cache 时 best-effort 保存 partial trace；
- provider usage metadata 正确求和；缺失、部分、负数、畸形字段被忽略；
- artifact v2 整体仍受现有 max_result_bytes 预算；
- v1 artifact 和未知 v3 artifact 保持当前 fallback；
- 未知/malformed nested namespace 被丢弃，绝不污染父 transcript。

### 11.3 Session 与 TUI

- CustomEntry 位于 active task call/result 路径，`SessionState.messages` 不包含它；
- duplicate trace update 不重复写 JSONL；
- branch 到 task 前不显示 abandoned trace，切回分支后恢复；
- resume/adopt/new session 不串 trace；
- TUI mount/resume/new/adopt/branch switch 都从 `session.subagent_traces` 重建，不从 messages 猜；
- compaction 不把 trace 内容加入 model summary；
- torn tail、坏 custom data、未知 version 不阻止 session 加载；
- live task block 原位附加 trace，不新增 chat row；
- Ctrl+O 展开顺序是 result 后 trace，再次切换恢复折叠；
- selection/copy 与视觉内容一致；
- v1 历史继续显示原结果，不出现空 trace header；
- 80x24、120x30、200x60 和三套主题可读；
- Esc 取消后 composer 恢复；LangChain 已发 terminal snapshot 时 partial trace 可见，否则明确允许
  trace unavailable，synthetic ToolMessage 仍保持合法配对。

## 12. 验证命令

每个 Phase 完成后执行：

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

人工 TUI 验收：

1. 在临时项目创建 `.forge/agents/oracle/AGENT.md`；
2. `/agents` 确认 source/tools/description；
3. 自然语言要求父代理调用 oracle；
4. 运行时观察 compact activity；
5. 完成后 Ctrl+O 检查 result + trace；
6. 关闭并 resume session，确认 trace 恢复；
7. 修改 AGENT.md 后 `/reload`，确认下一次调用使用新 prompt；
8. 在 80x24 和 high-contrast 下重复展开、copy 和 Esc。

真实 provider smoke 仍是 opt-in。若用户允许，只验证一个 tool-less custom oracle 和一个带 `read`
的 custom reviewer；不读取、打印或记录凭据。

## 13. 验收标准

- 用户只需新增一个小型 AGENT.md，就能得到新的可调用 role；
- 内置和自定义角色走同一个 profile compiler、SubagentRunner、task tool 和 TUI block；
- 自定义角色不能扩大 session 工具、模型调用或结果大小上限；
- child 仍是 fresh、同步、串行、无 checkpointer、无递归；
- 父 transcript 仍只有 task call 和配对 ToolMessage；
- child trace 在 live TUI 和 session resume 后都可检查，但不会进入父 model context；
- trace 不保存 thinking、raw provider metadata、raw tool args/result 或 artifact；
- usage metadata 有则显示、无则正常降级；
- V1 session/artifact 无迁移可读取；
- 三包依赖方向、create_agent-only、JSONL 事实源和现有取消修复保持不变；
- 全量 lint、type check、tests、CLI 和 build 通过。

## 14. 风险、攻击面与回退

| 风险 | 处理 |
| --- | --- |
| LangChain nested values 形状变化 | trace unavailable；保留 root task start/end；未知 nested event 永远丢弃 |
| 自定义 prompt 诱导越权 | tools 与 session capability 取严格交集；task 禁止；说明 bash 不是 sandbox |
| 自定义角色过多使 tool prompt 变长 | V2 面向小 registry；`/agents` 提示保持约十个，不增加搜索工具 |
| trace 泄漏 tool 内容或凭据 | 只保存展示投影；禁 raw args/results/artifact；固定三层 UTF-8 预算 |
| trace CustomEntry 破坏 branch lineage | 作为 task call 与 ToolMessage 之间的 active tree entry；加入 branch/resume 回归测试 |
| usage metadata provider 差异 | 只读标准字段，畸形值忽略，不把 token 指标作为成功条件 |
| reload 与运行 task 交错 | `_run_active` 时拒绝；局部构造成功后无 await 原子提交；失败保留旧配置 |
| 计划方向不合适 | Phase 1/2 可独立回退；删除 custom profiles 恢复内置角色；忽略 CustomEntry 恢复 V1 TUI |

没有外部 API、服务、CLI 或凭据依赖。规模增长首先会击中动态 registry prompt 和单个 session JSONL；
这正是 V2 明确不做远程 registry 和 raw transcript 的原因。回退不需要数据迁移：旧 Forge 会把
v2 task artifact 当普通/未知 tool result，并忽略 `CustomEntry`。

## 15. 明确拒绝的相近方案

### 15.1 直接采用 Deep Agents `SubAgentMiddleware`

不采用。它会同时带入 Forge 已有的 planning/filesystem/memory/subagent harness 概念，模糊
`forge_agent`、`forge_coding` 和 JSONL 的唯一职责。Forge 继续用 LangChain 官方 subagent-as-tool
模式手写一份教学实现。

### 15.2 每个 child 一份 JSONL/session/checkpointer

不采用。它能支持 continuation/HITL，但会产生第二套 session lifecycle、索引、清理、branch 和
恢复语义。V2 的只读 trace 用现有 parent `CustomEntry` 足够。

### 15.3 把完整 child transcript 塞入 task artifact

不采用。artifact 会跟随父 ToolMessage 进入持久化和潜在模型上下文，重复 instruction/final output，
并绕过 V1 compact result 的目标。trace 必须是单独的 bounded CustomEntry。

### 15.4 增加 background/parallel/FleetView

不采用。后台需要 job id、watcher、status、steer、cancel、completion delivery 和 crash recovery；并行
writer 还需要 worktree 或所有权协议。它们应在未来以独立架构计划讨论，而不是藏在 V2 中。

### 15.5 增加通用 workflow DSL 或自动 review loop

不采用。推荐的 `clarify → scout → worker → fresh reviewer → worker` 可以继续写进项目 instructions，
由父 `create_agent` 自然编排。把它固化成新 DSL 会形成第二个控制流运行时。

## 16. 后续可单独研究的实验

以下想法有创造性，但不进入 V2 承诺：

- 只有经过工具 mutation metadata 证明为只读的角色才允许并行；
- custom role 的 provider-neutral model tier override；
- child 用 LangChain structured output 返回 typed review report；
- child 遇到决策时返回 `blocked/question` 结果，由父代理向用户提问；
- 只读 replay viewer 导出为 HTML；
- LangSmith trace id 与本地 task call id 的可选关联。

每一项都应先证明不需要新的执行循环或持久化事实源，再单独立计划。

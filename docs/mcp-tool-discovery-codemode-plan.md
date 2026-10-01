# Forge MCP、工具按需加载与 Codemode 实现方案

- 状态：A–E 已实现，Windows 本地验收通过；下文 A/B 配置记录保留为历史阶段记录。没有合并历史分支、提交或发布。
- 日期：2026-09-30。
- 代码基线：`0a0c05c1871c2e06e5993204821533753c021d56`；编写时工作区已有其他未提交文档和 `AGENTS.md` 修改，实施时保留它们。
- 对齐基线：Pi `v0.99.1`；QuickJS 底层评估版本 `quickjs-rs 0.2.5`。
- 范围：统一工具执行边界、MCP 接入、工具按需加载、Forge 自有 Codemode。实现会涉及超过八个源文件及对应验证面，按独立可交付阶段推进。

## 1. 目标与决策

三个设计是一套工具体系：MCP 提供外部能力，按需加载控制模型看到的工具集合，统一执行入口保证直接调用与 Codemode 内部调用使用相同的安全和结果契约。

采用现有 LangChain `create_agent` 和 `astream_events(version="v3")`。工具仍是原生 `BaseTool`，模型消息仍由 LangChain 管理；Codemode 是其中一个工具，不建立第二个模型循环。Forge 自行实现 Codemode 的宿主协议，直接复用 `quickjs-rs`，不依赖 `langchain-quickjs` 或 Deep Agents。

最小可交付版本：保持现有内置工具，添加安全的嵌套调用和无跨调用 JS 状态的 Codemode；MCP、搜索、分支数据逐阶段接入同一目录。完整范围包含 stdio、Streamable HTTP、OAuth、MCP 资源、工具声明恢复和显式 JSON 存储。分类器、自动模型路由、`models.*`、第三方扩展加载、MCP Apps、MCP prompts、SSE/WebSocket 不在本方案交付范围。

### 1.1 与项目规则的关系

实施需要在 `AGENTS.md` 和 `docs/architecture.md` 明确允许“仅用于 Codemode 的内嵌 QuickJS/WASM 执行环境”。这是现有“不得未经明确决策添加第二语言运行时”规则的有界例外；Python 分发包、包依赖方向及 LangChain 生产循环不变。实施已将此有界例外写入 AGENTS.md 与 architecture，并通过 uv 锁定依赖。

JS 内部依赖调用允许由代码排序；现有“依赖工具等待下一模型轮次”仍适用于模型直接产生的工具批次。内部调用不伪造助手消息或绕过批次校验。

### 1.2 历史实现复用

本地 `codex/mcp-integration` 分支及提交 `6c4eb6d895b75b5d67bd05e141c428d120631fc5` 已核实存在。它包含旧 MCP 配置、生命周期与离线服务器验证，但未获得合并批准，路径和框架入口也已变化。

选择性复用配置校验、结果映射和验证场景；不整体 cherry-pick，不复制旧 runtime/cache/gateway 架构。将项目配置授权、跨源凭据转发、在途调用清理、分页、目录刷新、CLI 路由列为必须重新验证的历史问题。旧分支的测试通过记录不证明本方案实现正确。

## 2. 当前事实与依赖路线

实施锁定版本为 LangChain `1.4.3`、LangChain Core `1.6.6`、LangGraph `1.2.12`、FastMCP `4.0.10`、MCP `2.2.0`、quickjs-rs `0.2.5`。Deep Agents 和 langchain-quickjs 未加入依赖。方案编写起点为 LangChain `1.3.14` / Core `1.5.3` / LangGraph `1.2.10`，彼时 langchain.mcp 尚不存在；以下以锁文件和当前源码为准。`ToolSet` 已提供稳定顺序和原生工具目录；`ToolCallBatchMiddleware` 校验最新模型消息的调用 ID；文件工具共用同文件队列；JSONL 已支持 `CustomEntry` 和活动分支恢复。

新 MCP 入口采用官方 `langchain.mcp.MCPAdapter` / `as_langchain_tool`。官方说明该入口需要 `langchain[mcp]>=1.4.0`，仍处于 beta；独立 `langchain-mcp-adapters` 仓库已归档。MCP 工具转换复用官方入口，连接、OAuth、资源操作复用其底层 FastMCP 公开接口，Forge 只管理授权和生命周期。

实施时将基础 LangChain 要求调整为 `langchain[mcp]>=1.4.0,<2.0`；Codemode 加入 `quickjs-rs==0.2.5`，其运行依赖为 Wasmtime。因本方案直接引用 FastMCP API，在 `pyproject.toml` 显式声明 `fastmcp>=4,<5`；由 `uv` 统一求解并生成锁文件，不手工编辑。新增包延迟导入；没有 MCP 配置时不启动连接，没有 Codemode 调用时不启动 worker。

若官方 API 升级影响现有 v3 流、ToolRuntime 或 HITL，必须修复窄范围兼容问题并完成回归后交付，不能以降级消息协议或添加第二个执行循环代替。依赖求解失败则该阶段不交付；保留已完成的独立阶段。

现已用本地 fake chat model、stdio/HTTP MCP 与模拟 OAuth 验证 Forge 集成、子进程 IPC、取消/超时、存储和恢复。真实 provider、第三方 MCP/OAuth 与 Linux 执行未运行，不能据此声称生产服务兼容性或模型质量。

## 3. 所有权与执行结构

```text
forge_cli：CLI /mcp、TUI、导出展示
    |
forge_coding：CodingSession、ToolSet、MCP、Codemode 宿主与 worker
    |                          |
    |                          +-- worker：QuickJS/WASM；仅交换 JSON 请求与输出
    |
forge_agent：原生工具执行校验、嵌套调用上下文、产品事件
    |
LangChain create_agent / ToolNode / BaseTool
```

工具、网络、文件和 provider 都留在父进程。worker 不接收 `ToolRuntime`、Python 对象、凭据或 provider 配置。`forge_agent` 不管理进程、文件、MCP 或 QuickJS，不反向导入 coding 层。

### 3.1 一个目录，三种集合

扩展现有 `ToolDefinition` 的产品元数据，原生对象仍是 schema、description、执行的唯一来源。新增 `exposure`、`namespace`、`annotations`；结构化输出 schema 从原生工具/MCP 元数据读取，不重复维护输入 schema。

| exposure | 模型直接声明 | JS 可调用 | 搜索可加载 |
| --- | --- | --- | --- |
| `direct` | 当前激活时声明 | 当前激活时允许 | 已注册且授权时允许 |
| `model-only` | 当前激活时声明 | 禁止 | 已注册且授权时允许直接声明 |
| `codemode` | 默认不声明 | 注册且授权时允许 | 可以 |
| `deferred` | 加载后声明 | 注册且授权时允许 | 可以 |
| `hidden` | 禁止 | 禁止 | 禁止 |

目录区分 registered、declared、callable。它们是原生工具的产品视图，不是第二份 LangChain runtime。当前内置工具默认 `direct`；`codemode`、`tool_search`、`task`、问卷和 Goal 控制工具为 `model-only`。子智能体能力与目录授权取交集，不能因目录共享扩大权限。

MCP 默认 `deferred`，不自动启用 Codemode；搜索工具在存在可加载工具时可用。用户主动启用 Codemode 后，JS 可调用已授权的 deferred MCP 工具，不要求先把每个 schema 放进模型请求。

名称以 native name 为事实；MCP 名称使用官方 server 前缀。JS 标识符采用 Pi 的规范化规则并检查碰撞；碰撞阻止相关目录发布，不能让后注册工具覆盖前者。命名空间来源与原始服务工具名保存在元数据。

### 3.2 共享执行入口

在 `forge_agent/tool_execution.py` 抽出单次调用的参数/结果校验和有界错误转换。外层 `ToolCallBatchMiddleware` 保持现有模型批次并行/顺序及配对规则；嵌套入口不参与该批次的成员 ID 检查，也不重复获取外层顺序锁。

嵌套入口接收注册的原生工具、父调用上下文和参数，通过原生工具调用路径执行，派生可信 `ToolRuntime`、config、context、子调用 ID；复用统一执行前后策略。将需逐调用生效的策略从“仅模型批次 wrapper”移到共享入口，而不是声称直接 `ainvoke` 自动执行全部中间件。HITL 和状态跳转工具保持 model-only。

子 ID 格式为 `<parent_id>/<递增序号>`。并发分配 ID 时使用父 run 私有计数器；每个实际请求只生成一次。当前原生工具的 schema 验证、同文件队列、超时和取消语义继续生效。普通失败不停止兄弟调用；JS 是否提前失败由 Promise 的语言语义决定。

`Command` 状态更新不允许通过 Codemode 内部调用隐式修改图；此类工具应 model-only，意外返回时给有界错误并停止该调用。嵌套调用不是额外模型 tool message；仅外层助手调用与 ToolMessage 配对。

### 3.3 嵌套事件与持久化

在现有工具产品事件中增加可选的 `parent_tool_call_id`，缺省为 `None`，保持旧 JSONL 可读。父结果 artifact 写入有界 `nested_calls`：名称、子 ID、参数预览、状态、耗时、错误和 `complete`；不持久化全部子结果。

预算：最多 256 条记录，单调用参数预览 8 KiB，父结果全部参数预览 32 KiB；超过预算设置 `complete=false`，继续记录聚合数量，不影响实际调用结果。错误/参数按现有脱敏规则处理；TUI 缺省只显示名称、状态和耗时，展开再显示安全预览。

实时记录通过现有异步事件/stream writer 投影；禁止将子回调再次当作父模型声明的工具调用，避免重复展示和配对错误。恢复、复制、导出使用父 artifact 中同一份记录。compaction 从允许的内置文件工具参数提取文件路径；不推断任意 MCP 参数为文件路径。

## 4. MCP 接入

### 4.1 配置与授权

新增 `~/.forge/mcp.json` 和项目 `.forge/mcp.json`，使用 `mcpServers` 对象。项目配置只有通过现有项目 trust 后才能读取；同名项目配置整体覆盖用户配置，不合并 command、URL、环境或 headers 字段。

支持 `command/args/cwd/env` 或 `url/headers` 二选一，及 `enabled`、`exposure`、`toolExposure`、`timeoutSeconds`。schema 严格拒绝无效字段。超时默认连接 10 秒、调用 60 秒；结果模型视图 20 KiB，脚本结构化视图上限 1 MiB，超限保留首尾和全文路径。env/headers 只允许环境变量名称引用，不把实际值写入项目文件、日志或 JSONL；缺失引用阻止该服务启动。

`enabled` 是配置意图，不是执行授权。第一版用户配置也不自动连接；`/mcp enable <server>` 表达会话授权，`forge --mcp <server>` 表达单次运行授权，可重复指定。非交互模式没有选中的服务就不连接。trust yes 不代替 MCP 授权。

授权绑定有效配置的来源、规范化 workspace 与指纹；command、args、cwd、URL、env 引用、headers 引用或认证方式变化使旧授权失效。曝光策略只影响已授权服务，不产生连接许可。项目 symlink 逃逸拒绝。

### 4.2 连接与结果

一份会话级 runtime 管理服务描述、授权、活动操作和日志。工具转换使用官方原生工具；其底层 Client/transport 由 runtime 持有。保留官方每调用连接语义，不另造持久 back-channel；只有可验证的服务器会话需求才能改为 SDK 已支持的显式连接，不实现自制重连协议。

连接和发现只发生在明确 enable/CLI 选择后。发现是对话外元数据操作，不能把整套 tools/list 结果发送给模型。分页一直读取到 nextCursor 缺省，重复 cursor/页数异常有界失败；单服务失败不影响其他服务和内置工具。目录变更在 idle `/reload` 或重新 enable 时刷新；第一版不承诺实时通知。

MCP 返回保留 text/image 内容、`structuredContent`、`isError` 和官方 artifact。脚本视图返回标准 `CallToolResult`；模型视图按现有 LangChain multimodal 格式处理。未知 binary 内容保存到受控 artifact 目录，返回路径。只脱敏文本但原样保留结构化内容属于泄漏，二者必须一致处理。

变更性工具不自动重试；超时后的外部副作用状态可能未知，必须报告，不能自动再执行。只给列表/资源读取一次瞬时失败重试，遵守总体操作 deadline。HTTP 不自动跨源重定向携带认证头；跨源请求重新认证。MCP 的 annotations 只作提示，不能替代授权或文件安全检查。

### 4.3 生命周期

每个活动操作归属 session generation 与调用 ID。close、取消、会话切换及 reload 先停止接收新调用，再取消并等待在途操作，最后关闭 transport/process。关闭等待最多 5 秒；超时终止拥有的 stdio 进程树并记录失败，无法确认资源已关闭时拒绝发布替代 runtime，不继续假装切换成功。

配置替换只在 idle 时进行；先解析校验新配置，不执行未授权配置。成功发现后原子替换目录，失败保留旧目录与诊断；已删除/撤权服务立即停止可调用，旧 schema 或旧 artifact 不能赋予权限。

### 4.4 OAuth 与资源

OAuth 通过底层 SDK 公开能力实现 `/mcp login <server>`、`logout`；登录只由人发起，不由模型或项目启动触发。认证材料存放在用户 `~/.forge/` 的独立 MCP 认证目录，限制本机访问，不复用 Forge provider credential 文件。登录与启用分别管理，登录不自动授权执行；URL 脱敏后显示，取消有界关闭。

提供 `list_mcp_resources`、`list_mcp_resource_templates`、`read_mcp_resource`，只访问已授权服务。读取 URI 必须来自当前服务的列表或已验证模板；资源操作标记只读，同样有 deadline、分页和输出限额。跳过 MCP Apps 的 HTML/UI 资源。官方 `langchain.mcp` 未封装资源，因此调用已有 FastMCP Client 的公开资源方法，不再添加一套 MCP 协议实现。

默认拒绝服务端 elicitation、sampling、roots 请求，不安装第二种问卷机制。官方适配器会默认接管 elicitation：传入配置了拒绝处理器的 prebuilt Client，保证不触发该适配器的 interrupt 循环。验收必须证明工具调用返回有界不支持错误，而不是挂起或重放副作用。

## 5. 工具按需加载

新增原生 `tool_search`，输入 `query`、可选 `namespace` 与 `limit`；默认 5，上限 20。目录规模上限每会话 2,000 个工具；超限拒绝发布超限服务并给诊断，内置工具保留。JSON schema 由原生工具生成，单工具声明上限 16 KiB，超过上限不可直接声明，可用 describeTool 分页查看。

`searchTools(query, {namespace?, limit?})` 与上述搜索契约一致，但不改变模型声明；`describeTool(name, {offset?, limit?})` 返回 schema 的 UTF-8 文本页及 `next_offset`，offset 是字符位置，limit 默认 4,096 字符、上限 8,192。大 schema 工具仍可由 JS 调用，模型直接加载时返回明确的大小限制错误。

对名称、namespace、description 用共享 BM25 搜索函数，稳定按原目录顺序破除同分；不使用向量库、LLM 分类或新搜索服务。搜索仅覆盖当前已授权且非 hidden 的工具；model-only 只可直接加载。

搜索成功将选中工具加入当前分支 declared 集合。新增原生 model-call middleware，只在下一次模型请求前选择/bind 原生工具。执行 ToolNode 必须能解析允许的整个 registered 集合；未声明直接调用按策略拒绝，JS 内部调用按 callable 集合检查。不能仅修改 provider schema 后忘记执行注册。

保存 `forge.tool_loadout.v1` CustomEntry：版本、增加的稳定工具名和 metadata/schema 指纹；恢复从活动分支重建集合，再与当前授权目录取交集。不持久化凭据、客户端或整套 schema。未知工具给诊断，保留历史结果；不会为恢复历史重新启动服务。fork/clone/import 沿现有活动路径规则处理，项目 trust 与 MCP 授权必须重新验证。

工具搜索返回有界名称、描述和加载状态，不返回整套 schema；实际 schema 进入下一请求。它不触发 provider 请求，不影响模型调用预算。动态 schema 的缓存前缀变化按真实情况记录，不声称能完全保留 provider cache。

## 6. Forge Codemode

### 6.1 执行环境与 IPC

父进程注册一个 `codemode(code: str)` StructuredTool；worker 使用当前 Python 解释器启动 `-m forge_coding.codemode.worker`，无 Node、Rust 编译或外部 shell。每次调用独立 worker/Context，运行结束立即关闭。子进程仅携带运行必需的最小环境，不继承 provider 密钥和 MCP 认证环境；cwd 是受控工作目录。

采用 stdio JSONL IPC，协议带版本、run ID 和 request ID。仅允许 init、tool_call、tool_result、output、done/error 消息；单行最大 2 MiB，单次工具参数最多 64 KiB。收发队列各最多 64 条且各有 8 MiB 总字节预算，容量不足时背压等待并受总 deadline 约束；调用数超限直接拒绝，不无限积压。未知消息、错配 ID、非 JSON 值和超限输入停止执行。stdout 只用于协议，stderr 最多保留末尾 64 KiB 诊断；JS console 转换为 output 消息。

QuickJS API 不能阻塞父事件循环。worker 拥有 runtime；工具请求回到父 asyncio loop 执行，保持现有锁的 event-loop 归属。父端并发子调用最多 32，超出排队；最多 256 次工具调用，超出返回有界 budget 错误。这是明确的 Forge 资源上限，Pi 脚本在超过上限时可能需要分批。

### 6.2 Pi 脚本契约

支持 `tools.<normalized_name>(args)`、`ALL_TOOLS`、`text`、`image`、`console.log/warn/error`、`exit`、`store/load`、`searchTools`、`describeTool`。工具名称规范化参考 Pi，不采用 LangChain PTC 的 camelCase 重命名。

源代码使用固定 async 包装支持 top-level await 和 return；用 QuickJS 正常解析，不通过正则重写 JS。首行可用 `// @options: {"max_output_tokens": 1000, "timeout_ms": 60000}`；严格 JSON、只允许两个键。`exit` 正常结束当前脚本，结束后不再接受新工具请求，并取消未完成的请求。

默认不声明 `codemode`，新增 `--codemode on|only` 单次启用。`on` 保留直接工具声明；`only` 将可嵌套调用的 direct 工具声明移入 Codemode API 提示，model-only 工具继续直接声明。无 MCP 时也能启用。启用状态记录在当前分支 loadout；SDK 通过 CodingSessionConfig 中对应字段选择模式。

API 提示按 namespace 分组，从原生 schema 生成 TypeScript 风格的调用说明，预算 3,000 估算 token；所有 namespace 和工具数量仍显示，超限部分由 searchTools/describeTool 获取。`tool_search` 与脚本搜索共用排序逻辑。

内置工具脚本结果读取 artifact 中公开结构化字段；普通没有结构化结果的工具返回文本；MCP 返回 CallToolResult。不要把 Forge 的全部私有 artifact、路径或 provider 调试信息暴露为 JS 对象。shell 脚本结果包含 output、truncated、full_output_path、exit_code、wall_time_seconds；保持现有完整输出文件机制，脚本视图上限 1 MiB。

### 6.3 超时、失败与输出

默认堆预算 256 MiB、输出预算 10,000 估算 token；单脚本输出收集另有 8 MiB byte 硬上限，达到后停止收集并标记丢失。token 估算复用现有计数规则，不伪称 provider 精确 tokenizer。

Pi CLI 未指定 timeout_ms 时无总 deadline；Forge 第一版默认总 deadline 300 秒，允许脚本设置 1..300,000 毫秒。这是取消/资源保护的明确差异；长任务拆成多个调用。JS 自身执行预算与父端真实耗时 deadline 分开实施；只用 QuickJS timeout 不能约束等待宿主工具。

取消/超时：父端停止派发请求、取消并等待子工具、终止 worker；截止后的 IPC 结果丢弃，不允许修改状态。普通脚本错误保留已有输出，标记 Script failed；Promise.allSettled 可收集局部失败。标记错误与 LangChain ToolMessage.status 一致，不能仅在正文里写 error 但结果仍显示 success。

输出包括 Script completed/failed、耗时、text/image 项、return 值及有界错误。超过 token 展示预算时保留首尾，已收集的完整内容通过现有安全写入工具保存到 artifact 目录，返回路径；超过 byte 收集上限的部分无法恢复，明确标记丢失，不能称为完整输出。image 仅接受已验证 data URL 或规范 image block，MIME 限于 PNG/JPEG/WebP/GIF，解码后每张最多 512 KiB、单脚本累计最多 4 MiB，并计入输出 byte 预算；不接受远程 URL 自动下载。较大的图片由工具返回受控文件路径，不经 IPC 内联。

### 6.4 JSON 分支存储

worker init 接收从当前分支恢复的 JSON store。`store(key, value)` 写入脚本私有变更集，undefined 删除；单值 64 KiB，总 store 1 MiB。复杂对象不能序列化、循环引用和超限均返回脚本错误。

只有脚本成功且 session generation 仍有效，才把 set/delete 追加为 `forge.codemode_store.v1` CustomEntry。append 成功之后才报告存储已提交；失败给持久化错误，不能伪称可以恢复。JS 全局变量、闭包和 promise 不跨调用保留。

从活动分支的全部可用 CustomEntry 重建 store，不只遍历模型 compaction 后剩余消息；fork/clone/import 遵循现有路径复制。失败脚本不提交 store，但已完成的工具副作用不会回滚。未知版本忽略并给诊断，解析失败不执行存储数据。

## 7. 文件落点

以下是实施落点；新增模块现已存在。输出落盘复用 tools/output.py 和现有安全 write 工具，未新增另一套文件写入机制。

| 现有/新增路径 | 工作 |
| --- | --- |
| `src/forge_agent/tool_execution.py` | 抽取共享执行校验，保留模型批次规则 |
| `src/forge_agent/context.py`、`events.py`、`langchain_runtime.py` | 嵌套上下文、可选父 ID、v3 事件投影 |
| `src/forge_coding/tools/definition.py`、`tool_set.py` | exposure、namespace、目录视图 |
| 新增 `src/forge_coding/features/tool_discovery.py` | BM25、tool_search、声明 middleware、loadout 数据验证 |
| 新增 `src/forge_coding/mcp/{__init__,config,runtime}.py` | 配置/授权、官方 SDK 连接、资源与 OAuth |
| 新增 `src/forge_coding/codemode/{__init__,runtime,worker}.py` | StructuredTool、父宿主与有界 IPC、QuickJS worker |
| `src/forge_coding/tools/base.py`、`shell.py`、`truncation.py` | 原生结果复用、公开脚本视图、输出预算 |
| `src/forge_coding/sessions/session.py`、`reload.py`、`compaction.py`、`export.py` | 生命周期、分支数据、文件操作与导出 |
| `src/forge_coding/paths.py`、`resources/trust.py` | 新配置/认证/artifact 路径与项目边界 |
| `src/forge_coding/commands/default_registry.py` | /mcp 领域命令 |
| `src/forge_cli/cli.py`、`tui/app.py`、`tui/widgets.py` | flags、状态、嵌套工具展示 |
| `pyproject.toml`、`uv.lock`、CI、README、NOTICE、AGENTS、architecture | 依赖、规则例外、使用与归因 |

## 8. 公共接口与配置增量

新增模型工具：`codemode`、`tool_search`、三项 MCP resource 工具。MCP server tools 本身按目录动态注册。新增 CLI 参数：可重复 `--mcp <server>`、`--codemode on|only`；不用再造通用 settings 系统或读取不存在的 settings.py。

新增 `/mcp list|enable|disable|tools|logs|login|logout`；其中 tools/logs/login/logout 接收 server 名，enable/disable 必须接收 server 名。无参数 `/mcp` 等同 list，显示来源、授权、连接、曝光和错误摘要。logs 默认最后 50 条、最多 200 条，脱敏后输出。disable 撤销会话授权并清理该服务，不更改项目配置。

已有 `/reload` 更新目录和资源，运行期间拒绝；branch loadout 在新用户轮次前应用。没有新增第三方账号要求：离线交付使用本地 fake MCP server 和 fake chat model；真实 HTTP/OAuth 仅在用户明确选择的服务上人工验收。

## 9. 独立交付阶段

| 阶段 | 内容 | 独立完成的条件 |
| --- | --- | --- |
| A：目录与统一边界 | exposure、共享单次执行、父子事件及 artifact；当前工具默认行为保持 | 原有 CLI/TUI 可用，新的嵌套调用离线场景能执行和导出；没有未使用的可配置框架 |
| B：MCP 工具 | 官方 MCP 依赖、配置、显式授权、stdio/HTTP、结果和生命周期、/mcp 基础命令；此阶段默认 direct | 配置并授权本地服务器即可直接使用工具，不依赖搜索或 Codemode |
| C：按需声明 | BM25、tool_search、deferred 默认、loadout 恢复与分支 | MCP 可搜索后直接调用；阶段 B 的直接曝光配置仍可用 |
| D：Codemode | QuickJS worker、IPC、Pi 脚本接口、结构化输出与 JSON store、on/only | 不配置 MCP 也可编排内置工具，接入 MCP 后共用目录；取消与恢复验收通过 |
| E：MCP 完整度 | SDK OAuth、resource 工具、诊断完善 | 工具、搜索、Codemode 均不依赖 OAuth 或资源；本地模拟认证与资源分页可重复验证 |

每阶段落地前更新代码基线并复核目录/API；不新增“研究阶段”代替未完成设计。每阶段结束可独立交付，后续未完成时不在 README 声称已支持。

## 10. 验收与可重复工件

以端到端离线验证为新功能主机制，复用现有 owning-module 检查。先记录下表失败模式和预期，再实现；不在实现后追加镜像实现的单元测试。开发期间只运行窄范围检查，全套检查在阶段完成时执行一次。

新增 `tests/test_tool_ecosystem_e2e.py`，共用 `tests/fixtures/tool_ecosystem/` 的 stdio/loopback HTTP 服务器与 fake chat model。HTTP 端口自动分配，不访问外网、不读取用户认证或真实会话。输出到 pytest 临时目录的 `tool-ecosystem-evidence/`：机器可读 summary.json、人工 session.jsonl、导出 HTML、脱敏事件 trace，以及取消后 process/task 存活检查；报告给出 artifact 绝对路径。测试失败也生成工件。

| 场景 | 必须断言 |
| --- | --- |
| 普通调用与嵌套调用 | 同参数错误得到相同拒绝；父子 ID 唯一，只有外层模型调用配对 |
| 并行、顺序与依赖 | 独立调用可并行，同文件串行；代码依赖顺序成立；外层顺序锁无死锁 |
| 错误与状态 | MCP isError、JS throw、schema 错误、未知工具正确标记，不污染兄弟结果 |
| 配置与授权 | 项目 trust 通过但 MCP 未 enable 时不启动进程/网络；配置指纹变化撤权 |
| HTTP | 超时、跨源重定向、取消不泄漏认证；变更调用不自动重试 |
| 目录 | 分页、重复 cursor、碰撞、超限、撤权、reload 与 search 的稳定结果 |
| 声明 | 搜索后下一模型请求含选中 schema；未声明或 hidden 直接调用被拒绝 |
| JS 边界 | 没有 fs/process/fetch/import；非法 IPC、无限循环、微任务循环、OOM 有界结束 |
| 总 deadline | 等待宿主工具也受 deadline 限制；所有待处理请求和 worker 清理完成 |
| 文件安全 | symlink/绝对路径逃逸和 final symlink 写入保持拒绝 |
| 输出 | 文本首尾截断、完整文件、图片大小、结构化数据和内容同时脱敏 |
| 会话 | store 成功才提交；失败/取消不提交；clone/fork/import/compaction 不串分支 |
| 展示 | TUI 取消、恢复、展开；/copy 不泄漏内部原始结果；HTML 显示有界嵌套记录 |
| OAuth/资源 | 模拟登录成功/失败/取消/登出、资源模板/分页；elicitation 明确拒绝无挂起 |

沿用 `tests/test_tool_execution_middleware.py`、`test_tool_set.py`、`test_coding_session.py`、`test_langchain_runtime.py`、`test_session_export.py`、`test_tui_app.py` 和 `test_cli.py` 的窄范围验证。真实 provider/OAuth smoke 只在明确授权后执行，离线通过不推断真实服务兼容。

阶段结束执行项目规定命令：

```powershell
uv sync --dev --extra providers --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run forge --help
uv run forge --version
```

最终人工验收：本地服务器授权与撤权；搜索后调用；独立与依赖脚本；取消长脚本；切换/恢复分支并读取 store；打开导出检查嵌套调用。Windows 与 Linux 都验证 worker 启动、取消和进程清理；CI 保持现有检查并增加对应系统的离线场景，所有新依赖锁定。

## 11. 风险、降级与回退

- 最脆弱前提：QuickJS worker 的取消能终止失控代码，并且父进程工具取消不会遗留任务。若 E2E 证明任一项失败，该阶段不交付；不能用 TUI 隐藏错误或仅取消等待来代替资源清理。
- 外部服务失败：单服务标记 unavailable，其他工具继续；模型下一请求移除不可用工具声明；恢复后仅 idle 发布新目录。
- 规模：工具数、schema、并发、调用数、store、IPC、输出各有上述上限；超过上限有界失败，不引入数据库或后台索引服务。
- 安全：WASM 限制 JS 的内存能力，宿主桥接开放的工具仍拥有原来的外部权限；shell 仍不是 sandbox。不能宣传为整个 Forge/外部 MCP 的系统沙箱。
- 回退：关闭 Codemode、撤销服务授权即可保留原内置工具体验；未知 CustomEntry 保持旧读取行为。保留历史结果，loadout/store 不启动服务、不执行代码；不删除用户历史数据来降级。
- 发布：按 Conventional Commit 分阶段；提交、创建分支、推送和合并仅按用户另行指令执行。更新 README、architecture、release notes；复用/移植 Pi 或 LangChain 代码时更新 NOTICE 并保留许可证归因。

## 12. 参考来源

- [Pi v0.99.1 Codemode 接口](https://github.com/earendil-works/pi/blob/v0.99.1/packages/coding-agent/docs/cli.md#how-codemode-works)
- [Pi 工具曝光与嵌套调用](https://github.com/earendil-works/pi/blob/v0.99.1/packages/coding-agent/docs/extensions.md#tools)
- [Pi MCP 配置和资源](https://github.com/earendil-works/pi/blob/v0.99.1/packages/coding-agent/docs/mcp.md)
- [Pi worker 生命周期](https://github.com/earendil-works/pi/blob/v0.99.1/packages/codemode/src/runtime/host.ts)
- [Pi Codemode 宿主及 store 提交](https://github.com/earendil-works/pi/blob/v0.99.1/packages/coding-agent/src/extensions/codemode/execute.ts)
- [LangChain MCP 当前官方入口](https://docs.langchain.com/oss/python/langchain/mcp)
- [MCP 迁移、elicitation 与资源边界](https://docs.langchain.com/oss/python/migrate/langchain-mcp-adapters)
- [Deep Agents Interpreter 实践与 PTC 边界](https://docs.langchain.com/oss/python/deepagents/interpreters)
- [quickjs-rs 0.2.5 对应发布源码](https://github.com/langchain-ai/quickjs-rs/tree/278cf32d17b07a9ba2951ebc826256eef703182d)

官方文档是编写日快照，实施时以锁定版本的公开接口和离线证据复核；历史 Forge MCP 分支只作选择性复用来源。

## 13. A 阶段实施记录

- 日期：2026-09-30；基于上列提交的未提交工作区，不创建分支、不合并旧 MCP 分支。
- 已实现：原生 metadata 的 exposure/namespace/annotations；registered/declared/callable 目录视图；模型声明筛选；共享单次调用边界和原生 ToolNode 嵌套注入；父子事件和有界 trace；取消后子任务清理；TUI、恢复、HTML 导出和文件 compaction。
- 当前可通过自定义原生工具内的 `get_nested_tool_executor().call(...)` 使用嵌套能力。内部子 ToolMessage 不进入模型消息，状态控制工具禁止嵌套；没有新增 CLI flags、依赖或 JS 运行环境。
- 有意收紧：持久化参数预览只保留 path/offset/limit，省略文件内容、shell command 和任意业务字段；256 次调用为硬上限。此阶段默认声明只包含 direct/model-only，deferred/codemode 工具保留在执行目录中，但没有搜索加载接口。
- 离线入口：`uv run pytest tests/test_tool_ecosystem_e2e.py -q -s`。每个场景打印 pytest 临时工件目录；分别包含父子事件/JSONL/HTML、取消清理状态、并行和预算结果。验收覆盖受控文件操作、伪造 runtime、越界、schema 错误、不可嵌套工具、Command 拒绝、并发和预算，以及实时与恢复展示一致性。
- 后续：B 接 MCP 官方 SDK 和授权/生命周期，C 接搜索与分支 loadout，D 接 JS worker 和 store，E 接 OAuth 与 resources。当前 README 不宣称它们已经可用。
- 验证结果：Windows / Python 3.12.11 全套 `uv run pytest` 为 1,236 passed、10 skipped；上述七项阶段结束命令全部通过。跳过项涉及 POSIX、不可用 symlink 和需显式启用的 rg/fd 集成；没有运行真实 provider 或 Linux。验收工件副本保存在被 Git 忽略的 `docs/dev/tool-ecosystem-evidence/phase-a-20260930-*/`，不提交生成的会话数据。

## 14. B 阶段配置层实施记录

- 日期：2026-09-30；代码基线同上，保留 A 阶段与用户未提交修改。
- 已实现 `forge_coding.mcp.config` 严格解析和 `ForgePaths` 配置路径。独立加载入口不创建客户端、不启动进程、不读取环境变量值。当前没有接入 session 启动流程，因此 CLI 尚不读取 MCP 配置。
- `env` / `headers` 值统一使用 `{"env": "VARIABLE_NAME"}`。同名项目服务整体覆盖用户服务；项目文件仅在匹配 workspace 的 trust 允许时读取，配置路径和进程 cwd 不允许逃出 workspace。服务器名限制为字母开头的 64 字符 ASCII 标识。
- 配置文件最大 256 KiB，最多 256 个服务；拒绝重复 JSON 键、未知字段、混合传输、内联环境/认证 header 值、URL 用户信息/query/fragment。错误不包含配置内容。URL 查询参数暂不支持；后续确有服务需要时应先定义敏感参数处理规则。
- 配置指纹绑定规范化 workspace、来源、有效配置和进程 cwd。这只是运行时撤权所需的数据，尚未实现授权状态或授权校验，`enabled` 不代表获得授权。
- 离线入口：`uv run pytest tests/test_mcp_config_e2e.py -q -s`；先写验收再实现，生成 `mcp-config-evidence.json`。当前覆盖 trust 拒绝读取、整体覆盖、指纹变化和无内容泄漏的配置拒绝。
- 官方入口在隔离依赖环境复核为 LangChain 1.4.3 / FastMCP 4.0.10；项目依赖和锁文件未升级。下一步仍是 B 阶段：授权状态、stdio/HTTP、官方工具适配、取消和关闭清理、session/CLI/TUI 接入与 `/mcp`，不能视为 MCP 已交付。
- 验证结果：Windows 全套 pytest 为 1,245 passed、10 skipped、3 warnings；`uv sync --dev --extra providers --locked`、Ruff check/format、mypy、CLI help/version 均通过。没有运行真实 provider 或 Linux。配置验收工件副本：`docs/dev/tool-ecosystem-evidence/phase-b-config-20260930/summary.json`（Git 忽略）。

## 15. B–E 完整实现记录

- 日期：2026-09-30；基线同上，全部修改仍在本地工作区。
- B：原生 LangChain MCP 转换；严格配置和显式授权；stdio/HTTP 每操作连接；分页、重复游标拒绝、撤权/刷新、跨源重定向禁用；内容和结构化结果一致脱敏，未知二进制保存为受控 JSON artifact。
- C：共享 BM25、namespace 筛选、原生 tool_search；下一模型请求加载 schema；大 schema 只能经 Codemode 调用和分页面查询；分支 loadout 只恢复有效目录交集。会话持有自己的原生工具元数据副本，防止共享工具的加载状态串扰。
- D：独立 Python/QuickJS/WASM worker；版本/run/request IPC；嵌套原生工具；on/only 模式；父进程存储复核和成功后追加 CustomEntry；可捕获 JS 异常也无法绕过 exit 停止派发；取消、超时、全局隔离、输出/图片/存储预算与安全落盘。
- E：SDK OAuth 的浏览器/回调/PKCE/刷新和私有 token storage；登录不授权工具，登出撤权；资源和模板分页、URI 校验、HTML 跳过；默认拒绝 elicitation，未注册 sampling/roots。Windows token 文件名使用 SDK 的公开安全编码策略，目录使用当前用户 ACL。
- 实现选择：IPC 接收逐行处理，最多 64 个在途请求，容量不足等待空位（每个参数最多 64 KiB），避免额外接收队列；回复使用 8 MiB 字节背压、写锁及 drain。stderr 持续排空并丢弃，失败只显示有界类别，避免宿主诊断泄漏。代码没有添加可配置队列框架。
- 已更新 README、architecture、NOTICE、AGENTS 和 Unreleased release notes；CI 增加 Windows/Linux 矩阵及 CLI 检查，尚未推送运行远程 CI。
- 离线入口：`uv run pytest tests/test_tool_ecosystem_e2e.py tests/test_mcp_config_e2e.py tests/test_mcp_runtime_e2e.py tests/test_mcp_http_e2e.py tests/test_tool_features_e2e.py -q -s`。验收生成 JSON、HTML、挂载 TUI 的 SVG；只使用 fake 模型、本地进程与 loopback OAuth。
- 验收范围包括实际 stdio PID 退出、调用消息配对、父子展示/恢复/导出、结构化脱敏、图像和未知二进制、工具/资源分页、资源 URI 拒绝、SDK elicitation 拒绝、CLI+MCP+Codemode、恢复不授权、100 次并发调用背压和挂载 TUI enable/disable。
- 最终 Windows 验收：1266 passed、10 skipped、4 warnings；依赖同步、Ruff check/format、mypy、CLI help/version 均通过。受控工件：`docs/dev/tool-ecosystem-evidence/final-20260930/summary.json`，只复制明确的 fake 验收 JSON/HTML/SVG，不复制 OAuth token storage 或真实用户会话。
- 保留风险：此前一次全套运行中的 exit 用例失败未能复现，根因尚未定位；加入不含原始内容的异常类别/IPC 阶段诊断后，单独 8 次、完整测试内 8 次和底层 worker 并行 40 次均通过，不声称该偶发问题已经修复。真实 provider、第三方 MCP、Linux 和远程 CI 尚未验证。
- 收尾修正：大 schema 降级保留 hidden 策略；目录和 schema 使用同一脱敏；官方 MCP 适配器内部回调关闭，避免重复产品事件。

## 16. 2026-10-01 审查修复

- 状态：7 项 P2 已修复，Windows 本地验收通过；基线仍为 `0a0c05c1871c2e06e5993204821533753c021d56`，未提交或推送。
- 统一刷新子智能体授权目录，复制父目录元数据并排除父控制闭包；子模型直接声明角色选中的 deferred 原生工具。
- 两处上下文计数使用 declared 目录；目录与搜索快照更新使缓存失效。
- 调用方的 tool_search/codemode 保留名明确拒绝；MCP 资源只替换自己发布的工具，不静默删除调用方工具。
- MCP 操作成功、失败和取消均同步已提交目录；指纹未变的发现失败保留旧工具可用性。
- HTTP JSON、单条 SSE 与 stdio 行在解析前执行 8 MiB 字节预算；禁止 HTTP 压缩响应；工具目录跨服务、资源目录跨分页累计预算。stdio 仅使用 Forge 的有界进程管道，JSON-RPC 和会话仍由 SDK 处理。
- 验证：首次 10 项新回归在未修代码上全部失败，后续同类路径亦观察红绿；新增 18 项最终全部通过。全套为 1284 passed、10 skipped、4 warnings；依赖同步、Ruff check/format、mypy、CLI help/version 均通过。
- 本地历史工件：`docs/dev/check-20261001/repairs.md`、`repairs-verification.json`、18 个 `repair-evidence/*.json`（Git 忽略）。当前实现事实仍以 source/README/architecture 为准；不复制认证缓存或真实用户会话。
- Linux、真实 provider/第三方 MCP、远程 CI 未执行；历史一次 exit 波动根因未定位，本次没有声称修复该问题。
- 关联回归：子目录元数据复制后，执行不可用状态仍由 MCP runtime 共同入口校验；第二次连接被拒绝，不会因副本陈旧绕过状态。

## 17. 2026-10-01 复查遗漏修复

- 状态：剩余三项 P2 已修复；基线 `0a0c05c1871c2e06e5993204821533753c021d56`，未提交或推送。
- 组合目录冲突时撤销全部 MCP 授权，同步清空 runtime/session 和子目录，保留原始取消异常；修正冲突后显式重新 enable。
- 资源列表、模板发现和内容读取采用只读失败策略，不因临时连接错误禁用正常工具。
- SSE 媒体类型规范化后按事件计费，跨块 CRLF 保留完整字节预算。
- 先写回归观察 14 项失败，修复后 32 项回归通过；Windows 全套 1298 passed、10 skipped、4 warnings。依赖同步、Ruff check/format、mypy、CLI help/version 均通过。
- 本地历史工件：`docs/dev/check-20261001/recheck-repairs.md`、`recheck-repairs-verification.json` 和 32 个 `recheck-repair-evidence/*.json`。Linux、真实服务、远程 CI 未验证；历史退出波动根因仍未定位。

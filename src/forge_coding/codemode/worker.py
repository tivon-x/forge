"""One-shot QuickJS/WASM worker. Stdout is reserved for versioned JSONL IPC."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from quickjs_rs import Runtime, SourceTransform

MAX_LINE = 2 * 1024 * 1024


def _read() -> dict[str, Any]:
    line = sys.stdin.buffer.readline(MAX_LINE + 1)
    if not line or len(line) > MAX_LINE or not line.endswith(b"\n"):
        raise ValueError("Invalid IPC input")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("Invalid IPC object")
    return value


async def run(initial: dict[str, Any]) -> None:
    run_id = initial["run"]
    pending: dict[int, asyncio.Future[Any]] = {}
    counter = 0
    slots = asyncio.Semaphore(64)
    finished = False

    def finish(snapshot: str) -> None:
        nonlocal finished
        if not finished:
            finished = True
            emit("done", ok=True, value=None, store=json.loads(snapshot))

    def emit(kind: str, **payload: Any) -> None:
        value = {"v": 1, "run": run_id, "type": kind, **payload}
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        if len(data) > MAX_LINE:
            raise ValueError("IPC output exceeded limit")
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()

    async def request(name: str, arguments: Any, target: str = "tool") -> Any:
        nonlocal counter
        async with slots:
            if finished or counter >= 256:
                raise ValueError("IPC call budget exhausted")
            if len(json.dumps(arguments, allow_nan=False).encode()) > 65536:
                raise ValueError("Tool arguments exceeded limit")
            counter += 1
            ident = counter
            future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            pending[ident] = future
            emit("tool_call", id=ident, name=name, args=arguments, target=target)
            try:
                return await future
            finally:
                pending.pop(ident, None)

    async def receive() -> None:
        while True:
            message = await asyncio.to_thread(_read)
            if (
                message.get("v") != 1
                or message.get("run") != run_id
                or message.get("type") != "tool_result"
            ):
                raise ValueError("Invalid IPC result")
            ident = message.get("id")
            if type(ident) is not int:
                raise ValueError("Invalid IPC result id")
            future = pending.get(ident)
            if future is None or future.done():
                raise ValueError("Unknown IPC result id")
            if message.get("ok") is True:
                future.set_result(message.get("value"))
            else:
                future.set_exception(ValueError("Tool call failed"))

    receiver = asyncio.create_task(receive())
    try:
        with (
            Runtime(
                memory_limit=256 * 1024 * 1024, transform_flags=SourceTransform.NONE
            ) as runtime,
            runtime.new_context(timeout=initial["timeout"]) as context,
        ):
            context.register("__forge_call", request)
            context.register("__forge_finish", finish)
            context.register("__forge_output", lambda item: emit("output", item=item))
            context.globals["__forge_catalog"] = initial["tools"]
            context.globals["__forge_store"] = initial["store"]
            context.eval(BOOTSTRAP)
            code = initial["code"]
            source = (
                "await (async()=>{try{const value=await(async()=>{\n"
                + code
                + "\n})();return JSON.stringify({ok:true,value:value??null,"
                + "store:__forge_export()});}catch(e){if(e===__forge_exit)"
                + "return JSON.stringify({ok:true,value:null,store:__forge_export()});"
                + "return JSON.stringify({ok:false,error:String(e).slice(0,256)});}})()"
            )
            result = await context.eval_async(source, timeout=initial["timeout"])
            if not finished:
                emit("done", **json.loads(result))
    except Exception:
        if not finished:
            emit("done", ok=False, error="JavaScript execution failed")
    finally:
        receiver.cancel()
        for future in pending.values():
            future.cancel()
        # stdin may still be waiting in a thread; the parent terminates this one-shot worker.


BOOTSTRAP = r"""
const __forge_exit=Object.freeze({});
const tools=Object.create(null);
const ALL_TOOLS=Object.freeze(__forge_catalog.map(t=>Object.freeze({...t})));
for(const t of ALL_TOOLS) tools[t.jsName]=(args={})=>__forge_call(t.name,args,'tool');
Object.freeze(tools);
let __forge_bytes=0;
let __forge_lost=false;
function text(value){
    const item={type:'text',
        text:typeof value==='string'?value:JSON.stringify(value)??String(value)};
    const bytes=JSON.stringify(item).length*3;
    if(__forge_bytes+bytes>8388608){__forge_lost=true;return;}
    __forge_bytes+=bytes; __forge_output(item);
}
function image(value){__forge_output({type:'image',image:value});}
const console=Object.freeze({
    log:(...v)=>text(v.length===1?v[0]:v),warn:(...v)=>text(v),error:(...v)=>text(v)
});
function exit(){__forge_finish(JSON.stringify(__forge_export()));throw __forge_exit;}
function store(key,value){
    if(typeof key!=='string'||key.length>256)throw Error('Invalid store key');
    if(value===undefined){delete __forge_store[key];return;}
    const encoded=JSON.stringify(value);
    if(encoded===undefined||encoded.length*3>65536)throw Error('Store value exceeded limit');
    const previous=__forge_store[key];
    Object.defineProperty(__forge_store,key,{
        value:JSON.parse(encoded),writable:true,enumerable:true,configurable:true
    });
    if(JSON.stringify(__forge_store).length*3>1048576){
        if(previous===undefined)delete __forge_store[key];else __forge_store[key]=previous;
        throw Error('Store exceeded limit');
    }
}
function load(key){
    if(!Object.hasOwn(__forge_store,key))return undefined;
    return JSON.parse(JSON.stringify(__forge_store[key]));
}
function __forge_export(){return {values:__forge_store,lost:__forge_lost};}
async function searchTools(query,options={}){
    const opts=typeof options==='number'?{limit:options}:options;
    return __forge_call('search',{query,...opts},'metadata');
}
async function describeTool(name,options={}){
    return __forge_call('describe',{name,...options},'metadata');
}
void 0;
"""


if __name__ == "__main__":
    try:
        initial = _read()
        if initial.get("type") != "init" or initial.get("v") != 1:
            raise ValueError("Invalid worker init")
        asyncio.run(run(initial))
    except Exception:
        sys.exit(1)

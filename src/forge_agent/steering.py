"""LangChain middleware that injects harness steering into the agent loop."""

from __future__ import annotations

from collections import deque
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AnyMessage

QueueMode = str


class SteeringMiddleware(AgentMiddleware):
    """Drain harness steering messages before the next model call.

    Uses the official ``before_model`` hook: the drained ``HumanMessage`` rows
    are returned as a ``messages`` state update, so the graph's message reducer
    appends them (assigning missing ids) and the immediately following model
    call sees them.  Steering never preempts the currently executing
    model/tool call -- it only takes effect at the next ``before_model``.

    The middleware never touches the ``ModelCallLimitMiddleware`` counters:
    it only adds messages to the state, so steering does not consume or reset
    completed model-call counts.

    Contract tests pin the reducer behavior this relies on: the returned
    messages must appear in the v3 ``values``/``messages`` projections with a
    stable id.  If an upstream LangChain change breaks that contract, the
    middleware tests fail first; Forge must not fall back to a private loop.
    """

    def __init__(
        self, queue: deque[AnyMessage], *, queue_mode: QueueMode = "one_at_a_time"
    ) -> None:
        self._queue = queue
        self._queue_mode = queue_mode

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        del state, runtime
        drained = self._drain()
        if not drained:
            return None
        return {"messages": list(drained)}

    def _drain(self) -> tuple[AnyMessage, ...]:
        if not self._queue:
            return ()
        if self._queue_mode == "all":
            messages = tuple(self._queue)
            self._queue.clear()
            return messages
        return (self._queue.popleft(),)

"""Process driver for D3 committed observation restart checks."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from forge_agent.session import CustomEntry
from forge_coding.features.observation import CommittedView
from forge_coding.sessions.storage import JsonlSessionStorage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("seed", "resume"))
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    return parser


async def _run(args: argparse.Namespace) -> None:
    storage = JsonlSessionStorage(args.session)
    if args.action == "seed":
        await storage.commit_records(
            [
                {
                    "type": "document.create",
                    "record": {"id": 1, "kind": "forge.live.v1", "scope": {"kind": "session"}},
                    "content": {
                        "kind": "base",
                        "version": 1,
                        "value": {"assistant": "restarted", "background": "running"},
                    },
                },
                {
                    "type": "task",
                    "value": {
                        "id": 2,
                        "kind": "d3.background",
                        "version": 1,
                        "input": {"case": "restart"},
                        "conversationId": 0,
                        "background": True,
                        "abortRequested": False,
                        "state": {"status": "running", "checkpoint": {"phase": "observe"}},
                    },
                },
            ]
        )
        await storage.append(CustomEntry(namespace="d3.marker", data={"committed": True}))
    view = await CommittedView.open(storage)
    snapshot = view.snapshot()
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps({"action": args.action, "snapshot": snapshot}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    await view.close()
    await storage.close()


def main() -> None:
    args = _parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

"""Event renderers for Forge coding frontends and print modes."""

from __future__ import annotations

from forge_cli.rendering.base import EventRenderer, PrintOutputMode
from forge_cli.rendering.json import JsonEventRenderer
from forge_cli.rendering.plain import FinalTextRenderer
from forge_cli.rendering.transcript import TranscriptRenderer


def create_event_renderer(mode: PrintOutputMode) -> EventRenderer:
    """Create a renderer for a print output mode."""
    if mode is PrintOutputMode.text:
        return FinalTextRenderer()
    if mode is PrintOutputMode.json:
        return JsonEventRenderer()
    return TranscriptRenderer()


__all__ = [
    "EventRenderer",
    "FinalTextRenderer",
    "JsonEventRenderer",
    "PrintOutputMode",
    "TranscriptRenderer",
    "create_event_renderer",
]

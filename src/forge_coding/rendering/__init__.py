"""Event renderers for Forge coding frontends and print modes."""

from __future__ import annotations

from forge_coding.rendering.base import EventRenderer, PrintOutputMode
from forge_coding.rendering.json import JsonEventRenderer
from forge_coding.rendering.plain import FinalTextRenderer
from forge_coding.rendering.transcript import TranscriptRenderer


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

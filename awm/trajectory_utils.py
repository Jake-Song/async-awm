"""Shared message-window helpers for AWM collection and training."""

from __future__ import annotations

from typing import Any


def windowed_messages(
    prompt: list[dict[str, Any]],
    completion: list[dict[str, Any]],
    context_window_turns: int,
) -> list[dict[str, Any]]:
    """Return the AWM context used to generate the next assistant turn.

    The system/user prompt and the first ``list_tools`` exchange stay pinned.
    Older exchanges after that prefix are dropped once they fall outside the
    recent-turn window. ``completion`` must contain complete assistant/tool
    pairs; it may also be empty.
    """
    if context_window_turns < 0:
        raise ValueError("context_window_turns must be non-negative")

    pin_end = min(2, len(completion))
    for i in range(0, len(completion), 2):
        calls = completion[i].get("tool_calls") or []
        if any(call.get("function", {}).get("name") == "list_tools" for call in calls):
            pin_end = min(i + 2, len(completion))
            break

    recent_start = max(0, len(completion) - 2 * context_window_turns)
    if recent_start <= pin_end:
        context = completion
    else:
        context = completion[:pin_end] + completion[recent_start:]
    return [*prompt, *context]

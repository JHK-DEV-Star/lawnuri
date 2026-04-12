"""
Pause/stop check utilities for debate graph nodes.

Two flavors:
- ``check_pause_interrupt(debate_id)`` — uses LangGraph ``interrupt()``
  for nodes running inside the **main graph** (which has a checkpointer).
- ``check_pause(debate_id)`` — raises ``DebatePausedError`` for nodes
  running inside a **subgraph** or manually-invoked graph without a
  checkpointer.  Callers should catch this and return an early-exit dict.
"""

from __future__ import annotations

from langgraph.types import interrupt

from app.utils.logger import logger


class DebatePausedError(Exception):
    """Raised by check_pause() when the debate is paused or stopped."""

    def __init__(self, status: str, debate_id: str):
        self.status = status
        self.debate_id = debate_id
        super().__init__(f"Debate {debate_id} is {status}")


async def _load_status(debate_id: str) -> str:
    """Load debate status from the store. Returns '' on failure."""
    if not debate_id:
        return ""
    try:
        from app.api.debate import DebateStore
        return await DebateStore.aload_status(debate_id)
    except Exception as exc:
        logger.debug("[pause_check] Failed to check pause for %s: %s", debate_id, exc)
        return ""


async def check_pause(debate_id: str) -> bool:
    """
    Check if the debate is paused/stopped and raise ``DebatePausedError``.

    Safe to use inside subgraph nodes (no checkpointer required).
    The caller (or wrapper) should catch ``DebatePausedError`` and
    return an appropriate early-exit state.

    Returns False when the debate is NOT paused (normal path).
    """
    status = await _load_status(debate_id)
    if status in ("paused", "stopped"):
        logger.info("[pause_check] %s detected for %s — raising DebatePausedError", status, debate_id)
        raise DebatePausedError(status, debate_id)
    return False


async def check_pause_interrupt(debate_id: str) -> bool:
    """
    Check if the debate is paused/stopped and fire a LangGraph ``interrupt()``.

    Only use this inside **main graph** nodes (where a checkpointer is active).
    ``interrupt()`` suspends the graph and the runner can later resume with
    ``Command(resume=True)``.

    Returns False when the debate is NOT paused (normal path).
    """
    status = await _load_status(debate_id)
    if status in ("paused", "stopped"):
        logger.info("[pause_check] %s detected for %s — firing interrupt", status, debate_id)
        interrupt({"reason": status, "debate_id": debate_id})
    return False

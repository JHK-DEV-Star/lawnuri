"""
WebSocket endpoint for real-time debate event streaming.

Replaces 5-second HTTP polling with push-based updates. Each debate
gets its own set of connected clients. Nodes call ``broadcast()`` to
push events (discussion messages, phase changes, etc.) to all
connected frontends.

Falls back gracefully: if no WebSocket clients are connected,
broadcast() is a no-op.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.utils.logger import logger

router = APIRouter()

# debate_id → set of connected WebSocket clients
_connections: dict[str, set[WebSocket]] = {}


@router.websocket("/api/debate/{debate_id}/ws")
async def debate_ws(websocket: WebSocket, debate_id: str):
    """Accept a WebSocket connection and keep it alive for event streaming."""
    await websocket.accept()
    _connections.setdefault(debate_id, set()).add(websocket)
    logger.info("[ws] Client connected for debate %s (%d total)",
                debate_id, len(_connections[debate_id]))
    try:
        while True:
            # Keep-alive: wait for client pings or disconnection
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _connections.get(debate_id, set()).discard(websocket)
        logger.info("[ws] Client disconnected from debate %s", debate_id)


async def broadcast(debate_id: str, event: dict[str, Any]) -> None:
    """
    Send a JSON event to ALL connected WebSocket clients for a debate.

    Safe to call even if no clients are connected (no-op).
    Dead connections are automatically pruned.
    """
    clients = _connections.get(debate_id, set()).copy()
    if not clients:
        return

    payload = json.dumps(event, ensure_ascii=False, default=str)
    dead: list[WebSocket] = []
    for ws in clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)

    # Prune dead connections
    for ws in dead:
        _connections.get(debate_id, set()).discard(ws)


def get_connection_count(debate_id: str) -> int:
    """Return the number of active WebSocket connections for a debate."""
    return len(_connections.get(debate_id, set()))

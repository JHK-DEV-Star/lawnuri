"""FastAPI application entry point for LawNuri backend."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import config
from app.utils.logger import logger
from app.api import debate_router, rag_router, report_router, settings_router
from app.api.ws import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("LawNuri backend starting up...")

    # Ensure data directories exist
    data_dirs = [
        Path(config.DATA_DIR),
        Path(config.UPLOADS_DIR),
    ]
    for d in data_dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s", d)

    # Initialize database (SQLite / PostgreSQL)
    try:
        from app.db import init_db
        await init_db()
        logger.info("Database initialized (%s).", config.DB_BACKEND)
    except Exception as exc:
        logger.warning("Database init failed (falling back to JSON): %s", exc)

    # Reset stale running/paused debates to stopped (from unclean shutdown)
    try:
        from app.db import get_db_connection
        db = await get_db_connection()
        if db is not None:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM debates WHERE status IN ('running', 'paused')"
            )
            row = await cursor.fetchone()
            stale_count = row[0] if row else 0
            if stale_count > 0:
                await db.execute(
                    "UPDATE debates SET status = 'stopped', updated_at = datetime('now') "
                    "WHERE status IN ('running', 'paused')"
                )
                await db.commit()
                logger.info("Reset %d stale running/paused debates to stopped on startup.", stale_count)
    except Exception as exc:
        logger.warning("Startup stale debate reset failed: %s", exc)

    logger.info("LawNuri backend ready.")

    yield

    logger.info("LawNuri backend shutting down...")

    # Close checkpointer connection
    try:
        from app.graph.checkpointer import close_checkpointer
        await close_checkpointer()
    except Exception as exc:
        logger.debug("Checkpointer shutdown: %s", exc)

    # Mark running debates as stopped via DB
    try:
        from app.db import get_db_connection, close_db
        db = await get_db_connection()
        if db is not None:
            await db.execute(
                "UPDATE debates SET status = 'stopped', updated_at = datetime('now') "
                "WHERE status IN ('running', 'paused')"
            )
            await db.commit()
            logger.info("Stopped running debates on shutdown (via DB).")
            await close_db()
    except Exception as exc:
        logger.warning("DB shutdown cleanup failed: %s", exc)

    import os, json
    debates_dir = os.path.join(config.DATA_DIR, "debates")
    if os.path.isdir(debates_dir):
        for entry in os.listdir(debates_dir):
            state_path = os.path.join(debates_dir, entry, "state.json")
            if not os.path.isfile(state_path):
                continue
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if state.get("status") in ("running", "paused"):
                    state["status"] = "stopped"
                    from datetime import datetime, timezone
                    state["updated_at"] = datetime.now(timezone.utc).isoformat()
                    with open(state_path, "w", encoding="utf-8") as f:
                        json.dump(state, f, ensure_ascii=False, indent=2, default=str)
                    logger.info("Stopped running debate %s on shutdown", entry)
            except Exception as exc:
                logger.warning("Failed to save debate %s on shutdown: %s", entry, exc)


app = FastAPI(
    title="LawNuri API",
    description="Backend API for the LawNuri multi-agent legal debate simulator.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(debate_router)
app.include_router(rag_router)
app.include_router(report_router)
app.include_router(settings_router)
app.include_router(ws_router)


@app.get("/health", tags=["system"])
async def health_check():
    return {"status": "ok", "service": "lawnuri-backend"}

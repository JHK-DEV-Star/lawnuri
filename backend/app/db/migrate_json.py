"""
One-time migration: JSON files → SQLite database.

Reads all existing JSON-based data (settings, debates, anonymization maps,
legal API cache) and inserts them into the SQLite database.

Usage:
    cd backend
    python -m app.db.migrate_json

Safe to re-run (uses INSERT OR IGNORE / INSERT OR REPLACE).
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure backend/ is on sys.path
_backend_dir = str(Path(__file__).resolve().parent.parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from app.config import config
from app.db.database import init_db, close_db, get_db_connection


async def migrate():
    """Migrate legacy JSON file storage to SQLite database."""
    print("=" * 60)
    print("  LawNuri: JSON → SQLite Migration")
    print("=" * 60)

    # Initialize database (creates tables)
    await init_db()
    db = await get_db_connection()
    if db is None:
        print("[ERROR] Database not available.")
        return

    # 1. Migrate settings.json
    settings_path = Path(config.SETTINGS_FILE)
    if settings_path.exists():
        try:
            raw = json.loads(settings_path.read_text(encoding="utf-8"))
            data_json = json.dumps(raw, ensure_ascii=False)
            await db.execute(
                "INSERT OR REPLACE INTO settings (id, data) VALUES (1, ?)",
                (data_json,),
            )
            await db.commit()
            print(f"[OK] Settings migrated from {settings_path}")
        except Exception as exc:
            print(f"[WARN] Failed to migrate settings: {exc}")
    else:
        print("[SKIP] No settings.json found.")

    # 2. Migrate debates
    debates_dir = Path(config.DATA_DIR) / "debates"
    debate_count = 0
    if debates_dir.is_dir():
        for entry in os.listdir(debates_dir):
            state_path = debates_dir / entry / "state.json"
            if not state_path.is_file():
                continue

            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                debate_id = state.get("debate_id", entry)
                state_json = json.dumps(state, ensure_ascii=False, default=str)

                await db.execute(
                    """
                    INSERT OR REPLACE INTO debates
                        (debate_id, situation_brief, analysis, default_model,
                         status, current_round, state, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        debate_id,
                        state.get("situation_brief", ""),
                        json.dumps(state.get("analysis"), default=str) if state.get("analysis") else None,
                        state.get("default_model", ""),
                        state.get("status", "created"),
                        state.get("current_round", 0),
                        state_json,
                        state.get("created_at", ""),
                        state.get("updated_at", ""),
                    ),
                )
                debate_count += 1

                # 2b. Migrate anonymization map
                anon_path = debates_dir / entry / "anonymization_map.json"
                if anon_path.is_file():
                    try:
                        anon_map = json.loads(anon_path.read_text(encoding="utf-8"))
                        await db.execute(
                            "INSERT OR REPLACE INTO anonymization_maps (debate_id, mapping) VALUES (?, ?)",
                            (debate_id, json.dumps(anon_map, ensure_ascii=False)),
                        )
                    except Exception as exc:
                        print(f"  [WARN] Anonymization map for {debate_id}: {exc}")

                # 2c. Migrate legal API cache
                cache_dir = debates_dir / entry / "cache" / "legal"
                if cache_dir.is_dir():
                    cache_count = 0
                    for cache_file in cache_dir.glob("*.json"):
                        try:
                            cache_data = json.loads(cache_file.read_text(encoding="utf-8"))
                            cache_key = cache_file.stem
                            await db.execute(
                                "INSERT OR IGNORE INTO legal_api_cache (cache_key, debate_id, response) VALUES (?, ?, ?)",
                                (cache_key, debate_id, json.dumps(cache_data, ensure_ascii=False, default=str)),
                            )
                            cache_count += 1
                        except Exception:
                            pass
                    if cache_count > 0:
                        print(f"  [OK] {cache_count} cache entries for {debate_id}")

            except Exception as exc:
                print(f"  [WARN] Failed to migrate debate {entry}: {exc}")

        await db.commit()
        print(f"[OK] {debate_count} debates migrated.")
    else:
        print("[SKIP] No debates directory found.")

    # 3. Summary
    print()
    print("--- Migration Summary ---")
    for table in ["settings", "debates", "anonymization_maps", "legal_api_cache",
                   "vector_chunks", "graph_entities"]:
        async with db.execute(f"SELECT COUNT(*) FROM {table}") as cur:
            row = await cur.fetchone()
            count = row[0] if row else 0
        print(f"  {table}: {count} rows")

    await close_db()
    print()
    print("[DONE] Migration complete.")
    print(f"  Database: {config.DB_PATH}")


if __name__ == "__main__":
    asyncio.run(migrate())

"""
Debate API endpoints for LawNuri backend.

Orchestration layer connecting the Flutter frontend to the LangGraph
debate engine. Handles debate creation, analysis, agent generation,
execution control (start/pause/resume/stop), live log streaming,
graph visualization, and human-in-the-loop interventions.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.agents.debater import build_profile_generator_prompt
from app.agents.language import get_language_instruction
from app.api.settings import PRESET_PROVIDERS, settings_mgr, aload_settings
from app.config import config
from app.graph.main_graph import build_debate_graph
from app.models.agent import AgentProfile
from app.models.debate import DebateAnalysis, DebateCreate
from app.models.task import TaskManager, TaskStatus
from app.rag.legal_api import LegalAPIClient
from app.rag.searcher import Searcher
from app.utils.embedding_client import EmbeddingClient
from app.utils.llm_client import LLMClient
from app.utils.logger import logger

router = APIRouter(prefix="/api/debate", tags=["debate"])

_task_mgr = TaskManager()
_running_tasks: dict[str, asyncio.Task] = {}


class InterruptBody(BaseModel):
    """Body for the interrupt endpoint."""
    target_team: str
    content: str
    type: str = "hint"  # "hint" or "document"


class ConfigUpdateBody(BaseModel):
    """Body for mid-session config update."""
    default_model: Optional[str] = None
    agent_overrides: Optional[dict[str, str]] = None


class ExtendBody(BaseModel):
    """Body for extending a debate."""
    additional_rounds: int
    new_evidence: Optional[str] = None
    target_team: Optional[str] = None


def _get_debate_repo():
    """Get SQLite debate repository and connection, or (None, None) if unavailable."""
    try:
        from app.db.database import _connection
        from app.db.sqlite.debate import SQLiteDebateRepo
        if _connection is not None:
            return SQLiteDebateRepo(), _connection
    except Exception:
        pass
    return None, None


class DebateStore:
    """Manages debate state persistence in backend/data/debates/{id}/state.json."""

    @staticmethod
    def get_state_path(debate_id: str) -> str:
        """Return the absolute path to a debate's state.json file."""
        return os.path.join(config.DATA_DIR, "debates", debate_id, "state.json")

    @staticmethod
    def load(debate_id: str) -> dict:
        """
        Load state from disk.

        Raises:
            HTTPException 404 if the debate directory does not exist.
        """
        path = DebateStore.get_state_path(debate_id)
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail=f"Debate {debate_id} not found.")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Failed to load debate state %s: %s", debate_id, exc)
            raise HTTPException(status_code=500, detail="Failed to load debate state.") from exc

    @staticmethod
    def save(debate_id: str, state: dict) -> None:
        """Persist the full state dict to disk."""
        path = DebateStore.get_state_path(debate_id)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def update(debate_id: str, **kwargs: Any) -> dict:
        """Load state, merge kwargs, save, and return the updated state."""
        state = DebateStore.load(debate_id)
        state.update(kwargs)
        state["updated_at"] = datetime.now().isoformat()
        DebateStore.save(debate_id, state)
        return state

    @staticmethod
    def list_all() -> list[dict]:
        """Scan debates directory and return summary of each debate."""
        debates_dir = os.path.join(config.DATA_DIR, "debates")
        if not os.path.isdir(debates_dir):
            return []
        results = []
        for entry in os.listdir(debates_dir):
            state_path = os.path.join(debates_dir, entry, "state.json")
            if not os.path.isfile(state_path):
                continue
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                analysis = state.get("analysis") or {}
                results.append({
                    "debate_id": state.get("debate_id", entry),
                    "situation_brief": (state.get("situation_brief") or "")[:120],
                    "topic": analysis.get("topic", ""),
                    "status": state.get("status", "unknown"),
                    "current_round": state.get("current_round", 0),
                    "max_rounds": state.get("max_rounds", 0),
                    "created_at": state.get("created_at", ""),
                    "updated_at": state.get("updated_at", ""),
                })
            except Exception:
                continue
        results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return results

    @staticmethod
    async def aload(debate_id: str) -> dict:
        """Load state: try SQLite first, fall back to JSON."""
        repo, db = _get_debate_repo()
        if repo and db:
            try:
                state = await repo.load(db, debate_id)
                if state:
                    return state
            except Exception as exc:
                logger.warning("SQLite load failed for debate %s, falling back to JSON: %s", debate_id, exc)
        return DebateStore.load(debate_id)

    @staticmethod
    async def aload_status(debate_id: str) -> str:
        """Load only the debate status (lightweight, no full state deserialization)."""
        repo, db = _get_debate_repo()
        if repo and db:
            try:
                status = await repo.load_status(db, debate_id)
                if status is not None:
                    return status
            except Exception:
                pass
        # Fallback: load from JSON
        state = DebateStore.load(debate_id)
        return state.get("status", "")

    @staticmethod
    async def asave(debate_id: str, state: dict) -> None:
        """Save state: write to JSON and SQLite (dual-write)."""
        DebateStore.save(debate_id, state)
        repo, db = _get_debate_repo()
        if repo and db:
            try:
                await repo.save(db, debate_id, state)
            except Exception as exc:
                logger.warning("SQLite save failed for debate %s: %s", debate_id, exc)

    @staticmethod
    async def aupdate(debate_id: str, **kwargs: Any) -> dict:
        """Load state, merge kwargs, save to both stores, return updated state."""
        state = await DebateStore.aload(debate_id)
        state.update(kwargs)
        state["updated_at"] = datetime.now().isoformat()
        await DebateStore.asave(debate_id, state)
        return state

    @staticmethod
    async def alist_all() -> list[dict]:
        """List all debates: try SQLite first, fall back to JSON."""
        repo, db = _get_debate_repo()
        if repo and db:
            try:
                results = await repo.list_all(db)
                if results:
                    return results
            except Exception as exc:
                logger.warning("SQLite list_all failed, falling back to JSON: %s", exc)
        return DebateStore.list_all()

    @staticmethod
    async def adelete(debate_id: str) -> bool:
        """Delete a debate from both stores."""
        path = DebateStore.get_state_path(debate_id)
        json_deleted = False
        if os.path.isfile(path):
            try:
                import shutil
                debate_dir = os.path.dirname(path)
                shutil.rmtree(debate_dir, ignore_errors=True)
                json_deleted = True
            except Exception:
                pass
        repo, db = _get_debate_repo()
        db_deleted = False
        if repo and db:
            try:
                db_deleted = await repo.delete(db, debate_id)
            except Exception as exc:
                logger.warning("SQLite delete failed for debate %s: %s", debate_id, exc)
        return json_deleted or db_deleted


def _resolve_provider_for_model(model: str, settings: dict) -> tuple[str, dict]:
    """
    Given a model name (possibly in "provider/model" format) and the
    full settings dict, determine which provider owns the model.
    """
    providers = settings.get("llm_providers", {})

    # Handle "provider/model" format from frontend
    explicit_provider = None
    model_name = model
    if "/" in model:
        explicit_provider, model_name = model.split("/", 1)

    # If provider was explicitly specified, use it directly
    if explicit_provider:
        entry = providers.get(explicit_provider, {})
        if entry.get("api_key") and entry.get("enabled", False):
            return explicit_provider, entry

    # Search by model name in presets
    for prov_name, preset in PRESET_PROVIDERS.items():
        if model_name in preset.get("models", {}):
            entry = providers.get(prov_name, {})
            if entry.get("api_key") and entry.get("enabled", False):
                return prov_name, entry
    # Check custom provider
    custom = providers.get("custom", {})
    if custom.get("api_key") and custom.get("enabled", False):
        return "custom", custom
    # Fallback: return first enabled provider
    for prov_name, entry in providers.items():
        if entry.get("api_key") and entry.get("enabled", False):
            return prov_name, entry
    raise HTTPException(status_code=400, detail="No enabled LLM provider found. Configure one in Settings.")


def _get_base_url(provider_name: str, entry: dict) -> str:
    """Return the correct base_url for a given provider."""
    from app.models.llm_config import PRESET_PROVIDERS as MODEL_PRESETS
    if provider_name == "custom":
        return entry.get("base_url", "")
    preset = MODEL_PRESETS.get(provider_name, {})
    return preset.get("base_url", "")


def _build_llm_client(model: str | None = None) -> LLMClient:
    """Build an LLMClient from current settings, optionally overriding the model."""
    settings = settings_mgr.load()

    if not model:
        # Pick the first available model from enabled providers
        providers = settings.get("llm_providers", {})
        for prov_name, entry in providers.items():
            if entry.get("api_key") and entry.get("enabled", False):
                preset = PRESET_PROVIDERS.get(prov_name, {})
                models = preset.get("models", [])
                if prov_name == "custom":
                    model = entry.get("model", "")
                elif models:
                    model = models[0]
                if model:
                    break
        if not model:
            raise HTTPException(status_code=400, detail="No LLM model available.")

    prov_name, entry = _resolve_provider_for_model(model, settings)
    base_url = _get_base_url(prov_name, entry)
    api_key = entry.get("api_key", "")

    # Strip "provider/" prefix from model name if present
    pure_model = model.split("/", 1)[-1] if "/" in model else model

    return LLMClient(
        api_key=api_key, base_url=base_url, model=pure_model, provider=prov_name,
        vertex_project_id=entry.get("project_id", ""),
        vertex_location=entry.get("location", "global"),
    )


def _build_llm_config_dict(model: str | None = None) -> dict:
    """
    Build the llm_config dict expected by build_debate_graph.

    Returns dict with keys: api_key, base_url, model.
    """
    settings = settings_mgr.load()

    if not model:
        providers = settings.get("llm_providers", {})
        for prov_name, entry in providers.items():
            if entry.get("api_key") and entry.get("enabled", False):
                preset = PRESET_PROVIDERS.get(prov_name, {})
                models = preset.get("models", [])
                if prov_name == "custom":
                    model = entry.get("model", "")
                elif models:
                    model = models[0]
                if model:
                    break
        if not model:
            raise HTTPException(status_code=400, detail="No LLM model available.")

    prov_name, entry = _resolve_provider_for_model(model, settings)
    base_url = _get_base_url(prov_name, entry)

    # Strip "provider/" prefix from model name if present
    pure_model = model.split("/", 1)[-1] if "/" in model else model

    return {
        "api_key": entry.get("api_key", ""),
        "base_url": base_url,
        "model": pure_model,
        "provider": prov_name,
        "vertex_project_id": entry.get("project_id", ""),
        "vertex_location": entry.get("location", "global"),
    }


def _build_searcher(debate_id: str) -> Searcher | None:
    """Build a Searcher instance from current settings. Returns None on failure."""
    try:
        settings = settings_mgr.load()
        embedding_cfg = settings.get("embedding", {})
        providers = settings.get("llm_providers", {})

        # Use the embedding provider's API key
        emb_provider = embedding_cfg.get("provider", "openai")
        emb_entry = providers.get(emb_provider, {})
        api_key = emb_entry.get("api_key", "")
        if not api_key:
            return None

        from app.models.llm_config import PRESET_PROVIDERS as MODEL_PRESETS
        preset = MODEL_PRESETS.get(emb_provider, {})
        base_url = preset.get("base_url", "")

        emb_client = EmbeddingClient(
            api_key=api_key,
            base_url=base_url,
            model=embedding_cfg.get("model", "text-embedding-3-small"),
        )
        return Searcher(embedding_client=emb_client, data_dir=config.DATA_DIR)
    except Exception as exc:
        logger.warning("Failed to build Searcher: %s", exc)
        return None


async def _build_legal_api(debate_id: str) -> LegalAPIClient | None:
    """Build a LegalAPIClient from current settings (async, SQLite-aware)."""
    try:
        settings = await aload_settings()
        legal = settings.get("legal_api", {})
        logger.info("[_build_legal_api] legal_api section keys: %s", list(legal.keys()))
        cache_dir = os.path.join(config.DATA_DIR, "debates", debate_id, "cache", "legal")

        law_key = legal.get("law_api_key", "")
        prec_key = legal.get("precedent_api_key", "")
        logger.info(
            "[_build_legal_api] law_key=%d chars, prec_key=%d chars",
            len(law_key), len(prec_key),
        )

        api_key = (law_key or prec_key).strip()
        if not api_key:
            logger.warning("[_build_legal_api] No legal API key found in settings. Cannot create client.")
            return None

        logger.info("[_build_legal_api] Creating LegalAPIClient with key length=%d", len(api_key))
        return LegalAPIClient(
            api_key=api_key,
            law_api_key=law_key.strip() if law_key else "",
            precedent_api_key=prec_key.strip() if prec_key else "",
            cache_dir=cache_dir,
            debate_id=debate_id,
        )
    except Exception as exc:
        logger.warning("[_build_legal_api] Failed to build LegalAPIClient: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Background debate runner
# ---------------------------------------------------------------------------

async def _run_debate(debate_id: str, task_id: str, graph: Any, initial_state: dict) -> None:
    """
    Execute the compiled debate graph as a background coroutine.

    Uses astream() to broadcast node completions via WebSocket in
    real-time.  Falls back gracefully when no WS clients are connected.
    Persists final state to DebateStore after completion.
    """
    from app.api.ws import broadcast

    try:
        _task_mgr.update_task(task_id, status=TaskStatus.PROCESSING, message="Debate running...")

        # Pre-save agent lists before graph execution (in case graph crashes)
        await DebateStore.aupdate(debate_id,
            team_a_agents=initial_state.get("team_a_agents", []),
            team_b_agents=initial_state.get("team_b_agents", []),
            judge_agents=initial_state.get("judge_agents", []),
        )

        # Stream the graph for real-time broadcasts; retrieve authoritative
        # final state from the checkpointer afterwards (not from partial events,
        # because astream yields raw node outputs without applying reducers).
        import datetime as _dt
        import uuid as _uuid

        # Each invocation needs a unique thread_id to avoid stale checkpoint
        # data from a previous run being merged via Annotated[list, operator.add]
        _thread_id = f"{debate_id}_{_uuid.uuid4().hex[:8]}"
        _graph_config = {"configurable": {"thread_id": _thread_id}}
        _was_interrupted = False
        async for event in graph.astream(initial_state, config=_graph_config):
            for node_name, _node_output in event.items():
                if node_name == "__interrupt__":
                    _was_interrupted = True
                    continue
                # Broadcast node completion to WebSocket clients
                await broadcast(debate_id, {
                    "type": "node_complete",
                    "node": node_name,
                    "timestamp": _dt.datetime.now().isoformat(),
                })

        if _was_interrupted:
            # Graph was paused via interrupt() — save progress from checkpoint
            logger.info("Debate %s interrupted (paused/stopped).", debate_id)
            try:
                _snapshot = await graph.aget_state(_graph_config)
                if _snapshot and _snapshot.values:
                    _int_state = await DebateStore.aload(debate_id)
                    _sv = _snapshot.values
                    # Persist progress made before interrupt
                    _int_state["debate_log"] = _sv.get("debate_log", _int_state.get("debate_log", []))
                    _int_state["all_evidences"] = _sv.get("all_evidences", _int_state.get("all_evidences", []))
                    _int_state["current_round"] = _sv.get("round", _int_state.get("current_round", 0))
                    _int_state["judge_notes"] = _sv.get("judge_notes", _int_state.get("judge_notes", []))
                    _int_state["internal_discussions"] = _sv.get("internal_discussions", _int_state.get("internal_discussions", []))
                    _int_state["judge_qa_log"] = _sv.get("judge_qa_log", _int_state.get("judge_qa_log", []))
                    _int_state["team_a_state"] = _sv.get("team_a_state", _int_state.get("team_a_state", {}))
                    _int_state["team_b_state"] = _sv.get("team_b_state", _int_state.get("team_b_state", {}))
                    _int_state["blacklisted_evidence"] = _sv.get("blacklisted_evidence", _int_state.get("blacklisted_evidence", []))
                    _int_state["agent_memories"] = _sv.get("agent_memories", _int_state.get("agent_memories", {}))
                    _int_state["current_team"] = _sv.get("current_team", _int_state.get("current_team", "team_a"))
                    _int_state["verdicts"] = _sv.get("verdicts", _int_state.get("verdicts", []))
                    _int_state["early_stop_votes"] = _sv.get("early_stop_votes", _int_state.get("early_stop_votes", []))
                    _int_state["judge_improvement_feedback"] = _sv.get("judge_improvement_feedback", _int_state.get("judge_improvement_feedback", {}))
                    _int_state["max_rounds"] = _sv.get("max_rounds", _int_state.get("max_rounds", 10))
                    _int_state["status"] = "paused"
                    await DebateStore.asave(debate_id, _int_state)
                else:
                    await DebateStore.aupdate(debate_id, status="paused")
            except Exception:
                await DebateStore.aupdate(debate_id, status="paused")
            _task_mgr.complete_task(task_id, result={"status": "paused"})
            return

        # Retrieve the fully-reduced final state from the checkpointer
        final_state = dict(initial_state)
        try:
            _snapshot = await graph.aget_state(_graph_config)
            if _snapshot and _snapshot.values:
                final_state.update(_snapshot.values)
        except ValueError:
            # No checkpointer — fall back to initial_state (unchanged keys preserved)
            logger.warning("Debate %s: no checkpointer, using initial_state as final", debate_id)

        # Persist the final state
        state = await DebateStore.aload(debate_id)
        state["debate_log"] = final_state.get("debate_log", [])
        state["all_evidences"] = final_state.get("all_evidences", [])
        state["verdicts"] = final_state.get("verdicts", [])
        state["current_round"] = final_state.get("round", state.get("current_round", 0))
        state["max_rounds"] = final_state.get("max_rounds", state.get("max_rounds", 10))
        state["judge_notes"] = final_state.get("judge_notes", [])
        state["early_stop_votes"] = final_state.get("early_stop_votes", [])
        state["blacklisted_evidence"] = final_state.get("blacklisted_evidence", [])
        state["agent_memories"] = final_state.get("agent_memories", {})
        state["current_team"] = final_state.get("current_team", "team_a")
        state["judge_improvement_feedback"] = final_state.get("judge_improvement_feedback", {})

        # Determine final status
        graph_status = final_state.get("status", "completed")
        if graph_status == "paused":
            state["status"] = "paused"
        elif final_state.get("verdicts"):
            state["status"] = "completed"
        else:
            state["status"] = graph_status

        # Preserve team states for potential resume
        state["team_a_state"] = final_state.get("team_a_state", {})
        state["team_b_state"] = final_state.get("team_b_state", {})

        # Persist agent lists for graph visualization
        # Use initial_state (not final_state) because LangGraph may not
        # include unchanged keys in the final output
        state["team_a_agents"] = initial_state.get("team_a_agents", [])
        state["team_b_agents"] = initial_state.get("team_b_agents", [])
        state["judge_agents"] = initial_state.get("judge_agents", [])

        # Persist internal discussions for graph visualization
        state["internal_discussions"] = final_state.get("internal_discussions", [])

        # Persist judge Q&A log
        state["judge_qa_log"] = final_state.get("judge_qa_log", [])

        await DebateStore.asave(debate_id, state)

        # Auto-generate report after debate completion
        if state["status"] == "completed":
            try:
                from app.api.report import _generate_report
                report = await _generate_report(state)
                state["report"] = report
                await DebateStore.asave(debate_id, state)
                logger.info("Auto-generated report for debate %s", debate_id)
            except Exception as e:
                logger.warning("Auto report generation failed for %s: %s", debate_id, e)

        _task_mgr.complete_task(task_id, result={"status": state["status"]})

        logger.info("Debate %s finished with status: %s", debate_id, state["status"])

    except asyncio.CancelledError:
        # Task was cancelled (e.g. pause_debate called task.cancel()).
        # Save whatever progress we can from the checkpoint before exiting.
        logger.info("Debate %s task cancelled — saving checkpoint state.", debate_id)
        try:
            _snapshot = await graph.aget_state(_graph_config)
            if _snapshot and _snapshot.values:
                _canc_state = await DebateStore.aload(debate_id)
                _sv = _snapshot.values
                _canc_state["debate_log"] = _sv.get("debate_log", _canc_state.get("debate_log", []))
                _canc_state["all_evidences"] = _sv.get("all_evidences", _canc_state.get("all_evidences", []))
                _canc_state["current_round"] = _sv.get("round", _canc_state.get("current_round", 0))
                _canc_state["judge_notes"] = _sv.get("judge_notes", _canc_state.get("judge_notes", []))
                _canc_state["internal_discussions"] = _sv.get("internal_discussions", _canc_state.get("internal_discussions", []))
                _canc_state["judge_qa_log"] = _sv.get("judge_qa_log", _canc_state.get("judge_qa_log", []))
                _canc_state["team_a_state"] = _sv.get("team_a_state", _canc_state.get("team_a_state", {}))
                _canc_state["team_b_state"] = _sv.get("team_b_state", _canc_state.get("team_b_state", {}))
                _canc_state["blacklisted_evidence"] = _sv.get("blacklisted_evidence", _canc_state.get("blacklisted_evidence", []))
                _canc_state["agent_memories"] = _sv.get("agent_memories", _canc_state.get("agent_memories", {}))
                _canc_state["current_team"] = _sv.get("current_team", _canc_state.get("current_team", "team_a"))
                _canc_state["verdicts"] = _sv.get("verdicts", _canc_state.get("verdicts", []))
                _canc_state["early_stop_votes"] = _sv.get("early_stop_votes", _canc_state.get("early_stop_votes", []))
                _canc_state["judge_improvement_feedback"] = _sv.get("judge_improvement_feedback", _canc_state.get("judge_improvement_feedback", {}))
                _canc_state["max_rounds"] = _sv.get("max_rounds", _canc_state.get("max_rounds", 10))
                _canc_state["status"] = "paused"
                await DebateStore.asave(debate_id, _canc_state)
            else:
                await DebateStore.aupdate(debate_id, status="paused")
        except Exception:
            logger.debug("Debate %s: checkpoint save after cancel failed", debate_id, exc_info=True)
            try:
                await DebateStore.aupdate(debate_id, status="paused")
            except Exception:
                pass
        _task_mgr.complete_task(task_id, result={"status": "paused"})

    except Exception as exc:
        logger.error("Debate %s failed: %s", debate_id, exc, exc_info=True)
        try:
            await DebateStore.aupdate(debate_id, status="stopped")
        except Exception:
            pass
        _task_mgr.fail_task(task_id, error=str(exc))
    finally:
        _running_tasks.pop(debate_id, None)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/list")
async def list_debates():
    """Return a summary list of all saved debate sessions."""
    return await DebateStore.alist_all()


@router.delete("/{debate_id}")
async def delete_debate(debate_id: str):
    """Delete a debate and all associated data (files + DB)."""
    try:
        await DebateStore.adelete(debate_id)
        return {"status": "ok", "message": f"Debate {debate_id} deleted"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Debate {debate_id} not found")
    except Exception as exc:
        logger.error("Failed to delete debate %s: %s", debate_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/create")
async def create_debate(body: DebateCreate):
    """Create a new debate session and return the debate_id."""
    debate_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    settings = await aload_settings()
    debate_settings = settings.get("debate", {})

    state = {
        "debate_id": debate_id,
        "situation_brief": body.situation_brief,
        "default_model": body.default_model,
        "analysis": None,
        "agents": [],
        "status": "created",
        "min_rounds": debate_settings.get("min_rounds", 3),
        "max_rounds": debate_settings.get("max_rounds", 10),
        "current_round": 0,
        "debate_log": [],
        "all_evidences": [],
        "verdicts": [],
        "judge_notes": [],
        "early_stop_votes": [],
        "team_a_state": {},
        "team_b_state": {},
        "user_interrupt": None,
        "report": None,
        "created_at": now,
        "updated_at": now,
    }

    _settings = settings_mgr.load()
    _debate_cfg = _settings.get("debate", {})
    state["team_a_name"] = _debate_cfg.get("team_a_name", "Team A")
    state["team_b_name"] = _debate_cfg.get("team_b_name", "Team B")

    debate_dir = os.path.join(config.DATA_DIR, "debates", debate_id)
    for sub in ["rag/common/chroma", "rag/team_a/chroma", "rag/team_b/chroma", "cache/legal"]:
        Path(os.path.join(debate_dir, sub)).mkdir(parents=True, exist_ok=True)

    await DebateStore.asave(debate_id, state)
    logger.info("Created debate %s", debate_id)

    return {"debate_id": debate_id, "status": "created"}


# ---------------------------------------------------------------------------
# Analysis step prompts (split into 3 calls for better quality on smaller models)
# ---------------------------------------------------------------------------

def _build_analysis_step1_prompt(situation: str, uploaded_section: str, language: str) -> str:
    """Step 1: Parties + Timeline + Causal Chains."""
    prompt = f"""\
You are a senior legal analyst. Analyze the following situation.
Focus on identifying the parties, reconstructing the timeline, and tracing causal chains.

## Step 1: Identify ALL Parties
- Who are the people/organizations involved?
- What is each party's role in the situation?

## Step 2: Reconstruct the Timeline (CRITICAL — affects entire debate accuracy)
- List ALL actions/events in chronological order
- For each event: WHO did WHAT, WHEN, and its legal SIGNIFICANCE
- If a date is not mentioned, mark it as "unknown"

### RELATIVE TIME RESOLUTION (MANDATORY — FAILURE TO COMPLY WILL PRODUCE INCORRECT LEGAL ANALYSIS)
When the situation uses ANY relative time expression, you MUST:
1. Identify the BASE date/year that the relative expression refers to
2. CALCULATE the resulting absolute date/year by adding/subtracting the offset
3. Output ONLY the calculated absolute date — NEVER output the base date for both events

Common relative expressions (in any language):
- "X years/months/days later/after" → base_date + X
- "the following year / next year" → base_year + 1
- "about/approximately X years later" → base_year + X (note as approximate)
- "subsequently / thereafter / since then" → determine from context

Examples:
- "Contract in 2011, terminated about 2 years later" → contract: 2011, termination: approximately 2013
- "Hired Jan 2020, fired 3 months later" → hired: Jan 2020, fired: Apr 2020
- "Accident in 2018, sued the following year" → accident: 2018, lawsuit: 2019
- "Signed in 2015, renewed 1 year later, dispute 6 months after renewal"
  → signed: 2015, renewed: 2016, dispute: mid-2016

COMMON ERROR TO AVOID: Assigning the SAME year to all events when relative offsets clearly
indicate different dates. Each event MUST have its own independently calculated date.

### TIMELINE VERIFICATION (perform before outputting)
After constructing the timeline, verify:
- Are all dates in chronological order?
- Does each event with a relative time expression have a DIFFERENT date from its reference event?
- If two events have the same date, is that actually correct or did you fail to apply an offset?

## Step 3: Trace Causal Chains
- How did each action lead to the next?
- What are the cause-and-effect relationships?
- What was the critical turning point?

## ANTI-HALLUCINATION RULES (CRITICAL)
- ONLY extract information that is EXPLICITLY stated or DIRECTLY implied in the situation brief or uploaded evidence
- If a date, name, or specific detail is NOT mentioned, mark it as "unknown" — NEVER invent one
- For the timeline: only include events actually described. Do NOT fill gaps with assumed events
- For parties: only name parties that are explicitly mentioned or clearly implied
- NEVER assume legal outcomes, court decisions, or procedural steps that are not stated

Situation (User's perspective and direction):
{situation}{uploaded_section}

Output ONLY valid JSON:
{{
    "parties": [
        {{"id": "party_a", "name": "...", "role": "...", "description": "..."}},
        {{"id": "party_b", "name": "...", "role": "...", "description": "..."}}
    ],
    "timeline": [
        {{"date": "...", "actor": "party_a or party_b", "action": "...", "significance": "..."}}
    ],
    "causal_chain": ["action1 -> consequence1 -> action2 -> consequence2"]
}}
"""
    prompt += get_language_instruction(language)
    return prompt


def _build_analysis_step2_prompt(
    situation: str, uploaded_section: str, step1_result: dict, language: str,
) -> str:
    """Step 2: Key Facts extraction, referencing step 1 results."""
    _step1_ref = json.dumps(step1_result, ensure_ascii=False, indent=2)
    prompt = f"""\
You are a senior legal analyst. Extract ALL key facts from the following situation and evidence.

## Reference: Previous analysis (parties, timeline, causal chain)
{_step1_ref}

## Your Task: Identify Key Facts (BE MAXIMALLY SPECIFIC)
Extract ALL factual details from the situation. Each fact must be a concrete, specific statement.

For each fact, include:
- The SPECIFIC action or event (what exactly happened)
- WHO was involved (by name/role)
- WHEN it happened (exact or calculated date)
- Any AMOUNTS, QUANTITIES, or MEASURABLE details (money, duration, distance, etc.)
- The LEGAL SIGNIFICANCE of this fact (why it matters for the case)

Rate each fact:
- "critical": Directly determines a legal element (constitutive requirement) of the claim
- "high": Strongly supports or weakens a key argument
- "medium": Provides useful context but does not determine outcome

Mark "disputed": true only if the situation brief presents conflicting accounts of the same event.

DO NOT summarize or generalize. Extract facts as specifically as stated in the situation.
BAD: "There was a contract dispute"
GOOD: "Party A signed a commercial lease agreement on 2023-03-15 for $2,400/month with a 12-month term"

## ANALYSIS APPROACH
1. The user's situation brief provides DIRECTION and PERSPECTIVE — what they want to argue.
2. The uploaded evidence documents (if any) provide FACTUAL DETAILS — what actually happened.
3. You MUST extract ALL concrete facts from BOTH sources.

## CRITICAL: FACTS ONLY, NO ABSTRACTION
- Extract facts at the CONCRETE level: WHO did WHAT, WHEN, WHERE, HOW MUCH
- DO NOT summarize groups of facts into abstract categories
- DO NOT replace specific facts with legal concepts
- BAD: "There were issues with identity verification"
- GOOD: "The bank clerk processed the loan application on March 15, 2023 without requiring a government-issued photo ID from the applicant"
- BAD: "The contract had procedural defects"
- GOOD: "Section 4.2 of the lease agreement was left unsigned by the tenant, and the notarization stamp is dated two weeks after the claimed signing date"

## KEY FACTS EXTRACTION — MAXIMIZE CLUES FROM EVIDENCE
For EVERY factual claim in the situation brief AND evidence documents:
- Include it as a separate key_fact entry
- Preserve the SPECIFIC details (dates, amounts, names, actions)
- Do NOT merge multiple facts into one summary
- Do NOT omit facts that seem minor — they may be legally critical

EVIDENCE MINING (CRITICAL):
- Read uploaded documents LINE BY LINE to find every possible clue
- Dates, names, amounts, contract terms, communications, procedural steps — ALL are relevant
- Implicit facts matter: if a document mentions "no record of X", that ABSENCE is a key fact
- Contradictions between the user's account and the evidence are CRITICAL to note
- Even metadata clues (document dates, sender/receiver, reference numbers) should be extracted
- The more facts you extract, the better the debate will be — DO NOT be selective, be EXHAUSTIVE

## ANTI-HALLUCINATION RULES (CRITICAL)
- ONLY extract information that is EXPLICITLY stated or DIRECTLY implied in the situation brief or uploaded evidence
- If a date, name, or specific detail is NOT mentioned, mark it as "unknown" — NEVER invent one
- For key_facts: mark "disputed": true if sources present conflicting accounts
- NEVER assume legal outcomes, court decisions, or procedural steps that are not stated

Situation (User's perspective and direction):
{situation}{uploaded_section}

Output ONLY valid JSON:
{{
    "key_facts": [
        {{"fact": "...", "disputed": true/false, "importance": "critical/high/medium"}}
    ]
}}
"""
    prompt += get_language_instruction(language)
    return prompt


def _build_analysis_step3_prompt(
    situation: str, step1_result: dict, step2_result: dict, language: str,
) -> str:
    """Step 3: Positions, strategy, and issues — referencing all previous results."""
    _prev_ref = json.dumps({**step1_result, **step2_result}, ensure_ascii=False, indent=2)
    prompt = f"""\
You are a senior legal analyst. Based on the analysis below, define the debate positions, key issues, and strategic warnings.

## Reference: Full analysis so far
{_prev_ref}

## Your Task

### Define Positions
- topic: A concise title for the core debate
- opinion_a: The position SUPPORTING the person who described this situation
- opinion_b: The OPPOSING position
- focus_points: What should each team strategically FOCUS on?

RULES for opinion_a vs opinion_b:
- opinion_a always represents the perspective of the person who wrote this situation
- opinion_b always represents the opposing side
- If the situation describes a dispute, opinion_a = the person filing/complaining
- If unclear, opinion_a = the position most sympathetic to the situation as described

### Strategic Warnings
- team_a_cautions: Weak points and traps to avoid (2-4 items)
- team_b_cautions: Weak points and traps to avoid (2-4 items)

### Key Issues
- List the core legal issues that will determine the outcome

### Missing Information
- List any details that would help the analysis but are not present in the situation

## NO EMOTIONAL OR SUBJECTIVE OPINIONS
- opinion_a and opinion_b must be based ONLY on verified facts, not emotional framing
- DO NOT include language like "the victim was wronged" or "the defendant acted maliciously" — these are emotional, not factual
- BAD: "The innocent party deserves protection" (emotional appeal)
- GOOD: "A contract signed without the guarantor's consent is voidable under Article 107" (fact-based legal position)
- BAD: "The defendant maliciously defrauded the plaintiff" (emotional judgment)
- GOOD: "The defendant transferred $50,000 from the joint account on June 3 without co-signer authorization" (verifiable fact)
- Opinions must be positions that can be PROVEN or DISPROVEN with evidence
- If a fact is disputed, note it as disputed — do NOT assume one side's emotional framing

Situation (for reference):
{situation[:3000]}

Output ONLY valid JSON:
{{
    "topic": "...",
    "opinion_a": "...",
    "opinion_b": "...",
    "key_issues": ["issue1", "issue2"],
    "team_a_cautions": ["caution1", "caution2"],
    "team_b_cautions": ["caution1", "caution2"],
    "focus_points": {{
        "team_a": "what team a should focus on",
        "team_b": "what team b should focus on"
    }},
    "missing_information": ["detail1 that would help the analysis"]
}}
"""
    prompt += get_language_instruction(language)
    return prompt


async def _achat_json_with_pdf(
    llm, prompt: str, pdf_parts: list[dict], max_tokens: int = 8192,
) -> dict:
    """Call achat_json with optional PDF multimodal parts."""
    if pdf_parts:
        content_array: list[dict] = [{"type": "text", "text": prompt}]
        for pdf in pdf_parts:
            content_array.append({
                "type": "pdf",
                "filename": pdf["filename"],
                "data": pdf["data"],
            })
        multimodal_messages = [{"role": "user", "content": content_array}]
        try:
            return await llm.achat_json(
                multimodal_messages, temperature=0.3, max_tokens=max_tokens,
            )
        except Exception as pdf_exc:
            logger.warning("[analyze] Native PDF failed, falling back to text: %s", pdf_exc)
            return await llm.achat_json(
                [{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=max_tokens,
            )
    else:
        return await llm.achat_json(
            [{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=max_tokens,
        )


@router.post("/{debate_id}/analyze")
async def analyze_debate(debate_id: str):
    """Use LLM to analyze the situation brief and extract debate structure."""
    state = await DebateStore.aload(debate_id)

    llm = _build_llm_client(state.get("default_model"))

    # Read uploaded evidence files (if any)
    from app.utils.file_parser import FileParser
    from pathlib import Path

    uploaded_section = ""
    pdf_parts: list[dict] = []       # PDF binaries for multimodal pass-through
    uploaded_texts: list[str] = []   # Text extracted from non-PDF files
    uploads_dir = Path(f"data/debates/{debate_id}/uploads")
    if uploads_dir.exists():
        for pool_dir in sorted(uploads_dir.iterdir()):
            if not pool_dir.is_dir():
                continue
            if pool_dir.name != "common":
                continue  # team_a/team_b files handled separately below
            for file_path in sorted(pool_dir.iterdir()):
                ext = file_path.suffix.lower()
                if ext not in FileParser.SUPPORTED_EXTENSIONS:
                    continue
                try:
                    if ext == ".pdf":
                        # Keep PDF as binary for multimodal LLM
                        pdf_bytes = FileParser.read_pdf_bytes(file_path)
                        pdf_parts.append({
                            "filename": file_path.name,
                            "data": pdf_bytes,
                            "pool": pool_dir.name,
                        })
                        # Also extract text as fallback
                        text = FileParser.extract_text(file_path)
                        if text.strip():
                            uploaded_texts.append(
                                f"[File: {file_path.name} (pool: {pool_dir.name})]\n{text}"
                            )
                    else:
                        text = FileParser.extract_text(file_path)
                        if text.strip():
                            uploaded_texts.append(
                                f"[File: {file_path.name} (pool: {pool_dir.name})]\n{text}"
                            )
                except Exception as e:
                    logger.warning("Failed to parse uploaded file %s: %s", file_path, e)
        if uploaded_texts:
            combined = "\n\n---\n\n".join(uploaded_texts)
            uploaded_section = f"\n\nUploaded Evidence Documents:\n{combined}\n"
            logger.info("Included %d uploaded files in analysis (%d chars)", len(uploaded_texts), len(combined))
        if pdf_parts:
            logger.info("Found %d PDF files for multimodal pass-through", len(pdf_parts))

    settings = settings_mgr.load()
    language = settings.get("debate", {}).get("language", "ko")
    situation = state["situation_brief"]

    state["status"] = "analyzing"
    await DebateStore.asave(debate_id, state)

    try:
        # If evidence is too long, pre-extract facts in chunks then rebuild prompt
        _CHUNK_THRESHOLD = 40000  # chars (~10k tokens)
        _total_input = len(situation) + len(uploaded_section)

        if _total_input > _CHUNK_THRESHOLD and uploaded_section:
            _chunk_size = 30000
            _evidence_text = uploaded_section
            _chunks: list[str] = []
            for i in range(0, len(_evidence_text), _chunk_size):
                _chunks.append(_evidence_text[i:i + _chunk_size])

            accumulated_facts: list[str] = []
            for ci, chunk in enumerate(_chunks):
                _prev_facts = "\n".join(accumulated_facts) if accumulated_facts else "(none yet)"
                _extract_prompt = (
                    f"You are extracting facts from evidence documents for a legal case analysis.\n\n"
                    f"User's situation:\n{situation[:3000]}\n\n"
                    f"Facts extracted so far from previous chunks:\n{_prev_facts}\n\n"
                    f"Evidence chunk {ci+1}/{len(_chunks)}:\n{chunk}\n\n"
                    f"Extract ALL concrete facts from this chunk. For each fact:\n"
                    f"- WHO did WHAT, WHEN, WHERE, HOW MUCH\n"
                    f"- Preserve specific details (dates, amounts, names, actions)\n"
                    f"- Note absences ('no record of X') as facts too\n"
                    f"- Note contradictions with the user's account\n"
                    f"- Do NOT repeat facts already extracted above\n\n"
                    f"Output a numbered list of new facts found in this chunk only."
                )
                _extract_prompt += get_language_instruction(language)
                _chunk_facts = await llm.achat(
                    [{"role": "user", "content": _extract_prompt}],
                    temperature=0.2,
                    max_tokens=4096,
                )
                accumulated_facts.append(f"[Chunk {ci+1}]\n{_chunk_facts}")
                logger.info("[analyze] Evidence chunk %d/%d processed (%d chars)", ci+1, len(_chunks), len(_chunk_facts))

            # Use extracted facts instead of raw evidence for step prompts
            _facts_summary = "\n\n".join(accumulated_facts)
            uploaded_section = f"\n\nExtracted Facts from Uploaded Evidence:\n{_facts_summary}\n"
            logger.info("[analyze] Chunked extraction complete: %d chunks, %d total fact chars", len(_chunks), len(_facts_summary))

        # --- 3-step analysis ---
        # Call 1: Parties + Timeline + Causal Chain
        step1_prompt = _build_analysis_step1_prompt(situation, uploaded_section, language)
        step1_result = await _achat_json_with_pdf(llm, step1_prompt, pdf_parts, max_tokens=8192)
        logger.info("[analyze] Step 1 complete (parties/timeline/causal_chain) for debate %s", debate_id)

        # Call 2: Key Facts (references step 1, includes evidence)
        step2_prompt = _build_analysis_step2_prompt(situation, uploaded_section, step1_result, language)
        step2_result = await llm.achat_json(
            [{"role": "user", "content": step2_prompt}],
            temperature=0.3, max_tokens=12288,
        )
        logger.info("[analyze] Step 2 complete (key_facts: %d) for debate %s",
                     len(step2_result.get("key_facts", [])), debate_id)

        # Call 3: Positions + Strategy (references step 1 + 2)
        step3_prompt = _build_analysis_step3_prompt(situation, step1_result, step2_result, language)
        step3_result = await llm.achat_json(
            [{"role": "user", "content": step3_prompt}],
            temperature=0.3, max_tokens=8192,
        )
        logger.info("[analyze] Step 3 complete (positions/strategy) for debate %s", debate_id)

        # Merge all steps
        analysis = {**step1_result, **step2_result, **step3_result}

    except Exception as exc:
        logger.error("Analysis failed for debate %s: %s", debate_id, exc)
        await DebateStore.aupdate(debate_id, status="created")
        raise HTTPException(status_code=502, detail=f"LLM analysis failed: {exc}") from exc

    validated = DebateAnalysis(**analysis)
    analysis_data = validated.model_dump()
    state["analysis"] = analysis_data
    # Also save individual fields for LangGraph state compatibility
    state["topic"] = analysis_data.get("topic", "")
    state["opinion_a"] = analysis_data.get("opinion_a", "")
    state["opinion_b"] = analysis_data.get("opinion_b", "")
    state["key_issues"] = analysis_data.get("key_issues", [])
    state["team_a_cautions"] = analysis_data.get("team_a_cautions", [])
    state["team_b_cautions"] = analysis_data.get("team_b_cautions", [])
    state["parties"] = analysis_data.get("parties", [])
    state["timeline"] = analysis_data.get("timeline", [])
    state["causal_chain"] = analysis_data.get("causal_chain", [])
    state["key_facts"] = analysis_data.get("key_facts", [])
    state["focus_points"] = analysis_data.get("focus_points", {})
    state["missing_information"] = analysis_data.get("missing_information", [])
    # --- Team-specific file analysis (team_a / team_b only) ---
    if uploads_dir.exists():
        for team_id in ["team_a", "team_b"]:
            team_upload_dir = uploads_dir / team_id
            if not team_upload_dir.exists():
                continue
            team_texts: list[str] = []
            for file_path in sorted(team_upload_dir.iterdir()):
                ext = file_path.suffix.lower()
                if ext not in FileParser.SUPPORTED_EXTENSIONS:
                    continue
                try:
                    text = FileParser.extract_text(file_path)
                    if text.strip():
                        team_texts.append(f"[File: {file_path.name}]\n{text}")
                except Exception as e:
                    logger.warning("Failed to parse team file %s: %s", file_path, e)

            if not team_texts:
                continue

            combined_team = "\n\n---\n\n".join(team_texts)
            team_extract_prompt = (
                f"You are a legal analyst. Extract ALL key facts from the following evidence documents.\n"
                f"These documents are confidential evidence for {team_id}.\n\n"
                f"Case topic: {situation[:1000]}\n\n"
                f"Evidence:\n{combined_team}\n\n"
                f"Extract every concrete fact: WHO did WHAT, WHEN, WHERE, HOW MUCH.\n"
                f"Output a numbered list of facts."
            )
            team_extract_prompt += get_language_instruction(language)

            try:
                team_facts = await llm.achat(
                    [{"role": "user", "content": team_extract_prompt}],
                    temperature=0.3,
                    max_tokens=4096,
                )
            except Exception as e:
                logger.error("[analyze] Team %s file analysis failed: %s", team_id, e)
                continue

            team_state_key = f"{team_id}_state"
            team_state = state.get(team_state_key, {})
            if team_state is None:
                team_state = {}
            extra = list(team_state.get("extra_evidence", []))
            extra.append({
                "content": team_facts,
                "source_type": "team_uploaded_analysis",
                "source_detail": f"팀 전용 증거 분석 ({len(team_texts)} files)",
                "submitted_by": team_id,
                "round": 0,
                "speaker": "system",
                "type": "evidence",
            })
            team_state["extra_evidence"] = extra
            state[team_state_key] = team_state

            logger.info(
                "[analyze] Team %s evidence analyzed: %d files, %d chars extracted",
                team_id, len(team_texts), len(team_facts),
            )

    state["status"] = "created"
    await DebateStore.asave(debate_id, state)

    logger.info("Analysis complete for debate %s: topic=%s", debate_id, validated.topic)
    return {"analysis": validated.model_dump()}


@router.post("/{debate_id}/agents/generate")
async def generate_agents(debate_id: str):
    """Auto-generate 13 agent profiles (5+5 debaters, 3 judges) using LLM."""
    state = await DebateStore.aload(debate_id)

    analysis = state.get("analysis")
    if not analysis:
        raise HTTPException(status_code=400, detail="Run analysis first before generating agents.")

    settings = await aload_settings()
    debate_settings = settings.get("debate", {})
    team_size = debate_settings.get("team_size", 5)
    judge_count = debate_settings.get("judge_count", 3)

    llm = _build_llm_client(state.get("default_model"))

    language = debate_settings.get("language", "ko")

    team_a_name = state.get("team_a_name", "Team A")
    team_b_name = state.get("team_b_name", "Team B")

    prompt = build_profile_generator_prompt(
        situation_brief=state["situation_brief"],
        analysis=analysis,
        team_size=team_size,
        judge_count=judge_count,
        language=language,
        team_a_name=team_a_name,
        team_b_name=team_b_name,
    )

    try:
        profiles_raw = await llm.achat_json(
            [{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4096,
        )
    except Exception as exc:
        logger.error("Agent generation failed for debate %s: %s", debate_id, exc)
        raise HTTPException(status_code=502, detail=f"LLM agent generation failed: {exc}") from exc

    agents: list[dict] = []

    for i, raw in enumerate(profiles_raw.get("team_a", [])):
        raw.setdefault("agent_id", f"team_a_{i+1}")
        raw.setdefault("role", "debater")
        raw.setdefault("team", "team_a")
        profile = AgentProfile(**raw)
        agents.append(profile.model_dump())

    for i, raw in enumerate(profiles_raw.get("team_b", [])):
        raw.setdefault("agent_id", f"team_b_{i+1}")
        raw.setdefault("role", "debater")
        raw.setdefault("team", "team_b")
        profile = AgentProfile(**raw)
        agents.append(profile.model_dump())

    for i, raw in enumerate(profiles_raw.get("judges", [])):
        raw.setdefault("agent_id", f"judge_{i+1}")
        raw.setdefault("role", "judge")
        raw.setdefault("team", None)
        profile = AgentProfile(**raw)
        agents.append(profile.model_dump())

    state["agents"] = agents
    state["status"] = "ready"
    await DebateStore.asave(debate_id, state)

    logger.info(
        "Generated %d agents for debate %s (%d team_a, %d team_b, %d judges)",
        len(agents), debate_id,
        len(profiles_raw.get("team_a", [])),
        len(profiles_raw.get("team_b", [])),
        len(profiles_raw.get("judges", [])),
    )
    return {"agents": agents}


@router.get("/{debate_id}/agents")
async def get_agents(debate_id: str):
    """Return all agent profiles for a debate."""
    state = await DebateStore.aload(debate_id)
    # Combine agents from all three team lists (state splits them)
    agents = (
        state.get("team_a_agents", [])
        + state.get("team_b_agents", [])
        + state.get("judge_agents", [])
    )
    if not agents:
        agents = state.get("agents", [])
    return {"agents": agents}


@router.put("/{debate_id}/agents/{agent_id}")
async def update_agent(debate_id: str, agent_id: str, body: dict):
    """Update a single agent's profile fields."""
    state = await DebateStore.aload(debate_id)

    # Only allow edits when debate is paused or stopped
    status = state.get("status", "")
    if status in ("running", "extended"):
        raise HTTPException(status_code=400, detail="Cannot edit agents while debate is running. Pause first.")

    found_agent = None
    for list_key in ["team_a_agents", "team_b_agents", "judge_agents"]:
        for agent in state.get(list_key, []):
            if agent.get("agent_id") == agent_id:
                for field in ("name", "specialty", "personality", "debate_style", "background", "llm_override"):
                    if field in body:
                        agent[field] = body[field]
                found_agent = agent
                break
        if found_agent:
            break

    # Fallback: legacy "agents" key
    if not found_agent:
        for agent in state.get("agents", []):
            if agent.get("agent_id") == agent_id:
                for field in ("name", "specialty", "personality", "debate_style", "background", "llm_override"):
                    if field in body:
                        agent[field] = body[field]
                found_agent = agent
                break

    if not found_agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found.")

    await DebateStore.asave(debate_id, state)
    return {"status": "ok", "agent": found_agent}


@router.post("/{debate_id}/start")
async def start_debate(debate_id: str):
    """Start the debate execution as a background task."""
    state = await DebateStore.aload(debate_id)

    if not state.get("analysis"):
        raise HTTPException(status_code=400, detail="Analysis required before starting.")
    if not state.get("agents"):
        raise HTTPException(status_code=400, detail="Agent profiles required before starting.")
    if debate_id in _running_tasks:
        raise HTTPException(status_code=409, detail="Debate is already running.")
    if _running_tasks:
        _other_id = next(iter(_running_tasks))
        raise HTTPException(
            status_code=409,
            detail=f"Another debate ({_other_id}) is already running. Stop it first.",
        )

    try:
        _settings = await aload_settings()
        _legal = _settings.get("legal_api", {})
        _law_key = _legal.get("law_api_key", "")
        _prec_key = _legal.get("precedent_api_key", "")
        if not _law_key and not _prec_key:
            raise HTTPException(
                status_code=400,
                detail="법령/판례 API key(OC)가 설정되지 않았습니다. Settings에서 입력해주세요.",
            )
    except HTTPException:
        raise
    except Exception:
        pass  # If settings can't load, let it proceed and fail later

    default_model = state.get("default_model")
    llm_config = _build_llm_config_dict(default_model)
    searcher = _build_searcher(debate_id)
    legal_api = await _build_legal_api(debate_id)

    from app.graph.checkpointer import get_checkpointer
    checkpointer = await get_checkpointer()
    graph = build_debate_graph(
        llm_config=llm_config,
        searcher=searcher,
        legal_api=legal_api,
        checkpointer=checkpointer,
    )

    agents = state.get("agents", [])
    team_a_agents = [a for a in agents if a.get("team") == "team_a"]
    team_b_agents = [a for a in agents if a.get("team") == "team_b"]
    judge_agents = [a for a in agents if a.get("role") == "judge"]

    analysis = state["analysis"]

    initial_state: dict[str, Any] = {
        "debate_id": debate_id,
        "situation_brief": state["situation_brief"],
        "analysis": analysis,  # Preserve full analysis for status API
        "topic": analysis.get("topic", ""),
        "opinion_a": analysis.get("opinion_a", ""),
        "opinion_b": analysis.get("opinion_b", ""),
        "key_issues": analysis.get("key_issues", []),
        "round": 1,
        "min_rounds": state.get("min_rounds", 3),
        "max_rounds": state.get("max_rounds", 10),
        "debate_log": state.get("debate_log", []),
        "all_evidences": state.get("all_evidences", []),
        "team_a_state": state.get("team_a_state", {}),
        "team_b_state": state.get("team_b_state", {}),
        "team_a_agents": team_a_agents,
        "team_b_agents": team_b_agents,
        "judge_agents": judge_agents,
        "judge_notes": state.get("judge_notes", []),
        "current_team": "team_a",
        "next_action": "",
        "early_stop_votes": state.get("early_stop_votes", []),
        "verdicts": [],
        "status": "running",
        "default_model": default_model or llm_config["model"],
        "llm_config": llm_config,
        "user_interrupt": None,
        "team_a_name": state.get("team_a_name", "Team A"),
        "team_b_name": state.get("team_b_name", "Team B"),
        "internal_discussions": [],
        "judge_qa_log": [],
        "pending_judge_questions": [],
        "blacklisted_evidence": [],
        "agent_memories": {},
        "judge_improvement_feedback": {},
        "parties": analysis.get("parties", []),
        "timeline": analysis.get("timeline", []),
        "causal_chain": analysis.get("causal_chain", []),
        "key_facts": analysis.get("key_facts", []),
        "focus_points": analysis.get("focus_points", {}),
        "missing_information": analysis.get("missing_information", []),
        "team_a_cautions": analysis.get("team_a_cautions", []),
        "team_b_cautions": analysis.get("team_b_cautions", []),
    }

    task = _task_mgr.create_task(task_type="debate_run", message="Starting debate...")
    state["status"] = "running"
    state["current_round"] = 1
    await DebateStore.asave(debate_id, state)

    bg_task = asyncio.create_task(_run_debate(debate_id, task.task_id, graph, initial_state))
    _running_tasks[debate_id] = bg_task

    logger.info("Debate %s started (task_id=%s)", debate_id, task.task_id)
    return {"task_id": task.task_id, "status": "running"}


@router.get("/{debate_id}/status")
async def get_debate_status(debate_id: str):
    """Return current debate status, round, and task progress."""
    state = await DebateStore.aload(debate_id)

    # Auto-correct stale status: if DB says running/paused but no task exists,
    # the process was lost (server restart, crash). Mark as stopped.
    _db_status = state.get("status", "unknown")
    if _db_status in ("running", "paused") and debate_id not in _running_tasks:
        logger.info(
            "[status] Debate %s status '%s' but no running task — correcting to 'stopped'.",
            debate_id, _db_status,
        )
        state["status"] = "stopped"
        await DebateStore.aupdate(debate_id, status="stopped")

    result: dict[str, Any] = {
        "debate_id": debate_id,
        "status": state.get("status", "unknown"),
        "current_round": state.get("current_round", 0),
        "max_rounds": state.get("max_rounds", 10),
        "log_count": len(state.get("debate_log", [])),
        "evidence_count": len(state.get("all_evidences", [])),
        "current_phase": state.get("current_phase", ""),
        "current_team": state.get("current_team", ""),
        "analysis": state.get("analysis") or {
            "topic": state.get("topic", ""),
            "opinion_a": state.get("opinion_a", ""),
            "opinion_b": state.get("opinion_b", ""),
            "key_issues": state.get("key_issues", []),
            "team_a_cautions": state.get("team_a_cautions", []),
            "team_b_cautions": state.get("team_b_cautions", []),
            "parties": state.get("parties", []),
            "timeline": state.get("timeline", []),
            "causal_chain": state.get("causal_chain", []),
            "key_facts": state.get("key_facts", []),
            "focus_points": state.get("focus_points", {}),
            "missing_information": state.get("missing_information", []),
        },
        "situation_brief": state.get("situation_brief", ""),
        "discussion_progress": state.get("discussion_progress", 0),
        "discussion_total": state.get("discussion_total") or settings_mgr.load().get("debate", {}).get("team_discussion_turns", 15),
        "discussion_extensions_remaining": state.get("discussion_extensions_remaining", 3),
        "verdicts": state.get("verdicts", []),
        "team_a_name": state.get("team_a_name", "Team A"),
        "team_b_name": state.get("team_b_name", "Team B"),
    }

    # Attach task progress if a debate task is running
    if debate_id in _running_tasks:
        tasks = _task_mgr.list_tasks(task_type="debate_run")
        for t in reversed(tasks):
            result["task"] = t
            break

    return result


@router.get("/{debate_id}/log")
async def get_debate_log(debate_id: str, from_index: int = Query(0, ge=0)):
    """Return debate log entries starting from a given index (for incremental loading)."""
    state = await DebateStore.aload(debate_id)
    full_log = state.get("debate_log", [])
    return {
        "total": len(full_log),
        "from_index": from_index,
        "entries": full_log[from_index:],
    }


@router.get("/{debate_id}/graph")
async def get_debate_graph(debate_id: str):
    """
    Return a relation graph for visualization.

    Nodes represent agents; edges represent rebuttal/agreement relationships
    extracted from the debate log.
    """
    state = await DebateStore.aload(debate_id)
    # Combine agents from all three lists (state has no single "agents" key)
    agents = (
        state.get("team_a_agents", [])
        + state.get("team_b_agents", [])
        + state.get("judge_agents", [])
    )
    # Fallback to legacy "agents" key if present
    if not agents:
        agents = state.get("agents", [])

    debate_log = state.get("debate_log", [])
    all_evidences = state.get("all_evidences", [])

    logger.info(
        "[graph] debate_id=%s, agents=%d, debate_log=%d, all_evidences=%d",
        debate_id, len(agents), len(debate_log), len(all_evidences),
    )

    # Build node list
    nodes: list[dict] = []
    statement_counts: dict[str, int] = {}
    for entry in debate_log:
        speaker = entry.get("speaker", "")
        if speaker:
            statement_counts[speaker] = statement_counts.get(speaker, 0) + 1

    for agent in agents:
        aid = agent.get("agent_id", "")
        team = agent.get("team")
        color = "#4A90D9" if team == "team_a" else "#E74C3C" if team == "team_b" else "#FFC107"
        nodes.append({
            "id": aid,
            "name": agent.get("name", aid),
            "team": team,
            "role": agent.get("role", "debater"),
            "color": color,
            "specialty": agent.get("specialty", ""),
            "personality": agent.get("personality", ""),
            "debate_style": agent.get("debate_style", ""),
            "background": agent.get("background", ""),
            "llm_override": agent.get("llm_override", ""),
            "statement_count": statement_counts.get(aid, 0),
        })

    # Build edge list: track each team's last speaker for cross-team edges
    edges: list[dict] = []
    edge_id = 0
    team_last_speaker: dict[str, str] = {}

    for entry in debate_log:
        team = entry.get("team", "")
        speaker = entry.get("speaker", "")
        if not speaker:
            continue
        rnd = entry.get("round", 0)

        # Cross-team rebuts edges
        for other_team, other_speaker in team_last_speaker.items():
            if other_team and other_team != team and other_speaker and other_speaker != speaker:
                edge_id += 1
                edges.append({
                    "id": f"e{edge_id}",
                    "source": speaker,
                    "target": other_speaker,
                    "relation_type": "rebuts",
                    "round": rnd,
                })

        # Same team supports edge
        if team in team_last_speaker and team_last_speaker[team] != speaker:
            edge_id += 1
            edges.append({
                "id": f"e{edge_id}",
                "source": speaker,
                "target": team_last_speaker[team],
                "relation_type": "supports",
                "round": rnd,
            })

        team_last_speaker[team] = speaker

    # Add evidence nodes (using already-loaded all_evidences)
    current_round = state.get("current_round", state.get("round", 0))

    evidence_cite_count: dict[str, int] = {}
    current_round_cite_count: dict[str, int] = {}
    cited_rounds: dict[str, set] = {}  # evidence_id → set of rounds where cited
    for ev in all_evidences:
        eid = ev.get("evidence_id", "")
        rnd = ev.get("round", 0)
        evidence_cite_count[eid] = evidence_cite_count.get(eid, 0) + 1
        cited_rounds.setdefault(eid, set()).add(rnd)
        if rnd == current_round:
            current_round_cite_count[eid] = current_round_cite_count.get(eid, 0) + 1

    # Count citations that appeared in representative statements (not internal discussion)
    statement_cite_count: dict[str, int] = {}
    for entry in debate_log:
        for ev in entry.get("evidence", []):
            if isinstance(ev, dict):
                eid = ev.get("evidence_id", "")
                if eid:
                    statement_cite_count[eid] = statement_cite_count.get(eid, 0) + 1

    evidence_color_map = {
        "legal_statute": "#9E9E9E",
        "court_precedent": "#9E9E9E",
        "constitutional_decision": "#9E9E9E",
        "uploaded_document": "#9E9E9E",
        "graph_relation": "#9E9E9E",
    }

    import hashlib as _hashlib

    seen_evidence: set[str] = set()
    auto_ev_idx = 0
    for ev in all_evidences:
        ev_id = ev.get("evidence_id", "")
        # Safety: dict/list IDs → hash, empty → auto
        if isinstance(ev_id, (dict, list)):
            ev_id = _hashlib.md5(json.dumps(ev_id, sort_keys=True, default=str).encode()).hexdigest()[:12]
        elif not isinstance(ev_id, str):
            ev_id = str(ev_id)
        if not ev_id:
            ev_id = f"auto_{auto_ev_idx}"
            auto_ev_idx += 1
        # Hash stringified dicts/lists and very long IDs
        if ev_id.startswith('{') or ev_id.startswith('['):
            ev_id = _hashlib.md5(ev_id.encode()).hexdigest()[:12]
        elif len(ev_id) > 80:
            ev_id = _hashlib.md5(ev_id.encode()).hexdigest()[:12]
        node_id = f"ev_{ev_id}"
        if node_id in seen_evidence:
            continue
        seen_evidence.add(node_id)

        source_type = ev.get("source_type", "")
        source_detail = ev.get("source_detail", "Unknown")
        label = source_detail[:40] + "..." if len(source_detail) > 40 else source_detail

        nodes.append({
            "id": node_id,
            "name": label,
            "team": "",
            "role": "evidence",
            "color": evidence_color_map.get(source_type, "#999999"),
            "specialty": source_type,
            "statement_count": evidence_cite_count.get(ev_id, 1),
            "current_round_count": current_round_cite_count.get(ev_id, 0),
            "uncited_rounds": max(0, current_round - len(cited_rounds.get(ev_id, set()))),
            "cited_in_statement": statement_cite_count.get(ev_id, 0),
            "url": ev.get("url", ""),
        })

    # Add "cites" edges from agents to evidence (dedup by speaker+evidence pair)
    seen_cites: set[str] = set()
    for ev in all_evidences:
        ev_id = ev.get("evidence_id", "")
        speaker = ev.get("speaker", "")
        cite_key = f"{speaker}_{ev_id}"
        if speaker and ev_id and cite_key not in seen_cites:
            seen_cites.add(cite_key)
            edges.append({
                "id": f"cite_{cite_key}",
                "source": speaker,
                "target": f"ev_{ev_id}",
                "relation_type": "cites",
                "round": ev.get("round", 0),
            })

    # Add "rebuts" edges between opposing teams in same round
    from collections import defaultdict
    round_speakers = defaultdict(lambda: {"team_a": None, "team_b": None})
    for entry in debate_log:
        r = entry.get("round", 0)
        team = entry.get("team", "")
        speaker = entry.get("speaker", "")
        if team and speaker:
            round_speakers[r][team] = speaker

    for r, teams in round_speakers.items():
        a_speaker = teams.get("team_a")
        b_speaker = teams.get("team_b")
        if a_speaker and b_speaker:
            edges.append({
                "id": f"rebut_r{r}",
                "source": b_speaker,
                "target": a_speaker,
                "relation_type": "rebuts",
                "round": r,
            })

    # Add internal discussion "discusses_with" edges
    # Build name → agent_id mapping (discussion speakers use names, not IDs)
    name_to_id: dict[str, str] = {}
    for agent in agents:
        name_to_id[agent.get("name", "")] = agent.get("agent_id", "")

    for entry in debate_log:
        discussion = entry.get("internal_discussion", [])
        rnd = entry.get("round", 0)
        participants: list[str] = []
        for d in discussion:
            spk_name = d.get("speaker", "")
            spk_id = name_to_id.get(spk_name, spk_name)  # name→ID conversion
            if spk_id and spk_id not in participants:
                participants.append(spk_id)
        for i in range(len(participants)):
            for j in range(i + 1, len(participants)):
                edges.append({
                    "id": f"disc_r{rnd}_{participants[i]}_{participants[j]}",
                    "source": participants[i],
                    "target": participants[j],
                    "relation_type": "discusses_with",
                    "round": rnd,
                })

    # Add judge Q&A "questions" edges
    qa_log = state.get("judge_qa_log", [])
    logger.info("[graph] judge_qa_log: %d entries", len(qa_log))
    for qi, qa in enumerate(qa_log):
        judge_id = qa.get("judge_id", "")
        target_id = qa.get("target_agent_id", "")
        if judge_id and target_id:
            edges.append({
                "id": f"qa_{qi}",
                "source": judge_id,
                "target": target_id,
                "relation_type": "questions",
                "round": qa.get("round", 0),
            })

    # Extract user interventions from all_evidences
    interventions = [
        {
            "type": ev.get("type", "hint"),
            "content": ev.get("content", ""),
            "round": ev.get("round", 0),
            "target_team": ev.get("submitted_by", ""),
        }
        for ev in all_evidences
        if ev.get("source_type") == "user_injected"
    ]

    return {"nodes": nodes, "edges": edges, "interventions": interventions}


@router.post("/{debate_id}/interrupt")
async def interrupt_debate(debate_id: str, body: InterruptBody):
    """Inject a human-in-the-loop intervention into the running debate."""
    state = await DebateStore.aload(debate_id)

    if state.get("status") != "running":
        raise HTTPException(status_code=400, detail="Can only interrupt a running debate.")

    interrupt_data = {
        "target_team": body.target_team,
        "content": body.content,
        "type": body.type,
        "timestamp": datetime.now().isoformat(),
    }

    # Write to state so the graph picks it up on the next iteration
    await DebateStore.aupdate(debate_id, user_interrupt=interrupt_data)

    logger.info("User interrupt set for debate %s: team=%s type=%s", debate_id, body.target_team, body.type)
    return {"status": "ok", "interrupt": interrupt_data}


@router.post("/{debate_id}/pause")
async def pause_debate(debate_id: str):
    """Pause the running debate — immediately cancels the background task."""
    state = await DebateStore.aload(debate_id)

    if state.get("status") not in ("running", "extended"):
        raise HTTPException(status_code=400, detail="Debate is not running.")

    # 1. Save paused status first (before cancelling the task)
    await DebateStore.aupdate(debate_id, status="paused")

    # 2. Cancel the running task immediately
    task = _running_tasks.pop(debate_id, None)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        logger.info("Debate %s task cancelled on pause.", debate_id)

    logger.info("Debate %s paused.", debate_id)
    return {"status": "paused"}


@router.put("/{debate_id}/config")
async def update_debate_config(debate_id: str, body: ConfigUpdateBody):
    """Update model config while the debate is paused."""
    state = await DebateStore.aload(debate_id)

    if state.get("status") in ("running", "extended"):
        raise HTTPException(status_code=400, detail="Cannot change config while debate is running. Pause first.")

    if body.default_model:
        state["default_model"] = body.default_model

    if body.agent_overrides:
        agents = state.get("agents", [])
        for agent in agents:
            aid = agent.get("agent_id", "")
            if aid in body.agent_overrides:
                agent["llm_override"] = body.agent_overrides[aid]
        state["agents"] = agents

    await DebateStore.asave(debate_id, state)
    logger.info("Config updated for paused debate %s", debate_id)

    return {
        "status": "ok",
        "default_model": state.get("default_model"),
        "agents": state.get("agents", []),
    }


@router.post("/{debate_id}/resume")
async def resume_debate(debate_id: str):
    """Resume a paused or stopped debate by restarting the graph from current state."""
    state = await DebateStore.aload(debate_id)

    if state.get("status") not in ("paused", "stopped"):
        raise HTTPException(status_code=400, detail="Debate is not paused or stopped.")

    # Prevent parallel execution — only one debate can run at a time
    _other_running = {k: v for k, v in _running_tasks.items() if k != debate_id}
    if _other_running:
        _other_id = next(iter(_other_running))
        raise HTTPException(
            status_code=409,
            detail=f"Another debate ({_other_id}) is already running. Stop it first.",
        )

    # Cancel lingering task from before pause (it may still be running
    # until it hits a pause checkpoint).
    old_task = _running_tasks.pop(debate_id, None)
    if old_task and not old_task.done():
        old_task.cancel()
        try:
            await old_task
        except (asyncio.CancelledError, Exception):
            pass
        logger.info("Cancelled lingering task for %s before resume", debate_id)

    # Rebuild graph with checkpointer
    default_model = state.get("default_model")
    llm_config = _build_llm_config_dict(default_model)
    searcher = _build_searcher(debate_id)
    legal_api = await _build_legal_api(debate_id)
    from app.graph.checkpointer import get_checkpointer
    checkpointer = await get_checkpointer()
    graph = build_debate_graph(
        llm_config=llm_config, searcher=searcher, legal_api=legal_api,
        checkpointer=checkpointer,
    )

    # Use team-specific agent lists (updated by update_agent endpoint during pause)
    # Fallback to legacy "agents" key only if team-specific lists are empty
    _team_a = state.get("team_a_agents", [])
    _team_b = state.get("team_b_agents", [])
    _judges = state.get("judge_agents", [])
    if not _team_a and not _team_b and not _judges:
        agents = state.get("agents", [])
        _team_a = [a for a in agents if a.get("team") == "team_a"]
        _team_b = [a for a in agents if a.get("team") == "team_b"]
        _judges = [a for a in agents if a.get("role") == "judge"]

    analysis = state.get("analysis", {})

    initial_state: dict[str, Any] = {
        "debate_id": debate_id,
        "situation_brief": state["situation_brief"],
        "analysis": analysis,
        "topic": analysis.get("topic", ""),
        "opinion_a": analysis.get("opinion_a", ""),
        "opinion_b": analysis.get("opinion_b", ""),
        "key_issues": analysis.get("key_issues", []),
        "round": state.get("current_round", 1),
        "min_rounds": state.get("min_rounds", 3),
        "max_rounds": state.get("max_rounds", 10),
        "debate_log": state.get("debate_log", []),
        "all_evidences": state.get("all_evidences", []),
        "team_a_state": state.get("team_a_state", {}),
        "team_b_state": state.get("team_b_state", {}),
        "team_a_agents": _team_a,
        "team_b_agents": _team_b,
        "judge_agents": _judges,
        "judge_notes": state.get("judge_notes", []),
        "current_team": state.get("current_team", "team_a"),
        "next_action": "",
        "early_stop_votes": state.get("early_stop_votes", []),
        "verdicts": state.get("verdicts", []),
        "status": "running",
        "default_model": default_model or llm_config["model"],
        "llm_config": llm_config,
        "user_interrupt": None,
        "team_a_name": state.get("team_a_name", "Team A"),
        "team_b_name": state.get("team_b_name", "Team B"),
        "internal_discussions": state.get("internal_discussions", []),
        "judge_qa_log": state.get("judge_qa_log", []),
        "pending_judge_questions": [],
        "blacklisted_evidence": state.get("blacklisted_evidence", []),
        "agent_memories": state.get("agent_memories", {}),
        "judge_improvement_feedback": state.get("judge_improvement_feedback", {}),
        "parties": analysis.get("parties", []),
        "timeline": analysis.get("timeline", []),
        "causal_chain": analysis.get("causal_chain", []),
        "key_facts": analysis.get("key_facts", []),
        "focus_points": analysis.get("focus_points", {}),
        "missing_information": analysis.get("missing_information", []),
        "team_a_cautions": analysis.get("team_a_cautions", []),
        "team_b_cautions": analysis.get("team_b_cautions", []),
    }

    state["status"] = "running"
    await DebateStore.asave(debate_id, state)

    task = _task_mgr.create_task(task_type="debate_run", message="Resuming debate...")
    bg_task = asyncio.create_task(_run_debate(debate_id, task.task_id, graph, initial_state))
    _running_tasks[debate_id] = bg_task

    logger.info("Debate %s resumed (task_id=%s)", debate_id, task.task_id)
    return {"task_id": task.task_id, "status": "running"}


@router.post("/{debate_id}/stop")
async def stop_debate(debate_id: str):
    """Force stop the debate and trigger final judgment."""
    state = await DebateStore.aload(debate_id)

    if state.get("status") not in ("running", "paused", "extended"):
        raise HTTPException(status_code=400, detail="Debate is not active.")

    # Signal the graph to stop
    await DebateStore.aupdate(debate_id, status="stopped", next_action="stop")

    # Cancel the background task if running
    bg = _running_tasks.pop(debate_id, None)
    if bg and not bg.done():
        bg.cancel()

    logger.info("Debate %s stopped.", debate_id)
    return {"status": "stopped"}


@router.get("/{debate_id}/verdict")
async def get_verdict(debate_id: str):
    """Return the final verdicts from the debate."""
    state = await DebateStore.aload(debate_id)
    verdicts = state.get("verdicts", [])
    if not verdicts:
        raise HTTPException(status_code=404, detail="No verdicts available. Debate may not have finished.")
    return {"verdicts": verdicts}


@router.post("/{debate_id}/extend")
async def extend_debate(debate_id: str, body: ExtendBody):
    """
    Extend the debate by adding more rounds and optionally injecting evidence.
    Restarts the graph from the current state.
    """
    state = await DebateStore.aload(debate_id)
    cur_status = state.get("status")

    is_running = cur_status == "running"
    if is_running:
        current_round = state.get("current_round", 0)
        max_rounds = state.get("max_rounds", 10)
        if current_round < max_rounds:
            raise HTTPException(
                status_code=400,
                detail="Can only extend during the last round.",
            )
    elif cur_status not in ("completed", "paused", "stopped"):
        raise HTTPException(
            status_code=400,
            detail="Debate must be running (last round), completed, paused, or stopped to extend.",
        )

    # Update rounds
    old_max = state.get("max_rounds", 10)
    new_max = old_max + body.additional_rounds
    state["max_rounds"] = new_max

    if is_running:
        await DebateStore.asave(debate_id, state)
        logger.info(
            "Debate %s extended while running: max_rounds %d -> %d",
            debate_id, old_max, new_max,
        )
        return {"status": "extended", "max_rounds": new_max}

    if debate_id in _running_tasks:
        raise HTTPException(status_code=409, detail="Debate task still active.")

    state["status"] = "extended"

    # Inject new evidence if provided
    if body.new_evidence and body.target_team:
        evidence_entry = {
            "content": body.new_evidence,
            "source": "user_injected",
            "team": body.target_team,
            "round": state.get("current_round", 0),
        }
        team_state_key = f"{body.target_team}_state"
        ts = state.get(team_state_key, {})
        extra = ts.get("extra_evidence", [])
        extra.append(evidence_entry)
        ts["extra_evidence"] = extra
        state[team_state_key] = ts

    await DebateStore.asave(debate_id, state)

    # Restart graph with checkpointer
    default_model = state.get("default_model")
    llm_config = _build_llm_config_dict(default_model)
    searcher = _build_searcher(debate_id)
    legal_api = await _build_legal_api(debate_id)
    from app.graph.checkpointer import get_checkpointer
    checkpointer = await get_checkpointer()
    graph = build_debate_graph(
        llm_config=llm_config, searcher=searcher, legal_api=legal_api,
        checkpointer=checkpointer,
    )

    analysis = state.get("analysis", {})

    # Use team-specific agent lists (same as resume_debate)
    _team_a = state.get("team_a_agents", [])
    _team_b = state.get("team_b_agents", [])
    _judges = state.get("judge_agents", [])
    if not _team_a and not _team_b and not _judges:
        agents = state.get("agents", [])
        _team_a = [a for a in agents if a.get("team") == "team_a"]
        _team_b = [a for a in agents if a.get("team") == "team_b"]
        _judges = [a for a in agents if a.get("role") == "judge"]

    initial_state: dict[str, Any] = {
        "debate_id": debate_id,
        "situation_brief": state["situation_brief"],
        "analysis": analysis,
        "topic": analysis.get("topic", ""),
        "opinion_a": analysis.get("opinion_a", ""),
        "opinion_b": analysis.get("opinion_b", ""),
        "key_issues": analysis.get("key_issues", []),
        "round": state.get("current_round", 1),
        "min_rounds": state.get("min_rounds", 3),
        "max_rounds": new_max,
        "debate_log": state.get("debate_log", []),
        "all_evidences": state.get("all_evidences", []),
        "team_a_state": state.get("team_a_state", {}),
        "team_b_state": state.get("team_b_state", {}),
        "team_a_agents": _team_a,
        "team_b_agents": _team_b,
        "judge_agents": _judges,
        "judge_notes": state.get("judge_notes", []),
        "current_team": state.get("current_team", "team_a"),
        "next_action": "",
        "early_stop_votes": state.get("early_stop_votes", []),
        "verdicts": state.get("verdicts", []),
        "status": "running",
        "default_model": default_model or llm_config["model"],
        "llm_config": llm_config,
        "user_interrupt": None,
        "team_a_name": state.get("team_a_name", "Team A"),
        "team_b_name": state.get("team_b_name", "Team B"),
        "internal_discussions": state.get("internal_discussions", []),
        "judge_qa_log": state.get("judge_qa_log", []),
        "pending_judge_questions": [],
        "blacklisted_evidence": state.get("blacklisted_evidence", []),
        "agent_memories": state.get("agent_memories", {}),
        "judge_improvement_feedback": state.get("judge_improvement_feedback", {}),
        "parties": analysis.get("parties", []),
        "timeline": analysis.get("timeline", []),
        "causal_chain": analysis.get("causal_chain", []),
        "key_facts": analysis.get("key_facts", []),
        "focus_points": analysis.get("focus_points", {}),
        "missing_information": analysis.get("missing_information", []),
        "team_a_cautions": analysis.get("team_a_cautions", []),
        "team_b_cautions": analysis.get("team_b_cautions", []),
    }

    task = _task_mgr.create_task(task_type="debate_run", message="Extending debate...")
    bg_task = asyncio.create_task(_run_debate(debate_id, task.task_id, graph, initial_state))
    _running_tasks[debate_id] = bg_task

    logger.info(
        "Debate %s extended: max_rounds %d -> %d (task_id=%s)",
        debate_id, old_max, new_max, task.task_id,
    )
    return {"task_id": task.task_id, "status": "extended", "max_rounds": new_max}

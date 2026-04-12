"""
RAG (Retrieval-Augmented Generation) API endpoints for LawNuri backend.

Handles document upload, indexing (chunking + embedding + graph building),
hybrid search (vector + graph), and knowledge graph visualization.
"""

from __future__ import annotations

import asyncio
import os
from glob import glob
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import config
from app.models.llm_config import PRESET_PROVIDERS
from app.models.task import TaskStatus, task_manager
from app.utils.logger import logger

router = APIRouter(prefix="/api/rag", tags=["rag"])

_ALLOWED_EXTENSIONS = {"pdf", "md", "txt", "markdown"}

_VALID_POOLS = {"common", "team_a", "team_b"}



class IndexRequest(BaseModel):
    """Request body for starting an indexing task."""
    debate_id: str
    pool: Optional[str] = None  # If None, index all pools with uploaded files


class SearchRequest(BaseModel):
    """Request body for performing a search."""
    query: str
    debate_id: str
    pool: str = "common"
    search_type: str = "both"  # "vector", "graph", "both"
    top_k: int = 5



def _get_clients():
    """
    Build LLMClient and EmbeddingClient from the current settings.

    Reads the settings.json via SettingsManager to determine the
    default provider, API key, and base URL. Uses PRESET_PROVIDERS
    from models/llm_config.py to resolve base URLs.

    Returns:
        Tuple of (LLMClient, EmbeddingClient).

    Raises:
        HTTPException: If no LLM provider is configured and enabled.
    """
    from app.api.settings import SettingsManager
    from app.utils.embedding_client import EmbeddingClient
    from app.utils.llm_client import LLMClient

    settings_mgr = SettingsManager()
    settings = settings_mgr.load()

    providers = settings.get("llm_providers", {})
    llm_api_key = ""
    llm_base_url = ""
    llm_model = ""
    llm_provider = ""
    llm_provider_info: dict = {}

    for provider_id, info in providers.items():
        if info.get("api_key") and info.get("enabled", False):
            llm_api_key = info["api_key"]
            llm_provider = provider_id
            llm_provider_info = info

            if provider_id == "custom":
                llm_base_url = info.get("base_url", "")
                llm_model = info.get("model", "")
            else:
                preset = PRESET_PROVIDERS.get(provider_id, {})
                llm_base_url = preset.get("base_url", "")
                llm_model = info.get("model", "")
                if not llm_model:
                    models = preset.get("models", [])
                    llm_model = models[0] if models else ""
            break

    if not llm_api_key:
        raise HTTPException(
            status_code=400,
            detail="No LLM provider configured. Please set up an API key in Settings.",
        )

    embedding_cfg = settings.get("embedding", {})
    embedding_provider = embedding_cfg.get("provider", "openai")
    embedding_model = embedding_cfg.get("model", "text-embedding-3-small")

    embedding_provider_info = providers.get(embedding_provider, {})
    embedding_api_key = embedding_provider_info.get("api_key", llm_api_key)

    if embedding_provider == "custom":
        embedding_base_url = embedding_provider_info.get("base_url", "")
    else:
        preset = PRESET_PROVIDERS.get(embedding_provider, {})
        embedding_base_url = preset.get("base_url", "https://api.openai.com/v1")

    llm_client = LLMClient(
        api_key=llm_api_key,
        base_url=llm_base_url,
        model=llm_model,
        provider=llm_provider,
        vertex_project_id=llm_provider_info.get("project_id", ""),
        vertex_location=llm_provider_info.get("location", "global"),
    )
    embedding_client = EmbeddingClient(
        api_key=embedding_api_key,
        base_url=embedding_base_url,
        model=embedding_model,
    )

    return llm_client, embedding_client



@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    pool: str = Form("common"),
    debate_id: str = Form(...),
):
    """
    Upload a legal document (PDF, Markdown, TXT) for RAG processing.

    The file is saved to the debate's upload directory, organized by pool.
    Supported extensions: pdf, md, txt, markdown.

    Args:
        file: The uploaded file.
        pool: Target pool (common, team_a, team_b).
        debate_id: The debate this document belongs to.

    Returns:
        File metadata including saved path, size, and pool.
    """
    # Validate pool
    if pool not in _VALID_POOLS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid pool '{pool}'. Must be one of: {sorted(_VALID_POOLS)}",
        )

    # Validate file extension
    filename = file.filename or "unnamed"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file extension '.{ext}'. "
                f"Allowed: {sorted(_ALLOWED_EXTENSIONS)}"
            ),
        )

    # Build save path: data/debates/{debate_id}/uploads/{pool}/
    upload_dir = os.path.join(
        config.DATA_DIR, "debates", debate_id, "uploads", pool
    )
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, filename)

    # Write file to disk
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    file_size = len(content)
    logger.info(
        "Uploaded file '%s' (%d bytes) to pool='%s' debate='%s'.",
        filename, file_size, pool, debate_id,
    )

    return {
        "status": "ok",
        "file": {
            "filename": filename,
            "size": file_size,
            "pool": pool,
            "debate_id": debate_id,
            "path": file_path,
        },
    }



@router.post("/index")
async def start_indexing(body: IndexRequest):
    """
    Start the indexing pipeline as an async background task.

    Scans the upload directory for the specified debate (and optionally
    a specific pool), then runs the full indexing pipeline: parsing,
    anonymization, chunking, embedding, vector storage, entity/relation
    extraction, and graph building.

    Args:
        body: IndexRequest with debate_id and optional pool.

    Returns:
        A task_id that can be used to track progress via
        GET /api/rag/index/status/{task_id}.
    """
    debate_id = body.debate_id
    pool = body.pool

    # Build the list of (pool, file_paths) to index
    pools_to_index: list[tuple[str, list[str]]] = []

    if pool:
        if pool not in _VALID_POOLS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid pool '{pool}'. Must be one of: {sorted(_VALID_POOLS)}",
            )
        file_paths = _get_upload_files(debate_id, pool)
        if not file_paths:
            raise HTTPException(
                status_code=404,
                detail=f"No files found to index in pool '{pool}' for debate '{debate_id}'.",
            )
        pools_to_index.append((pool, file_paths))
    else:
        # Index all pools that have uploaded files
        for p in _VALID_POOLS:
            file_paths = _get_upload_files(debate_id, p)
            if file_paths:
                pools_to_index.append((p, file_paths))

        if not pools_to_index:
            raise HTTPException(
                status_code=404,
                detail=f"No uploaded files found for debate '{debate_id}'.",
            )

    # Create a task for progress tracking
    task = task_manager.create_task(
        task_type="indexing",
        message=f"Starting indexing for debate {debate_id}",
    )
    task_manager.update_task(
        task.task_id,
        status=TaskStatus.PROCESSING,
        progress=0.0,
        message="Initializing indexing pipeline...",
    )

    # Launch the background indexing coroutine
    asyncio.create_task(
        _run_indexing(task.task_id, debate_id, pools_to_index)
    )

    logger.info(
        "Indexing task '%s' created for debate '%s' (pools: %s).",
        task.task_id,
        debate_id,
        [p for p, _ in pools_to_index],
    )

    return {
        "status": "ok",
        "task_id": task.task_id,
        "debate_id": debate_id,
        "pools": [p for p, _ in pools_to_index],
        "total_files": sum(len(fps) for _, fps in pools_to_index),
    }


async def _run_indexing(
    task_id: str,
    debate_id: str,
    pools_to_index: list[tuple[str, list[str]]],
) -> None:
    """
    Background coroutine that runs the indexing pipeline.

    Creates LLM/embedding clients from current settings and invokes
    the Indexer for each pool's files.

    Args:
        task_id: TaskManager task ID for progress reporting.
        debate_id: The debate identifier.
        pools_to_index: List of (pool_name, file_paths) tuples.
    """
    from app.rag.indexer import Indexer

    try:
        llm_client, embedding_client = _get_clients()

        indexer = Indexer(
            llm_client=llm_client,
            embedding_client=embedding_client,
            data_dir=config.DATA_DIR,
        )

        total_pools = len(pools_to_index)
        all_results: list[dict] = []

        for idx, (pool, file_paths) in enumerate(pools_to_index):
            base_progress = idx / total_pools
            task_manager.update_task(
                task_id,
                progress=base_progress,
                message=f"Indexing pool '{pool}' ({idx + 1}/{total_pools})...",
            )

            result = await indexer.index_documents(
                file_paths=file_paths,
                pool=pool,
                debate_id=debate_id,
                task_id=None,  # We manage progress at the pool level
            )
            all_results.append({"pool": pool, **result})

        task_manager.complete_task(
            task_id,
            result={
                "debate_id": debate_id,
                "pools": all_results,
            },
        )
        logger.info("Indexing task '%s' completed successfully.", task_id)

    except Exception as exc:
        logger.exception("Indexing task '%s' failed: %s", task_id, exc)
        task_manager.fail_task(task_id, error=str(exc))


def _get_upload_files(debate_id: str, pool: str) -> list[str]:
    """
    List all uploaded files for a given debate and pool.

    Args:
        debate_id: The debate identifier.
        pool: The pool name.

    Returns:
        List of absolute file paths found in the upload directory.
    """
    upload_dir = os.path.join(
        config.DATA_DIR, "debates", debate_id, "uploads", pool
    )
    if not os.path.isdir(upload_dir):
        return []

    files: list[str] = []
    for ext in _ALLOWED_EXTENSIONS:
        files.extend(glob(os.path.join(upload_dir, f"*.{ext}")))

    return sorted(files)



@router.get("/index/status/{task_id}")
async def get_index_status(task_id: str):
    """
    Check the status and progress of an indexing task.

    Args:
        task_id: The task identifier returned by POST /api/rag/index.

    Returns:
        Task state including status, progress, message, and result.
    """
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found.",
        )

    return {"task": task.to_dict()}



@router.post("/search")
async def search(body: SearchRequest):
    """
    Perform a hybrid vector + graph search over indexed documents.

    Supports three search modes:
        - "vector": embedding-based similarity search (ChromaDB).
        - "graph": entity label matching with neighbor traversal (NetworkX).
        - "both": combined and deduplicated results from both backends.

    Args:
        body: SearchRequest with query, debate_id, pool, search_type, top_k.

    Returns:
        List of ranked search results.
    """
    if body.pool not in _VALID_POOLS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid pool '{body.pool}'. Must be one of: {sorted(_VALID_POOLS)}",
        )

    if body.search_type not in ("vector", "graph", "both"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid search_type '{body.search_type}'. Must be 'vector', 'graph', or 'both'.",
        )

    try:
        _, embedding_client = _get_clients()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize search clients: {exc}",
        ) from exc

    from app.rag.searcher import Searcher

    searcher = Searcher(
        embedding_client=embedding_client,
        data_dir=config.DATA_DIR,
    )

    try:
        results = await searcher.search(
            query=body.query,
            debate_id=body.debate_id,
            pool=body.pool,
            search_type=body.search_type,
            top_k=body.top_k,
        )
    except Exception as exc:
        logger.exception("Search failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {exc}",
        ) from exc

    return {
        "status": "ok",
        "query": body.query,
        "debate_id": body.debate_id,
        "pool": body.pool,
        "search_type": body.search_type,
        "count": len(results),
        "results": results,
    }



@router.get("/graph/{debate_id}/{pool_id}")
async def get_knowledge_graph(debate_id: str, pool_id: str):
    """
    Get the knowledge graph visualization data for a document pool.

    Returns the full graph (nodes and edges) as JSON, suitable for
    rendering in a frontend graph visualization library.

    Args:
        debate_id: The debate identifier.
        pool_id: The pool name (common, team_a, team_b).

    Returns:
        Graph data with nodes and edges lists.
    """
    if pool_id not in _VALID_POOLS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid pool '{pool_id}'. Must be one of: {sorted(_VALID_POOLS)}",
        )

    from app.rag.graph_store import GraphStore

    try:
        graph_store = GraphStore(debate_id=debate_id, pool=pool_id)
        await graph_store.load()
        graph_data = graph_store.to_dict()

        if not graph_data.get("nodes"):
            return {
                "status": "ok",
                "debate_id": debate_id,
                "pool": pool_id,
                "graph": {"nodes": [], "edges": []},
                "message": "No graph data found. Run indexing first.",
            }
    except Exception as exc:
        logger.exception("Failed to load graph for debate=%s pool=%s: %s", debate_id, pool_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load knowledge graph: {exc}",
        ) from exc

    return {
        "status": "ok",
        "debate_id": debate_id,
        "pool": pool_id,
        "graph": graph_data,
        "stats": {
            "nodes": len(graph_data.get("nodes", [])),
            "edges": len(graph_data.get("edges", [])),
        },
    }

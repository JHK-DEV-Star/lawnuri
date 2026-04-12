"""
Embedding client supporting multiple providers (OpenAI, Gemini, Anthropic fallback).
Provides both synchronous and asynchronous methods, plus relevance scoring.

Includes batch embedding via provider Batch APIs (50% cost discount).
"""

from typing import List, Union
import asyncio
import json
import math

from app.utils.logger import logger


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingClient:
    """
    Embedding client for generating text embeddings.
    Supports OpenAI SDK and Google GenAI SDK based on provider.

    Args:
        api_key: API key for the provider.
        base_url: Base URL of the API endpoint (OpenAI only).
        model: Embedding model identifier.
        provider: "openai" or "gemini".
        timeout: Request timeout in seconds.
        max_retries: Maximum number of automatic retries.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        provider: str = "openai",
        base_url: str = "",
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._provider = provider
        self._api_key = api_key

        if provider == "gemini":
            # Use Google GenAI SDK for Gemini embeddings (API key auth works)
            from google import genai
            self._genai_client = genai.Client(api_key=api_key)
            self._sync_client = None
            self._async_client = None
        else:
            # Use OpenAI SDK for OpenAI + OpenAI-compatible providers
            from openai import AsyncOpenAI, OpenAI
            self._genai_client = None
            self._sync_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
            )
            self._async_client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
            )

        logger.info(
            "EmbeddingClient initialized: model=%s, provider=%s",
            model,
            provider,
        )

    def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings synchronously."""
        inputs = [text] if isinstance(text, str) else text

        logger.debug(
            "[Embedding.embed] Requesting embeddings for %d text(s) with %s (%s)",
            len(inputs), self.model, self._provider,
        )

        if self._provider == "gemini":
            # Gemini: max 100 per batch request
            _GEMINI_BATCH = 100
            vectors: List[List[float]] = []
            for _start in range(0, len(inputs), _GEMINI_BATCH):
                _batch = inputs[_start:_start + _GEMINI_BATCH]
                result = self._genai_client.models.embed_content(
                    model=self.model,
                    contents=_batch,
                )
                vectors.extend([e.values for e in result.embeddings])
        else:
            response = self._sync_client.embeddings.create(
                model=self.model,
                input=inputs,
            )
            embeddings = sorted(response.data, key=lambda x: x.index)
            vectors = [item.embedding for item in embeddings]

        logger.debug(
            "[Embedding.embed] Got %d vector(s), dimension=%d",
            len(vectors),
            len(vectors[0]) if vectors else 0,
        )
        return vectors

    async def aembed(self, text: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings asynchronously."""
        inputs = [text] if isinstance(text, str) else text

        logger.debug(
            "[Embedding.aembed] Requesting embeddings for %d text(s) with %s (%s)",
            len(inputs), self.model, self._provider,
        )

        if self._provider == "gemini":
            # Gemini: max 100 per batch request
            _GEMINI_BATCH = 100
            vectors: List[List[float]] = []
            for _start in range(0, len(inputs), _GEMINI_BATCH):
                _batch = inputs[_start:_start + _GEMINI_BATCH]
                result = await self._genai_client.aio.models.embed_content(
                    model=self.model,
                    contents=_batch,
                )
                vectors.extend([e.values for e in result.embeddings])
        else:
            response = await self._async_client.embeddings.create(
                model=self.model,
                input=inputs,
            )
            embeddings = sorted(response.data, key=lambda x: x.index)
            vectors = [item.embedding for item in embeddings]

        logger.debug(
            "[Embedding.aembed] Got %d vector(s), dimension=%d",
            len(vectors),
            len(vectors[0]) if vectors else 0,
        )
        return vectors

    # ------------------------------------------------------------------
    # Batch embedding (50% discount, no rate limits)
    # ------------------------------------------------------------------

    async def aembed_batch(
        self,
        texts: List[str],
        poll_interval: int = 10,
        max_wait: int = 1800,
    ) -> List[List[float]]:
        """Generate embeddings via Batch API (50% cost, no rate limit).

        Submits a batch job, polls until completion, and returns vectors.
        Falls back to chunked synchronous embedding on failure.

        Args:
            texts: List of texts to embed.
            poll_interval: Seconds between status polls.
            max_wait: Maximum seconds to wait before timeout.

        Returns:
            List of embedding vectors (same order as input texts).
        """
        if not texts:
            return []

        logger.info(
            "[Embedding.aembed_batch] Submitting %d texts via Batch API (%s).",
            len(texts), self._provider,
        )

        try:
            if self._provider == "gemini":
                return await self._gemini_batch_embed(texts, poll_interval, max_wait)
            else:
                return await self._openai_batch_embed(texts, poll_interval, max_wait)
        except Exception as exc:
            logger.warning(
                "[Embedding.aembed_batch] Batch API failed (%s), falling back to sync batches.",
                exc,
            )
            return await self._sync_batch_fallback(texts)

    async def _gemini_batch_embed(
        self, texts: List[str], poll_interval: int, max_wait: int,
    ) -> List[List[float]]:
        """Gemini Batch API for embeddings."""
        from google.genai import types

        job = await self._genai_client.aio.batches.create_embeddings(
            model=self.model,
            src=types.EmbeddingsBatchJobSource(
                inlined_requests=types.EmbedContentBatch(contents=texts),
            ),
        )
        logger.info("[Embedding.batch] Gemini batch job created: %s", job.name)

        _terminal = {
            "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
        }
        elapsed = 0
        while elapsed < max_wait:
            job = await self._genai_client.aio.batches.get(name=job.name)
            state = str(job.state) if job.state else "UNKNOWN"
            # Handle both enum and string state values
            state_str = state.split(".")[-1] if "." in state else state
            if state_str in _terminal:
                break
            logger.info("[Embedding.batch] Job %s state: %s (elapsed %ds)", job.name, state_str, elapsed)
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        state_str = str(job.state).split(".")[-1] if job.state else "UNKNOWN"
        if state_str != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Gemini batch job ended with state: {state_str}")

        # Extract vectors from inlined responses
        responses = job.dest.inlined_embed_content_responses or []
        if len(responses) != len(texts):
            raise RuntimeError(
                f"Batch result count mismatch: expected {len(texts)}, got {len(responses)}"
            )

        vectors = []
        for resp in responses:
            if resp.error:
                raise RuntimeError(f"Batch item error: {resp.error}")
            vectors.append(list(resp.response.embedding.values))

        logger.info("[Embedding.batch] Gemini batch completed: %d vectors.", len(vectors))
        return vectors

    async def _openai_batch_embed(
        self, texts: List[str], poll_interval: int, max_wait: int,
    ) -> List[List[float]]:
        """OpenAI Batch API for embeddings."""
        # 1. Build JSONL input
        lines = []
        for i, text in enumerate(texts):
            lines.append(json.dumps({
                "custom_id": f"emb-{i}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": self.model, "input": text},
            }))
        jsonl_content = "\n".join(lines)

        # 2. Upload input file
        input_file = await self._async_client.files.create(
            file=("batch_embed.jsonl", jsonl_content.encode("utf-8")),
            purpose="batch",
        )
        logger.info("[Embedding.batch] OpenAI input file uploaded: %s", input_file.id)

        # 3. Create batch job
        batch = await self._async_client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
        )
        logger.info("[Embedding.batch] OpenAI batch job created: %s", batch.id)

        # 4. Poll until completion
        elapsed = 0
        while elapsed < max_wait:
            batch = await self._async_client.batches.retrieve(batch.id)
            if batch.status in ("completed", "failed", "expired", "cancelled"):
                break
            logger.info("[Embedding.batch] Batch %s status: %s (elapsed %ds)", batch.id, batch.status, elapsed)
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        if batch.status != "completed":
            raise RuntimeError(f"OpenAI batch ended with status: {batch.status}")

        # 5. Download and parse output file
        output_file_id = batch.output_file_id
        if not output_file_id:
            raise RuntimeError("OpenAI batch completed but no output_file_id")

        file_response = await self._async_client.files.content(output_file_id)
        output_text = file_response.text

        # Parse JSONL output — results may not be in order
        result_map: dict[int, List[float]] = {}
        for line in output_text.strip().split("\n"):
            if not line.strip():
                continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id", "")
            idx = int(custom_id.replace("emb-", ""))
            embedding_data = obj.get("response", {}).get("body", {}).get("data", [])
            if embedding_data:
                result_map[idx] = embedding_data[0].get("embedding", [])

        vectors = [result_map[i] for i in range(len(texts))]
        logger.info("[Embedding.batch] OpenAI batch completed: %d vectors.", len(vectors))
        return vectors

    async def _sync_batch_fallback(self, texts: List[str]) -> List[List[float]]:
        """Fallback: chunked synchronous embedding with rate-limit delays."""
        _BATCH = 100
        vectors: List[List[float]] = []
        total_batches = (len(texts) + _BATCH - 1) // _BATCH
        for b_idx in range(0, len(texts), _BATCH):
            batch = texts[b_idx:b_idx + _BATCH]
            batch_num = b_idx // _BATCH + 1
            logger.info("[Embedding.fallback] Sync batch %d/%d (%d texts)...", batch_num, total_batches, len(batch))
            batch_vecs = await self.aembed(batch)
            vectors.extend(batch_vecs)
            if b_idx + _BATCH < len(texts):
                await asyncio.sleep(1.5)
        return vectors

    async def compute_relevance(
        self,
        topic_text: str,
        results: list[dict],
        auto_remove_threshold: float = 0.55,
    ) -> list[dict]:
        """
        Score search results by embedding similarity to the topic.

        - Results with score < auto_remove_threshold are removed.
        - Remaining results get '_relevance_score' field added.
        - Returns sorted by relevance (highest first).
        """
        if not results:
            return results

        texts = [
            f"{r.get('title', '')} {str(r.get('content', ''))[:100]}"
            for r in results
        ]

        try:
            all_texts = [topic_text] + texts
            all_vectors = await self.aembed(all_texts)
            topic_vec = all_vectors[0]
            result_vecs = all_vectors[1:]

            scored = []
            removed = 0
            for r, vec in zip(results, result_vecs):
                sim = _cosine_similarity(topic_vec, vec)
                label = r.get("title", r.get("source_detail", ""))[:80]
                src_type = r.get("source_type", r.get("type", "unknown"))
                if sim < auto_remove_threshold:
                    removed += 1
                    logger.info(
                        "[relevance] REMOVED (score=%.3f < threshold=%.2f): %s | source=%s",
                        sim, auto_remove_threshold, label, src_type,
                    )
                    continue
                r["_relevance_score"] = round(sim, 3)
                scored.append(r)
                logger.info(
                    "[relevance] KEPT (score=%.3f): %s | source=%s",
                    sim, label, src_type,
                )

            scored.sort(key=lambda x: x.get("_relevance_score", 0), reverse=True)
            logger.info(
                "[relevance] %d/%d results passed (removed %d below %.2f)",
                len(scored), len(results), removed, auto_remove_threshold,
            )
            return scored

        except Exception as exc:
            logger.warning("[relevance] Embedding failed, returning unfiltered: %s", exc)
            return results


def build_embedding_client() -> "EmbeddingClient | None":
    """Build an EmbeddingClient from current settings. Returns None on failure."""
    try:
        from app.api.settings import settings_mgr
        settings = settings_mgr.load()
        providers = settings.get("llm_providers", {})

        # Try Gemini first (free embedding via GenAI SDK)
        gemini = providers.get("gemini", {})
        if gemini.get("api_key"):
            return EmbeddingClient(
                api_key=gemini["api_key"],
                model="gemini-embedding-001",
                provider="gemini",
            )

        # Try Vertex AI (same genai SDK, vertexai=True)
        vertex_cfg = providers.get("vertex_ai", {})
        if vertex_cfg.get("api_key") or vertex_cfg.get("enabled"):
            from google import genai
            _vx_kwargs: dict = {"vertexai": True}
            if vertex_cfg.get("api_key"):
                _vx_kwargs["api_key"] = vertex_cfg["api_key"]
            if vertex_cfg.get("project_id"):
                _vx_kwargs["project"] = vertex_cfg["project_id"]
            if vertex_cfg.get("location") and vertex_cfg["location"] != "global":
                _vx_kwargs["location"] = vertex_cfg["location"]
            client = EmbeddingClient.__new__(EmbeddingClient)
            client.model = "gemini-embedding-001"
            client.base_url = ""
            client._provider = "gemini"
            client._api_key = vertex_cfg.get("api_key", "")
            client._genai_client = genai.Client(**_vx_kwargs)
            client._sync_client = None
            client._async_client = None
            logger.info("EmbeddingClient initialized: model=gemini-embedding-001, provider=vertex_ai")
            return client

        # Then OpenAI
        openai_cfg = providers.get("openai", {})
        if openai_cfg.get("api_key"):
            return EmbeddingClient(
                api_key=openai_cfg["api_key"],
                model="text-embedding-3-small",
                provider="openai",
                base_url="https://api.openai.com/v1",
            )

        # Anthropic → fallback to OpenAI embedding
        anthropic_cfg = providers.get("anthropic", {})
        if anthropic_cfg.get("api_key") and openai_cfg.get("api_key"):
            return EmbeddingClient(
                api_key=openai_cfg["api_key"],
                model="text-embedding-3-small",
                provider="openai",
                base_url="https://api.openai.com/v1",
            )

        logger.warning("No embedding provider configured")
        return None
    except Exception as exc:
        logger.warning("Failed to build embedding client: %s", exc)
        return None

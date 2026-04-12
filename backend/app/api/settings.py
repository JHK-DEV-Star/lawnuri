"""
Settings API for LawNuri backend.

Manages LLM provider configuration, API keys (encrypted at rest),
debate parameters, and embedding settings. All settings are persisted
in backend/data/settings.json.
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

from cryptography.fernet import Fernet
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import config
from app.utils.logger import logger

router = APIRouter(prefix="/api/settings", tags=["settings"])


# ---------------------------------------------------------------------------
# SQLite dual-write helpers
# ---------------------------------------------------------------------------

async def _get_settings_repo():
    """Get SQLite settings repository and connection, or (None, None) if unavailable."""
    try:
        from app.db.database import _connection
        from app.db.sqlite.settings import SQLiteSettingsRepo
        if _connection is not None:
            return SQLiteSettingsRepo(), _connection
    except Exception:
        pass
    return None, None


async def _sync_settings_to_db(data: dict) -> None:
    """Best-effort write of settings to SQLite (fire-and-forget)."""
    repo, db = await _get_settings_repo()
    if repo and db:
        try:
            await repo.save(db, data)
        except Exception as exc:
            logger.warning("Failed to sync settings to SQLite: %s", exc)


async def _load_settings_from_db() -> dict | None:
    """Try to load settings from SQLite. Returns None on failure or if empty."""
    repo, db = await _get_settings_repo()
    if repo and db:
        try:
            data = await repo.load(db)
            if data:
                return data
        except Exception as exc:
            logger.warning("Failed to load settings from SQLite: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class ProviderKeyUpdate(BaseModel):
    """Request body for updating a provider's API key."""
    api_key: str = ""
    base_url: Optional[str] = None
    model: Optional[str] = None
    enabled: Optional[bool] = True
    project_id: Optional[str] = None
    location: Optional[str] = None
    tier: Optional[str] = None


class DebateSettingsUpdate(BaseModel):
    """Request body for updating debate settings."""
    min_rounds: Optional[int] = None
    max_rounds: Optional[int] = None
    team_discussion_turns: Optional[int] = None
    max_review_more: Optional[int] = None
    accept_ratio: Optional[float] = None
    review_more_ratio: Optional[float] = None
    judge_early_stop_votes: Optional[int] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    rag_search_top_k: Optional[int] = None
    team_size: Optional[int] = None
    judge_count: Optional[int] = None
    llm_temperature: Optional[float] = None


class LegalApiUpdate(BaseModel):
    """Request body for updating legal API settings."""
    law_api_key: Optional[str] = None
    precedent_api_key: Optional[str] = None
    max_api_calls_per_round: Optional[int] = None


# ---------------------------------------------------------------------------
# Preset provider/model definitions
# ---------------------------------------------------------------------------

PRESET_PROVIDERS = {
    "openai": {
        "label": "OpenAI",
        "base_url": None,
        "test_model": "gpt-5.4-nano",
        "models": {
            "gpt-5.4":      {"input": 2.50, "output": 15.00},
            "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
            "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
            "gpt-5.2":      {"input": 1.75, "output": 14.00},
            "gpt-5-mini":   {"input": 0.25, "output": 2.00},
            "gpt-4.1":      {"input": 2.00, "output": 8.00},
            "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
            "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
            "gpt-4o":       {"input": 2.50, "output": 10.00},
            "gpt-4o-mini":  {"input": 0.15, "output": 0.60},
            "o3":           {"input": 2.00, "output": 8.00},
            "o4-mini":      {"input": 1.10, "output": 4.40},
        },
    },
    "gemini": {
        "label": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "test_model": "gemini-2.5-flash-lite",
        "paid_only_test_model": "gemini-3.1-pro-preview",  # billing 여부 감지용
        "models": {
            "gemini-3.1-pro-preview":        {"input": 2.00, "output": 12.00, "free_available": False},
            "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50,  "free_available": True},
            "gemini-3-flash-preview":        {"input": 0.50, "output": 3.00,  "free_available": True},
            "gemini-2.5-pro":        {"input": 1.25, "output": 10.00, "free_available": True},
            "gemini-2.5-flash":      {"input": 0.30, "output": 2.50,  "free_available": True},
            "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40,  "free_available": True},
            "gemini-2.0-flash":      {"input": 0.10, "output": 0.40,  "free_available": True},
        },
    },
    "anthropic": {
        "label": "Anthropic",
        "base_url": "https://api.anthropic.com/v1/",
        "test_model": "claude-haiku-4-5",
        "models": {
            "claude-opus-4-6":   {"input": 5.00, "output": 25.00},
            "claude-opus-4-5":   {"input": 5.00, "output": 25.00},
            "claude-opus-4-1":   {"input": 15.00, "output": 75.00},
            "claude-opus-4":     {"input": 15.00, "output": 75.00},
            "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
            "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
            "claude-sonnet-4":   {"input": 3.00, "output": 15.00},
            "claude-haiku-4-5":  {"input": 1.00, "output": 5.00},
        },
    },
    "vertex_ai": {
        "label": "Vertex AI (Google Cloud)",
        "base_url": None,  # user must provide project-specific URL
        "test_model": "gemini-2.5-flash",
        "models": {
            "gemini-3.1-pro-preview":        {"input": 2.00, "output": 12.00},
            "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50},
            "gemini-3-flash-preview":        {"input": 0.50, "output": 3.00},
            "gemini-3-pro-preview":          {"input": 2.00, "output": 12.00},
            "gemini-2.5-pro":       {"input": 1.25, "output": 10.00},
            "gemini-2.5-flash":     {"input": 0.30, "output": 2.50},
            "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
            "gemini-2.0-flash":     {"input": 0.15, "output": 0.60},
        },
    },
    "custom": {
        "label": "Custom (OpenAI-compatible)",
        "base_url": None,
        "test_model": None,
        "models": {},
    },
}


# ---------------------------------------------------------------------------
# Default settings template
# ---------------------------------------------------------------------------

DEFAULT_SETTINGS: dict[str, Any] = {
    "llm_providers": {
        "openai": {"api_key": "", "enabled": False},
        "gemini": {"api_key": "", "enabled": False, "tier": "free"},
        "anthropic": {"api_key": "", "enabled": False},
        "vertex_ai": {"api_key": "", "project_id": "", "location": "global", "enabled": False},
        "custom": {"api_key": "", "base_url": "", "model": "", "enabled": False},
    },
    "embedding": {
        "provider": "openai",
        "model": "text-embedding-3-small",
    },
    "debate": {
        "min_rounds": 3,
        "max_rounds": 5,
        "team_discussion_turns": 10,
        "max_review_more": 3,
        "accept_ratio": 0.4,
        "review_more_ratio": 0.6,
        "judge_early_stop_votes": 2,
        "chunk_size": 1500,
        "chunk_overlap": 200,
        "rag_search_top_k": 10,
        "team_size": 5,
        "judge_count": 3,
        "llm_temperature": 0.7,
    },
    "legal_api": {
        "law_api_key": "",
        "precedent_api_key": "",
        "max_api_calls_per_round": 15,
    },
}


# ---------------------------------------------------------------------------
# SettingsManager - handles file I/O, encryption, and masking
# ---------------------------------------------------------------------------

class SettingsManager:
    """
    Manages loading, saving, encryption, and masking of settings.json.

    API keys are encrypted at rest using Fernet symmetric encryption.
    A key file (.fernet.key) is auto-generated in the data directory on
    first use and must NOT be committed to version control.
    """

    # Fields that contain sensitive API keys and should be encrypted/masked
    _SENSITIVE_FIELDS = {"api_key", "law_api_key", "precedent_api_key"}

    # TTL for in-memory settings cache (seconds)
    _CACHE_TTL = 30.0

    def __init__(self, settings_path: str | None = None):
        self._path = Path(settings_path or config.SETTINGS_FILE)
        self._key_path = self._path.parent / ".fernet.key"
        self._fernet: Fernet | None = None
        self._cache: dict | None = None
        self._cache_ts: float = 0.0

    # -- Fernet helpers ----------------------------------------------------

    def _get_fernet(self) -> Fernet:
        """Return (and lazily initialise) the Fernet instance."""
        if self._fernet is not None:
            return self._fernet

        if self._key_path.exists():
            key = self._key_path.read_bytes().strip()
        else:
            key = Fernet.generate_key()
            self._key_path.parent.mkdir(parents=True, exist_ok=True)
            self._key_path.write_bytes(key)
            logger.info("Generated new Fernet key at %s", self._key_path)

        self._fernet = Fernet(key)
        return self._fernet

    def _encrypt(self, plaintext: str) -> str:
        """Encrypt a plaintext string. Returns empty string for empty input."""
        if not plaintext:
            return ""
        return self._get_fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")

    def _decrypt(self, token: str) -> str:
        """Decrypt a Fernet token string. Returns empty string on failure."""
        if not token:
            return ""
        try:
            return self._get_fernet().decrypt(token.encode("utf-8")).decode("utf-8")
        except Exception:
            logger.warning("Failed to decrypt a stored value; returning empty.")
            return ""

    # -- Masking -----------------------------------------------------------

    @staticmethod
    def _mask_key(value: str) -> str:
        """Mask an API key, showing only the last 4 characters."""
        if not value or len(value) <= 4:
            return "****" if value else ""
        return "*" * (len(value) - 4) + value[-4:]

    def _mask_dict(self, data: dict) -> dict:
        """Recursively mask sensitive fields in a dict (for GET responses)."""
        masked = {}
        for k, v in data.items():
            if isinstance(v, dict):
                masked[k] = self._mask_dict(v)
            elif k in self._SENSITIVE_FIELDS and isinstance(v, str):
                masked[k] = self._mask_key(v)
            else:
                masked[k] = v
        return masked

    # -- Encrypt / decrypt entire settings dict ----------------------------

    def _encrypt_sensitive(self, data: dict) -> dict:
        """Encrypt all sensitive fields before writing to disk."""
        out = {}
        for k, v in data.items():
            if isinstance(v, dict):
                out[k] = self._encrypt_sensitive(v)
            elif k in self._SENSITIVE_FIELDS and isinstance(v, str) and v:
                out[k] = self._encrypt(v)
            else:
                out[k] = v
        return out

    def _decrypt_sensitive(self, data: dict) -> dict:
        """Decrypt all sensitive fields after reading from disk."""
        out = {}
        for k, v in data.items():
            if isinstance(v, dict):
                out[k] = self._decrypt_sensitive(v)
            elif k in self._SENSITIVE_FIELDS and isinstance(v, str) and v:
                out[k] = self._decrypt(v)
            else:
                out[k] = v
        return out

    # -- File I/O ----------------------------------------------------------

    def _invalidate_cache(self) -> None:
        """Clear the in-memory settings cache."""
        self._cache = None
        self._cache_ts = 0.0

    def load(self) -> dict:
        """
        Load settings from disk with TTL caching. Creates default settings
        file if missing. Returns decrypted settings dict.
        """
        now = time.monotonic()
        if self._cache is not None and (now - self._cache_ts) < self._CACHE_TTL:
            return self._cache

        if not self._path.exists():
            logger.info("Settings file not found, creating defaults at %s", self._path)
            self.save(DEFAULT_SETTINGS)
            return json.loads(json.dumps(DEFAULT_SETTINGS))  # deep copy

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            result = self._decrypt_sensitive(raw)
            self._cache = result
            self._cache_ts = now
            return result
        except Exception as exc:
            logger.error("Failed to load settings: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to load settings.") from exc

    def save(self, data: dict) -> None:
        """Encrypt sensitive fields and write settings to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        encrypted = self._encrypt_sensitive(data)
        self._path.write_text(
            json.dumps(encrypted, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._invalidate_cache()
        logger.debug("Settings saved to %s", self._path)

    def load_masked(self) -> dict:
        """Load settings with sensitive fields masked (for API responses)."""
        data = self.load()
        return self._mask_dict(data)


# Module-level singleton
settings_mgr = SettingsManager()


# ---------------------------------------------------------------------------
# Async load/save wrappers (SQLite primary, JSON fallback + dual-write)
# ---------------------------------------------------------------------------

async def aload_settings() -> dict:
    """
    Load settings: try SQLite first, fall back to JSON.

    The returned dict has sensitive fields already decrypted (via SettingsManager).
    """
    db_data = await _load_settings_from_db()
    if db_data:
        # DB stores data already encrypted by SettingsManager, so decrypt it
        return settings_mgr._decrypt_sensitive(db_data)
    # Fallback to JSON
    return settings_mgr.load()


async def asave_settings(data: dict) -> None:
    """
    Save settings: write to JSON *and* SQLite (dual-write).

    Accepts the decrypted settings dict. Encryption is handled internally.
    """
    # Always write to JSON (backward compat)
    settings_mgr.save(data)
    # Also write to SQLite (encrypted, same as JSON on-disk format)
    encrypted = settings_mgr._encrypt_sensitive(data)
    await _sync_settings_to_db(encrypted)


async def aload_settings_masked() -> dict:
    """Load settings with sensitive fields masked (for API responses)."""
    data = await aload_settings()
    return settings_mgr._mask_dict(data)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/presets")
async def get_llm_presets():
    """List preset LLM providers and their available models."""
    return {"presets": PRESET_PROVIDERS}


@router.get("/providers")
async def get_llm_providers():
    """
    Get saved provider configuration with masked API keys.
    Shows whether each provider has a key configured and is enabled.
    """
    data = await aload_settings()
    providers = data.get("llm_providers", {})
    result = {}
    for name, info in providers.items():
        key_raw = info.get("api_key", "")
        has_key = bool(key_raw)
        result[name] = {
            "has_key": has_key,
            "api_key": key_raw,  # actual key (localhost only, safe)
            "api_key_masked": SettingsManager._mask_key(key_raw),
            "enabled": info.get("enabled", False),
        }
        # Include extra fields per provider
        if name == "vertex_ai":
            result[name]["project_id"] = info.get("project_id", "")
            result[name]["location"] = info.get("location", "global")
        if name == "custom":
            result[name]["base_url"] = info.get("base_url", "")
            result[name]["model"] = info.get("model", "")
        if name == "gemini":
            result[name]["tier"] = info.get("tier", "free")
    return {"providers": result}


@router.put("/providers/{provider}")
async def update_llm_provider(provider: str, body: ProviderKeyUpdate):
    """Save or update the API key (and optional config) for a provider."""
    data = await aload_settings()
    providers = data.setdefault("llm_providers", {})

    if provider not in PRESET_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    entry = providers.get(provider, {})
    entry["api_key"] = body.api_key
    entry["enabled"] = body.enabled if body.enabled is not None else True

    if provider == "vertex_ai":
        if body.project_id is not None:
            entry["project_id"] = body.project_id
        if body.location is not None:
            entry["location"] = body.location

    if provider == "custom":
        if body.base_url is not None:
            entry["base_url"] = body.base_url
        if body.model is not None:
            entry["model"] = body.model

    if provider == "gemini":
        if body.tier is not None and body.tier in ("free", "billing"):
            entry["tier"] = body.tier
        elif "tier" not in entry:
            entry["tier"] = "free"

    providers[provider] = entry
    data["llm_providers"] = providers
    await asave_settings(data)

    logger.info("Updated LLM provider '%s' (enabled=%s)", provider, entry["enabled"])
    return {"status": "ok", "provider": provider}


@router.post("/providers/{provider}/test")
async def test_llm_provider(provider: str):
    """
    Test connectivity for a configured provider.
    Attempts a lightweight API call to verify the key works.
    """
    data = await aload_settings()
    providers = data.get("llm_providers", {})
    entry = providers.get(provider)

    if not entry or not entry.get("api_key"):
        raise HTTPException(status_code=400, detail=f"No API key configured for {provider}.")

    # Attempt a test call
    try:
        api_key = entry["api_key"]
        preset = PRESET_PROVIDERS.get(provider, {})

        if provider == "custom":
            base_url = entry.get("base_url") or None
            model = entry.get("model", "")
            if not model:
                raise HTTPException(
                    status_code=400,
                    detail="Custom provider requires a model name.",
                )
        else:
            # Use user-provided base_url if preset has None (e.g. vertex_ai)
            base_url = preset.get("base_url") or entry.get("base_url")
            model = preset.get("test_model", "")

        if provider == "anthropic":
            # Anthropic uses its own SDK (not OpenAI-compatible)
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=api_key)
            await client.messages.create(
                model=model,
                max_tokens=5,
                messages=[{"role": "user", "content": "Hi"}],
            )
        elif provider == "gemini":
            # Google AI Studio: Gen AI SDK with API key
            from google import genai
            client = genai.Client(api_key=api_key)
            # Step 1: free-tier 모델로 키 유효성 확인
            client.models.generate_content(
                model=model,
                contents="Hi",
            )
            # Step 2: paid-only 모델로 billing 여부 확인
            detected_tier = "free"
            paid_model = preset.get("paid_only_test_model", "")
            if paid_model:
                try:
                    client.models.generate_content(
                        model=paid_model,
                        contents="Hi",
                    )
                    detected_tier = "billing"
                    logger.info("[test] Gemini paid model accessible — tier=billing")
                except Exception as _paid_exc:
                    logger.info(
                        "[test] Gemini paid model unavailable (tier=free): %s",
                        _paid_exc,
                    )
                    detected_tier = "free"
            # 감지된 tier를 settings에 저장
            entry["tier"] = detected_tier
            providers[provider] = entry
            data["llm_providers"] = providers
            await asave_settings(data)
            logger.info(
                "Provider 'gemini' test succeeded (model=%s, tier=%s).",
                model, detected_tier,
            )
            return {
                "status": "ok",
                "provider": provider,
                "model": model,
                "tier": detected_tier,
            }
        elif provider == "vertex_ai":
            # Vertex AI: Gen AI SDK with API key
            from google import genai

            _vx_kwargs: dict = {"vertexai": True, "api_key": api_key}
            project_id = entry.get("project_id", "")
            location = entry.get("location", "")
            if project_id:
                _vx_kwargs["project"] = project_id
            if location and location != "global":
                _vx_kwargs["location"] = location

            client = genai.Client(**_vx_kwargs)
            client.models.generate_content(
                model=model,
                contents="Hi",
            )
        else:
            # OpenAI and other OpenAI-compatible providers
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )

        logger.info("Provider '%s' test succeeded (model=%s).", provider, model)
        return {"status": "ok", "provider": provider, "model": model}

    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Provider '%s' test failed: %s", provider, exc)
        raise HTTPException(
            status_code=502,
            detail=f"Connection test failed for {provider}: {exc}",
        ) from exc


@router.put("/legal-api")
async def update_legal_api_settings(body: LegalApiUpdate):
    """Update legal API settings (OC key, max API calls)."""
    data = await aload_settings()
    legal = data.get("legal_api", dict(DEFAULT_SETTINGS["legal_api"]))
    update = body.model_dump(exclude_none=True)
    legal.update(update)
    data["legal_api"] = legal
    await asave_settings(data)
    logger.info("Legal API settings updated: %s", list(update.keys()))
    masked = settings_mgr._mask_dict(legal)
    return {"status": "ok", "legal_api": masked}


@router.get("/legal-api")
async def get_legal_api_settings():
    """Get legal API settings with both raw and masked keys."""
    data = await aload_settings()
    legal = data.get("legal_api", {})
    law_key = legal.get("law_api_key", "")
    prec_key = legal.get("precedent_api_key", "")
    return {
        "legal_api": {
            "law_api_key": law_key,
            "law_api_key_masked": settings_mgr._mask_key(law_key),
            "precedent_api_key": prec_key,
            "precedent_api_key_masked": settings_mgr._mask_key(prec_key),
            "max_api_calls_per_round": legal.get("max_api_calls_per_round", 10),
            "external_search_enabled": legal.get("external_search_enabled", True),
        },
    }


@router.post("/legal-api/test")
async def test_legal_api():
    """Test the legal API OC key by making a simple search request."""
    data = await aload_settings()
    legal = data.get("legal_api", {})
    api_key = (legal.get("law_api_key") or legal.get("precedent_api_key", "")).strip()

    if not api_key:
        raise HTTPException(status_code=400, detail="법률 API OC 키가 설정되지 않았습니다. Settings에서 OC 키를 입력해주세요.")

    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://www.law.go.kr/DRF/lawSearch.do",
                params={
                    "OC": api_key,
                    "target": "law",
                    "type": "JSON",
                    "query": "민법",
                    "display": "1",
                },
            )
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"Legal API returned HTTP {resp.status_code}",
                )
            result = resp.json()
            # Detect error response from Korean legal API
            if isinstance(result, dict) and "result" in result and "msg" in result:
                msg = result.get("msg", "")
                return {"status": "error", "message": f"{result['result']}: {msg}"}
            if isinstance(result, dict) and "LawSearch" in result:
                total = result["LawSearch"].get("totalCnt", 0)
                return {"status": "ok", "message": f"법률 API 연결 성공 ({total}건)"}
            return {"status": "error", "message": "Unexpected response format"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Legal API test failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Legal API test failed: {exc}") from exc


@router.get("/models")
async def get_available_models():
    """
    List available models, filtered to only providers that have a valid
    (non-empty) API key configured and are enabled.

    Gemini: only listed when a key is present. If tier=="free", only models
    marked `free_available=True` are returned with zero pricing. If tier==
    "billing", all models are returned with paid pricing.
    """
    data = await aload_settings()
    providers = data.get("llm_providers", {})
    available: list[dict[str, Any]] = []

    for name, info in providers.items():
        preset = PRESET_PROVIDERS.get(name, {})
        model_prices = preset.get("models", {})
        has_key = bool(info.get("api_key")) and info.get("enabled", False)

        if not has_key:
            continue

        # Gemini: tier에 따라 모델/가격 필터링 (키가 있을 때만 도달).
        if name == "gemini":
            gemini_tier = info.get("tier", "free")
            for m, prices in model_prices.items():
                if gemini_tier == "free" and not prices.get("free_available", True):
                    continue  # free에서는 paid-only 모델 숨김
                if gemini_tier == "free":
                    input_p = 0
                    output_p = 0
                else:
                    input_p = prices.get("input", 0)
                    output_p = prices.get("output", 0)
                available.append({
                    "id": f"{name}/{m}",
                    "name": m,
                    "provider": name,
                    "input_price": input_p,
                    "output_price": output_p,
                    "tier": gemini_tier,
                    "has_key": True,
                })
            continue

        if name == "custom" and info.get("model"):
            custom_model = info["model"]
            available.append({
                "id": f"{name}/{custom_model}",
                "name": custom_model,
                "provider": name,
                "input_price": 0,
                "output_price": 0,
            })
        else:
            for m, prices in model_prices.items():
                available.append({
                    "id": f"{name}/{m}",
                    "name": m,
                    "provider": name,
                    "input_price": prices.get("input", 0),
                    "output_price": prices.get("output", 0),
                })

    return {"models": available}


@router.get("/debate")
async def get_debate_settings():
    """Get current debate configuration."""
    data = await aload_settings()
    return {"debate": data.get("debate", DEFAULT_SETTINGS["debate"])}


@router.put("/debate")
async def update_debate_settings(body: DebateSettingsUpdate):
    """Update debate configuration (partial update supported)."""
    data = await aload_settings()
    debate = data.get("debate", dict(DEFAULT_SETTINGS["debate"]))

    # Apply only provided fields
    update = body.model_dump(exclude_none=True)
    debate.update(update)

    data["debate"] = debate
    await asave_settings(data)
    logger.info("Debate settings updated: %s", list(update.keys()))
    return {"status": "ok", "debate": debate}


@router.get("")
async def get_all_settings():
    """Get all settings with sensitive fields masked."""
    return {"settings": await aload_settings_masked()}

"""
Multi-LLM provider registry for LawNuri.

Defines preset LLM providers (OpenAI, Gemini, Anthropic) and
Pydantic models for managing provider configurations, including
user-defined custom providers.
"""

from typing import Optional

from pydantic import BaseModel


PRESET_PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": [
            "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano",
            "gpt-5.2", "gpt-5-mini",
            "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
            "gpt-4o", "gpt-4o-mini",
            "o3", "o4-mini",
        ],
    },
    "gemini": {
        "name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": [
            "gemini-3.1-pro-preview", "gemini-3-flash-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
        ],
    },
    "anthropic": {
        "name": "Anthropic",
        "base_url": "https://api.anthropic.com/v1/",
        "models": [
            "claude-opus-4-6", "claude-opus-4-5",
            "claude-opus-4-1", "claude-opus-4",
            "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-sonnet-4",
            "claude-haiku-4-5", "claude-haiku-3-5",
        ],
    },
    "vertex_ai": {
        "name": "Vertex AI (Google Cloud)",
        "base_url": None,  # user must provide project-specific URL
        "models": [
            "gemini-3.1-pro-preview", "gemini-3-flash-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
        ],
    },
}


class LLMProviderConfig(BaseModel):
    """Configuration for a registered LLM provider."""

    provider_id: str
    name: str
    base_url: str
    api_key: str
    models: list[str]


class CustomProvider(BaseModel):
    """User-defined custom LLM provider."""

    provider_id: str
    name: str
    base_url: str
    api_key: Optional[str] = None
    models: list[str] = []

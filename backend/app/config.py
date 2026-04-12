"""Environment configuration for LawNuri backend.

Most settings (LLM providers, API keys, agent profiles) are managed through
the Settings UI and stored in backend/data/settings.json. This module only
handles minimal server-level configuration.
"""

import os
import sys
from dotenv import load_dotenv

if getattr(sys, 'frozen', False):
    _base_dir = os.path.dirname(sys.executable)
    dotenv_path = os.path.join(_base_dir, '..', '.env')
else:
    _base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    dotenv_path = os.path.join(_base_dir, '..', '.env')

load_dotenv(dotenv_path)


class Config:
    """Minimal server configuration. Most settings live in Settings UI."""

    UVICORN_PORT: int = int(os.getenv("UVICORN_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

    DATA_DIR: str = os.path.join(_base_dir, 'data')
    UPLOADS_DIR: str = os.path.join(_base_dir, 'data', 'uploads')
    SETTINGS_FILE: str = os.path.join(_base_dir, 'data', 'settings.json')

    DB_BACKEND: str = os.getenv("DB_BACKEND", "sqlite")
    DB_PATH: str = os.path.join(_base_dir, 'data', 'lawnuri.db')
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://lawnuri:lawnuri@localhost:5432/lawnuri",
    )

    MAX_CONTENT_LENGTH: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: set = {'pdf', 'md', 'txt', 'markdown'}


config = Config()

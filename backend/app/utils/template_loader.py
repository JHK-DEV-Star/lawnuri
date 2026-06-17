"""
Complaint (소장) template loader for LawNuri.

Reads complaint-document templates from a folder (config.TEMPLATES_DIR, or a
settings override) as PLAIN content — NOT through the RAG indexer (which would
anonymize and chunk, destroying template structure).

Supported formats:
  - .json  → structured template parsed directly (preferred). See SCHEMA below.
  - .docx  → full text via FileParser (python-docx).
  - .pdf / .md / .txt → full text via FileParser.

JSON template schema (one file = one template):
    {
      "id": "civil_loan_claim",            # optional; defaults to filename slug
      "title": "대여금 청구의 소",
      "category": "민사",                   # legal domain
      "subcategory": "이행의 소 · 금전지급청구",
      "doc_type": "소장",                   # 소장 | 신청서 | 고소장 | ...
      "description": "...",                 # used as the summary
      "when_to_use": "...",                 # shown to the selection LLM
      "sections": [{"name": "청구취지", "guide": "..."}]
    }

A module-level singleton caches parsed templates so files are read once at
startup (warm load in the FastAPI lifespan) and can be reloaded on demand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from app.config import config
from app.utils.file_parser import FileParser
from app.utils.logger import logger

# Extensions we treat as templates (superset handled below; .json special-cased).
_TEMPLATE_EXTS = {".json", ".docx", ".pdf", ".md", ".txt"}


def _slugify(name: str) -> str:
    """Filename stem → a stable id slug."""
    stem = Path(name).stem.strip().lower()
    return "".join(c if c.isalnum() else "_" for c in stem).strip("_") or "template"


class TemplateLoader:
    """Singleton loader/cache for complaint templates."""

    _instance: Optional["TemplateLoader"] = None

    def __init__(self) -> None:
        self._templates: dict[str, dict] = {}   # id -> template dict
        self._loaded: bool = False
        self._loaded_dir: str = ""

    # ---- singleton access -------------------------------------------------
    @classmethod
    def instance(cls) -> "TemplateLoader":
        if cls._instance is None:
            cls._instance = TemplateLoader()
        return cls._instance

    # ---- directory resolution --------------------------------------------
    @staticmethod
    def _resolve_dir() -> str:
        """Settings override (complaint.templates_dir) if non-empty, else config default."""
        templates_dir = config.TEMPLATES_DIR
        try:
            from app.api.settings import settings_mgr
            complaint_cfg = settings_mgr.load().get("complaint", {})
            override = (complaint_cfg.get("templates_dir") or "").strip()
            if override:
                templates_dir = override
        except Exception:
            # Settings not available yet (e.g. early startup) — use config default.
            pass
        return templates_dir

    # ---- loading ----------------------------------------------------------
    def load_all(self, templates_dir: str | None = None) -> int:
        """Read every template file in the folder into the cache. Returns count."""
        target = templates_dir or self._resolve_dir()
        path = Path(target)
        templates: dict[str, dict] = {}

        if not path.exists():
            logger.info("[TemplateLoader] Templates dir does not exist yet: %s", target)
            self._templates = {}
            self._loaded = True
            self._loaded_dir = target
            return 0

        for fp in sorted(path.iterdir()):
            if not fp.is_file() or fp.suffix.lower() not in _TEMPLATE_EXTS:
                continue
            if fp.name.startswith("."):  # skip sidecars like .summaries.json
                continue
            try:
                tpl = self._parse_file(fp)
                if tpl:
                    templates[tpl["id"]] = tpl
            except Exception as exc:  # one bad file must not break the rest
                logger.warning("[TemplateLoader] Failed to parse %s: %s", fp.name, exc)

        self._templates = templates
        self._loaded = True
        self._loaded_dir = target
        logger.info("[TemplateLoader] Loaded %d template(s) from %s", len(templates), target)
        return len(templates)

    def _parse_file(self, fp: Path) -> dict | None:
        """Parse one template file into the cache dict shape."""
        ext = fp.suffix.lower()
        if ext == ".json":
            data = json.loads(fp.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                logger.warning("[TemplateLoader] %s is not a JSON object — skipped.", fp.name)
                return None
            tid = (data.get("id") or _slugify(fp.name)).strip()
            sections = data.get("sections", [])
            return {
                "id": tid,
                "title": data.get("title") or fp.stem,
                "category": data.get("category", ""),
                "subcategory": data.get("subcategory", ""),
                "doc_type": data.get("doc_type", "소장"),
                "description": data.get("description", ""),
                "when_to_use": data.get("when_to_use", ""),
                "sections": sections,
                "full_text": json.dumps(data, ensure_ascii=False, indent=2),
                "source_file": fp.name,
            }
        # Non-JSON: read full text, derive a heuristic summary.
        text = FileParser.extract_text(fp)
        summary = self._heuristic_summary(text)
        return {
            "id": _slugify(fp.name),
            "title": fp.stem,
            "category": "",
            "subcategory": "",
            "doc_type": "소장",
            "description": summary,
            "when_to_use": "",
            "sections": [],
            "full_text": text,
            "source_file": fp.name,
        }

    @staticmethod
    def _heuristic_summary(text: str, limit: int = 400) -> str:
        """Cheap fallback summary for non-JSON templates: first non-empty lines."""
        snippet = " ".join(line.strip() for line in text.splitlines() if line.strip())
        return snippet[:limit]

    def ensure_loaded(self) -> None:
        """Lazy-load if the cache is empty (e.g. fresh install / first request)."""
        if not self._loaded:
            self.load_all()

    def refresh(self) -> int:
        """Force a reload from disk (e.g. after the user drops new template files)."""
        return self.load_all()

    # ---- access -----------------------------------------------------------
    def list_catalog(self) -> list[dict]:
        """Compact catalog for the selection LLM and the settings UI."""
        self.ensure_loaded()
        return [
            {
                "id": t["id"],
                "title": t["title"],
                "category": t.get("category", ""),
                "doc_type": t.get("doc_type", ""),
                "description": t.get("description", ""),
                "when_to_use": t.get("when_to_use", ""),
            }
            for t in self._templates.values()
        ]

    def catalog_text(self) -> str:
        """Render the catalog as a prompt-ready string for template selection."""
        lines = []
        for t in self.list_catalog():
            meta = "/".join(x for x in (t.get("category"), t.get("doc_type")) if x)
            desc = t.get("when_to_use") or t.get("description") or ""
            lines.append(f"[{t['id']}] {t['title']} ({meta}) — {desc}")
        return "\n".join(lines)

    def get(self, template_id: str) -> dict | None:
        self.ensure_loaded()
        return self._templates.get(template_id)

    def count(self) -> int:
        self.ensure_loaded()
        return len(self._templates)


def get_template_loader() -> TemplateLoader:
    return TemplateLoader.instance()

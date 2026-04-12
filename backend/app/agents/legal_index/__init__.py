"""
Legal system index loader.

Provides jurisdiction-specific legal system reference texts that help
LLM agents formulate accurate search queries. Index files are plain text
stored alongside this module and loaded on demand.

Usage:
    from app.agents.legal_index import get_legal_index
    index_text = get_legal_index("korea")
"""

from pathlib import Path

_INDEX_DIR = Path(__file__).parent


def get_legal_index(jurisdiction: str = "korea") -> str:
    """
    Load the legal system index text for the given jurisdiction.

    Args:
        jurisdiction: Country/region code matching a .txt file name
                      (e.g., "korea", "usa", "japan").

    Returns:
        The index text, or empty string if the file does not exist.
    """
    path = _INDEX_DIR / f"{jurisdiction}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def list_jurisdictions() -> list[str]:
    """Return a list of available jurisdiction codes."""
    return [p.stem for p in _INDEX_DIR.glob("*.txt")]

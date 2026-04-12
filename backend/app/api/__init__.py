"""
API router package for LawNuri backend.
Exports all routers for inclusion in the main FastAPI app.
"""

from app.api.debate import router as debate_router
from app.api.rag import router as rag_router
from app.api.report import router as report_router
from app.api.settings import router as settings_router

__all__ = [
    "debate_router",
    "rag_router",
    "report_router",
    "settings_router",
]

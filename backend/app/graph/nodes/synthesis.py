"""
Synthesis node — complaint (소장) mode only.

Runs AFTER final_judgment. Based on the full debate over the chosen template's
validity and the judges' verdicts, it:
  1) produces a drafting strategy (complaint_analysis / 작성 방향), and
  2) drafts the actual 소장 (drafted_complaint) following the template structure.

In debate mode this node is never reached (gated by should_synthesize edge).
"""

from __future__ import annotations

from app.agents.complaint_drafter import (
    build_synthesis_prompt,
    build_draft_prompt,
    COMPLAINT_DISCLAIMER,
)
from app.utils.llm_client import LLMClient
from app.utils.logger import logger


def _build_debate_summary(debate_log: list, max_entries: int = 12) -> str:
    """Compact transcript for the synthesis prompts."""
    lines = []
    for entry in (debate_log or [])[-max_entries:]:
        team = entry.get("team", "?")
        speaker = entry.get("speaker", "?")
        rnd = entry.get("round", "?")
        statement = str(entry.get("statement", entry.get("content", "")))
        lines.append(f"[R{rnd}|{team}|{speaker}] {statement[:1200]}")
    return "\n\n".join(lines) if lines else "(토론 기록 없음)"


def _resolve_language() -> str:
    """complaint.language → debate.language → 'ko'."""
    try:
        from app.api.settings import settings_mgr
        s = settings_mgr.load()
        return (
            (s.get("complaint", {}) or {}).get("language")
            or (s.get("debate", {}) or {}).get("language")
            or "ko"
        )
    except Exception:
        return "ko"


async def synthesis_node(state: dict, llm_client: LLMClient, searcher=None) -> dict:
    """Generate the drafting strategy and the drafted 소장."""
    selected_template = state.get("selected_template", {}) or {}
    situation_brief = state.get("situation_brief", "")
    opinion_a = state.get("opinion_a", "")
    opinion_b = state.get("opinion_b", "")
    verdicts = state.get("verdicts", [])
    debate_summary = _build_debate_summary(state.get("debate_log", []))
    language = _resolve_language()

    logger.info(
        "[synthesis] Drafting 소장 (template=%s) for debate %s",
        selected_template.get("id", "?"), state.get("debate_id", "?"),
    )

    # Step 1: drafting strategy (종합 결과)
    synthesis_prompt = build_synthesis_prompt(
        situation_brief=situation_brief,
        selected_template=selected_template,
        opinion_a=opinion_a,
        opinion_b=opinion_b,
        verdicts=verdicts,
        debate_summary=debate_summary,
        language=language,
    )
    try:
        complaint_analysis = await llm_client.achat(
            [{"role": "user", "content": synthesis_prompt}],
            temperature=0.4, max_tokens=4000,
        )
    except Exception as exc:
        logger.warning("[synthesis] Strategy generation failed: %s", exc)
        complaint_analysis = ""

    # Step 2: the actual 소장 draft
    draft_prompt = build_draft_prompt(
        situation_brief=situation_brief,
        selected_template=selected_template,
        synthesis=complaint_analysis,
        opinion_a=opinion_a,
        debate_summary=debate_summary,
        language=language,
    )
    try:
        drafted_complaint = await llm_client.achat(
            [{"role": "user", "content": draft_prompt}],
            temperature=0.5, max_tokens=8000,
        )
    except Exception as exc:
        logger.warning("[synthesis] Draft generation failed: %s", exc)
        drafted_complaint = ""

    # Always append the disclaimer to the generated 소장 (deterministic).
    if drafted_complaint.strip():
        drafted_complaint = drafted_complaint.rstrip() + "\n\n" + COMPLAINT_DISCLAIMER

    return {
        "complaint_analysis": {
            "strategy": complaint_analysis,
            "template_id": selected_template.get("id", ""),
            "template_title": selected_template.get("title", ""),
            "selection_reason": selected_template.get("selection_reason", ""),
        },
        "drafted_complaint": drafted_complaint,
        "current_phase": "completed",
    }

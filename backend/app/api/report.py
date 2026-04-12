"""
Report API endpoints for LawNuri backend.

Generates comprehensive debate reports using LLM analysis of the completed
debate state, including executive summary, evidence inventory, argument
analysis, and judge verdicts. Reports are cached in the debate state once
generated and can be downloaded as PDF.

Report data (judge verdicts, evidence inventory, situation analysis, and
transcript) is built directly from debate state, not LLM-generated.
"""

from __future__ import annotations

import io
import os
from urllib.parse import quote
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from fpdf import FPDF

from app.agents.language import get_language_instruction
from app.api.debate import DebateStore, _build_llm_client
from app.api.settings import settings_mgr
from app.utils.logger import logger

router = APIRouter(prefix="/api/report", tags=["report"])


def _build_debate_summary_for_prompt(state: dict) -> str:
    """
    Build a condensed textual summary of the debate state suitable
    for inclusion in an LLM prompt (keeps token count manageable).
    """
    analysis = state.get("analysis", {})
    debate_log = state.get("debate_log", [])
    all_evidences = state.get("all_evidences", [])
    verdicts = state.get("verdicts", [])
    agents = state.get("agents", [])

    lines: list[str] = []
    lines.append(f"Topic: {analysis.get('topic', 'N/A')}")
    lines.append(f"Opinion A: {analysis.get('opinion_a', 'N/A')}")
    lines.append(f"Opinion B: {analysis.get('opinion_b', 'N/A')}")
    lines.append(f"Key Issues: {', '.join(analysis.get('key_issues', []))}")
    lines.append(f"Total Rounds: {state.get('current_round', 0)}")
    lines.append(f"Total Log Entries: {len(debate_log)}")
    lines.append(f"Total Evidence Pieces: {len(all_evidences)}")
    lines.append("")

    lines.append("=== Agents ===")
    for a in agents:
        lines.append(
            f"  {a.get('agent_id')}: {a.get('name')} "
            f"(role={a.get('role')}, team={a.get('team')}, "
            f"specialty={a.get('specialty')})"
        )
    lines.append("")

    agent_names: dict[str, str] = {}
    for a in agents:
        agent_names[a.get("agent_id", "")] = a.get("name", a.get("agent_id", ""))

    lines.append("=== Debate Log ===")
    for entry in debate_log:
        content = entry.get("statement", entry.get("content", ""))
        speaker_id = entry.get("speaker", "?")
        speaker_name = agent_names.get(speaker_id, speaker_id)
        lines.append(
            f"[Round {entry.get('round', '?')}] "
            f"{entry.get('team', '?')} ({speaker_name}): {content}"
        )
    lines.append("")

    lines.append("=== Evidence Inventory ===")
    for ev in all_evidences:
        if isinstance(ev, dict):
            lines.append(
                f"  - [{ev.get('source_type', ev.get('source', 'unknown'))}] "
                f"by {ev.get('submitted_by', ev.get('team', '?'))}: "
                f"{ev.get('source_detail', '')}"
            )
    lines.append("")

    if verdicts:
        lines.append("=== Judge Verdicts ===")
        for v in verdicts:
            lines.append(
                f"  Judge {v.get('judge_id', '?')}: winner={v.get('winner', '?')} "
                f"confidence={v.get('confidence', '?')}"
            )
            lines.append(f"    Reasoning: {v.get('reasoning', 'N/A')}")
            for team_key in ("team_a", "team_b"):
                ts = v.get(f"score_{team_key}", {})
                if isinstance(ts, dict) and ts:
                    lines.append(
                        f"    {team_key}: evidence={ts.get('evidence_quality', '?')} "
                        f"logic={ts.get('legal_reasoning', '?')} "
                        f"persuasion={ts.get('persuasiveness', '?')}"
                    )

    return "\n".join(lines)


async def _generate_report(state: dict) -> dict:
    """
    Generate a full debate report by calling the LLM with the debate state.

    Returns the report dict with all sections.
    """
    summary_text = _build_debate_summary_for_prompt(state)

    prompt = f"""\
You are a legal debate analyst. Based on the following debate data,
generate a comprehensive report.

{summary_text}

Output ONLY valid JSON with the following structure:
{{
    "executive_summary": {{
        "result": "team_a or team_b or draw (which team won)",
        "summary": "A detailed 5-8 sentence overview of the debate, outcome, and key takeaways"
    }},
    "decisive_evidence": Rank the most impactful evidence BY CATEGORY.
    For each category, list the top 3 items that most influenced the debate outcome.
    - "precedent": Court precedents cited during the debate
    - "statute": Laws and statutes cited
    - "statement": Key arguments or statements made by debaters
    Each item needs: rank (1-3), description, source, team (team_a/team_b), impact
    {{
        "precedent": [
            {{"rank": 1, "description": "What the evidence is", "source": "Where it came from", "team": "team_a or team_b", "impact": "How it affected the debate outcome"}}
        ],
        "statute": [
            {{"rank": 1, "description": "What the evidence is", "source": "Where it came from", "team": "team_a or team_b", "impact": "How it affected the debate outcome"}}
        ],
        "statement": [
            {{"rank": 1, "description": "What the evidence is", "source": "Where it came from", "team": "team_a or team_b", "impact": "How it affected the debate outcome"}}
        ]
    }},
    "argument_analysis": {{
        "team_a": {{
            "strongest": ["Best argument 1", "Best argument 2"],
            "weakest": ["Weakest point 1"],
            "missing_evidence": ["Evidence they should have cited"]
        }},
        "team_b": {{
            "strongest": ["Best argument 1", "Best argument 2"],
            "weakest": ["Weakest point 1"],
            "missing_evidence": ["Evidence they should have cited"]
        }}
    }},
    "debate_flow_summary": [
        {{
            "round": 1,
            "team_a_summary": "What team A argued",
            "team_b_summary": "What team B argued",
            "key_moment": "Notable event in this round"
        }}
    ],
    "recommendations": [
        "Recommendation 1 for further legal action or analysis",
        "Recommendation 2"
    ]
}}
"""

    settings = settings_mgr.load()
    language = settings.get("debate", {}).get("language", "ko")
    prompt += get_language_instruction(language)

    llm = _build_llm_client(state.get("default_model"))

    try:
        report = await llm.achat_json(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4096,
        )
    except Exception as exc:
        logger.error("Report generation failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Report generation failed: {exc}") from exc

    result_map = {"team_a": "Team A 승리", "team_b": "Team B 승리", "draw": "무승부"}
    exec_sum = report.get("executive_summary", {})
    if isinstance(exec_sum, dict):
        raw = exec_sum.get("result", "")
        exec_sum["result"] = result_map.get(raw, raw)

    def _normalize_scores(v: dict) -> dict:
        """Convert score_team_a/score_team_b to unified scores dict."""
        result = {}
        for team_key in ("team_a", "team_b"):
            raw = v.get(f"score_{team_key}", {})
            if isinstance(raw, dict) and raw:
                result[team_key] = {
                    "evidence_quality": raw.get("evidence_quality", 0),
                    "argument_logic": raw.get("legal_reasoning", 0),
                    "persuasiveness": raw.get("persuasiveness", 0),
                    "rebuttal_effectiveness": raw.get("rebuttal_effectiveness", 0),
                    "overall": raw.get("overall", 0),
                }
        return result

    report["judge_verdicts"] = []
    for v in state.get("verdicts", []):
        if not isinstance(v, dict):
            continue
        scores = v.get("scores", {})
        if not scores:
            scores = _normalize_scores(v)
        report["judge_verdicts"].append({
            "judge_id": v.get("judge_id", ""),
            "judge_name": v.get("judge_name", v.get("judge_id", "Judge")),
            "winner": v.get("winner", v.get("verdict", "")),
            "confidence": v.get("confidence", 0),
            "reasoning_summary": v.get("reasoning", ""),
            "scores": scores,
        })

    all_ev = state.get("all_evidences", [])
    by_source: dict[str, int] = {}
    by_team = {"team_a": 0, "team_b": 0}
    for ev in all_ev:
        if not isinstance(ev, dict):
            continue
        src = ev.get("source_type", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
        submitted = str(ev.get("submitted_by", ""))
        if "team_a" in submitted:
            by_team["team_a"] += 1
        elif "team_b" in submitted:
            by_team["team_b"] += 1
    report["evidence_inventory"] = {"by_source": by_source, "by_team": by_team}

    all_evidences = state.get("all_evidences", [])
    evidence_items = []
    seen = set()
    for ev in all_evidences:
        if not isinstance(ev, dict):
            continue
        detail = ev.get("source_detail", "")
        if detail and detail not in seen:
            seen.add(detail)
            evidence_items.append({
                "source_type": ev.get("source_type", ""),
                "source_detail": detail,
                "submitted_by": ev.get("submitted_by", ""),
                "url": ev.get("url", ""),
            })
    report["evidence_items"] = evidence_items

    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["debate_id"] = state.get("debate_id", "")
    report["topic"] = state.get("analysis", {}).get("topic", "")

    all_evidences = state.get("all_evidences", [])
    url_map = {}
    for e in all_evidences:
        url = e.get("url", "")
        detail = e.get("source_detail", "")
        if url and detail:
            url_map[detail] = url

    decisive = report.get("decisive_evidence", {})
    if isinstance(decisive, dict):
        # New categorized format: iterate through each category's list
        for category_items in decisive.values():
            if isinstance(category_items, list):
                for item in category_items:
                    source = item.get("source", "")
                    for detail, url in url_map.items():
                        if detail in source or source in detail:
                            item["url"] = url
                            break
    elif isinstance(decisive, list):
        # Backward compatibility: old flat list format
        for item in decisive:
            source = item.get("source", "")
            for detail, url in url_map.items():
                if detail in source or source in detail:
                    item["url"] = url
                    break

    report["situation_analysis"] = {
        "topic": state.get("topic", ""),
        "opinion_a": state.get("opinion_a", ""),
        "opinion_b": state.get("opinion_b", ""),
        "key_issues": state.get("key_issues", []),
        "parties": state.get("parties", []),
        "timeline": state.get("timeline", []),
        "causal_chain": state.get("causal_chain", []),
        "key_facts": state.get("key_facts", []),
        "situation_brief": state.get("situation_brief", ""),
    }

    # Build agent_id → name mapping for speaker name resolution
    agent_name_map: dict[str, str] = {}
    for agent_list_key in ("team_a_agents", "team_b_agents", "judge_agents"):
        for agent in state.get(agent_list_key, []):
            if isinstance(agent, dict):
                aid = agent.get("agent_id", "")
                aname = agent.get("name", aid)
                if aid:
                    agent_name_map[aid] = aname

    # Add debate transcript (representative statements + judge Q&A from debate_log only)
    transcript = []
    for entry in state.get("debate_log", []):
        if not isinstance(entry, dict):
            continue
        speaker_id = entry.get("speaker", "")
        speaker_name = agent_name_map.get(speaker_id, speaker_id)
        entry_type = entry.get("entry_type", "")

        # Q&A entries: add "Q: " or "A: " prefix to statement
        statement = entry.get("statement", "")
        team = entry.get("team", "")
        if entry_type == "judge_question":
            statement = f"Q: {statement}"
            team = "judge"
        elif entry_type == "qa_answer":
            statement = f"A: {statement}"

        transcript.append({
            "round": entry.get("round", 0),
            "team": team,
            "speaker": speaker_name,
            "statement": statement,
            "evidence": entry.get("evidence", []),
            "timestamp": entry.get("timestamp", ""),
        })
    report["transcript"] = transcript

    return report


def _report_to_markdown(report: dict, state: dict) -> str:
    """Convert a report dict to a readable Markdown document."""
    analysis = state.get("analysis", {})
    lines: list[str] = []

    lines.append(f"# Debate Report: {report.get('topic', analysis.get('topic', 'N/A'))}")
    lines.append("")
    lines.append(f"**Generated:** {report.get('generated_at', 'N/A')}")
    lines.append(f"**Debate ID:** {report.get('debate_id', 'N/A')}")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    exec_summary = report.get("executive_summary", "N/A")
    if isinstance(exec_summary, dict):
        result = exec_summary.get("result", "")
        if result:
            lines.append(f"**Result:** {result}")
            lines.append("")
        lines.append(exec_summary.get("summary", "N/A"))
    else:
        lines.append(str(exec_summary))
    lines.append("")

    # Decisive Evidence
    lines.append("## Decisive Evidence")
    lines.append("")
    decisive = report.get("decisive_evidence", {})

    def _md_evidence_items(items):
        for ev in items:
            lines.append(f"#### Rank {ev.get('rank', '?')}")
            lines.append(f"- **Description:** {ev.get('description', 'N/A')}")
            lines.append(f"- **Source:** {ev.get('source', 'N/A')}")
            url = ev.get("url", "")
            if url:
                lines.append(f"- **원문 보기:** [{url}]({url})")
            lines.append(f"- **Team:** {ev.get('team', 'N/A')}")
            lines.append(f"- **Impact:** {ev.get('impact', 'N/A')}")
            lines.append("")

    if isinstance(decisive, dict):
        category_labels = {
            "precedent": "Precedents",
            "statute": "Statutes",
            "statement": "Statements",
        }
        for cat_key, cat_label in category_labels.items():
            items = decisive.get(cat_key, [])
            if items:
                lines.append(f"### {cat_label}")
                lines.append("")
                _md_evidence_items(items)
    elif isinstance(decisive, list):
        for ev in decisive:
            lines.append(f"### Rank {ev.get('rank', '?')}")
            lines.append(f"- **Description:** {ev.get('description', 'N/A')}")
            lines.append(f"- **Source:** {ev.get('source', 'N/A')}")
            url = ev.get("url", "")
            if url:
                lines.append(f"- **원문 보기:** [{url}]({url})")
            lines.append(f"- **Team:** {ev.get('team', 'N/A')}")
            lines.append(f"- **Impact:** {ev.get('impact', 'N/A')}")
            lines.append("")

    # Judge Verdicts
    lines.append("## Judge Verdicts")
    lines.append("")
    for v in report.get("judge_verdicts", []):
        lines.append(f"### {v.get('judge_name', v.get('judge_id', 'Judge'))}")
        lines.append(f"- **Winner:** {v.get('winner', 'N/A')}")
        lines.append(f"- **Confidence:** {v.get('confidence', 'N/A')}")
        lines.append(f"- **Reasoning:** {v.get('reasoning_summary', 'N/A')}")
        scores = v.get("scores", {})
        for team_key in ("team_a", "team_b"):
            ts = scores.get(team_key, {})
            if ts:
                label = "Team A" if team_key == "team_a" else "Team B"
                lines.append(
                    f"- **{label} Scores:** "
                    f"Evidence={ts.get('evidence_quality', '?')}, "
                    f"Logic={ts.get('argument_logic', '?')}, "
                    f"Persuasiveness={ts.get('persuasiveness', '?')}"
                )
        lines.append("")

    # Argument Analysis
    lines.append("## Argument Analysis")
    lines.append("")
    arg_analysis = report.get("argument_analysis", {})
    for team_key, label in [("team_a", "Team A"), ("team_b", "Team B")]:
        ta = arg_analysis.get(team_key, {})
        lines.append(f"### {label}")
        lines.append("")
        lines.append("**Strongest Arguments:**")
        for s in ta.get("strongest", []):
            lines.append(f"- {s}")
        lines.append("")
        lines.append("**Weakest Points:**")
        for w in ta.get("weakest", []):
            lines.append(f"- {w}")
        lines.append("")
        lines.append("**Missing Evidence:**")
        for m in ta.get("missing_evidence", []):
            lines.append(f"- {m}")
        lines.append("")

    # Evidence Inventory
    lines.append("## Evidence Inventory")
    lines.append("")
    inv = report.get("evidence_inventory", {})
    by_source = inv.get("by_source", {})
    if by_source:
        lines.append("**By Source:**")
        lines.append("")
        lines.append("| Source Type | Count |")
        lines.append("|---|---|")
        for src, count in by_source.items():
            lines.append(f"| {src} | {count} |")
        lines.append("")
    by_team = inv.get("by_team", {})
    if by_team:
        lines.append("**By Team:**")
        lines.append("")
        lines.append("| Team | Count |")
        lines.append("|---|---|")
        for team, count in by_team.items():
            lines.append(f"| {team} | {count} |")
        lines.append("")

    # Debate Flow Summary
    lines.append("## Debate Flow Summary")
    lines.append("")
    for rd in report.get("debate_flow_summary", []):
        lines.append(f"### Round {rd.get('round', '?')}")
        lines.append(f"- **Team A:** {rd.get('team_a_summary', 'N/A')}")
        lines.append(f"- **Team B:** {rd.get('team_b_summary', 'N/A')}")
        key_moment = rd.get("key_moment", "")
        if key_moment:
            lines.append(f"- **Key Moment:** {key_moment}")
        lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    for rec in report.get("recommendations", []):
        lines.append(f"- {rec}")
    lines.append("")

    return "\n".join(lines)


def _report_to_pdf(report: dict, state: dict) -> bytes:
    """Convert a report dict to a styled PDF matching the Flutter report screen."""
    analysis = state.get("analysis", {})

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    korean_font_loaded = False
    font_candidates = [
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "malgun.ttf"),
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/nanum/NanumGothic.ttf",
    ]
    for font_path in font_candidates:
        if os.path.isfile(font_path):
            try:
                pdf.add_font("KoreanFont", "", font_path, uni=True)
                pdf.add_font("KoreanFont", "B", font_path, uni=True)
                korean_font_loaded = True
                break
            except Exception:
                continue

    C_BLUE = (33, 150, 243)
    C_RED = (244, 67, 54)
    C_AMBER = (255, 160, 0)
    C_GREEN = (76, 175, 80)
    C_ORANGE = (255, 152, 0)
    C_GREY = (158, 158, 158)
    C_BLACK87 = (38, 38, 38)
    C_BORDER = (229, 229, 229)
    C_BG_LIGHT = (245, 245, 245)

    def _set_font(style: str = "", size: int = 10):
        if korean_font_loaded:
            pdf.set_font("KoreanFont", style=style, size=size)
        else:
            pdf.set_font("Helvetica", style=style, size=size)

    def _reset_colors():
        pdf.set_text_color(0, 0, 0)
        pdf.set_draw_color(0, 0, 0)
        pdf.set_fill_color(255, 255, 255)

    def _content_width() -> float:
        return pdf.w - pdf.l_margin - pdf.r_margin

    def _rounded_rect(x, y, w, h, r=3, style="D", fill_color=None, draw_color=None):
        """Draw a rounded rectangle. style: D=draw, F=fill, DF=both."""
        from fpdf.enums import RenderStyle, Corner
        corners = (Corner.TOP_LEFT, Corner.TOP_RIGHT, Corner.BOTTOM_LEFT, Corner.BOTTOM_RIGHT)
        if fill_color:
            pdf.set_fill_color(*fill_color)
        if draw_color:
            pdf.set_draw_color(*draw_color)
        rs = RenderStyle.coerce(style)
        pdf._draw_rounded_rect(x, y, w, h, rs, corners, r)

    def _ensure_space(needed_mm: float):
        """Add page if not enough vertical space remains."""
        if pdf.get_y() + needed_mm > pdf.h - pdf.b_margin:
            pdf.add_page()

    def _section_header(number: str, title: str):
        """Render numbered section header matching Flutter's MiroFish style."""
        _ensure_space(20)
        pdf.set_x(pdf.l_margin)
        pdf.ln(8)
        box_size = 14
        x = pdf.get_x()
        y = pdf.get_y()
        _rounded_rect(x, y, box_size, box_size, r=4, style="F", fill_color=C_BLACK87)
        _set_font("B", 10)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(x, y)
        pdf.cell(box_size, box_size, number, align="C")
        pdf.set_xy(x + box_size + 6, y)
        pdf.set_text_color(*C_BLACK87)
        _set_font("B", 14)
        pdf.cell(0, box_size, title)
        pdf.set_y(y + box_size + 4)
        _reset_colors()

    def _card_start() -> tuple:
        """Begin a card container. Returns (y_position, page_number)."""
        pdf.set_x(pdf.l_margin)
        return (pdf.get_y(), pdf.page)

    def _card_end(start_info: tuple):
        """Draw a bordered rounded-rect card from start_y to current y."""
        start_y, start_page = start_info
        end_y = pdf.get_y() + 6
        # Only draw border if still on the same page
        if pdf.page == start_page:
            h = end_y - start_y
            if h > 0:
                _rounded_rect(
                    pdf.l_margin, start_y, _content_width(), h,
                    r=4, style="D", draw_color=C_BORDER,
                )
        pdf.set_y(end_y + 2)
        _reset_colors()

    def _body(text: str, indent: float = 0):
        _set_font("", 10)
        pdf.set_x(pdf.l_margin + indent)
        w = _content_width() - indent * 2
        pdf.multi_cell(w if w > 10 else 0, 6, str(text))
        pdf.ln(1)

    def _label_value(label: str, value: str, indent: float = 0):
        _set_font("B", 10)
        pdf.set_x(pdf.l_margin + indent)
        pdf.set_text_color(153, 153, 153)
        pdf.cell(0, 5, label, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*C_BLACK87)
        _set_font("", 10)
        pdf.set_x(pdf.l_margin + indent)
        w = _content_width() - indent * 2
        pdf.multi_cell(w if w > 10 else 0, 6, str(value))
        pdf.ln(2)

    def _badge(text: str, color: tuple, x: float = None, light: bool = False):
        """Draw a colored badge with rounded corners.
        light=True: colored text on light background. light=False: white text on colored bg."""
        if x is not None:
            pdf.set_x(x)
        _set_font("B", 9)
        tw = pdf.get_string_width(text) + 10
        bh = 8
        bx = pdf.get_x()
        by = pdf.get_y()
        if light:
            light_bg = (
                min(color[0] + int((255 - color[0]) * 0.85), 255),
                min(color[1] + int((255 - color[1]) * 0.85), 255),
                min(color[2] + int((255 - color[2]) * 0.85), 255),
            )
            _rounded_rect(bx, by, tw, bh, r=3, style="DF", fill_color=light_bg, draw_color=color)
            pdf.set_text_color(*color)
        else:
            _rounded_rect(bx, by, tw, bh, r=3, style="F", fill_color=color)
            pdf.set_text_color(255, 255, 255)
        pdf.set_xy(bx, by)
        pdf.cell(tw, bh, text, align="C")
        pdf.set_xy(bx + tw + 3, by)
        _reset_colors()
        return tw

    def _colored_dot(color: tuple):
        """Draw a small colored circle indicator."""
        x = pdf.get_x()
        y = pdf.get_y() + 2.5
        pdf.set_fill_color(*color)
        pdf.ellipse(x, y, 3, 3, style="F")
        pdf.set_x(x + 5)
        _reset_colors()

    def _progress_bar(value: float, max_val: float, color: tuple, width: float = 80):
        """Draw a horizontal progress bar with rounded ends."""
        x = pdf.get_x()
        y = pdf.get_y() + 0.5
        bar_h = 6
        fraction = min(value / max_val, 1.0) if max_val > 0 else 0
        _rounded_rect(x, y, width, bar_h, r=3, style="F", fill_color=C_BG_LIGHT)
        if fraction > 0:
            fill_w = max(width * fraction, 6)  # min width for rounded rect
            _rounded_rect(x, y, fill_w, bar_h, r=3, style="F", fill_color=color)
        pdf.set_x(x + width + 4)
        _reset_colors()

    def _bullet_list(items: list, indent: float = 6):
        """Render a bullet list."""
        for item in items:
            text = item.get("text", str(item)) if isinstance(item, dict) else str(item)
            _set_font("", 10)
            pdf.set_text_color(*C_BLACK87)
            pdf.set_x(pdf.l_margin + indent)
            w = _content_width() - indent
            pdf.multi_cell(w if w > 10 else 0, 6, f"\u2022 {text}")
            pdf.ln(1)

    def _team_color(team: str) -> tuple:
        t = team.lower()
        if "team_a" in t or "team a" in t:
            return C_BLUE
        if "team_b" in t or "team b" in t:
            return C_RED
        if "judge" in t:
            return C_AMBER
        return C_GREY

    pdf.add_page()
    CARD_PAD = 6  # horizontal padding inside cards

    topic = report.get("topic", analysis.get("topic", "Debate Report"))
    _set_font("B", 20)
    pdf.set_text_color(*C_BLACK87)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 10, topic)
    pdf.ln(2)
    _set_font("", 9)
    pdf.set_text_color(153, 153, 153)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 5, f"Generated: {report.get('generated_at', 'N/A')}  |  ID: {report.get('debate_id', 'N/A')}")
    pdf.ln(8)
    _reset_colors()

    sit = report.get("situation_analysis", {})
    if sit:
        _section_header("00", "\uc0c1\ud669 \ubd84\uc11d")
        cy = _card_start()
        pdf.ln(4)
        if sit.get("situation_brief"):
            _label_value("\uc0c1\ud669 \uc124\uba85", sit["situation_brief"], indent=CARD_PAD)
        if sit.get("topic"):
            _set_font("B", 13)
            pdf.set_text_color(*C_BLACK87)
            pdf.set_x(pdf.l_margin + CARD_PAD)
            pdf.multi_cell(_content_width() - CARD_PAD * 2, 7, sit["topic"])
            pdf.ln(3)
        if sit.get("opinion_a"):
            pdf.set_x(pdf.l_margin + CARD_PAD)
            _colored_dot(C_BLUE)
            _set_font("", 10)
            pdf.set_text_color(*C_BLUE)
            pdf.multi_cell(_content_width() - CARD_PAD * 2 - 5, 6, sit["opinion_a"])
            pdf.ln(2)
        if sit.get("opinion_b"):
            pdf.set_x(pdf.l_margin + CARD_PAD)
            _colored_dot(C_RED)
            _set_font("", 10)
            pdf.set_text_color(*C_RED)
            pdf.multi_cell(_content_width() - CARD_PAD * 2 - 5, 6, sit["opinion_b"])
            pdf.ln(2)
        _reset_colors()
        for issue in sit.get("key_issues", []):
            _set_font("", 10)
            pdf.set_x(pdf.l_margin + CARD_PAD + 4)
            pdf.multi_cell(_content_width() - CARD_PAD * 2 - 4, 6, f"\u2022 {issue}")
        if sit.get("parties"):
            pdf.ln(2)
            _set_font("B", 10)
            pdf.set_x(pdf.l_margin + CARD_PAD)
            pdf.cell(0, 6, "Parties", new_x="LMARGIN", new_y="NEXT")
            _set_font("", 10)
            for p in sit["parties"]:
                pdf.set_x(pdf.l_margin + CARD_PAD + 4)
                text = f"\u2022 {p.get('name', '')} - {p.get('role', '')}" if isinstance(p, dict) else f"\u2022 {p}"
                pdf.multi_cell(_content_width() - CARD_PAD * 2 - 4, 6, text)
        if sit.get("timeline"):
            pdf.ln(2)
            _set_font("B", 10)
            pdf.set_x(pdf.l_margin + CARD_PAD)
            pdf.cell(0, 6, "Timeline", new_x="LMARGIN", new_y="NEXT")
            _set_font("", 10)
            for t in sit["timeline"]:
                pdf.set_x(pdf.l_margin + CARD_PAD + 4)
                if isinstance(t, dict):
                    _action = t.get("action", "")
                    _sig = t.get("significance", "")
                    text = f"\u2022 [{t.get('date', '')}] {_action}"
                    pdf.multi_cell(_content_width() - CARD_PAD * 2 - 4, 6, text)
                    if _sig:
                        pdf.set_x(pdf.l_margin + CARD_PAD + 8)
                        _set_font("I", 9)  # italic subtext for significance
                        pdf.multi_cell(_content_width() - CARD_PAD * 2 - 8, 5, _sig)
                        _set_font("", 10)  # restore default body font
                else:
                    pdf.multi_cell(_content_width() - CARD_PAD * 2 - 4, 6, f"\u2022 {t}")
        _card_end(cy)

    _section_header("01", "\uc885\ud569 \uc694\uc57d")
    cy = _card_start()
    pdf.ln(4)
    exec_summary = report.get("executive_summary", "N/A")
    if isinstance(exec_summary, dict):
        result = exec_summary.get("result", "")
        if result:
            color = _team_color(result)
            pdf.set_x(pdf.l_margin + CARD_PAD)
            _badge(result, color, light=True)
            pdf.ln(8)
        summary_text = exec_summary.get("summary", exec_summary.get("text", exec_summary.get("content", "N/A")))
        _body(str(summary_text), indent=CARD_PAD)
    else:
        _body(str(exec_summary), indent=CARD_PAD)
    _card_end(cy)

    _section_header("02", "\ud575\uc2ec \uc99d\uac70")
    decisive = report.get("decisive_evidence", {})
    medal_labels = ["\ud83e\udd47", "\ud83e\udd48", "\ud83e\udd49"]
    cat_meta = {
        "precedent": ("\ud310\ub840 (Precedents)", C_BLUE),
        "statute": ("\ubc95\ub839 (Statutes)", C_GREEN),
        "statement": ("\ubc1c\uc5b8 (Statements)", C_ORANGE),
    }

    def _render_decisive_items(items, indent=CARD_PAD):
        iw = _content_width() - indent * 2
        for i, ev in enumerate(items):
            if not isinstance(ev, dict):
                continue
            desc = str(ev.get("description", ev.get("content", ev.get("text", ""))))
            impact = str(ev.get("impact", ""))
            source = str(ev.get("source", ""))
            team = str(ev.get("team", ""))
            url = ev.get("url", "")
            rank_label = medal_labels[i] if i < len(medal_labels) else f"#{i+1}"

            pdf.set_x(pdf.l_margin + indent)
            _set_font("", 12)
            pdf.cell(10, 6, rank_label)
            _set_font("", 10)
            pdf.set_text_color(*C_BLACK87)
            pdf.multi_cell(iw - 10, 6, desc)
            if impact:
                pdf.set_x(pdf.l_margin + indent + 10)
                pdf.set_text_color(117, 117, 117)
                _set_font("", 9)
                pdf.multi_cell(iw - 10, 5, f"Impact: {impact}")
            if source or team:
                pdf.set_x(pdf.l_margin + indent + 10)
                _set_font("", 8)
                pdf.set_text_color(117, 117, 117)
                tags = []
                if source:
                    tags.append(f"Source: {source}")
                if team:
                    tags.append(f"Team: {team}")
                pdf.cell(iw - 10, 5, "  |  ".join(tags), new_x="LMARGIN", new_y="NEXT")
            if url:
                pdf.set_x(pdf.l_margin + indent + 10)
                pdf.set_text_color(33, 150, 243)
                _set_font("", 9)
                # fpdf2 writes the link string into the PDF /URI action as-is
                # without percent-encoding, so raw Korean characters (e.g.
                # `법령/민법`, `query=2022다203804`) produce non-RFC 3986 URIs
                # that most PDF viewers fail to open. Percent-encode the
                # non-ASCII parts while preserving URL structural characters.
                # `%` is included in `safe` so already-encoded URLs are not
                # double-encoded.
                _safe_url = quote(url, safe=":/?&=#%[]@!$'()*+,;")
                pdf.cell(iw - 10, 5, "\uc6d0\ubb38 \ubcf4\uae30 \u2192", link=_safe_url, new_x="LMARGIN", new_y="NEXT")
            _reset_colors()
            pdf.ln(3)

    cy = _card_start()
    pdf.ln(4)
    if isinstance(decisive, dict):
        for cat_key in ["precedent", "statute", "statement"]:
            items = decisive.get(cat_key, decisive.get(f"{cat_key}s", []))
            if not items:
                continue
            cat_label, accent = cat_meta.get(cat_key, (cat_key, C_GREY))
            pdf.set_x(pdf.l_margin + CARD_PAD)
            x = pdf.get_x()
            y = pdf.get_y()
            _rounded_rect(x, y, 3, 8, r=1.5, style="F", fill_color=accent)
            pdf.set_xy(x + 7, y)
            _set_font("B", 11)
            pdf.set_text_color(*C_BLACK87)
            pdf.cell(0, 8, cat_label)
            pdf.ln(11)
            _reset_colors()
            _render_decisive_items(items)
    elif isinstance(decisive, list):
        _render_decisive_items(decisive)
    _card_end(cy)

    _section_header("03", "\uc2ec\ud310 \ud310\uacb0")
    for v in report.get("judge_verdicts", []):
        if not isinstance(v, dict):
            continue
        judge_name = str(v.get("judge_name", v.get("judge_id", "Judge")))
        verdict_val = str(v.get("winner", v.get("verdict", "")))
        confidence = v.get("confidence")
        reasoning = str(v.get("reasoning_summary", v.get("reasoning", "")))
        scores = v.get("scores", {})

        _ensure_space(15)

        C_AMBER_LIGHT = (255, 248, 225)
        hy = pdf.get_y()
        _rounded_rect(pdf.l_margin, hy, _content_width(), 12,
                      r=4, style="F", fill_color=C_AMBER_LIGHT)

        pdf.set_xy(pdf.l_margin + CARD_PAD, hy + 2)
        _badge(judge_name, C_AMBER, light=True)
        vcolor = _team_color(verdict_val)
        _badge(verdict_val if verdict_val else "N/A", vcolor, light=True)
        if confidence is not None:
            conf_text = f"{int(confidence * 100) if isinstance(confidence, float) and confidence <= 1 else confidence}%"
            _set_font("B", 16)
            pdf.set_text_color(*C_BLACK87)
            cw = pdf.get_string_width(conf_text)
            pdf.set_x(pdf.w - pdf.r_margin - cw - CARD_PAD)
            pdf.cell(cw, 8, conf_text)
        _reset_colors()
        pdf.set_y(hy + 16)

        if reasoning:
            _set_font("", 10)
            pdf.set_x(pdf.l_margin + CARD_PAD)
            pdf.set_text_color(*C_BLACK87)
            pdf.multi_cell(_content_width() - CARD_PAD * 2, 6, reasoning)
            pdf.ln(3)

        if scores:
            score_label_w = 35
            score_bar_w = 80
            for team_key in ("team_a", "team_b"):
                ts = scores.get(team_key, {})
                if not ts or not isinstance(ts, dict):
                    continue
                tc = C_BLUE if team_key == "team_a" else C_RED
                label = "Team A" if team_key == "team_a" else "Team B"
                pdf.set_x(pdf.l_margin + CARD_PAD)
                _set_font("B", 9)
                pdf.set_text_color(*tc)
                pdf.cell(score_label_w, 6, label)
                pdf.ln(7)
                for score_key in ("evidence_quality", "argument_logic", "persuasiveness", "rebuttal_effectiveness", "overall"):
                    score_val = ts.get(score_key)
                    if score_val is None:
                        continue
                    num_score = float(score_val) if isinstance(score_val, (int, float)) else 0
                    pdf.set_x(pdf.l_margin + CARD_PAD + 4)
                    _set_font("", 8)
                    pdf.set_text_color(117, 117, 117)
                    pdf.cell(score_label_w, 6, score_key.replace("_", " ").title())
                    _progress_bar(num_score, 100, tc, score_bar_w)
                    _set_font("B", 9)
                    pdf.set_text_color(*C_BLACK87)
                    pdf.cell(12, 6, str(score_val), align="R")
                    pdf.ln(8)
                pdf.ln(2)
        _reset_colors()

        pdf.set_draw_color(*C_BORDER)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(6)
        _reset_colors()

    _section_header("04", "\ub17c\uc99d \ubd84\uc11d")
    arg_analysis = report.get("argument_analysis", {})
    for team_key, label in [("team_a", "Team A"), ("team_b", "Team B")]:
        ta = arg_analysis.get(team_key, {})
        if not ta:
            continue
        tc = C_BLUE if team_key == "team_a" else C_RED

        cy = _card_start()
        pdf.ln(4)
        pdf.set_x(pdf.l_margin + CARD_PAD)
        x = pdf.get_x()
        y = pdf.get_y()
        _rounded_rect(x, y, 3, 8, r=1.5, style="F", fill_color=tc)
        pdf.set_xy(x + 7, y)
        _set_font("B", 12)
        pdf.set_text_color(*tc)
        pdf.cell(0, 8, label)
        pdf.ln(11)
        _reset_colors()

        strengths = ta.get("strongest", ta.get("strengths", []))
        weaknesses = ta.get("weakest", ta.get("weaknesses", []))
        missing = ta.get("missing_evidence", ta.get("missing", []))

        if strengths:
            _set_font("B", 10)
            pdf.set_text_color(*C_GREEN)
            pdf.set_x(pdf.l_margin + CARD_PAD + 4)
            pdf.cell(0, 6, "Strengths", new_x="LMARGIN", new_y="NEXT")
            _reset_colors()
            _bullet_list(strengths, indent=CARD_PAD + 8)
        if weaknesses:
            _set_font("B", 10)
            pdf.set_text_color(*C_ORANGE)
            pdf.set_x(pdf.l_margin + CARD_PAD + 4)
            pdf.cell(0, 6, "Weaknesses", new_x="LMARGIN", new_y="NEXT")
            _reset_colors()
            _bullet_list(weaknesses, indent=CARD_PAD + 8)
        if missing:
            _set_font("B", 10)
            pdf.set_text_color(244, 67, 54)
            pdf.set_x(pdf.l_margin + CARD_PAD + 4)
            pdf.cell(0, 6, "Missing Evidence", new_x="LMARGIN", new_y="NEXT")
            _reset_colors()
            _bullet_list(missing, indent=CARD_PAD + 8)
        _card_end(cy)
        pdf.ln(2)

    _section_header("05", "\uc99d\uac70 \ubaa9\ub85d")
    inv = report.get("evidence_inventory", {})
    cy = _card_start()
    pdf.ln(4)
    if inv:
        for top_key, top_val in inv.items():
            if isinstance(top_val, dict):
                for sub_key, sub_val in top_val.items():
                    pdf.set_x(pdf.l_margin + CARD_PAD)
                    _set_font("", 9)
                    pdf.set_text_color(117, 117, 117)
                    pdf.cell(45, 6, str(sub_key))
                    _set_font("B", 11)
                    pdf.set_text_color(*C_BLACK87)
                    pdf.cell(20, 6, str(sub_val))
                    pdf.ln(7)
            else:
                pdf.set_x(pdf.l_margin + CARD_PAD)
                _set_font("", 9)
                pdf.set_text_color(117, 117, 117)
                pdf.cell(45, 6, str(top_key))
                _set_font("B", 11)
                pdf.set_text_color(*C_BLACK87)
                pdf.cell(20, 6, str(top_val))
                pdf.ln(7)
        _reset_colors()

    evidence_items = report.get("evidence_items", [])
    if evidence_items:
        pdf.ln(2)
        pdf.set_draw_color(*C_BORDER)
        pdf.line(pdf.l_margin + CARD_PAD, pdf.get_y(), pdf.w - pdf.r_margin - CARD_PAD, pdf.get_y())
        pdf.ln(4)
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            source_type = str(item.get("source_type", "")).lower()
            source_detail = str(item.get("source_detail", ""))
            if source_type.startswith("prec") or "court" in source_type:
                ic = C_BLUE
            elif source_type.startswith("stat") or "legal" in source_type:
                ic = C_GREEN
            else:
                ic = C_GREY
            pdf.set_x(pdf.l_margin + CARD_PAD)
            _set_font("", 9)
            pdf.set_text_color(*ic)
            pdf.cell(0, 5, f"\u25cf {source_detail}", new_x="LMARGIN", new_y="NEXT")
        _reset_colors()
    _card_end(cy)

    rounds_data = report.get("debate_flow_summary", report.get("round_summary", []))
    if rounds_data:
        _section_header("06", "\ub77c\uc6b4\ub4dc\ubcc4 \uc694\uc57d")
        cy = _card_start()
        pdf.ln(4)
        for idx, rd in enumerate(rounds_data):
            _ensure_space(20)
            round_num = rd.get("round", idx + 1) if isinstance(rd, dict) else idx + 1
            summary = ""
            if isinstance(rd, dict):
                summary = str(rd.get("summary", rd.get("team_a_summary", rd.get("key_moment", ""))))
            else:
                summary = str(rd)

            pdf.set_x(pdf.l_margin + CARD_PAD)
            x = pdf.get_x()
            y = pdf.get_y()
            pdf.set_fill_color(*C_BLUE)
            pdf.ellipse(x, y, 10, 10, style="F")
            _set_font("B", 8)
            pdf.set_text_color(255, 255, 255)
            rn_text = str(round_num)
            pdf.set_xy(x, y)
            pdf.cell(10, 10, rn_text, align="C")
            _reset_colors()

            pdf.set_xy(x + 14, y)
            _set_font("", 10)
            pdf.set_text_color(*C_BLACK87)
            pdf.multi_cell(_content_width() - CARD_PAD * 2 - 14, 6, summary)
            pdf.ln(4)
        _card_end(cy)

    recs = report.get("recommendations", [])
    rec_items = []
    if isinstance(recs, list):
        rec_items = recs
    elif isinstance(recs, dict):
        for v in recs.values():
            if isinstance(v, list):
                rec_items.extend(v)
            else:
                rec_items.append(v)
    if rec_items:
        _section_header("07", "\uad8c\uace0\uc0ac\ud56d")
        cy = _card_start()
        pdf.ln(4)
        _bullet_list(rec_items, indent=CARD_PAD + 4)
        _card_end(cy)

    transcript = report.get("transcript", [])
    if transcript:
        pdf.add_page()
        _section_header("08", "\ud1a0\ub860 \uae30\ub85d")

        # Group by round
        by_round: dict[int, list] = {}
        for entry in transcript:
            if not isinstance(entry, dict):
                continue
            rnd = entry.get("round", 0)
            by_round.setdefault(rnd, []).append(entry)
        sorted_rounds = sorted(by_round.keys())

        for rnd in sorted_rounds:
            _ensure_space(20)
            rh_y = pdf.get_y()
            _rounded_rect(
                pdf.l_margin, rh_y, _content_width(), 9,
                r=4, style="F", fill_color=C_BG_LIGHT,
            )
            _set_font("B", 10)
            pdf.set_text_color(*C_BLACK87)
            pdf.set_xy(pdf.l_margin + CARD_PAD, rh_y + 1)
            pdf.cell(0, 7, f"Round {rnd}")
            pdf.set_y(rh_y + 12)
            _reset_colors()

            for entry in by_round[rnd]:
                _ensure_space(15)
                team = str(entry.get("team", ""))
                speaker = str(entry.get("speaker", ""))
                statement = str(entry.get("statement", ""))
                tc = _team_color(team)

                badge_text = f"{speaker}" if speaker else team.upper()
                pdf.set_x(pdf.l_margin + CARD_PAD)
                _badge(badge_text, tc)
                if team:
                    _set_font("", 8)
                    pdf.set_text_color(117, 117, 117)
                    pdf.cell(0, 8, f"  {team.upper()}")
                pdf.ln(11)

                if statement:
                    _set_font("", 10)
                    pdf.set_text_color(*C_BLACK87)
                    pdf.set_x(pdf.l_margin + CARD_PAD)
                    pdf.multi_cell(_content_width() - CARD_PAD * 2, 6, statement)
                    pdf.ln(4)
                _reset_colors()

            if rnd != sorted_rounds[-1]:
                pdf.set_draw_color(*C_BORDER)
                pdf.line(pdf.l_margin + CARD_PAD, pdf.get_y(), pdf.w - pdf.r_margin - CARD_PAD, pdf.get_y())
                pdf.ln(6)
                _reset_colors()

    return pdf.output()


def _remap_speakers(report: dict, state: dict) -> None:
    """Re-map transcript speaker IDs to display names using agent profiles."""
    agent_name_map: dict[str, str] = {}
    for key in ("team_a_agents", "team_b_agents", "judge_agents"):
        for agent in state.get(key, []):
            if isinstance(agent, dict):
                aid = agent.get("agent_id", "")
                if aid:
                    agent_name_map[aid] = agent.get("name", aid)
    for entry in report.get("transcript", []):
        speaker = entry.get("speaker", "")
        entry["speaker"] = agent_name_map.get(speaker, speaker)


@router.get("/{debate_id}")
async def get_report(debate_id: str):
    """
    Get the generated report for a debate.

    If the report has not been generated yet, it will be generated on first
    request and cached in the debate state for subsequent calls.
    """
    state = DebateStore.load(debate_id)

    status = state.get("status", "")
    if status not in ("completed", "stopped", "paused"):
        raise HTTPException(
            status_code=400,
            detail=f"Report unavailable: debate status is '{status}'. "
                   "Debate must be completed, stopped, or paused.",
        )

    cached_report = state.get("report")
    if cached_report:
        _remap_speakers(cached_report, state)
        return {"report": cached_report}

    logger.info("Generating report for debate %s...", debate_id)
    report = await _generate_report(state)

    state["report"] = report
    DebateStore.save(debate_id, state)

    logger.info("Report generated and cached for debate %s", debate_id)
    return {"report": report}


@router.post("/{debate_id}/regenerate")
async def regenerate_report(debate_id: str):
    """
    Regenerate the report by clearing the cache and generating a new one.
    """
    state = DebateStore.load(debate_id)

    status = state.get("status", "")
    if status not in ("completed", "stopped", "paused"):
        raise HTTPException(
            status_code=400,
            detail=f"Report unavailable: debate status is '{status}'. "
                   "Debate must be completed, stopped, or paused.",
        )

    state.pop("report", None)

    logger.info("Regenerating report for debate %s...", debate_id)
    report = await _generate_report(state)

    state["report"] = report
    DebateStore.save(debate_id, state)

    logger.info("Report regenerated and cached for debate %s", debate_id)
    return {"report": report}


@router.get("/{debate_id}/download")
async def download_report(debate_id: str):
    """
    Download the debate report as a Markdown file.

    Generates the report first if it has not been cached yet.
    """
    state = DebateStore.load(debate_id)

    status = state.get("status", "")
    if status not in ("completed", "stopped", "paused"):
        raise HTTPException(
            status_code=400,
            detail=f"Report unavailable: debate status is '{status}'.",
        )

    report = state.get("report")
    if not report:
        logger.info("Report not cached, generating for download: %s", debate_id)
        report = await _generate_report(state)
        state["report"] = report
        DebateStore.save(debate_id, state)

    _remap_speakers(report, state)

    pdf_bytes = _report_to_pdf(report, state)

    topic_slug = (
        report.get("topic", "debate")
        .replace(" ", "_")
        .replace("/", "_")[:50]
    )
    filename = f"report_{debate_id[:8]}_{topic_slug}.pdf"
    ascii_filename = f"report_{debate_id[:8]}.pdf"
    encoded_filename = quote(filename)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=\"{ascii_filename}\"; filename*=UTF-8''{encoded_filename}",
            "Content-Length": str(len(pdf_bytes)),
        },
    )

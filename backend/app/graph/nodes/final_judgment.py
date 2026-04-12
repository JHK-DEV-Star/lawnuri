"""
LangGraph node for final judgment by all judges.

After the debate concludes (max rounds reached, early stop, or user stop),
each judge produces a comprehensive verdict with decisive evidence analysis,
scoring, and a final ruling.
"""

from __future__ import annotations

import asyncio
import json

from app.agents.language import get_language_instruction
from app.graph.state import DebateState
from app.utils.llm_client import LLMClient
from app.utils.logger import logger

# Maximum retries for JSON verdict parsing
_MAX_JSON_RETRIES = 3

# System prompt for final verdict generation
_VERDICT_SYSTEM_PROMPT = """\
IMPORTANT: You must ONLY evaluate the public debate statements presented during the debate.
Internal team discussions, background preparation, and claims made outside the formal debate are NOT admissible evidence.
Base your verdict SOLELY on what was stated in the debate rounds.

You are {judge_name}, a distinguished legal judge delivering your final verdict.
Background: {background}
Judgment style: {judgment_style}

You have observed the entire debate and accumulated evaluation notes.
Now produce a comprehensive, well-reasoned verdict.

When writing your verdict, follow this judicial reasoning structure:

1. **Factual Background**: Summarize undisputed facts and contested facts.

2. **Issues for Determination**: List 1-3 core legal questions to resolve.

3. **Analysis per Issue**:
   For each issue:
   a) Summarize Team A's position and key evidence
   b) Summarize Team B's position and key evidence
   c) Applicable legal principles
   d) Your analysis and finding
   e) Which team's argument prevails and why

4. **Decisive Evidence**: Identify the 2-3 pieces of evidence that most
   influenced your decision. Explain their impact.

5. **Final Determination**: Overall winner with confidence level.

**Evidence Quality Scoring Rules:**
- Citations with verified, real case numbers from search results = full credit
- Vague citations without specific case numbers = reduced credit
- Fabricated/placeholder case numbers (e.g., 20XX다XXXXX patterns) = PENALTY
- If an agent was challenged about a citation and could not provide the real
  case number, reduce their evidence_quality score significantly.
- Consider ONLY what was actually said in the formal debate statements.
  Internal team discussions and preliminary analysis should NOT influence scoring.

Your reasoning field should read like a judicial opinion — structured,
evidence-based, and explaining WHY one side prevails, not just THAT it does.

Output ONLY a valid JSON object with this exact structure:
{{
    "judge_id": "{judge_id}",
    "judge_name": "{judge_name}",
    "verdict": "team_a" | "team_b" | "draw",
    "confidence": 0.0 to 1.0,
    "score_team_a": {{
        "legal_reasoning": 0 to 100,
        "evidence_quality": 0 to 100,
        "persuasiveness": 0 to 100,
        "rebuttal_effectiveness": 0 to 100,
        "overall": 0 to 100
    }},
    "score_team_b": {{
        "legal_reasoning": 0 to 100,
        "evidence_quality": 0 to 100,
        "persuasiveness": 0 to 100,
        "rebuttal_effectiveness": 0 to 100,
        "overall": 0 to 100
    }},
    "decisive_evidence": [
        {{
            "description": "Brief description of the key evidence",
            "team": "team_a | team_b",
            "impact": "How this evidence influenced the verdict"
        }}
    ],
    "reasoning": "Detailed reasoning for the verdict (2-3 paragraphs)",
    "suggestions": "Suggestions for both teams to improve their arguments"
}}

Rules:
- Be objective and fair.
- Base your verdict strictly on arguments and evidence presented during the debate.
- The winner must have demonstrably stronger arguments overall.
- Only declare "draw" if both sides are genuinely equal in quality.
- Provide specific references to statements and evidence from the debate.

## Mandatory Counter-Analysis (Internal reflection — do NOT include in output)

Before finalizing your verdict, mentally complete this counter-analysis.
This is an INTERNAL CHECK only — do NOT write it in your reasoning.
This does NOT mean the other side is correct. It is solely a bias-prevention exercise.

1. Identify which side you are initially inclined to favor.
2. Mentally consider: "If I had to rule for the OTHER side,
   what would be the strongest LEGAL justifications?"
   Think of at least 3 legal arguments.
3. For each, consider whether the side you favor has a stronger
   legal response.
4. Only after this internal reflection, render your final verdict.
   If you cannot find strong legal responses to the other side's
   arguments, reconsider whether your initial inclination was based
   on legal merit or unconscious preference.

IMPORTANT: This counter-analysis is NOT suggesting the other side
has merit. It is a technique to ensure your verdict is grounded in
LEGAL REASONING, not emotional or ethical bias.
"""


def _build_verdict_context(state: DebateState, judge: dict) -> str:
    """
    Build the full context string for a judge's verdict generation.

    Assembles the debate topic, all statements, evidence, and the judge's
    accumulated notes into a single context prompt.

    Args:
        state: Current DebateState with full debate history.
        judge: Judge agent profile dict.

    Returns:
        Formatted context string for the verdict prompt.
    """
    judge_id = judge.get("agent_id", "unknown")

    # Debate metadata
    lines = [
        f"DEBATE TOPIC: {state.get('topic', '?')}",
        f"Opinion A (team_a advocates): {state.get('opinion_a', '?')}",
        f"Opinion B (team_b advocates): {state.get('opinion_b', '?')}",
        f"Key issues: {', '.join(state.get('key_issues', []))}",
        f"Total rounds: {state.get('round', 0)}",
        "",
        "=" * 60,
        "FULL DEBATE LOG",
        "=" * 60,
    ]

    # Full debate log
    debate_log = state.get("debate_log", [])
    for i, entry in enumerate(debate_log):
        team = entry.get("team", "?")
        speaker = entry.get("speaker", "?")
        round_num = entry.get("round", "?")
        content = entry.get("statement", entry.get("content", ""))
        lines.append(f"\n--- Round {round_num} | {team} | Speaker: {speaker} ---")
        lines.append(content)

    # Judge Q&A section
    qa_log = state.get("judge_qa_log", [])
    if qa_log:
        lines.append("\n=== Judge Questions & Answers ===")
        lines.append("(These exchanges occurred during the debate and are admissible.)")
        for qa in qa_log:
            judge_name = qa.get("judge_name", qa.get("judge_id", "Judge"))
            target = qa.get("target_agent_name", qa.get("target_agent_id", "Agent"))
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            lines.append(f"\nJudge {judge_name} → {target}:")
            lines.append(f"  Q: {question}")
            if answer:
                lines.append(f"  A: {answer}")

    # All evidence submitted
    lines.append("")
    lines.append("=" * 60)
    lines.append("ALL SUBMITTED EVIDENCE")
    lines.append("=" * 60)

    all_evidences = state.get("all_evidences", [])
    for ev in all_evidences:
        src = ev.get("source_type", "?")
        detail = ev.get("source_detail", "")
        team = ev.get("submitted_by", "?")
        content = str(ev.get("content", ""))[:300]
        lines.append(f"\n[{src}] by {team}: {detail}")
        lines.append(f"  Content: {content}")

    # This judge's accumulated notes
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"YOUR ACCUMULATED NOTES (Judge {judge.get('name', '?')})")
    lines.append("=" * 60)

    judge_notes = state.get("judge_notes", [])
    my_notes = [n for n in judge_notes if n.get("judge_id") == judge_id]
    if my_notes:
        for note in my_notes:
            r = note.get("round", "?")
            team_eval = note.get("team_evaluated", "?")
            content = note.get("content", "")
            lines.append(f"\n[Round {r}, evaluating {team_eval}]")
            lines.append(content)
    else:
        lines.append("(No accumulated notes)")

    return "\n".join(lines)


async def _generate_single_verdict(
    judge: dict,
    state: DebateState,
    llm_client: LLMClient,
) -> dict:
    """
    Generate a single judge's verdict.

    Retries JSON parsing up to _MAX_JSON_RETRIES times on failure.

    Args:
        judge: Judge agent profile dict.
        state: Current DebateState.
        llm_client: LLM client for verdict generation.

    Returns:
        Parsed verdict dict. On complete failure, returns a fallback
        verdict with error information.
    """
    judge_name = judge.get("name", "Judge")
    judge_id = judge.get("agent_id", "unknown")
    background = judge.get("background", "legal professional")
    judgment_style = judge.get("debate_style", "balanced")

    # Load language from settings
    try:
        from app.api.settings import settings_mgr
        _settings = settings_mgr.load()
        _language = _settings.get("debate", {}).get("language", "ko")
    except Exception:
        _language = "ko"

    system_msg = _VERDICT_SYSTEM_PROMPT.format(
        judge_name=judge_name,
        judge_id=judge_id,
        background=background,
        judgment_style=judgment_style,
    ) + get_language_instruction(_language)

    context = _build_verdict_context(state, judge)
    user_msg = (
        f"{context}\n\n"
        f"Based on everything above, produce your final verdict as a JSON object."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    for attempt in range(_MAX_JSON_RETRIES + 1):
        try:
            verdict = await llm_client.achat_json(
                messages, temperature=0.3, max_tokens=3000
            )

            # Ensure required fields exist
            verdict.setdefault("judge_id", judge_id)
            verdict.setdefault("judge_name", judge_name)
            verdict.setdefault("verdict", "draw")
            verdict.setdefault("confidence", 0.5)
            verdict.setdefault("reasoning", "")

            # Validate winner value
            if verdict["verdict"] not in ("team_a", "team_b", "draw"):
                verdict["verdict"] = "draw"

            # Validate confidence range
            try:
                verdict["confidence"] = max(0.0, min(1.0, float(verdict["confidence"])))
            except (ValueError, TypeError):
                verdict["confidence"] = 0.5

            logger.info(
                "[final_judgment] Judge %s verdict: winner=%s (confidence=%.2f).",
                judge_name, verdict["verdict"], verdict["confidence"],
            )
            return verdict

        except json.JSONDecodeError:
            if attempt < _MAX_JSON_RETRIES:
                logger.warning(
                    "[final_judgment] Judge %s verdict JSON failed (attempt %d/%d).",
                    judge_name, attempt + 1, _MAX_JSON_RETRIES + 1,
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "Your response was not valid JSON. Please output ONLY "
                        "the JSON object matching the required structure, with "
                        "no additional text before or after."
                    ),
                })
            else:
                logger.error(
                    "[final_judgment] Judge %s verdict failed after %d retries.",
                    judge_name, _MAX_JSON_RETRIES + 1,
                )

    # Fallback verdict on total failure
    return {
        "judge_id": judge_id,
        "judge_name": judge_name,
        "verdict": "draw",
        "confidence": 0.0,
        "score_team_a": {
            "legal_reasoning": 50,
            "evidence_quality": 50,
            "persuasiveness": 50,
            "rebuttal_effectiveness": 50,
            "overall": 50,
        },
        "score_team_b": {
            "legal_reasoning": 50,
            "evidence_quality": 50,
            "persuasiveness": 50,
            "rebuttal_effectiveness": 50,
            "overall": 50,
        },
        "decisive_evidence": [],
        "reasoning": f"Judge {judge_name} was unable to produce a structured verdict due to parsing errors.",
        "suggestions": "",
        "error": "Failed to parse verdict JSON after maximum retries.",
    }


async def final_judgment_node(
    state: DebateState,
    llm_client: LLMClient,
) -> dict:
    """
    Produce final verdicts from all judges.

    Each judge independently evaluates the full debate and produces a
    comprehensive verdict with scoring, decisive evidence analysis,
    and detailed reasoning. All judges run in parallel.

    Args:
        state: Current DebateState with the complete debate history,
            all evidence, and accumulated judge notes.
        llm_client: LLM client for verdict generation.

    Returns:
        Partial state update with ``verdicts`` list and
        ``status`` set to "completed".
    """
    judges = state.get("judge_agents", [])

    if not judges:
        logger.warning("[final_judgment] No judge agents configured.")
        return {
            "verdicts": [],
            "status": "completed",
        }

    logger.info(
        "[final_judgment] Generating final verdicts from %d judges.", len(judges)
    )

    # Retry verdict generation up to 3 times
    verdicts: list[dict] = []
    for attempt in range(3):
        verdict_tasks = [
            _generate_single_verdict(judge, state, llm_client)
            for judge in judges
        ]
        results = await asyncio.gather(*verdict_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, dict):
                verdicts.append(result)
            elif isinstance(result, Exception):
                logger.error("[final_judgment] Verdict exception: %s", result)

        if verdicts:
            break
        logger.warning("[final_judgment] Attempt %d: no verdicts, retrying in 5s...", attempt + 1)
        await asyncio.sleep(5)

    # Log verdict summary
    if verdicts:
        winners = [v.get("verdict", "?") for v in verdicts]
        logger.info(
            "[final_judgment] Verdicts collected: %s",
            ", ".join(f"{v.get('judge_name', '?')}->{v.get('verdict', '?')}" for v in verdicts),
        )

        # Determine majority verdict
        from collections import Counter
        winner_counts = Counter(winners)
        majority_winner, majority_count = winner_counts.most_common(1)[0]
        logger.info(
            "[final_judgment] Majority verdict: %s (%d/%d judges).",
            majority_winner, majority_count, len(verdicts),
        )

    # Log accumulated token usage at debate completion
    if hasattr(llm_client, 'log_usage_summary'):
        llm_client.log_usage_summary(label=f"debate-{state.get('debate_id', '?')}")

    return {
        "verdicts": verdicts,
        "status": "completed",
    }

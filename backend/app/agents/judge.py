"""
Judge agent prompt templates for LawNuri.

Contains system prompt builders for judge agents who evaluate
debate statements, vote on early termination, and produce
final verdicts.
"""

from __future__ import annotations


from app.agents.language import get_language_instruction, SIMULATION_FRAME_JUDGE


def build_judge_system_prompt(
    agent: dict,
    topic: str,
    opinion_a: str,
    opinion_b: str,
    language: str = "ko",
    team_a_name: str = "Team A",
    team_b_name: str = "Team B",
) -> str:
    """
    Build the base system prompt for a judge agent.

    Establishes the judge's identity, role, and the debate context.
    This prompt is used as the foundation for all judge interactions.

    Args:
        agent: Judge agent profile dict with keys: name, specialty,
            personality, debate_style (used as judgment_style),
            background, agent_id.
        topic: The debate topic string.
        opinion_a: Opinion advocated by team_a.
        opinion_b: Opinion advocated by team_b.

    Returns:
        Formatted system prompt string.
    """
    name = agent.get("name", "Judge")
    agent_id = agent.get("agent_id", "unknown")
    specialty = agent.get("specialty", "general law")
    personality = agent.get("personality", "balanced")
    judgment_style = agent.get("debate_style", "thorough and balanced")
    background = agent.get("background", "experienced legal professional")

    prompt = SIMULATION_FRAME_JUDGE + f"""\
# Judge Profile
- **Name**: {name}
- **ID**: {agent_id}
- **Specialty**: {specialty}
- **Personality**: {personality}
- **Judgment Style**: {judgment_style}
- **Background**: {background}

# Debate Context
- **Topic**: {topic}
- **Opinion A ({team_a_name} advocates)**: {opinion_a}
- **Opinion B ({team_b_name} advocates)**: {opinion_b}

# Judge Responsibilities
You are an impartial judge evaluating a legal debate. Your duties:

1. **Objectivity**: Evaluate arguments on their merits, not personal
   preference. Both sides deserve fair consideration.
2. **Evidence Focus**: Weight arguments supported by properly cited
   evidence more heavily than unsupported assertions.
3. **Legal Soundness**: Assess whether legal reasoning is correct,
   relevant statutes are properly applied, and precedents are
   appropriately invoked.
4. **Logical Coherence**: Evaluate the logical structure of arguments.
   Identify fallacies, non sequiturs, and unsupported leaps.
5. **Rebuttal Quality**: Consider how effectively each side addresses
   and counters the opponent's arguments.

# Citation Verification
When evaluating cited evidence ([CITE:type:id] tags):
- Verify that citations are used in proper context
- Note when evidence is misrepresented or taken out of context
- Give more weight to properly sourced legal authority
- Flag any citations that appear fabricated or unsupported
"""
    prompt += get_language_instruction(language)
    return prompt.strip()


def build_judge_accumulate_prompt(
    agent: dict,
    latest_statement: dict,
    debate_log: list,
    round: int,
    language: str = "ko",
    team_a_name: str = "Team A",
    team_b_name: str = "Team B",
) -> str:
    """
    Build the prompt for a judge to evaluate the latest debate statement.

    Used during the judge_accumulate phase after each team speaks.

    Args:
        agent: Judge agent profile dict.
        latest_statement: The most recent debate log entry dict with
            keys: team, speaker, content, round, evidence_count.
        debate_log: Full debate log (list of entry dicts) for context.
        round: Current round number.

    Returns:
        Formatted user prompt string for evaluation note generation.
    """
    name = agent.get("name", "Judge")
    judgment_style = agent.get("debate_style", "balanced")

    # Format the latest statement
    stmt_team = latest_statement.get("team", "unknown")
    stmt_speaker = latest_statement.get("speaker", "unknown")
    stmt_content = latest_statement.get("content", "(no content)")
    stmt_evidence_count = latest_statement.get("evidence_count", 0)

    # Format recent debate context (last 4 entries before the latest)
    context_lines = []
    recent_entries = debate_log[-5:-1] if len(debate_log) > 1 else []
    for entry in recent_entries:
        team = entry.get("team", "?")
        speaker = entry.get("speaker", "?")
        content_text = str(entry.get("statement", entry.get("content", "")))
        context_lines.append(f"[{team}] {speaker}: {content_text}")

    context_text = "\n".join(context_lines) if context_lines else "(no prior statements)"

    prompt = SIMULATION_FRAME_JUDGE + f"""\
# Evaluation Task
Round {round} - Evaluate the latest statement.

## Recent Debate Context
{context_text}

## Latest Statement to Evaluate
- **Team**: {stmt_team}
- **Speaker**: {stmt_speaker}
- **Evidence Items Cited**: {stmt_evidence_count}
- **Content**:

{stmt_content}

## Evaluation Criteria
As {name} (judgment style: {judgment_style}), evaluate using these
weighted criteria:

1. **Legal Accuracy (30%)** - Are cited statutes/precedents correct and
   applicable? Is the legal reasoning sound? Any misapplication of law?
2. **Evidence Quality (25%)** - Does the evidence support the claims?
   Are sources authoritative (Supreme Court > lower courts)?
   Are citations properly used in context?
3. **Argumentation Structure (20%)** - Is the argument logically organized
   (IRAC)? Does each claim flow from evidence to conclusion?
4. **Rebuttal Effectiveness (15%)** - Does it address opponent's strongest
   points? Are counterarguments substantive or merely dismissive?
5. **Persuasiveness (10%)** - Overall conviction and clarity of presentation.
6. **Temporal Accuracy (bonus)** - Did the team verify that cited precedents
   are still valid under current law? Did they note any law amendments since
   the precedent? Did they prefer recent rulings over older ones?

Additional temporal verification:
- Verify that cited precedents are based on current (현행) law, not outdated provisions.
- Give more weight to recent Supreme Court (대법원) decisions over older lower-court rulings.
- Penalize citations of precedents decided under since-amended statutes without acknowledging the amendment.

Output a structured evaluation with one assessment per criterion (5-6 lines).
Format: "[Criterion] (score/10): brief assessment"
"""
    prompt += get_language_instruction(language)
    return prompt.strip()


def build_judge_early_stop_prompt(
    agent: dict,
    debate_log: list,
    round: int,
    min_rounds: int,
    language: str = "ko",
    team_a_name: str = "Team A",
    team_b_name: str = "Team B",
) -> str:
    """
    Build the prompt for a judge's early-stop evaluation.

    Asks the judge to vote on whether the debate has reached sufficient
    depth to conclude early. Uses conservative criteria to prevent
    premature termination.

    Args:
        agent: Judge agent profile dict.
        debate_log: Full debate log for analysis.
        round: Current round number.
        min_rounds: Minimum rounds before early stop is allowed.

    Returns:
        Formatted prompt string. Expected output is a JSON object with
        "vote" ("sufficient" | "continue") and "reason" fields.
    """
    name = agent.get("name", "Judge")
    specialty = agent.get("specialty", "general law")

    # Summarize the full debate
    summary_lines = []
    for entry in debate_log:
        team = entry.get("team", "?")
        r = entry.get("round", "?")
        content_text = str(entry.get("statement", entry.get("content", "")))
        summary_lines.append(f"[Round {r}][{team}] {content_text}")
    debate_summary = "\n".join(summary_lines) if summary_lines else "(empty debate)"

    # Count statements per team
    team_a_count = sum(1 for e in debate_log if e.get("team") == "team_a")
    team_b_count = sum(1 for e in debate_log if e.get("team") == "team_b")

    prompt = SIMULATION_FRAME_JUDGE + f"""\
# Early Termination Vote
Judge: {name} (Specialty: {specialty})
Current Round: {round} (minimum required: {min_rounds})

## Debate Statistics
- {team_a_name} statements: {team_a_count}
- {team_b_name} statements: {team_b_count}
- Total exchanges: {len(debate_log)}

## Full Debate Summary
{debate_summary}

## Voting Criteria
Vote "sufficient" to end the debate early ONLY if ALL conditions are met:

1. **Thorough Presentation**: Both teams have fully articulated their
   core arguments with supporting evidence.
2. **Key Evidence Addressed**: All major evidence items have been
   introduced, cited, and challenged or acknowledged by the other side.
3. **Argument Saturation**: Arguments are repeating without introducing
   new substantive points or evidence. Diminishing returns are clear.
4. **Fairness**: Both teams have had equal opportunity to present and
   rebut. Neither side has been cut short.
5. **No Outstanding Issues**: There are no significant legal points or
   evidence items that remain unaddressed.

## Conservative Default
When in doubt, vote "continue". Premature termination deprives teams
of their right to fully argue their case. It is better to have one
extra round than to stop too early.

## Output Format
Output ONLY a valid JSON object:
{{
    "vote": "sufficient" | "continue",
    "reason": "Your specific justification referencing the criteria above"
}}
"""
    prompt += get_language_instruction(language)
    return prompt.strip()


def build_judge_verdict_prompt(
    agent: dict,
    topic: str,
    opinion_a: str,
    opinion_b: str,
    debate_log: list,
    all_evidences: list,
    judge_notes: list,
    language: str = "ko",
    team_a_name: str = "Team A",
    team_b_name: str = "Team B",
) -> str:
    """
    Build the prompt for a judge's final verdict.

    Assembles the complete debate record, all evidence, and the judge's
    accumulated notes into a comprehensive prompt for verdict generation.

    Args:
        agent: Judge agent profile dict.
        topic: The debate topic string.
        opinion_a: Opinion advocated by team_a.
        opinion_b: Opinion advocated by team_b.
        debate_log: Full debate log (list of entry dicts).
        all_evidences: All evidence items submitted during the debate.
        judge_notes: This judge's accumulated evaluation notes.

    Returns:
        Formatted prompt string. Expected output is a JSON object
        conforming to the verdict schema with winner, scores,
        decisive_evidences, reasoning, and suggestions.
    """
    name = agent.get("name", "Judge")
    agent_id = agent.get("agent_id", "unknown")
    specialty = agent.get("specialty", "general law")
    judgment_style = agent.get("debate_style", "balanced")
    background = agent.get("background", "legal professional")

    # Format full debate log
    debate_lines = []
    for entry in debate_log:
        team = entry.get("team", "?")
        speaker = entry.get("speaker", "?")
        r = entry.get("round", "?")
        content = entry.get("statement", entry.get("content", ""))
        debate_lines.append(
            f"\n--- Round {r} | {team} | Speaker: {speaker} ---\n{content}"
        )
    debate_text = "\n".join(debate_lines) if debate_lines else "(no debate entries)"

    # Format all evidence
    evidence_lines = []
    for ev in all_evidences:
        src_type = ev.get("source_type", "?")
        detail = ev.get("source_detail", "")
        team = ev.get("submitted_by", "?")
        content = str(ev.get("content", ""))
        evidence_lines.append(f"[{src_type}] by {team}: {detail}\n  Content: {content}")
    evidence_text = "\n".join(evidence_lines) if evidence_lines else "(no evidence submitted)"

    # Format this judge's accumulated notes
    my_notes = [n for n in judge_notes if n.get("judge_id") == agent_id]
    notes_lines = []
    for note in my_notes:
        r = note.get("round", "?")
        team_eval = note.get("team_evaluated", "?")
        content = note.get("content", "")
        notes_lines.append(f"[Round {r}, evaluating {team_eval}]\n{content}")
    notes_text = "\n".join(notes_lines) if notes_lines else "(no accumulated notes)"

    prompt = SIMULATION_FRAME_JUDGE + f"""\
# Final Verdict Generation

## Judge Profile
- **Name**: {name}
- **Specialty**: {specialty}
- **Judgment Style**: {judgment_style}
- **Background**: {background}

## Debate Information
- **Topic**: {topic}
- **Opinion A ({team_a_name} advocates)**: {opinion_a}
- **Opinion B ({team_b_name} advocates)**: {opinion_b}
- **Total Rounds**: {len(set(e.get("round", 0) for e in debate_log))}

## Complete Debate Log
{"=" * 60}
{debate_text}

## All Submitted Evidence
{"=" * 60}
{evidence_text}

## Your Accumulated Evaluation Notes
{"=" * 60}
{notes_text}

## Verdict Instructions
Based on your thorough review of the entire debate, produce your
final verdict. You must:

1. **Determine the winner** based on overall argument quality,
   evidence strength, and rebuttal effectiveness. Only declare
   "draw" if both sides are genuinely equal.

2. **Score both teams** (0-100) on each criterion:
   - legal_reasoning: Correctness and depth of legal analysis
   - evidence_quality: Relevance, authority, and proper use of evidence
   - persuasiveness: Clarity, structure, and compelling presentation
   - rebuttal_effectiveness: How well they countered opponent arguments
   - overall: Holistic assessment

3. **Temporal precedent verification**:
   - Verify that cited precedents are based on current (현행) law, not outdated provisions.
   - Give more weight to recent Supreme Court (대법원) decisions over older lower-court rulings.
   - Penalize citations of precedents decided under since-amended statutes without acknowledging the amendment.

4. **Identify decisive evidence** that most influenced your verdict.
   For each item, describe the evidence, which team submitted it,
   and how it impacted the outcome.

5. **Provide detailed reasoning** (2-3 paragraphs) explaining your
   verdict with specific references to debate statements and evidence.

6. **Offer suggestions** for how each team could improve.

## Output Format
Output ONLY a valid JSON object:
{{
    "judge_id": "{agent_id}",
    "judge_name": "{name}",
    "winner": "team_a" | "team_b" | "draw",
    "confidence": 0.0 to 1.0,
    "score_team_a": {{
        "legal_reasoning": 0-100,
        "evidence_quality": 0-100,
        "persuasiveness": 0-100,
        "rebuttal_effectiveness": 0-100,
        "overall": 0-100
    }},
    "score_team_b": {{
        "legal_reasoning": 0-100,
        "evidence_quality": 0-100,
        "persuasiveness": 0-100,
        "rebuttal_effectiveness": 0-100,
        "overall": 0-100
    }},
    "decisive_evidence": [
        {{
            "description": "Brief description of the key evidence",
            "team": "team_a" | "team_b",
            "impact": "How this evidence influenced your verdict"
        }}
    ],
    "reasoning": "Detailed reasoning for the verdict (2-3 paragraphs)",
    "suggestions": "Suggestions for both teams to improve"
}}

## Rules
- Be objective. Base your verdict strictly on arguments and evidence
  presented during the debate.
- The winner must have demonstrably stronger arguments overall.
- Do not let personal specialty bias your evaluation.
- Reference specific statements and evidence in your reasoning.
- Confidence should reflect how clear the winner is (0.5 = very close,
  0.9+ = decisive victory).
- When referencing precedents or laws in your reasoning, you MUST use citation tags:
  [판례: case_number] for precedents, [법령: law_name] for statutes.
  NEVER write case numbers or law names as plain text. Always use [판례: ...] or [법령: ...] tags.
  NEVER use [CITE:...], [case_citation:...], or any other format.
"""
    prompt += get_language_instruction(language)
    return prompt.strip()

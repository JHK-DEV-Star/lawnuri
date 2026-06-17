"""
Complaint (소장) drafting prompts for LawNuri — complaint mode only.

After the 3-round debate over a chosen template's validity and the judges'
verdict, the synthesis node uses these prompts to:
  1) analyze HOW the 소장 should be drafted (작성 방향), and
  2) draft the actual 소장 following the selected template's section structure.

Follows the same convention as debater.py / judge.py:
    SIMULATION_FRAME + body + get_language_instruction(language)
and reuses the strict anti-hallucination rules so no fake case numbers appear.
"""

from __future__ import annotations

from app.agents.language import get_language_instruction, SIMULATION_FRAME_ADVOCATE


# Shared anti-hallucination block (condensed from debater.py rules 1/3/9).
_ANTI_HALLUCINATION = """
# Citation Integrity (STRICT)
- ONLY cite precedents/statutes that actually appeared in the debate record or evidence.
- Use [판례: 사건번호] for precedents and [법령: 법령명 제○조] for statutes. No other tag formats.
- NEVER invent, approximate, or truncate a case number. If you lack a precedent, argue
  from statutory interpretation or general legal principle instead — do NOT fabricate one.
- Every factual claim must trace to the situation, the evidence, or a debate statement.
"""


def _template_block(selected_template: dict) -> str:
    """Render the selected template's structure for the drafting prompts."""
    if not selected_template:
        return "(선택된 소장 양식 없음 — 일반적인 소장 형식을 사용하세요.)"
    title = selected_template.get("title", "소장")
    doc_type = selected_template.get("doc_type", "소장")
    category = selected_template.get("category", "")
    sections = selected_template.get("sections", [])
    lines = [f"문서 종류: {doc_type} / 분야: {category}", f"양식 제목: {title}", "필수 섹션(이 순서·구조를 반드시 따를 것):"]
    if sections:
        for s in sections:
            name = s.get("name", "")
            guide = s.get("guide", "")
            lines.append(f"  - {name}: {guide}")
    else:
        # No structured sections (e.g. non-JSON template) — fall back to full text.
        full = selected_template.get("full_text", "")
        if full:
            lines.append("양식 원문(참고):")
            lines.append(full[:4000])
    return "\n".join(lines)


def _verdicts_summary(verdicts: list) -> str:
    parts = []
    for v in verdicts or []:
        name = v.get("judge_name", v.get("judge_id", "판사"))
        winner = v.get("verdict", "")
        reasoning = v.get("reasoning", "")
        parts.append(f"- {name}: 승자={winner}\n  사유: {reasoning}")
    return "\n".join(parts) if parts else "(판결 정보 없음)"


def build_synthesis_prompt(
    situation_brief: str,
    selected_template: dict,
    opinion_a: str,
    opinion_b: str,
    verdicts: list,
    debate_summary: str,
    language: str = "ko",
) -> str:
    """Build the prompt for the 'how to draft' analysis (종합 결과)."""
    prompt = SIMULATION_FRAME_ADVOCATE + f"""\
# Task: 소장 작성 방향 종합 (Complaint Drafting Strategy)

당신은 위 토론(선택된 소장의 유효성에 대한 다자 토론)과 판사들의 판결을 종합하여,
원고가 실제로 제출할 소장을 "어떻게 작성해야 하는지" 전략을 정리하는 선임 변호사입니다.

## 사건 개요
{situation_brief[:3000]}

## 선택된 소장 양식
{_template_block(selected_template)}

## 양측 입장
- 원고측(인용 가능): {opinion_a}
- 반대측(기각/각하 가능): {opinion_b}

## 판사 판결 요지
{_verdicts_summary(verdicts)}

## 토론 핵심 요약
{debate_summary[:4000]}
{_ANTI_HALLUCINATION}
# 출력
다음을 한국어로 정리하세요(JSON 아님, 서술형):
1. 핵심 청구 방향 — 어떤 청구취지로 가야 하는가
2. 청구원인의 논리 구성 — 어떤 사실·법리·판례를 어떤 순서로 배치할지
3. 반대측이 제기한 약점과 그 보강·반박 방법
4. 인용 가능성을 높이기 위한 작성 시 주의사항
"""
    prompt += get_language_instruction(language)
    return prompt.strip()


def build_draft_prompt(
    situation_brief: str,
    selected_template: dict,
    synthesis: str,
    opinion_a: str,
    debate_summary: str,
    language: str = "ko",
) -> str:
    """Build the prompt that produces the actual 소장 text."""
    prompt = SIMULATION_FRAME_ADVOCATE + f"""\
# Task: 소장 작성 (Draft the Complaint Document)

아래 작성 방향과 사건 내용을 바탕으로, 선택된 양식의 섹션 구조를 그대로 따라
실제 제출 가능한 수준의 정식 소장을 작성하세요. 격식 있는 한국 법률 문체를 사용합니다.

## 사건 개요
{situation_brief[:3000]}

## 선택된 소장 양식 (이 섹션 구조를 반드시 따를 것)
{_template_block(selected_template)}

## 작성 방향(종합 결과)
{synthesis[:4000]}

## 원고측 핵심 주장
{opinion_a}

## 토론에서 검증된 근거 요약
{debate_summary[:3000]}
{_ANTI_HALLUCINATION}
# 작성 규칙
- 양식의 각 섹션을 제목과 함께 빠짐없이 채울 것(당사자, 청구취지, 청구원인, 입증방법, 첨부서류 등).
- 청구취지는 법원이 그대로 주문으로 옮길 수 있을 만큼 명확·구체적으로.
- 청구원인은 사실관계 → 법리 → 적용 → 결론(IRAC) 순으로, 판례·법령은 위 인용 규칙에 따라.
- 구체적 수치·날짜·당사자가 불명확하면 [○○○], [OOOO.OO.OO.] 같은 표기로 비워 두되 형식은 유지.
- 완성된 소장 본문만 출력(설명·머리말 없이).
"""
    prompt += get_language_instruction(language)
    return prompt.strip()

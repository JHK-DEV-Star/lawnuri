"""Language instruction for system prompts. All instructions in English."""

LANGUAGE_INSTRUCTION = {
    "ko": "\n\n[IMPORTANT] All responses MUST be written in Korean (한국어). Legal terms, case citations, evidence descriptions, statements, and all other content must be in Korean.",
    "en": "\n\n[IMPORTANT] All responses MUST be written in English. Legal terms, case citations, evidence descriptions, statements, and all other content must be in English.",
    "ja": "\n\n[IMPORTANT] All responses MUST be written in Japanese (日本語). Legal terms, case citations, evidence descriptions, statements, and all other content must be in Japanese.",
    "zh": "\n\n[IMPORTANT] All responses MUST be written in Chinese (中文). Legal terms, case citations, evidence descriptions, statements, and all other content must be in Chinese.",
    "es": "\n\n[IMPORTANT] All responses MUST be written in Spanish (Español). Legal terms, case citations, evidence descriptions, statements, and all other content must be in Spanish.",
    "fr": "\n\n[IMPORTANT] All responses MUST be written in French (Français). Legal terms, case citations, evidence descriptions, statements, and all other content must be in French.",
    "de": "\n\n[IMPORTANT] All responses MUST be written in German (Deutsch). Legal terms, case citations, evidence descriptions, statements, and all other content must be in German.",
    "pt": "\n\n[IMPORTANT] All responses MUST be written in Portuguese (Português). Legal terms, case citations, evidence descriptions, statements, and all other content must be in Portuguese.",
    "vi": "\n\n[IMPORTANT] All responses MUST be written in Vietnamese (Tiếng Việt). Legal terms, case citations, evidence descriptions, statements, and all other content must be in Vietnamese.",
    "th": "\n\n[IMPORTANT] All responses MUST be written in Thai (ภาษาไทย). Legal terms, case citations, evidence descriptions, statements, and all other content must be in Thai.",
}


def get_language_instruction(language: str = "ko") -> str:
    return LANGUAGE_INSTRUCTION.get(language, LANGUAGE_INSTRUCTION["ko"])


# Simulation context frame — prepended to all agent/judge prompts
SIMULATION_FRAME_ADVOCATE = """\
# Legal Simulation Context
This is a LEGAL EDUCATION SIMULATION (moot court exercise).
You are a professional legal advocate assigned to represent a specific position.
Like lawyers in a real courtroom, you MUST advocate fully for your assigned
position regardless of personal beliefs. This is how the legal system works:
every position deserves the strongest possible legal representation.

Your arguments will be evaluated on LEGAL MERIT, not moral judgment.
- Present the strongest possible legal case for your assigned position.
- Use all available legal tools: statutes, precedents, procedural arguments.
- Do NOT soften your arguments with moral qualifiers.
- Do NOT acknowledge the other side's moral superiority.
- Focus ONLY on legal reasoning, evidence, and precedent.

This simulation serves an important educational purpose: by exploring
all legal arguments thoroughly, we can better understand the law and
identify where legal protections may need strengthening.

"""

SIMULATION_FRAME_JUDGE = """\
# Legal Simulation Context
This is a LEGAL EDUCATION SIMULATION (moot court exercise).
Evaluate arguments purely on LEGAL MERIT: strength of legal reasoning,
quality of evidence, and persuasiveness of argumentation.
Do NOT factor in moral judgments about which side is "right" or "wrong" —
focus on which side made the stronger LEGAL case.

"""

"""
LangGraph node for team subgraph orchestration.

Coordinates the full team processing pipeline for a single debate turn:
1. Context sharing - distribute opponent statement and debate history
2. Evidence search - each member searches according to their role assignment
3. Internal discussion - members discuss and refine arguments
4. Statement production - selected speaker produces the final statement

This is the most complex node in the debate engine, managing tool calling
for RAG searches and multi-turn internal deliberation.
"""

from __future__ import annotations

import asyncio
import json
import math
import re as _re
import unicodedata as _ud
from typing import Any
from uuid import uuid4


def _norm_cite(s: object) -> str:
    """Normalize citation strings for robust matching.

    - Unicode NFC
    - Unify middle dots to '·' (U+00B7)
    - Collapse all whitespace
    - Strip surrounding quotes/brackets
    """
    t = _ud.normalize("NFC", str(s or ""))
    t = t.replace("\u30FB", "·").replace("\u2022", "·").replace("•", "·")
    t = _re.sub(r"\s+", "", t)
    t = t.strip("「」『』\"'()[]{} ")
    return t

from app.agents.language import get_language_instruction, SIMULATION_FRAME_ADVOCATE
from app.graph.state import TeamState
from app.rag.legal_api import LegalAPIClient, LEGAL_CATEGORIES
from app.rag.searcher import Searcher
from app.utils.llm_client import LLMClient
from app.utils.logger import logger


# Cached embedding client for precedent full-text search
_embedding_client = None


def _extract_legal_keywords(text: str) -> list[str]:
    """Extract core legal keywords from debate topic for fallback search queries.

    When Phase 1 JSON parsing fails, this provides meaningful short queries
    instead of using the entire topic text as a search query.
    """
    keywords: list[str] = []
    # 1. Law names: ~법, ~령, ~규칙, ~특별법, ~조례
    law_names = _re.findall(r'[가-힣]{2,20}(?:특별법|기본법|시행령|시행규칙|조례|법률|법|령|규칙)', text)
    keywords.extend(law_names)
    # 2. Legal compound terms: noun + suffix patterns
    legal_terms = _re.findall(
        r'[가-힣]{2,6}(?:의무|권리|책임|절차|요건|위반|무효|취소|해지|이행|확인|인증|계약|보험|사기|거래|보호)',
        text,
    )
    keywords.extend(legal_terms)
    # 3. Core topic nouns (2-4 syllable nouns that appear important)
    short_terms = _re.findall(r'[가-힣]{2,4}(?:보험|서명|금융|통신|피해|환급|사기)', text)
    keywords.extend(short_terms)
    # Deduplicate preserving order, max 6
    seen: set[str] = set()
    unique: list[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique[:6] if unique else [text[:20]]




async def _llm_relevance_filter(
    items: list[dict],
    topic: str,
    team_opinion: str,
    llm_client,
    language: str = "ko",
) -> list[dict]:
    """Filter search results by LLM-judged direct relevance to the debate topic.

    Sends a batch of items to the LLM and asks which are directly relevant.
    Items judged irrelevant get '_llm_irrelevant' = True flag.
    Returns only relevant items.
    """
    if not items or not topic:
        return items

    # Only check items with case_number or law_name (precedents/statutes)
    _candidates: list[tuple[int, str]] = []
    for i, r in enumerate(items):
        cn = r.get("case_number", r.get("case_id", ""))
        ln = r.get("law_name", "")
        title = r.get("title", "")
        content = str(r.get("content", ""))[:800]
        label = cn or ln or title
        if not label:
            continue
        snippet = f"[{i}] {label}: {content}" if content else f"[{i}] {label}"
        _candidates.append((i, snippet))

    if not _candidates:
        return items

    # Batch into groups of 30 to fit context
    _BATCH = 15
    _irrelevant_indices: set[int] = set()

    for batch_start in range(0, len(_candidates), _BATCH):
        batch = _candidates[batch_start:batch_start + _BATCH]
        items_text = "\n".join(snippet for _, snippet in batch)

        prompt = (
            f"You are a legal relevance judge.\n\n"
            f"DEBATE TOPIC: {topic}\n"
            f"TEAM POSITION: {team_opinion[:300]}\n\n"
            f"## Evaluation Rules\n\n"
            f"### For PRECEDENTS (판례, case numbers like 2024다12345):\n"
            f"Evaluate using these 4 dimensions:\n"
            f"1. VICTIM — Is the victim's situation/role similar?\n"
            f"2. SITUATION — Is the context/background similar?\n"
            f"3. CONDUCT — Is the alleged wrongful act similar?\n"
            f"4. PERPETRATOR — Is the perpetrator's role/type similar?\n\n"
            f"ALL 4 dimensions must be similar for the precedent to be RELEVANT.\n"
            f"If even ONE dimension does not match, it is IRRELEVANT.\n"
            f"Compare at the level of CONCRETE FACTS, not abstract legal principles.\n\n"
            f"IRRELEVANT examples:\n"
            f"- Case: employer fired worker without notice → Precedent: employer reduced wages → IRRELEVANT\n"
            f"- Case: insurance fraud via forged identity → Precedent: insurance premium dispute → IRRELEVANT\n\n"
            f"### For STATUTES (법률, 법, 시행령, 시행규칙):\n"
            f"Evaluate whether the statute COULD APPLY to the legal issues in this case.\n"
            f"A statute is RELEVANT if:\n"
            f"- It governs the type of contract, transaction, or relationship in the case\n"
            f"- It defines rights, obligations, or procedures relevant to the dispute\n"
            f"- It provides legal grounds for claims or defenses in this case\n"
            f"A statute is IRRELEVANT if:\n"
            f"- It governs a completely different domain (e.g., refugee law for an insurance case)\n"
            f"- It has no connection to the legal issues, parties, or transactions involved\n\n"
            f"ITEMS:\n{items_text}\n\n"
            f"Return ONLY the indices of IRRELEVANT items as a comma-separated list.\n"
            f"Example: 0,3,7\n"
            f"If ALL items are relevant, return: NONE\n"
            f"Return ONLY the indices or NONE, nothing else."
        )

        try:
            response = await llm_client.achat(
                messages=[
                    {"role": "system", "content": "You judge legal relevance. Be strict — only items with direct legal relevance to the topic should pass."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            response = response.strip()
            if response.upper() != "NONE":
                for part in response.replace(" ", "").split(","):
                    try:
                        idx = int(part)
                        # Map back to original index
                        if 0 <= idx < len(batch):
                            orig_idx = batch[idx][0]
                            _irrelevant_indices.add(orig_idx)
                    except ValueError:
                        continue
        except Exception as exc:
            logger.warning("[llm_relevance] Batch relevance check failed: %s", exc)

    if _irrelevant_indices:
        for idx in _irrelevant_indices:
            if idx < len(items):
                items[idx]["_llm_irrelevant"] = True
                label = items[idx].get("case_number", "") or items[idx].get("law_name", "") or items[idx].get("title", "")
                logger.info("[llm_relevance] IRRELEVANT: [%d] %s", idx, label)

    filtered = [r for r in items if not r.get("_llm_irrelevant")]
    logger.info("[llm_relevance] %d/%d items passed relevance check (removed %d)",
                len(filtered), len(items), len(items) - len(filtered))
    return filtered


def _build_resolved_section(resolved_cases: set, accept_votes: dict, case_id_map: dict = None) -> str:
    """Build system prompt section listing already resolved precedents."""
    if not resolved_cases:
        return ""
    _accepted = [cn for cn in resolved_cases if cn in accept_votes and accept_votes[cn]]
    _rejected = [cn for cn in resolved_cases if cn not in _accepted]
    lines = ["## Already Resolved Precedents (DO NOT review again)\n"]
    if _accepted:
        lines.append(f"✓ ACCEPTED: {', '.join(_accepted)}")
    if _rejected:
        lines.append(f"✗ REJECTED/BLACKLISTED: {', '.join(_rejected)}")
    lines.append("These are DONE. DO NOT write any review tags "
                 "(ACCEPT/REJECT/BLACKLIST) for these cases. Move on.\n")
    if case_id_map:
        available = [cn for cn in case_id_map if cn not in resolved_cases]
        if available:
            lines.append(f"Available precedents for tool call: {', '.join(available[:10])}\n")
    return "\n".join(lines) + "\n"


def _build_pending_review_section(pending_reviews: list[dict], team_opinion: str = "") -> str:
    """Build system prompt section for pending precedent reviews."""
    if not pending_reviews:
        return ""
    lines = []

    # Tag instructions
    lines.append("## Precedent Review Tags (MANDATORY FORMAT)\n"
                 "You MUST use these exact tags when reviewing precedents:\n"
                 "- [ACCEPT: case_number] — SUPPORTS our argument. We SHOULD cite it.\n"
                 "- [REJECT: case_number] — NOT relevant (different legal issue/facts).\n"
                 "- [BLACKLIST: case_number] — HURTS our position. Ruling goes AGAINST our argument.\n"
                 "    ⚠ If the precedent is 'unfavorable' or 'harmful to us' → BLACKLIST, not ACCEPT.\n"
                 + "- [REVIEW_MORE: case_number] — Need more examination before deciding.\n\n"
                 + "Do NOT use any other format (e.g., [Result N], BLACKLIST: [name], etc.).\n")

    # Pending reviews
    if pending_reviews:
        lines.append("## PRECEDENT REVIEW REQUIRED\n")
        lines.append("The following precedent(s) were searched by your teammate(s). "
                     "You MUST evaluate each one and include your review tags in your response.\n")
        lines.append("\nWhen reviewing, check these 4 relevance dimensions:\n"
                     "1. VICTIM — Does the precedent involve a similar type of victim?\n"
                     "2. SITUATION — Is the factual context similar to our case?\n"
                     "3. CONDUCT — Is the wrongful act/violation the same type?\n"
                     "4. PERPETRATOR — Is the perpetrator in a similar role?\n"
                     "ALL 4 dimensions must match. If even ONE does not match, use [REJECT].\n"
                     "IMPORTANT: Compare at the level of CONCRETE FACTS, not abstract legal principles.\n"
                     "Ask yourself: 'If I remove the domain keywords, is the fact pattern still the same?'\n"
                     "IRRELEVANT examples:\n"
                     "- Case: employer fired without written notice / Precedent: employer reduced wages → IRRELEVANT\n"
                     "- Case: hospital wrong dosage / Precedent: hospital refused to disclose records → IRRELEVANT\n"
                     "RELEVANT: Case: employer fired without written notice / Precedent: employer terminated verbally without notice → RELEVANT\n"
                     "RULE: If the only way to connect is through an abstract principle (e.g., 'contract invalidity'), use [REJECT].\n"
                     "The specific WHO, WHAT, HOW must match — not just the legal category.\n"
                     "Passing all 4 dimensions does NOT mean you should [ACCEPT] — "
                     "it only means the precedent is not rejected on relevance grounds.\n"
                     "Next, check RULING DIRECTION: if the court's result goes AGAINST your team's position → [BLACKLIST]. NO EXCEPTIONS.\n"
                     "Do NOT try to 'distinguish' or 'salvage' an opposite-direction ruling — the opponent will use it against you.\n")
        if team_opinion:
            lines.append(f"\n**YOUR TEAM'S POSITION**: {team_opinion[:300]}\n")
            lines.append("Compare each precedent's holding against YOUR team's position above.\n"
                         "If the court ruled in a way that SUPPORTS THE OPPONENT, use [BLACKLIST].\n")
        for pr in pending_reviews:
            cn = pr.get("case_number", "")
            found_by = pr.get("found_by", "unknown")
            result = pr.get("result", "")
            is_review_more = pr.get("review_request", False)
            _history = pr.get("review_history", [])
            if _history:
                # Priority review item from previous round (REVIEW_MORE limit reached)
                lines.append(f"\n### [Priority Review] Precedent {cn} (REVIEW_MORE limit reached)")
                lines.append("Previous review notes:")
                for _h in _history:
                    lines.append(f"  - {_h[:300]}")
                lines.append("")
                if result:
                    lines.append(f"{result}\n")
                lines.append(f"→ Based on the above review history, make a FINAL decision: [ACCEPT: {cn}], [REJECT: {cn}], or [BLACKLIST: {cn}]\n")
            elif is_review_more:
                lines.append(f"\n### [REVIEW_MORE requested] Precedent {cn} (by {found_by})")
                lines.append(f"A teammate requested additional review of this precedent.\n")
                if result:
                    lines.append(f"{result}\n")
                lines.append(f"→ Respond with: [ACCEPT: {cn}], [REJECT: {cn}], [BLACKLIST: {cn}], or [REVIEW_MORE: {cn}]\n")
            else:
                lines.append(f"\n### Precedent {cn} (searched by {found_by})")
                if result:
                    lines.append(f"{result}\n")
                lines.append(f"→ Respond with: [ACCEPT: {cn}], [REJECT: {cn}], [BLACKLIST: {cn}], or [REVIEW_MORE: {cn}]\n")
        lines.append("\n⚠ You MUST include a review tag for EACH precedent above.\n")
        lines.append("If you REJECT or BLACKLIST, suggest a BETTER alternative precedent.\n"
                     "After reviewing, contribute your legal analysis from your assigned argument angle.\n")

    return "\n".join(lines) + "\n"


def _build_case_id_map(search_results: list[dict]) -> dict[str, str]:
    """Build {case_number: item_id} mapping from search results for on-demand lookup."""
    case_id_map: dict[str, str] = {}
    for sr in search_results:
        cn = sr.get("case_number", sr.get("case_id", ""))
        item_id = sr.get("_item_id", sr.get("item_id", ""))
        if cn and item_id and cn not in case_id_map:
            case_id_map[cn] = item_id
    return case_id_map


async def _on_demand_precedent_search(
    case_number: str,
    query: str,
    case_id_map: dict[str, str],
    legal_api,
) -> str:
    """On-demand: fetch full text, chunk, embed, store in shared VectorDB, and search.

    Only embeds the specific precedent requested (not all search results).
    Already-stored precedents are reused (dedup by case_number).

    Returns formatted search result text for the LLM.
    """
    import os
    from app.rag.vector_store import VectorStore
    from app.utils.file_parser import split_text_into_chunks

    global _embedding_client
    if _embedding_client is None:
        from app.utils.embedding_client import build_embedding_client
        _embedding_client = build_embedding_client()

    if _embedding_client is None:
        return f"[Embedding client unavailable for {case_number}]"

    # Open shared VectorDB
    _model_name = _embedding_client.model.replace("/", "_")
    _persist_dir = os.path.join("data", "shared", "precedent_vectors", _model_name, "chroma")
    os.makedirs(_persist_dir, exist_ok=True)
    precedent_vs = VectorStore(
        persist_dir=_persist_dir,
        collection_name="precedent_fulltext",
    )

    # 1. Check if already stored (dedup)
    q_emb = await _embedding_client.aembed([query])
    try:
        probe = await precedent_vs.search(q_emb[0], top_k=1, where={"case_number": case_number})
        already_stored = bool(probe)
    except Exception:
        already_stored = False

    # 2. If not stored, fetch full text → chunk → embed → store
    if not already_stored:
        item_id = case_id_map.get(case_number, "")
        if not item_id:
            return f"[No item_id found for case {case_number}]"

        try:
            detail = await legal_api.get_legal_detail("prec", item_id)
        except Exception as exc:
            return f"[Failed to fetch full text for {case_number}: {exc}]"

        full_text = detail.get("content", "") if isinstance(detail, dict) else ""
        if not full_text:
            return f"[No full text available for {case_number}]"

        chunks = split_text_into_chunks(full_text, chunk_size=3000, overlap=300)
        if not chunks:
            return f"[No chunks produced for {case_number}]"

        logger.info("[on-demand] Embedding %d chunks for %s...", len(chunks), case_number)
        embeddings = await _embedding_client.aembed(chunks)

        metadatas = [
            {"case_number": case_number, "chunk_index": ci, "source": "precedent_full_text"}
            for ci in range(len(chunks))
        ]
        ids = [f"{case_number}_{ci}" for ci in range(len(chunks))]
        await precedent_vs.add_chunks(chunks, metadatas, embeddings, ids)
        logger.info("[on-demand] Stored %d chunks for %s in shared VectorDB.", len(chunks), case_number)

    # 3. Search
    results = await precedent_vs.search(q_emb[0], top_k=5, where={"case_number": case_number})
    if not results:
        return f"[No results found for {case_number}: {query}]"

    result_text = f"\n[Precedent Full-Text Search: {case_number}]\n"
    for r in results:
        cn = r.get("metadata", {}).get("case_number", "")
        doc = r.get("document", "")
        sim = 1.0 - r.get("distance", 1.0)
        result_text += f"[{cn} | sim={sim:.2f}] {doc}\n"
    return result_text


# Tool definitions for precedent full-text search during discussion
_DISCUSSION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_precedent_detail",
            "description": (
                "Search the full text of court precedents stored in the vector database. "
                "Use this to verify holdings, court orders (주문: dismissal/grant), "
                "reasoning, and specific legal interpretations from the original ruling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query for the precedent full text "
                            "(e.g. 'dismissal order', 'identity verification obligation', "
                            "'court holding on contract validity')"
                        ),
                    },
                    "case_number": {
                        "type": "string",
                        "description": "The case number to search (e.g. '2020다12345')",
                    },
                },
                "required": ["query", "case_number"],
            },
        },
    }
]


# Tool definitions for LLM tool calling (search functions)
_SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search uploaded documents and knowledge graph for relevant "
                "legal information. Use for finding evidence from case files, "
                "contracts, and other uploaded materials."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in natural language.",
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["vector", "graph", "both"],
                        "description": (
                            "Type of search: 'vector' for semantic similarity, "
                            "'graph' for entity/relation lookup, 'both' for combined."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_legal",
            "description": (
                "Search Korean legal databases (law.go.kr). "
                "Category selection guide:\n"
                "- 'law': Statutes/decrees/regulations (e.g., 민법, 형법, 상법)\n"
                "- 'prec': Court precedents from 대법원/고등법원/지방법원\n"
                "- 'const': Constitutional Court decisions (헌재결정례)\n"
                "- 'interp': Legal interpretation opinions (법령해석례)\n"
                "- 'detc': Administrative tribunal decisions (행정심판재결례)\n"
                "- 'ordin': Local government ordinances (자치법규)\n"
                "- 'admrul': Administrative rules/guidelines (행정규칙)\n"
                "- 'treaty': International treaties (조약)\n"
                "- 'term': Legal terminology definitions (법령용어)\n"
                "- 'committee': Committee decisions (위원회결정문, requires 'org' code)\n\n"
                "Filter tips:\n"
                "- court: Filter by court name, e.g. '대법원', '서울고등법원' (prec only)\n"
                "- prncYd: Date range 'YYYYMMDD~YYYYMMDD' (e.g. '20200101~20241231')\n"
                "- jo: Filter by referenced statute, e.g. '민법' (useful for prec)\n"
                "- nb: Direct case number search, e.g. '2020다12345'\n"
                "- sort: 'ddes'(newest), 'dasc'(oldest)\n"
                "- search: 2 for full-text search (default 1 is title-only)\n\n"
                "SEARCH QUERY STRATEGY (CRITICAL):\n"
                "The Korean legal API uses keyword matching, NOT semantic search.\n"
                "- Use SHORT, CORE legal terms (2-4 words max).\n"
                "  GOOD: '하자 보수', '보증보험 해지', '소멸시효 기산점'\n"
                "  BAD: '도배 하자 보수 비용 청구', '보증보험 무단 설정에 따른 계약 무효'\n"
                "- For precedents: search by core legal concept, NOT full description.\n"
                "- Use search=2 (full-text) ONLY after title search returns 0.\n"
                "- For specific cases: use 'nb' param (e.g., nb='2019다229202').\n"
                "- For statutes: search exact law name (query='보험업법'), NOT articles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["law","prec","const","interp","detc","ordin","admrul","treaty","term","special","consulting","ministry","table","committee"],
                    },
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "description": "Max results (default 5, max 20)"},
                    "court": {"type": "string", "description": "Court name (prec only)"},
                    "sort": {"type": "string", "description": "Sort: ddes/dasc/ldes/lasc"},
                    "prncYd": {"type": "string", "description": "Date range YYYYMMDD~YYYYMMDD"},
                    "nb": {"type": "string", "description": "Case number"},
                    "jo": {"type": "string", "description": "Reference statute name"},
                    "org": {"type": "string", "description": "Organization code (committee: ftc/pipc/acrc/fsc/nlrc/kcc/nhrc/eir/iacr/clec/edrc/sfc)"},
                    "search": {"type": "integer", "description": "1=title, 2=fulltext"},
                    "efYd": {"type": "string", "description": "Enforcement date range YYYYMMDD~YYYYMMDD (law only)"},
                },
                "required": ["category", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_legal_detail",
            "description": (
                "Get the full text (본문) of a specific legal item. "
                "ALWAYS call this after search_legal to read the actual content. "
                "search_legal only returns titles and brief summaries — you MUST use this tool "
                "to see statute articles (조문), court holdings (판시사항), or ruling text. "
                "Use the case_number (사건번호) for precedents, or law_name (법령명) for statutes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["law","prec","const","interp","detc","ordin","admrul","treaty","term","special","consulting","ministry","table","committee"],
                    },
                    "reference": {
                        "type": "string",
                        "description": "Case number (사건번호) for precedents, or law name (법령명) for statutes",
                    },
                },
                "required": ["category", "reference"],
            },
        },
    },
]


# ------------------------------------------------------------------
# Tool execution dispatcher
# ------------------------------------------------------------------

async def _execute_tool_call(
    tool_name: str,
    arguments: dict,
    searcher: Searcher | None,
    legal_api: LegalAPIClient | None,
    debate_id: str,
    team_id: str,
    search_results_for_resolve: list[dict] | None = None,
    summary_only: bool = False,
) -> str:
    """
    Execute a tool call and return the result as a string.

    Dispatches to the appropriate search backend based on the tool name.

    Args:
        tool_name: Name of the tool function to execute.
        arguments: Parsed arguments dict from the LLM tool call.
        searcher: Searcher instance for document/graph search.
        legal_api: LegalAPIClient for statute/precedent search.
        debate_id: Current debate ID for scoping searches.
        team_id: Current team ID for pool access control.

    Returns:
        JSON-formatted string of search results.
    """
    try:
        if tool_name == "search_documents" and searcher:
            query = arguments.get("query", "")
            results = await searcher.search_all_pools(
                query=query,
                debate_id=debate_id,
                team=team_id,
                top_k=5,
            )
            return json.dumps(results, ensure_ascii=False, default=str)

        elif tool_name == "search_legal" and legal_api:
            category = arguments.get("category", "law")
            query = arguments.get("query", "")
            max_results = min(arguments.get("max_results", 5), 20)
            optional = {k: v for k, v in arguments.items()
                        if k not in ("category", "query", "max_results") and v}
            result = await legal_api.search_legal(category, query, max_results, **optional)
            return json.dumps(result, ensure_ascii=False, default=str)

        elif tool_name == "get_legal_detail" and legal_api:
            category = arguments.get("category", "law")
            # Accept both 'reference' (new) and 'item_id' (legacy) params
            reference = arguments.get("reference", "") or arguments.get("item_id", "")
            # Resolve reference (case_number/law_name) → internal item_id
            _resolved_id = ""
            for sr in (search_results_for_resolve or []):
                if not isinstance(sr, dict):
                    continue
                if (sr.get("case_number") == reference
                        or sr.get("law_name") == reference
                        or sr.get("case_id") == reference
                        or sr.get("title") == reference):
                    _resolved_id = sr.get("_item_id", sr.get("item_id", ""))
                    if _resolved_id:
                        break
            if not _resolved_id:
                # Fallback: treat reference itself as item_id (backward compat)
                _resolved_id = reference
            result = await legal_api.get_legal_detail(category, _resolved_id, summary_only=summary_only)
            if not result or not result.get("content"):
                return json.dumps({"error": f"No content found for {category}/{reference}. Check the case number or law name."}, ensure_ascii=False)
            return json.dumps(result, ensure_ascii=False, default=str)

        elif tool_name == "search_statutes" and legal_api:
            query = arguments.get("query", "")
            max_results = arguments.get("max_results", 5)
            results = await legal_api.search_statutes(
                query=query, max_results=max_results
            )
            return json.dumps(results, ensure_ascii=False, default=str)

        elif tool_name == "search_precedents" and legal_api:
            query = arguments.get("query", "")
            results = await legal_api.search_precedents(
                query=query,
                court=arguments.get("court", "all"),
                max_results=arguments.get("max_results", 5),
                search=arguments.get("search", 0),
                org=arguments.get("org", ""),
                jo=arguments.get("jo", ""),
                sort=arguments.get("sort", ""),
                prncYd=arguments.get("prncYd", ""),
                nb=arguments.get("nb", ""),
            )
            return json.dumps(results, ensure_ascii=False, default=str)

        else:
            return json.dumps({"error": f"Unknown tool or service unavailable: {tool_name}"})

    except Exception as exc:
        logger.error("[team_speak] Tool execution error (%s): %s", tool_name, exc)
        return json.dumps({"error": str(exc)})


# ------------------------------------------------------------------
# Search phase: each member searches according to their assignment
# ------------------------------------------------------------------

async def _execute_agent_search(
    agent: dict,
    assignment: dict,
    state: TeamState,
    llm_client: LLMClient,
    searcher: Searcher | None,
    legal_api: LegalAPIClient | None,
    max_tool_rounds: int = 10,
    language: str = "ko",
    unverified_citations: list[str] | None = None,
) -> list[dict]:
    """
    Execute search tasks for a single agent based on their role assignment.

    Uses a 3-phase approach:
    1. Strategy Planning - decide what to search for
    2. Exploration Search - broad search for candidate items
    3. Deep Analysis - retrieve full text of most promising results

    Args:
        agent: Agent profile dict.
        assignment: Role assignment dict with task and search_type.
        state: Current TeamState.
        llm_client: LLM client for the agent.
        searcher: Searcher instance.
        legal_api: Legal API client.
        max_tool_rounds: Maximum total tool-calling rounds across phases.

    Returns:
        List of search result dicts collected by this agent.
    """
    agent_name = agent.get("name", "Agent")
    agent_id = agent.get("agent_id", "unknown")
    task = assignment.get("task", "Search for relevant evidence")
    search_type = assignment.get("search_type", "document")
    specialty = agent.get("specialty", "general law")

    logger.info(
        "[team_speak] Agent %s (%s) starting search: %s [type=%s]",
        agent_name, agent_id, task[:80], search_type,
    )

    opponent_stmt = state.get("opponent_statement", "(no opponent statement yet)")
    team_opinion = state.get("team_opinion", "")
    team_cautions = state.get("team_cautions", [])
    cautions_text = ""
    if team_cautions:
        cautions_lines = "\n".join(f"  - {c}" for c in team_cautions)
        cautions_text = f"\n\nSTRATEGIC CAUTIONS for your team:\n{cautions_lines}\n"

    # Include extra evidence if available
    extra_evidence = state.get("extra_evidence", [])
    extra_text = ""
    if extra_evidence:
        extra_lines = [
            f"- [{e.get('type', 'hint')}] {e.get('content', '')}"
            for e in extra_evidence
        ]
        extra_text = "\nAdditional evidence from user:\n" + "\n".join(extra_lines)

    all_results: list[dict] = []
    debate_id = state.get("debate_id", "")
    team_id = state.get("team_id", "team_a")

    # Track used queries to skip duplicate API calls within and across rounds
    _seen_queries: set[str] = set(state.get("used_search_queries", []))

    # Build category description list for strategy prompt
    category_desc_lines = []
    for cat_key, cat_info in LEGAL_CATEGORIES.items():
        if isinstance(cat_info, dict):
            cat_label = cat_info.get("label", cat_key)
            category_desc_lines.append(f"- {cat_key}: {cat_label}")
        else:
            category_desc_lines.append(f"- {cat_key}: {cat_info}")
    category_desc = "\n".join(category_desc_lines)

    # ----------------------------------------------------------------
    # Phase 1: Strategy Planning
    # ----------------------------------------------------------------
    logger.info("[team_speak] Agent %s: Phase 1 - Strategy Planning", agent_name)

    # Load legal system index for search planning
    try:
        from app.agents.legal_index import get_legal_index
        _legal_index = get_legal_index("korea")
    except Exception:
        _legal_index = ""

    phase1_system = (
        f"You are a legal research strategist planning evidence searches for a debate.\n\n"
        f"Think through these steps systematically:\n\n"
        f"1. **Issue Identification**: What are the core legal issues in this debate?\n"
        f"   - What legal requirements must be met for the claimed legal effect?\n"
        f"   - What legal principles or doctrines are applicable?\n\n"
        f"2. **Legal Sources Needed**: What types of legal authority do you need?\n"
        f"   - Statutory law: Which specific statutes/articles are relevant?\n"
        f"   - Case law: Are there Supreme Court (대법원) precedents with similar facts?\n"
        f"   - Interpretations: Is there a dispute about how a provision should be interpreted?\n\n"
        f"3. **Search Plan**: Create a prioritized list of searches.\n"
        f"   - For each search, specify: category, query terms, expected outcome, and any filters.\n"
        f"   - Start with the most important/foundational searches.\n"
        f"   - Include both supporting evidence AND potential counterarguments to prepare rebuttals.\n"
        f"   - Use specific statute names and article numbers for accurate results.\n"
        f"   - The API uses substring matching, so '민법' may return '난민법'. Be specific!\n"
        f"   - **Prefer recent precedents**: Sort by date descending (sort=ddes). A 2023 ruling\n"
        f"     supersedes a 2010 ruling on the same legal question.\n"
        f"   - **Verify law amendments**: After finding a precedent, check if the cited statute\n"
        f"     is still current (시행일자). If amended, note which version the precedent was based on.\n"
        f"   - **En banc (전원합의체) decisions** carry special weight over panel decisions.\n\n"
        f"## Query Construction Strategy (CRITICAL for search success)\n"
        f"The API uses KEYWORD SUBSTRING MATCHING — NOT semantic/meaning search.\n"
        f"A query only returns results where the EXACT keyword appears in the document.\n\n"
        f"### Keyword Selection Rules:\n"
        f"- Extract CORE LEGAL TERMS from the debate topic (법률 용어 위주)\n"
        f"- Use the SHORTEST meaningful term that uniquely identifies the concept\n"
        f"- Combine 2-3 keywords that are likely to CO-OCCUR in relevant documents\n"
        f"- For statutes: use the EXACT statute name (e.g., '민법', '상법', '근로기준법')\n"
        f"- For precedents: use FACT-PATTERN keywords (e.g., '해고 부당', '계약 해지 손해배상')\n\n"
        f"### Good vs Bad Query Examples:\n"
        f"  ✓ '보증보험 해지' — short, specific legal terms\n"
        f"  ✓ '근로계약 해고 정당' — fact-pattern keywords for precedent search\n"
        f"  ✓ '소멸시효 기산점' — precise legal concept\n"
        f"  ✗ '보증보험 계약을 무단으로 체결한 경우의 효력' — too long, natural language\n"
        f"  ✗ '부당해고에 해당하는지 여부' — sentence, not keywords\n"
        f"  ✗ '보험법' — too broad, will match 난민법, 보험업법 etc.\n\n"
        f"### Keyword Decomposition:\n"
        f"Break complex issues into MULTIPLE separate short queries:\n"
        f"  Issue: '소프트웨어 개발 계약의 하자 보수 의무 범위'\n"
        f"  → query1: '소프트웨어 하자 보수' (prec)\n"
        f"  → query2: '도급 하자담보' (prec)\n"
        f"  → query3: '민법 도급' (law)\n\n"
        f"Context:\n"
        f"- Agent: {agent_name}, specialty: {specialty}\n"
        f"- Task: {task}\n"
        f"- Team opinion: {team_opinion}\n"
        f"- Opponent statement: {opponent_stmt}\n"
        + (cautions_text + "\n" if cautions_text else "\n")
        + (_legal_index + "\n\n" if _legal_index else "")
        + f"Available search categories:\n"
        f"{category_desc}\n\n"
        f"IMPORTANT: Output ONLY valid JSON. No markdown, no code fences, no explanation.\n"
        f"Each query MUST be 2-4 Korean words.\n"
        f"Generate at LEAST 6 search queries total (minimum 3 for precedents, minimum 3 for statutes).\n"
        f"For each legal issue, create MULTIPLE keyword variations:\n"
        f"- Original term AND shorter form (e.g., '손해경감의무' → also '손해경감')\n"
        f"- Synonym/related terms (e.g., '손해경감' → also '손해 방지')\n"
        f"- Always include at least one BROAD 2-character keyword per issue.\n\n"
        f"Example output:\n"
        f'{{"legal_analysis": "...", "search_plan": ['
        f'{{"category": "law", "query": "전자서명법", "purpose": "..."}}, '
        f'{{"category": "prec", "query": "본인확인 의무", "purpose": "..."}}, '
        f'{{"category": "law", "query": "보증보험 해지", "purpose": "..."}}]}}\n\n'
        f"Required JSON format:\n"
        f'{{\n'
        f'    "legal_analysis": "Brief analysis of key legal issues and applicable doctrines",\n'
        f'    "search_plan": [\n'
        f'        {{\n'
        f'            "category": "law|prec|const|...",\n'
        f'            "query": "2-4 Korean keywords ONLY",\n'
        f'            "purpose": "why this search is needed",\n'
        f'            "filters": {{"optional": "filter params"}}\n'
        f'        }}\n'
        f'    ]\n'
        f'}}'
        + get_language_instruction(language)
    )

    team_id = state.get("team_id", "unknown")
    _team_a_name = state.get("team_a_name", "Team A")
    _team_b_name = state.get("team_b_name", "Team B")
    _team_display = _team_a_name if team_id == "team_a" else _team_b_name
    phase1_user = (
        f"You are on {_team_display}. Your search must find evidence that SUPPORTS your position "
        f"and WEAKENS the opponent's position.\n\n"
        f"YOUR POSITION (support this): {team_opinion}\n"
        f"OPPONENT'S POSITION (counter this): {opponent_stmt}\n"
        f"{extra_text}\n\n"
        f"Plan your search strategy for: {task}\n\n"
        f"QUERY OPTIMIZATION RULES:\n"
        f"- The Korean legal API uses keyword matching, NOT semantic search.\n"
        f"- Use 2-4 core Korean legal terms per query (shorter is better).\n"
        f"- Plan MULTIPLE short queries instead of one long query.\n"
        f"- Example: Instead of a long complex query like the full issue description,\n"
        f"  Plan: query1='core_term1 core_term2', query2='core_term3 core_term4'"
    )

    # Append previous search queries — guide broad→narrow strategy
    _prev_queries = state.get("used_search_queries", [])
    if _prev_queries:
        phase1_user += "\n\n## PREVIOUS SEARCHES (refine, don't repeat)\n"
        phase1_user += f"Already tried queries: {', '.join(_prev_queries[:20])}\n\n"
        phase1_user += (
            "These searches have already been executed in previous rounds.\n"
            "DO NOT repeat them. Instead, NARROW your search scope:\n"
            "- Review the queries above to understand what has already been covered broadly.\n"
            "- Focus on GAPS: what specific legal points were NOT covered by previous searches?\n"
            "- Use MORE SPECIFIC keywords that target the exact disputed issues.\n"
            "- If previous searches were general (e.g., '계약 해지'), now search for specific\n"
            "  sub-issues (e.g., '해지 통보 방법', '해지 효력 발생 시점').\n"
            "- Address opponent's latest arguments or judge feedback with targeted queries.\n"
        )
    else:
        phase1_user += (
            "\n\n## FIRST ROUND — BROAD SEARCH\n"
            "This is the first search round. Search BROADLY:\n"
            "- Cover the main legal issues from multiple angles.\n"
            "- Search for foundational statutes AND relevant precedents.\n"
            "- Include searches for potential counter-arguments.\n"
        )

    # Append unverified citations to phase1 prompt if any
    if unverified_citations:
        phase1_system += (
            "\n\n## UNVERIFIED CITATIONS (Verification Required)\n"
            "The following citations from earlier rounds could NOT be verified:\n"
            + "\n".join(unverified_citations) + "\n\n"
            "During your search, PRIORITIZE finding these citations.\n"
            "If found, they will be automatically verified.\n"
            "If NOT found, they should NOT be used in further arguments.\n"
        )

    phase1_messages: list[dict[str, Any]] = [
        {"role": "system", "content": phase1_system},
        {"role": "user", "content": phase1_user},
    ]

    phase1_plan = ""
    try:
        phase1_plan = await llm_client.achat(
            phase1_messages,
            temperature=0.3,
            max_tokens=2000,
        )
    except Exception as exc:
        logger.error(
            "[team_speak] Phase 1 failed for agent %s: %s", agent_id, exc
        )

    # Validate phase1_plan is parseable JSON; if not, try extraction then fallback
    try:
        parsed_plan = json.loads(phase1_plan)
        if "search_plan" not in parsed_plan:
            raise ValueError("Missing search_plan key")
    except (json.JSONDecodeError, ValueError, TypeError):
        # Retry: try to extract JSON from markdown-wrapped response
        _json_match = _re.search(r'\{[\s\S]*\}', phase1_plan)
        if _json_match:
            try:
                parsed_plan = json.loads(_json_match.group())
                if "search_plan" in parsed_plan:
                    logger.info(
                        "[team_speak] Agent %s: Phase 1 JSON extracted from wrapped response.",
                        agent_name,
                    )
                else:
                    raise ValueError("Missing search_plan key after extraction")
            except (json.JSONDecodeError, ValueError, TypeError):
                parsed_plan = None
        else:
            parsed_plan = None

    if parsed_plan is None:
        logger.warning(
            "[team_speak] Agent %s: Phase 1 plan not valid JSON, using keyword fallback.",
            agent_name,
        )
        # Extract legal keywords from task + team_opinion for meaningful queries
        _combined_text = f"{task} {team_opinion} {opponent_stmt}"
        _keywords = _extract_legal_keywords(_combined_text)
        logger.info("[team_speak] Agent %s: Extracted keywords: %s", agent_name, _keywords)

        default_search_plan: list[dict] = []
        for _kw in _keywords:
            default_search_plan.append({"category": "law", "query": _kw, "purpose": f"법령 검색: {_kw}"})
            default_search_plan.append({"category": "prec", "query": _kw, "purpose": f"판례 검색: {_kw}"})
        # Also add search_type-specific fallback with short topic
        if search_type == "statute" and not any(p["category"] == "law" for p in default_search_plan):
            default_search_plan.append({"category": "law", "query": task[:20], "purpose": "General statute search"})
        elif search_type == "precedent" and not any(p["category"] == "prec" for p in default_search_plan):
            default_search_plan.append({"category": "prec", "query": task[:20], "purpose": "General precedent search"})

        parsed_plan = {"reasoning": "Keyword fallback (phase 1 parsing failed)", "search_plan": default_search_plan}
        phase1_plan = json.dumps(parsed_plan, ensure_ascii=False)

    logger.info(
        "[team_speak] Agent %s: Phase 1 plan: %s", agent_name, phase1_plan[:200]
    )

    # ----------------------------------------------------------------
    # Phase 2: Exploration Search
    # ----------------------------------------------------------------
    logger.info("[team_speak] Agent %s: Phase 2 - Exploration Search", agent_name)

    phase2_rounds = min(max_tool_rounds // 2, 3)  # cap at 3 rounds

    phase2_system = (
        f"You are {agent_name}. Execute your search plan.\n"
        f"Plan: {phase1_plan}\n\n"
        f"PHASE: EXPLORATION\n"
        f"Search for candidate legal items. After each result, evaluate relevance.\n"
        f"Adjust queries or try different categories if needed.\n\n"
        f"Search results already include 판결요지 (decision summary) for precedents.\n"
        f"You may call get_legal_detail for more context, but it will return summaries only at this stage.\n"
        f"Full text analysis will happen automatically in the next phase.\n"
        f"Focus on finding the RIGHT sources rather than reading every detail.\n\n"
        f"## Search Result Relevance Filter\n"
        f"After receiving search results, critically evaluate EACH result before using it.\n"
        f"Ask yourself for EVERY result:\n"
        f"1. Does this result directly address the SPECIFIC issue being debated?\n"
        f"2. Would citing this strengthen my argument about THIS particular case?\n"
        f"3. Is this result about the same legal domain as our debate topic?\n"
        f"If the answer to ANY question is NO → DISCARD the result. Do not cite it.\n"
        f"Principle: 3 highly relevant sources > 10 loosely related ones.\n"
        f"If results seem off-topic, refine your query with more specific terms and search again.\n\n"
        f"Concrete example of relevance filtering:\n"
        f"- Topic: \"교통사고 과실 비율과 손해배상 책임\"\n"
        f"  ✓ KEEP: \"교통사고 과실상계에 관한 판례\" (directly relevant)\n"
        f"  ✓ KEEP: \"자동차손해배상보장법 제3조\" (applicable statute)\n"
        f"  ✗ DISCARD: \"관세법 시행령 개정안\" (customs law, unrelated field)\n"
        f"  ✗ DISCARD: \"수출입 통관 절차 위반 사례\" (customs/trade, not traffic)\n\n"
        f"## Legal Relevance Criteria\n"
        f"Apply these LEGAL-SPECIFIC criteria to evaluate each search result:\n\n"
        f"1. **Same or related statute**: Does the result reference the same law, article, or a closely\n"
        f"   related provision as the one at issue? A result about a different article of the SAME law\n"
        f"   is relevant. A result about a completely unrelated law is NOT.\n\n"
        f"2. **Analogous facts**: Does the precedent involve similar factual circumstances?\n"
        f"   Look for: same type of transaction, same type of parties, same type of dispute.\n"
        f"   A precedent with materially different facts (different industry, different legal\n"
        f"   relationship) is NOT relevant even if it uses similar keywords.\n\n"
        f"3. **Same legal principle**: Does the result apply the same legal doctrine or principle?\n"
        f"   Examples of 'same principle': consent requirements, duty of notification,\n"
        f"   statute of limitations computation, contractual validity.\n"
        f"   Examples of 'different principle': antitrust, customs, tax, immigration.\n\n"
        f"4. **Temporal relevance**: Was the cited law or precedent in effect at the time of the events?\n"
        f"   Prefer current/applicable versions of statutes over repealed ones.\n"
        f"   More recent precedents on the same issue carry more weight than older ones.\n\n"
        f"5. **Jurisdictional authority**: Supreme Court precedents carry the highest weight.\n"
        f"   Lower court decisions are informative but less authoritative.\n"
        f"   Constitutional Court decisions on fundamental rights are highly relevant.\n\n"
        f"SCORING: A result must meet AT LEAST criteria 1 OR 2 OR 3 to be considered relevant.\n"
        f"Meeting none of these three = DISCARD regardless of keyword overlap.\n\n"
        f"## Evidence Position Evaluation (CRITICAL)\n"
        f"BEFORE adding any search result to your evidence, evaluate if it SUPPORTS or UNDERMINES your team:\n"
        f"- SUPPORTS: The ruling/reasoning aligns with YOUR team's argument → USE IT\n"
        f"- UNDERMINES: The ruling contradicts YOUR team's position → DO NOT USE.\n"
        f"  Note it internally so you can prepare a counter-argument if the opponent cites it.\n"
        f"- NEUTRAL: Related but not directly supportive → Only use if you can frame it to support YOUR position.\n\n"
        f"Your job is to find evidence for YOUR side. The opponent will find their own.\n"
        f"If a precedent ruled AGAINST the position you are advocating, DO NOT cite it.\n\n"
        f"### WARNING — COMMON MISINTERPRETATION\n"
        f"A precedent that MENTIONS your legal topic is NOT automatically favorable.\n"
        f"You MUST read the actual CONCLUSION/HOLDING direction:\n"
        f"- Did the court GRANT or DENY the claim?\n"
        f"- Did the court find the contract VALID or INVALID?\n"
        f"- Did the court find LIABILITY or NO LIABILITY?\n"
        f"- Did the court UPHOLD or OVERTURN the lower court?\n\n"
        f"Example of wrong interpretation:\n"
        f"- Your position: 'Identity verification is important in e-commerce'\n"
        f"- Precedent holds: 'With a certified digital certificate, NO additional identity verification is needed'\n"
        f"- This precedent UNDERMINES your position even though it mentions 'identity verification'\n\n"
        f"Read the HOLDING carefully — the direction of the ruling matters more than the keywords."
        + get_language_instruction(language)
    )

    phase2_user = (
        f"Team opinion (we advocate): {team_opinion}\n"
        f"Opponent's latest statement: {opponent_stmt}\n"
        f"{extra_text}\n\n"
        f"Execute your assigned task: {task}"
    )

    phase2_messages: list[dict[str, Any]] = [
        {"role": "system", "content": phase2_system},
        {"role": "user", "content": phase2_user},
    ]

    for _round in range(phase2_rounds):
        try:
            response = await llm_client.achat_with_tools(
                messages=phase2_messages,
                tools=_SEARCH_TOOLS,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=2000,
            )
        except Exception as exc:
            logger.error(
                "[team_speak] Phase 2 tool call failed for agent %s: %s", agent_id, exc
            )
            break

        choice = response.choices[0]
        assistant_message = choice.message

        # Append the assistant's response to the conversation
        msg_dict: dict[str, Any] = {"role": "assistant", "content": assistant_message.content or ""}
        if assistant_message.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_message.tool_calls
            ]
        phase2_messages.append(msg_dict)

        # If no tool calls, the agent is done with exploration
        if not assistant_message.tool_calls:
            break

        # Execute each tool call and feed results back
        for tool_call in assistant_message.tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            # Track search queries for duplicate avoidance in next round
            _query_used = ""
            if fn_name in ("search_legal", "search_precedents", "search_statutes"):
                _query_used = fn_args.get("query", "")

            # Skip duplicate search queries (already executed in this or previous rounds)
            if _query_used and _query_used.strip() in _seen_queries:
                logger.info("[team_speak] Skipping duplicate query: %s", _query_used)
                phase2_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": fn_name,
                    "content": json.dumps({"info": "Duplicate query skipped. Results already available from a previous search."}),
                })
                continue

            result_str = await _execute_tool_call(
                tool_name=fn_name,
                arguments=fn_args,
                searcher=searcher,
                legal_api=legal_api,
                debate_id=debate_id,
                team_id=team_id,
                search_results_for_resolve=all_results,
                summary_only=True,  # Phase 2: summary only
            )

            if _query_used:
                _seen_queries.add(_query_used.strip())

            # 0건 반환 시 자동 재시도: 검색어 단순화 (마지막 단어 제거)
            _is_empty = ('"total": 0' in result_str or '"results": []' in result_str)
            if _query_used and _is_empty:
                _words = _query_used.strip().split()
                if len(_words) >= 2:
                    _simplified = " ".join(_words[:-1])
                    if _simplified not in _seen_queries:
                        logger.info("[team_speak] 0 results for '%s', retrying with '%s'", _query_used, _simplified)
                        _retry_args = {**fn_args, "query": _simplified}
                        _retry_result = await _execute_tool_call(
                            tool_name=fn_name,
                            arguments=_retry_args,
                            searcher=searcher,
                            legal_api=legal_api,
                            debate_id=debate_id,
                            team_id=team_id,
                            search_results_for_resolve=all_results,
                            summary_only=True,
                        )
                        _seen_queries.add(_simplified)
                        if '"total": 0' not in _retry_result and '"results": []' not in _retry_result:
                            result_str = _retry_result
                            _query_used = _simplified
                            _is_empty = False

            # 0건 결과를 LLM에 전달 시 이전 실패 검색어 포함
            if _is_empty:
                _failed = [q for q in _seen_queries if q != _query_used.strip()]
                result_str = json.dumps({
                    "results": [], "total": 0,
                    "note": f"0 results. Previously tried queries: {', '.join(_failed) if _failed else 'none'}",
                })

            # Parse results and collect them (handle both dict and list)
            try:
                parsed_results = json.loads(result_str)
                if isinstance(parsed_results, dict):
                    if "results" in parsed_results:
                        items = parsed_results.get("results", [])
                        for r in items:
                            if isinstance(r, dict):
                                r["found_by"] = agent_id
                                if _query_used:
                                    r["_search_query"] = _query_used
                        all_results.extend(items)
                    elif "error" not in parsed_results:
                        parsed_results["found_by"] = agent_id
                        if _query_used:
                            parsed_results["_search_query"] = _query_used
                        all_results.append(parsed_results)
                elif isinstance(parsed_results, list):
                    for r in parsed_results:
                        if isinstance(r, dict):
                            r["found_by"] = agent_id
                            if _query_used:
                                r["_search_query"] = _query_used
                    all_results.extend(parsed_results)
            except (json.JSONDecodeError, TypeError):
                pass

            # Feed tool result back to the conversation
            phase2_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": fn_name,
                "content": result_str,
            })

    logger.info(
        "[team_speak] Agent %s: Phase 2 complete, %d results so far.", agent_name, len(all_results)
    )

    # ----------------------------------------------------------------
    # Phase 3: Deep Analysis via RAG + LLM 1회
    # ----------------------------------------------------------------
    logger.info("[team_speak] Agent %s: Phase 3 - RAG Deep Analysis", agent_name)

    # Build case_id_map from Phase 2 results
    _local_case_id_map: dict[str, str] = {}
    for r in all_results:
        cn = r.get("case_number", "")
        iid = r.get("_item_id", r.get("item_id", r.get("id", "")))
        if cn and iid and cn not in _local_case_id_map:
            _local_case_id_map[cn] = str(iid)

    # Select top precedents and statutes for RAG retrieval
    _precedent_results = [r for r in all_results if r.get("source") == "court_precedent" and r.get("case_number")]
    _top_precedents = _precedent_results[:3]

    rag_texts: list[str] = []
    if _top_precedents and legal_api:
        _rag_query = f"{team_opinion[:200]} {task}"
        for prec in _top_precedents:
            cn = prec.get("case_number", "")
            if not cn or cn not in _local_case_id_map:
                continue
            try:
                rag_result = await _on_demand_precedent_search(
                    cn, _rag_query, _local_case_id_map, legal_api,
                )
                if rag_result and not rag_result.startswith("["):
                    rag_texts.append(f"## {cn}\n{rag_result}")
                    prec["full_text_excerpt"] = rag_result
                    logger.info("[team_speak] Phase 3 RAG retrieved for %s (%d chars)", cn, len(rag_result))
            except Exception as exc:
                logger.warning("[team_speak] Phase 3 RAG failed for %s: %s", cn, exc)

    # Single LLM call to analyze all retrieved texts
    if rag_texts:
        phase3_prompt = (
            f"당신은 {agent_name}입니다. 다음 판례 원문을 분석하세요.\n"
            f"팀 입장: {team_opinion}\n"
            f"상대측 주장: {opponent_stmt}\n\n"
            + "\n\n".join(rag_texts)
            + f"\n\n각 판례에 대해 다음을 판단하세요:\n"
            f"1. 판결 방향: 청구 인용/기각, 책임 인정/부정\n"
            f"2. 우리 입장 지지 여부: 유리/불리/중립\n"
            f"3. 핵심 판시사항 요약 (2-3문장)\n"
            f"불리한 판례는 반드시 '불리'로 표시하고 사용하지 마세요."
            + get_language_instruction(language)
        )
        try:
            _phase3_analysis = await llm_client.achat(
                [{"role": "user", "content": phase3_prompt}],
                temperature=0.3,
                max_tokens=2000,
            )
            logger.info("[team_speak] Agent %s: Phase 3 analysis complete (%d chars)", agent_name, len(_phase3_analysis))
            # Attach analysis to top precedents for downstream use
            for prec in _top_precedents:
                prec["_phase3_analysis"] = _phase3_analysis
        except Exception as exc:
            logger.error("[team_speak] Phase 3 LLM analysis failed for %s: %s", agent_name, exc)
    else:
        logger.info("[team_speak] Agent %s: Phase 3 skipped (no precedents for RAG)", agent_name)

    logger.info(
        "[team_speak] Agent %s found %d results.", agent_name, len(all_results)
    )
    return all_results


# ------------------------------------------------------------------
# Discussion phase: internal team deliberation
# ------------------------------------------------------------------


def _build_qa_history_summary(qa_log: list) -> str:
    """Build a summary of previous judge Q&A for inclusion in prompts."""
    if not qa_log:
        return ""
    summary = (
        "\n\n## Previous Judge Q&A History (Important)\n"
        "The following questions were asked by judges. Judge questions reveal what judges\n"
        "consider CRITICAL to the case. You MUST proactively address these concerns in\n"
        "your arguments. Failing to address judge concerns weakens your position.\n\n"
    )
    for qa in qa_log:
        q = qa.get("question", "")
        a = qa.get("answer", "")
        judge = qa.get("judge_name", "?")
        target = qa.get("target_agent_id", "?")
        rnd = qa.get("round", "?")
        summary += f"- [Round {rnd}] Judge {judge} → {target}: {q}\n"
        if a:
            summary += f"  Answer: {a}\n"
    return summary


def _build_user_intervention_section(extra_evidence: list[dict] | None) -> str:
    """Build a priority section for user interventions in the discussion prompt."""
    if not extra_evidence:
        return ""
    _type_labels = {"hint": "Hint", "evidence": "Evidence"}
    lines = []
    for ev in extra_evidence:
        _utype = _type_labels.get(ev.get("type", "hint"), ev.get("type", "hint"))
        _ucontent = ev.get("content", "")
        _uround = ev.get("round", 0)
        lines.append(f"  - [{_utype}] (Round {_uround}): {_ucontent}")
    return (
        "\n## ⚡ USER INTERVENTION (PRIORITY — address FIRST)\n"
        "The user has provided the following interventions. Discuss these BEFORE other topics.\n"
        "- Hint: Strengthen arguments or adjust search strategy in this direction\n"
        "- Evidence: Incorporate this evidence into your arguments\n\n"
        + "\n".join(lines)
        + "\n\n"
    )


async def _conduct_discussion(
    members,
    search_results: list[dict],
    team_opinion: str,
    opponent_stmt: str,
    llm_client: LLMClient,
    language: str = "ko",
    debate_id: str = "",
    team_id: str = "unknown",
    judge_qa_log: list | None = None,
    assignments: list | None = None,
    situation_brief: str = "",
    analysis_summary: str = "",
    opponent_cited_summary: str = "",
    team_display_name: str = "",
    case_id_map: dict[str, str] | None = None,
    legal_api=None,
    topic: str = "",
    our_prev_stmt: str = "",
    judge_improvement_feedback: str = "",
    extra_evidence: list[dict] | None = None,
    prior_review_more_agents: dict[str, set] | None = None,
    prior_accept_votes: dict[str, set] | None = None,
) -> tuple[list[dict], set[str]]:
    """
    Conduct internal team discussion using a flat message-count loop.

    Total message count is controlled by ``team_discussion_turns`` in
    settings (NOT turns x members). Speaker selection uses a score-based
    system that considers participation balance, assignment priority,
    PASS history, and prevents consecutive turns by the same agent.

    The LLM may include ``[CONSENSUS]`` in its response to signal early
    agreement; the loop will break only when **all** members have spoken
    at least once.

    Progress is saved every 3 messages so the frontend can poll.

    Returns:
        List of discussion message dicts.
    """
    from app.api.settings import settings_mgr

    settings = settings_mgr.load()
    _debate_settings = settings.get("debate", {})
    max_messages = _debate_settings.get("team_discussion_turns", 15)
    max_review_more = _debate_settings.get("max_review_more", 3)
    accept_ratio = _debate_settings.get("accept_ratio", 0.4)
    review_more_ratio = _debate_settings.get("review_more_ratio", 0.6)

    discussion_log: list[dict] = []
    last_speaker_id = None
    consensus_votes: dict[str, bool] = {}  # agent_id → True if agreed

    # Precedent review tracking
    pending_reviews: list[dict] = []  # {"case_number", "result", "found_by"}
    review_queue: list[dict] = []  # 대기열: pending이 2개 미만이면 여기서 보충
    # accept_votes / review_more_agents는 라운드 간에도 유지되어야 함 → 호출자가 넘겨준 dict를 직접 사용(in-place mutate)
    accept_votes: dict[str, set] = prior_accept_votes if prior_accept_votes is not None else {}
    reject_votes: dict[str, set] = {}  # case_number → set of agent_ids who rejected
    # Track all tool call results for REVIEW_MORE re-injection
    all_tool_results: dict[str, str] = {}  # case_number → result text
    review_more_counts: dict[str, int] = {}  # case_number → REVIEW_MORE 횟수 (per-case, round limit용)
    review_more_agents: dict[str, set] = prior_review_more_agents if prior_review_more_agents is not None else {}
    next_round_priority: list[dict] = []  # REVIEW_MORE 제한 도달 → 다음 라운드 우선 논의
    review_history: dict[str, list[str]] = {}  # case_number → 이전 리뷰 발언 이력
    # Track resolved (accepted/rejected/blacklisted) cases to prevent re-review
    resolved_cases: set[str] = set()
    # Evidence sharing: case numbers cited in discussion become visible to all agents
    shared_case_numbers: set[str] = set()
    # Hallucination warning: injected into next agent's prompt then cleared
    _hallucination_warning: str = ""

    # Pre-populate review queue from Phase 2/3 search results
    # so agents review ALL found precedents during discussion
    _prec_results = [
        sr for sr in search_results
        if isinstance(sr, dict)
        and sr.get("case_number")
        and "prec" in str(sr.get("source", ""))
    ]
    for sr in _prec_results:
        cn = sr["case_number"]
        _review_text = str(sr.get("content", ""))
        _p3 = sr.get("_phase3_analysis", "")
        if _p3:
            _review_text += f"\n[Deep Analysis]\n{_p3[:300]}"
        review_entry = {
            "case_number": cn,
            "result": _review_text,
            "found_by": sr.get("found_by", "research"),
        }
        all_tool_results[cn] = _review_text
        if len(pending_reviews) < 2:
            pending_reviews.append(review_entry)
        else:
            review_queue.append(review_entry)
    if pending_reviews:
        logger.info(
            "[discussion] Pre-loaded %d precedents for review (%d queued)",
            len(pending_reviews), len(review_queue),
        )

    if not members:
        logger.warning("[discussion] No team members — skipping discussion")
        return [], set()

    for msg_idx in range(max_messages):

        # Score-based speaker selection
        def _speaker_score(m):
            aid = m.get("agent_id", "")
            if not aid:
                return float('-inf')  # skip members without agent_id
            score = 0
            spoke_count = sum(1 for d in discussion_log if d.get("agent_id") == aid)
            # Never-spoken agents get highest priority
            if spoke_count == 0:
                score += 10
            # Fewer speaks = higher priority (balance participation)
            score -= spoke_count * 2
            # Last speaker penalty — prevent consecutive turns
            if last_speaker_id and aid == last_speaker_id:
                score -= 20
            # Recent PASS penalty — agent that just passed is less likely to have new input
            for entry in discussion_log[-3:]:
                if entry.get("agent_id") == aid and "[PASS]" in entry.get("content", ""):
                    score -= 15
            # Assignment priority bonus (primary agents get slight boost)
            if assignments:
                _priority_rank = {"primary": 2, "supporting": 1}
                for a in assignments:
                    if a.get("agent_id") == aid:
                        score += _priority_rank.get(a.get("priority", ""), 0)
                        break
            # [ASK: name] tag detection — nominated agent gets priority
            if discussion_log:
                _last_entry = discussion_log[-1]
                _last_content = _last_entry.get("content", "")
                _last_aid = _last_entry.get("agent_id", "")
                _agent_name = m.get("name", "")
                if _last_aid != aid and _agent_name:
                    for _asked in _re.findall(r'\[ASK:\s*([^\]]+)\]', _last_content):
                        if _asked.strip() == _agent_name:
                            score += 15
                            break
            return score

        speaker = max(members, key=_speaker_score)

        agent_name = speaker.get("name", "Unknown")
        agent_id = speaker.get("agent_id", "")
        specialty = speaker.get("specialty", "")

        # Find this agent's argument angle from assignments
        agent_angle = ""
        for asgn in (assignments or []):
            if asgn.get("agent_id") == agent_id:
                angles = asgn.get("argument_angle", [])
                if isinstance(angles, list) and angles:
                    agent_angle = ", ".join(angles)
                elif isinstance(angles, str) and angles:
                    agent_angle = angles
                break

        all_spoke = len({d.get("agent_id", "") for d in discussion_log}) >= len(members)
        prev_context = ""
        if discussion_log:
            _history_lines: list[str] = []
            for d in discussion_log[-7:]:
                _sp = d.get("speaker", "?")
                _sid = d.get("agent_id", "")
                _marker = (
                    f"YOU ({agent_name}) said earlier"
                    if _sid == agent_id
                    else f"Teammate {_sp} said"
                )
                _content = str(d.get("content", "")).strip()
                _history_lines.append(f"--- {_marker} ---\n{_content}")
            prev_context = "\n\n".join(_history_lines) + "\n"

        # Build search summary — show THIS agent's results + shared results (cited by others)
        # Sliding window: REJECT/BLACKLIST'd items are removed, ACCEPT'd shown condensed,
        # remaining items slide up to fill the visible window (max 8).
        search_summary = ""
        _agent_results = [sr for sr in search_results if sr.get('found_by') == agent_id]
        # Include results from other agents that have been cited (shared via discussion)
        if shared_case_numbers:
            _own_case_nums = {
                sr.get('case_number', sr.get('case_id', '')) or sr.get('law_name', '')
                for sr in _agent_results
            }
            for sr in search_results:
                if sr.get('found_by') == agent_id:
                    continue
                _sr_id = sr.get('case_number', sr.get('case_id', '')) or sr.get('law_name', '')
                if _sr_id and _sr_id in shared_case_numbers and _sr_id not in _own_case_nums:
                    _agent_results.append(sr)

        # Separate accepted items (shown condensed) from available items
        _accepted_summary_lines: list[str] = []
        _available_results: list[dict] = []
        for sr in _agent_results:
            if not isinstance(sr, dict):
                continue
            _sr_case_num = sr.get('case_number', sr.get('case_id', ''))
            _sr_law_name = sr.get('law_name', '')
            _sr_id = _sr_case_num or _sr_law_name
            # Skip REJECT'd / BLACKLIST'd items (resolved but not accepted)
            if _sr_id and _sr_id in resolved_cases:
                if _sr_id in accept_votes and accept_votes[_sr_id]:
                    # ACCEPT'd → condensed summary only
                    _sr_title = sr.get('title', '')
                    _label = f"{_sr_id}"
                    if _sr_title:
                        _label += f" : {_sr_title}"
                    _accepted_summary_lines.append(f"  ✓ {_label}")
                # else: REJECT'd/BLACKLIST'd → skip entirely (sliding window)
                continue
            _available_results.append(sr)

        # Show accepted items as condensed summary
        if _accepted_summary_lines:
            search_summary += "\n[Already ACCEPTED — will be cited in statement]\n"
            search_summary += "\n".join(_accepted_summary_lines) + "\n"

        # Show remaining available items (sliding window, max 8)
        for i, sr in enumerate(_available_results[:8]):
            source_detail = sr.get('source_detail', '')
            content = str(sr.get('content', ''))
            source_type = sr.get('source_type', sr.get('type', ''))
            case_num = sr.get('case_number', sr.get('case_id', ''))
            law_name = sr.get('law_name', '')
            court = sr.get('court', '')
            date = sr.get('date', '')

            _title = sr.get('title', '')
            _is_prec = "prec" in str(sr.get("source", ""))

            # Precedent: case_number : title + content summary
            if _is_prec and case_num:
                header = case_num
                _sr_content_stripped = content.strip()
                _sr_title_stripped = _title.strip()
                # Ghost result check
                if _sr_content_stripped and (
                    _sr_content_stripped == _sr_title_stripped or len(_sr_content_stripped) < 20
                ):
                    search_summary += f"\n[{source_type}] {header}"
                    if _title:
                        search_summary += f" : {_title}"
                    search_summary += f"\n  ⚠ WARNING: NO actual legal content (only title). DO NOT cite this case.\n"
                else:
                    search_summary += f"\n[{source_type}] {header}"
                    if _title:
                        search_summary += f" : {_title}"
                    if content:
                        search_summary += f"\n  {content}\n"
                    else:
                        search_summary += "\n"
                    # Include Phase 3 deep analysis if available
                    _p3 = sr.get("_phase3_analysis", "")
                    if _p3:
                        search_summary += f"  [Deep Analysis] {_p3[:300]}\n"
            else:
                # Non-precedent (statutes, etc.)
                header = source_detail or law_name or case_num or _title or f"(unnamed item {i+1})"
                meta_parts = [p for p in [court, date, case_num] if p]
                meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
                search_summary += f"\n[{source_type}] {header}{meta}\n"
                if content:
                    search_summary += f"  Content: {content}\n"
        # Build evidence whitelist for verification (exclude REJECT/BLACKLIST'd)
        whitelist_lines = []
        for sr in _agent_results:
            if not isinstance(sr, dict):
                continue
            cn = sr.get('case_number', sr.get('case_id', ''))
            ln = sr.get('law_name', '')
            _wl_id = cn or ln
            # Skip items that are resolved as REJECT/BLACKLIST (not accepted)
            if _wl_id and _wl_id in resolved_cases:
                if _wl_id not in accept_votes or not accept_votes[_wl_id]:
                    continue  # REJECT'd/BLACKLIST'd → exclude from whitelist
            if cn:
                whitelist_lines.append(f"- [판례] {cn}")
            if ln:
                whitelist_lines.append(f"- [법령] {ln}")
        whitelist_text = "\n".join(sorted(set(whitelist_lines))) if whitelist_lines else "- (no verified evidence yet)"

        # --- Discussion citation rule branching based on retrieved evidence types ---
        _disc_prec_count = sum(1 for ln in whitelist_lines if ln.startswith("- [판례]"))
        _disc_stat_count = sum(1 for ln in whitelist_lines if ln.startswith("- [법령]"))
        if _disc_prec_count > 0:
            _disc_citation_rule = (
                "MANDATORY: Your response MUST cite at least 1 precedent using [판례: ID] tag.\n"
                "EXCEPTION: If you are currently reviewing a teammate's precedent "
                f"(using [ACCEPT]/[REJECT]/[BLACKLIST]/[REVIEW_MORE] tags), you may skip citing your own precedent in that response.\n"
                "If a precedent has already been proposed by a teammate, you may build on it instead of finding a new one.\n\n"
            )
        elif _disc_stat_count > 0:
            _disc_citation_rule = (
                "NOTE: No precedent was retrieved for this issue in this round's searches.\n"
                "You should NOT cite any [판례: ...] in this message — such a citation will be treated as fabricated.\n"
                "Instead:\n"
                "- Cite available statutes using [법령: ID] with specific articles, OR\n"
                "- Propose additional search queries the team should run to find precedents\n"
                "  (explicitly state the queries you would try), OR\n"
                "- Argue from statutory interpretation (textual, systematic, teleological).\n"
                "Acknowledging the precedent gap openly is better than inventing one.\n\n"
            )
        else:
            _disc_citation_rule = (
                "NOTE: Neither precedent nor statute has been retrieved yet for this issue.\n"
                "You MUST NOT emit any [판례:...], [법령:...], [헌재:...], [행심:...], or [문서:...] tag in this message\n"
                "(they will be flagged as fabricated citations).\n"
                "Instead, your response should focus on:\n"
                "- Proposing concrete additional search queries (exact keywords you would try), OR\n"
                "- Framing the issue in terms of general legal principles (clearly labeled as principles).\n"
                "Credibility requires honesty about the evidence gap.\n\n"
            )

        # Add situation brief, analysis, and opponent evidence to context
        # Frame differently per team to prevent position drift
        situation_section = ""
        if situation_brief:
            if team_id != "team_a":
                situation_section = (
                    f"\n## Situation Description (written by the OPPOSING PARTY)\n"
                    f"⚠ WARNING: This account is written from the opponent's perspective. "
                    f"It contains their subjective claims, emotional framing, and one-sided facts.\n"
                    f"Your job is to find legal grounds to DEFEND your client's position against these claims.\n"
                    f"Do NOT be swayed by emotional framing — focus on LEGAL MERITS of YOUR position.\n"
                    f"YOUR client's position: {team_opinion}\n\n"
                    f"{situation_brief}\n"
                )
            else:
                situation_section = (
                    f"\n## Situation Description (written by YOUR client)\n"
                    f"This is your client's account. While advocating for them, remember:\n"
                    f"- Focus on LEGAL ARGUMENTS, not emotional appeals\n"
                    f"- The opponent WILL challenge the facts here — be prepared to defend them with evidence\n"
                    f"- Do NOT let the opponent's counter-arguments weaken your position\n\n"
                    f"{situation_brief}\n"
                )
        if analysis_summary:
            situation_section += f"\n## Key Facts\n{analysis_summary}\n"
        if opponent_cited_summary:
            situation_section += (
                f"\n## Opponent's Cited Evidence (for reference/rebuttal ONLY)\n"
                f"{opponent_cited_summary}\n"
            )

        _display = team_display_name or team_id
        system_prompt = SIMULATION_FRAME_ADVOCATE + (
            f"You are {agent_name}, a legal expert specializing in {specialty}.\n"
            f"{situation_section}\n"
            f"You are in an internal team strategy meeting for {_display}.\n\n"
            f"IMPORTANT ROLE CONSTRAINTS:\n"
            f"- You are on {_display}. You MUST advocate for YOUR team's position.\n"
            f"- You must NEVER agree with or adopt the opponent's arguments.\n"
            f"- When discussing opponent's arguments, your goal is to find weaknesses and counter-arguments.\n"
            f"- If you find the opponent's evidence compelling, figure out how to reframe or distinguish it, NOT agree with it.\n"
            f"- NEVER make statements that strengthen the opponent. If you find a weakness in your team's argument, propose a solution alongside it.\n"
            f"- If a teammate cited a HARMFUL precedent, flag it and discuss how to counter — do NOT include it in the final statement.\n\n"
            f"## PERSONA LOCK — YOU ARE {agent_name} (CRITICAL)\n"
            f"- You are **{agent_name}** and ONLY {agent_name}. You speak for yourself, in the first person.\n"
            f"- You are in a meeting room with teammates: {', '.join(m.get('name', '') for m in members)}.\n"
            f"- NEVER write text as if another teammate is speaking. Do NOT put words in their mouth.\n"
            f"- NEVER prefix a line with another teammate's name (e.g. '[Park]:', 'Park:', '**Park**:', 'Park says:', 'As Lee would argue').\n"
            f"- NEVER fabricate a dialogue between multiple speakers in one response. You produce exactly ONE speaker's contribution: yours.\n"
            f"- If you want to respond to a teammate, refer to them in the third person (e.g. 'I disagree with Park's point about X because...') — do NOT simulate their voice.\n"
            f"- Before writing, silently confirm: 'I am {agent_name}. This response contains only my own words.'\n\n"
            f"## YOUR CORE POSITION (ANCHOR — NEVER ABANDON)\n"
            f"{team_opinion}\n\n"
            f"ALL your contributions must serve this position.\n"
            f"When the opponent makes a compelling point, your job is to\n"
            f"REFRAME or DISTINGUISH it, not concede.\n\n"
            f"⚠ POSITION CHECK: Before writing your response, verify:\n"
            f"- Am I arguing FOR: '{team_opinion[:100]}...'?\n"
            f"- Am I arguing AGAINST the opponent's claims?\n"
            f"- If my response could help the OPPONENT, I must REWRITE it.\n\n"
            f"=== OPPONENT'S LATEST ARGUMENT (you must COUNTER this, not agree) ===\n"
            f"{opponent_stmt}\n"
            f"=== END OF OPPONENT'S ARGUMENT ===\n\n"
            + (
                f"=== OUR TEAM'S LATEST STATEMENT (build upon this, don't repeat) ===\n"
                f"{our_prev_stmt}\n"
                f"=== END OF OUR STATEMENT ===\n\n"
                if our_prev_stmt else ""
            )
            + (
                f"## 4-DIMENSION RELEVANCE CHECK (CRITICAL)\n"
                f"Before citing ANY precedent, verify ALL 4 dimensions match our case:\n"
                f"1. VICTIM — Is the victim's situation/role similar?\n"
                f"2. SITUATION — Is the factual context similar?\n"
                f"3. CONDUCT — Is the wrongful act/violation similar?\n"
                f"4. PERPETRATOR — Is the perpetrator's role/type similar?\n"
                f"If even ONE dimension does not match, the precedent is IRRELEVANT — do NOT cite it.\n"
                f"Compare at the level of CONCRETE FACTS, not abstract legal principles.\n"
                f"Ask yourself: 'If I remove the domain keywords, is the fact pattern still the same?'\n"
                f"IRRELEVANT: same keyword but different facts (e.g., both 'employer' but fired vs. wage reduction)\n"
                f"RELEVANT: same concrete facts (e.g., both 'employer fired without written notice')\n"
                f"If the only way to connect is through an abstract principle (e.g., 'contract invalidity'), it is IRRELEVANT.\n"
                f"The specific WHO, WHAT, HOW must match — not just the legal category.\n\n"
                f"## RULING DIRECTION CHECK (CRITICAL)\n"
                f"A precedent can match your topic keywords but rule in the OPPOSITE direction.\n\n"
                f"BEFORE citing ANY precedent, answer these questions:\n"
                f"1. WHO won in this case? (plaintiff or defendant)\n"
                f"2. WHAT did the court hold? (contract valid/invalid, liability/no liability, etc.)\n"
                f"3. Does the court's holding SUPPORT or CONTRADICT your team's position?\n\n"
                f"EXAMPLE OF WRONG-DIRECTION PRECEDENT:\n"
                f"- Your argument: 'The insurance company committed fraud, so the contract is void'\n"
                f"- Precedent holds: 'The POLICYHOLDER committed fraud, so the contract is void'\n"
                f"- This precedent supports the OPPOSITE party's right to void the contract!\n"
                f"- → BLACKLIST: This ruling would actually strengthen the opponent's argument.\n\n"
                f"If the precedent's holding supports the OTHER side's argument, it is HARMFUL.\n"
                f"Use [BLACKLIST: case_number] for such cases.\n\n"
                f"ABSOLUTE RULE: If the court's RESULT goes against your team's position → BLACKLIST. NO EXCEPTIONS.\n"
                f"Do NOT try to 'distinguish' or 'salvage' an opposite-direction precedent.\n"
                f"'Our case is different because...' is NOT a valid reason to ACCEPT — the opponent will use the same precedent against you.\n\n"
                f"SALVAGING TRAP — DO NOT FALL FOR IT:\n"
                f"If you find yourself saying 'this precedent has unfavorable parts, BUT we can argue\n"
                f"our case is different...' → STOP. This is a sign the precedent should be BLACKLISTED.\n"
                f"The opponent can use that same precedent MORE EFFECTIVELY than you can.\n"
                f"Example: If the precedent held that verification procedures WERE adequate,\n"
                f"do NOT try to argue 'our case had even worse procedures.' The opponent will cite\n"
                f"the same precedent to show similar procedures have been upheld. BLACKLIST it instead.\n\n"
                f"## Precedent Full-Text Search\n"
                f"The system automatically searches precedent full texts when you cite a case number.\n"
                f"The auto-search uses the debate topic as query, showing the most relevant excerpts.\n"
                f"If you need to examine a SPECIFIC aspect (e.g., a particular holding or procedure),\n"
                f"use search_precedent_detail(query, case_number) with your own query.\n\n"
                f"IMPORTANT: Only propose precedents YOU found. Review teammates' precedents using the review tags.\n\n"
                if case_id_map else ""
            )
            + (
                _build_resolved_section(resolved_cases, accept_votes, case_id_map=case_id_map)
                if resolved_cases else ""
            )
            + (
                _build_pending_review_section(pending_reviews, team_opinion=team_opinion)
                if pending_reviews else ""
            )
            + (
                f"\n## CRITICAL: REVIEW BEFORE ANYTHING ELSE\n"
                f"There are {len(pending_reviews)} precedent(s) awaiting your review.\n"
                f"You MUST start your response with review tags for ALL pending precedents.\n"
                f"Your response will be REJECTED if it does not contain review tags.\n"
                f"Do NOT discuss strategy, cite new evidence, or use [ASK:] until ALL reviews are complete.\n\n"
                if pending_reviews else ""
            )
            + (
                _build_user_intervention_section(extra_evidence)
                if extra_evidence else ""
            )
            + f"## Research Results (NOT YET DISCUSSED — you must actively cite these)\n"
            f"The following were found during the research phase. They have NOT been discussed yet.\n"
            f"You MUST cite relevant items using [판례: ID] or [법령: ID] tags and explain how they support your argument.\n"
            f"{search_summary}\n"
            f"{_build_qa_history_summary(judge_qa_log or [])}\n"
            + (
                f"## Judge Improvement Feedback (CRITICAL — address these in this round)\n"
                f"The judges identified these weaknesses in your previous round's arguments.\n"
                f"You MUST address these points to strengthen your position:\n"
                f"{judge_improvement_feedback}\n\n"
                if judge_improvement_feedback else ""
            )
            + f"## Evidence Whitelist (ONLY these may be cited)\n"
            f"{whitelist_text}\n"
            f"ANY citation NOT in this list is INVALID and must be challenged.\n\n"
            f"## Prior Turns in This Meeting (REFERENCE ONLY — NOT a template to continue)\n"
            f"Below are past contributions from you and your teammates. They are shown so you know what has already been said.\n"
            f"They are NOT a format you should copy. Your response must be plain prose written in {agent_name}'s own first-person voice.\n\n"
            f"{prev_context}\n"
            f"## End of prior turns\n"
            f"You are {agent_name}. It is now YOUR turn. Write ONLY your own contribution — no teammate name prefixes, no simulated dialogue.\n\n"
            + (
                f"## YOUR ARGUMENT ANGLE: {agent_angle}\n"
                f"You MUST focus your analysis through this specific legal perspective.\n"
                f"Do NOT try to cover all angles — stay in YOUR lane.\n\n"
                if agent_angle else ""
            )
            + (
                f"This is your INITIAL position statement. You have NOT seen your "
                f"teammates' opinions yet. Present YOUR independent analysis from "
                f"your assigned argument angle.\n"
                f"You MUST cite specific precedents or statutes from the research results above "
                f"to support your position. Use [판례: ID] or [법령: ID] tags.\n\n"
                if not discussion_log else
                (
                    (
                        f"You can now see your teammates' positions.\n"
                        f"REVIEW GUIDELINES — For each cited precedent, apply these checks IN ORDER:\n\n"
                        f"**STEP 1 — CONCLUSION FIRST** (MOST IMPORTANT — check the RULING DIRECTION CHECK above):\n"
                        f"  Before looking at ANY reasoning or logic in the precedent, check the FINAL RESULT:\n"
                        f"  WHO won? WHAT was the court's CONCLUSION? (not the intermediate reasoning)\n"
                        f"  → If the court's CONCLUSION goes AGAINST our position → [BLACKLIST: case_number]. IMMEDIATELY. NO EXCEPTIONS.\n"
                        f"  ⚠ A precedent may contain favorable REASONING but reach an UNFAVORABLE CONCLUSION.\n"
                        f"  The opponent will cite the CONCLUSION, not the reasoning. Do NOT be deceived by intermediate logic.\n\n"
                        f"**STEP 2 — FACT-LEVEL RELEVANCE** (apply the 4-DIMENSION CHECK above):\n"
                        f"  Compare CONCRETE FACTS (who, what, how) — not abstract legal principles.\n"
                        f"  → If even ONE dimension does not match at the fact level → [REJECT: case_number]\n\n"
                        f"**STEP 3 — ACCEPT only if BOTH steps pass**:\n"
                        f"  The court's CONCLUSION favors our side AND the facts concretely match our case.\n"
                        f"  → [ACCEPT: case_number] and explain the specific holding that helps us.\n\n"
                        + (
                            f"**[REVIEW_MORE: case_number]** — Use ONLY when:\n"
                            f"  The retrieved excerpt does not contain enough information to determine the court's CONCLUSION or fact-level match.\n"
                            f"  This triggers an additional search to retrieve the missing portion.\n"
                            f"  If the conclusion is already visible and goes against you → BLACKLIST, not REVIEW_MORE.\n\n"
                        )
                        + f"Additional rules:\n"
                        f"- If a teammate cites something NOT in the whitelist → [REJECT: case_number]\n"
                        f"- When rejecting or blacklisting, suggest a better alternative from search results\n"
                        f"- Do not leave precedents unreviewed — every cited precedent needs a tag\n\n"
                        if pending_reviews else
                        f"You can now see your teammates' positions.\n"
                        f"Build on their arguments and contribute YOUR unique angle.\n\n"
                    )
                    + f"DO NOT repeat arguments or precedents already discussed by teammates in the conversation above.\n"
                    f"NOTE: Research-phase search results are NOT 'already discussed' — you MUST still cite and analyze them.\n"
                    f"Focus on what they MISSED, not what they already covered.\n\n"
                )
            )
            + f"## Evidence Impact Assessment (MANDATORY)\n"
            f"Before discussing strategy, FIRST rank ALL evidence items by their impact on the case:\n\n"
            f"**IMPACT RANKING**: For each evidence item, assess:\n"
            f"- How DIRECTLY does it prove/disprove a key element of the case?\n"
            f"- Does it address a CORE legal requirement (element of a cause of action) or just a peripheral point?\n"
            f"- Would a judge consider this DECISIVE or merely supplementary?\n\n"
            f"**PRIORITY RULE**:\n"
            f"- Evidence that proves/disproves a MANDATORY legal element (constitutive requirements, validity requirements) = HIGHEST priority\n"
            f"- Evidence about procedural requirements = HIGH priority\n"
            f"- Evidence about general legal principles = MEDIUM priority\n"
            f"- Evidence about peripheral or analogous situations = LOW priority\n\n"
            f"**EVIDENCE STRENGTH HIERARCHY**:\n"
            f"Stronger evidence can NEUTRALIZE weaker opposing evidence on the same point.\n"
            f"- A Supreme Court ruling on the exact issue OVERRIDES a lower court opinion or academic view\n"
            f"- Direct documentary proof (e.g., absence of required official certification) OVERRIDES circumstantial evidence (e.g., phone authentication records)\n"
            f"- Failure to satisfy a mandatory legal requirement OVERRIDES arguments about supplementary factors\n"
            f"- When you have a strong piece of evidence, explicitly argue that it neutralizes/weakens the opponent's related but weaker evidence\n\n"
            f"Focus your argument on the TOP 2-3 highest-impact evidence items.\n"
            f"Do NOT spread attention equally across all evidence — concentrate on what DECISIVELY wins the point.\n"
            f"If one strong evidence can neutralize multiple weaker opposing arguments, lead with it.\n\n"
            f"Contribute your analysis in 2-3 sentences. Focus on:\n"
            f"- What evidence supports OUR position from YOUR angle?\n"
            f"- What are the weaknesses in the OPPONENT'S argument that YOUR perspective reveals?\n"
            f"- What unique insight does YOUR angle bring that others might miss?\n\n"
            f"You MUST ground your discussion in the SPECIFIC evidence found above.\n\n"
            f"## Legal Authority Usage (CRITICAL)\n"
            f"Your team found legal authorities during research (shown in search results above).\n"
            f"You MUST actively USE them in your discussion — do not ignore search results.\n"
            f"For each search result:\n"
            f"1. Identify which part of your team's argument it supports\n"
            f"2. Cite it with the proper tag and explain the connection\n"
            f"3. If a result is NOT useful, explicitly state why (wrong direction, irrelevant facts)\n"
            f"Do NOT leave search results unaddressed.\n\n"
            f"{_disc_citation_rule}"
            f"When contributing:\n"
            f"- When citing a precedent, use [판례: ID] tag, then state what the court decided and how it supports your argument.\n"
            f"  If the holding CONTRADICTS your team's position, use [BLACKLIST: case_number] — do NOT cite it as support.\n"
            f"- When citing a statute, use [법령: ID] tag with the specific article number.\n"
            f"- Citation tags: [판례: ID], [법령: ID], [헌재: ID], [행심: ID], [문서: ID] ONLY. Other formats (e.g. [case_citation:], [법원:]) are IGNORED.\n"
            f"- Do NOT make generic claims without citing specific evidence.\n"
            f"- If evidence is insufficient, suggest what additional searches should be done.\n\n"
            f"## Argumentation Strategy Discussion\n"
            f"You MUST also discuss HOW to argue, not just WHAT to argue:\n"
            f"1. ARGUMENT ORDERING: Which argument should we lead with? Which is strongest?\n"
            f"2. BURDEN OF PROOF: Who has the burden of proof? How to use it strategically?\n"
            f"3. OPPONENT'S WEAKNESSES: Look at the opponent's latest statement carefully:\n"
            f"   - Did they contradict themselves? (Quote both statements if so)\n"
            f"   - Did they cite a source that actually supports OUR position?\n"
            f"   - Did they admit any facts we can use?\n"
            f"   - Did they use logical fallacies? (circular reasoning, false equivalence, straw man)\n"
            f"4. LOGICAL LOCK: Can we construct an argument where ANY response weakens their position?\n"
            f"5. COUNTERARGUMENT PREPARATION: What will the opponent likely argue next?\n\n"
            + (
                f"## Teammate Interaction\n"
                f"You can directly address a teammate to ask for their expert opinion:\n"
                f"- Use [ASK: teammate_name] tag to request their input on a specific point.\n"
                f"- Example: [ASK: Kim] Please review this precedent from your expertise in damages calculation.\n"
                f"- Example: [ASK: Park] Could you evaluate whether this statute applies to our case?\n"
                f"- When you are addressed by name or via [ASK:], you SHOULD respond to the specific request.\n"
                f"- Your teammates: {', '.join(m.get('name', '') for m in members)}\n\n"
                if not pending_reviews else ""
            )
            + f"## PASS Option\n"
            f"You may respond with [PASS] ONLY if ALL of the following conditions are met:\n"
            f"1. You have already contributed at least 2 substantive (non-PASS) responses in this discussion\n"
            f"2. You fully agree with the current direction\n"
            f"3. You have NO new insights, evidence, or counter-arguments from YOUR argument angle\n"
            f"4. There are no unreviewed precedents or unaddressed weaknesses\n\n"
            f"DO NOT [PASS] if:\n"
            f"- You haven't contributed your unique analysis from your assigned argument angle yet\n"
            f"- This is your first or second speaking turn\n"
            f"- There are pending precedent reviews\n"
            f"- The opponent raised points that haven't been addressed\n\n"
            f"When you believe the team has reached sufficient agreement on arguments "
            f"and evidence strategy, include [CONSENSUS] at the END of your message.\n\n"
            f"Rules for consensus:\n"
            f"1. ALL team members must agree — the discussion only ends when everyone signals [CONSENSUS]\n"
            f"2. All pending precedent reviews must be resolved before [CONSENSUS]\n"
            f"3. All team members must have contributed at least 1 substantive (non-PASS) response before consensus is valid\n"
            f"4. After all members have spoken at least twice, majority [CONSENSUS] is sufficient\n"
            f"4. The main arguments and supporting evidence must be clearly identified\n"
            f"5. If you include review tags ([ACCEPT]/[REJECT]/[BLACKLIST]) in your message,\n"
            f"   do not include [CONSENSUS] in the same message — review first, then discuss further.\n\n"
            f"## IMPORTANT: Citation Eligibility Rule\n"
            f"The representative statement can ONLY cite precedents and statutes\n"
            f"that were mentioned during this team discussion.\n"
            f"If you want a search result to be available for the final statement,\n"
            f"you MUST mention it using [판례: ID] or [법령: ID] format.\n"
            f"Any search result NOT mentioned in discussion will be EXCLUDED from the statement."
        )

        # Inject independent analysis if available (from multi-agent Phase 2)
        _own_analysis = speaker.get("_independent_analysis", "")
        if _own_analysis:
            system_prompt += (
                f"\n\n## YOUR INDEPENDENT ANALYSIS (from before discussion)\n"
                f"{_own_analysis}\n"
                f"Use this as YOUR foundation. You may refine your position based on "
                f"others' arguments, but maintain your unique perspective."
            )

        system_prompt += get_language_instruction(language)

        _user_msg = f"Share your thoughts on our legal strategy. (Message {msg_idx+1}/{max_messages})"
        if pending_reviews:
            _pending_cn_list = ", ".join(pr.get("case_number", "") for pr in pending_reviews if pr.get("case_number"))
            _user_msg = (
                f"⚠ MANDATORY REVIEW — You MUST review these precedents FIRST: {_pending_cn_list}\n"
                f"Start your response with [ACCEPT/REJECT/BLACKLIST/REVIEW_MORE] tags for EACH one.\n"
                f"Only after completing ALL reviews, add your strategic analysis.\n"
                f"(Message {msg_idx+1}/{max_messages})"
            )
        if _hallucination_warning:
            _user_msg += _hallucination_warning
            _hallucination_warning = ""  # deliver once, then clear
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _user_msg},
        ]

        # Use tool calling if precedent vector DB is available
        _agent_model = speaker.get("llm_override") or None
        if case_id_map and legal_api:
            try:
                tool_resp = await llm_client.achat_with_tools(
                    messages=messages,
                    tools=_DISCUSSION_TOOLS,
                    tool_choice="auto",
                    temperature=0.5,
                    max_tokens=4000,
                    model_override=_agent_model,
                )
                msg = tool_resp.choices[0].message
                tool_result_text = ""
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                            query = args.get("query", "")
                            case_number = args.get("case_number", "")
                            # Block resolved (blacklisted/accepted/rejected) cases
                            if case_number in resolved_cases:
                                available = [cn for cn in case_id_map if cn not in resolved_cases]
                                avail_str = ", ".join(available[:10]) if available else "none"
                                result = (
                                    f"[BLOCKED: {case_number} is already resolved. "
                                    f"Use a DIFFERENT case_number. "
                                    f"Available: {avail_str}]"
                                )
                            else:
                                result = await _on_demand_precedent_search(
                                    case_number=case_number,
                                    query=query,
                                    case_id_map=case_id_map,
                                    legal_api=legal_api,
                                )
                            tool_result_text += result
                            # Store for review by next agent (skip if already resolved)
                            if case_number and "[No " not in result and "[Failed" not in result and case_number not in resolved_cases:
                                all_tool_results[case_number] = result
                                # Manage pending_reviews: max 2 (new + 1 previous)
                                pending_reviews.append({
                                    "case_number": case_number,
                                    "result": result,
                                    "found_by": agent_name,
                                })
                                # Keep at most 2 pending reviews
                                while len(pending_reviews) > 2:
                                    review_queue.append(pending_reviews.pop(0))
                                logger.info("[discussion] Added %s to pending_reviews (by %s)", case_number, agent_name)
                        except Exception as _tc_exc:
                            logger.debug("[discussion] Tool call failed: %s", _tc_exc)
                response = (msg.content or "") + tool_result_text
            except Exception as _tool_exc:
                logger.warning("[discussion] Tool calling failed, falling back to plain chat: %s", _tool_exc)
                response = await llm_client.achat(messages, temperature=0.5, max_tokens=4000, model_override=_agent_model)
        else:
            response = await llm_client.achat(messages, temperature=0.5, max_tokens=4000, model_override=_agent_model)

        # PASS detection — agent declines to speak this turn
        _is_pass = "[PASS]" in response
        if _is_pass:
            discussion_log.append({
                "speaker": agent_name,
                "agent_id": agent_id,
                "content": "[PASS]",
                "turn": msg_idx,
                "is_pass": True,
            })
            last_speaker_id = agent_id
            logger.info("[discussion] %s passed at message %d/%d", agent_name, msg_idx + 1, max_messages)
            # Check if ALL agents passed consecutively → end discussion
            _recent = discussion_log[-len(members):] if len(discussion_log) >= len(members) else []
            if len(_recent) == len(members) and all(r.get("is_pass") for r in _recent):
                logger.info("[discussion] All agents passed — ending discussion")
                break
            continue

        discussion_log.append({
            "speaker": agent_name,
            "agent_id": agent_id,
            "content": response,
            "turn": msg_idx,
        })
        last_speaker_id = agent_id

        # Broadcast discussion message via WebSocket (no-op if no clients)
        if debate_id:
            try:
                from app.api.ws import broadcast as _ws_broadcast
                await _ws_broadcast(debate_id, {
                    "type": "discussion_message",
                    "team": team_id,
                    "speaker": agent_name,
                    "agent_id": agent_id,
                    "content": response[:500],
                    "turn": msg_idx,
                    "total": max_messages,
                })
            except Exception:
                pass

        # Save progress every 3 messages
        if debate_id and msg_idx % 3 == 0:
            try:
                from app.api.debate import DebateStore
                current = await DebateStore.aload(debate_id)
                current["discussion_progress"] = msg_idx + 1
                current["discussion_total"] = max_messages
                await DebateStore.asave(debate_id, current)
            except Exception:
                pass

        # Evidence sharing: any [판례:] or [법령:] citation makes that evidence shared
        for _share_m in _re.findall(r'\[판례:\s*([^\]]+)\]', response):
            shared_case_numbers.add(_share_m.strip())
        for _share_m in _re.findall(r'\[법령:\s*([^\]]+)\]', response):
            shared_case_numbers.add(_share_m.strip())

        # Auto-detect [판례: ...] chip tags in response and fetch full text
        if case_id_map and legal_api:
            _cited_tags = _re.findall(r'\[판례:\s*([^\]]+)\]', response)
            _cited_cases: set[str] = set()
            for _tag_val in _cited_tags:
                _tag_clean = _tag_val.strip()
                # Exact match first
                if _tag_clean in case_id_map:
                    _cited_cases.add(_tag_clean)
                    continue
                # Normalized match: extract digits and compare
                _tag_digits = _re.sub(r"[^0-9]", "", _tag_clean)
                if len(_tag_digits) >= 4:
                    for _map_key in case_id_map:
                        _key_digits = _re.sub(r"[^0-9]", "", _map_key)
                        if _tag_digits == _key_digits:
                            _cited_cases.add(_map_key)
                            break
            for cn in _cited_cases:
                if cn in resolved_cases:
                    continue
                # 1) review_queue에 있으면 → pending_reviews 맨 앞으로 승격
                _promoted = False
                for qi, qitem in enumerate(review_queue):
                    if qitem.get("case_number") == cn:
                        _item = review_queue.pop(qi)
                        pending_reviews.insert(0, _item)
                        while len(pending_reviews) > 2:
                            review_queue.append(pending_reviews.pop(-1))
                        logger.info("[discussion] Promoted %s from queue (cited by %s)", cn, agent_name)
                        _promoted = True
                        break
                if _promoted:
                    continue
                # 2) pending_reviews에 이미 있으면 → skip
                if any(pr.get("case_number") == cn for pr in pending_reviews):
                    continue
                # 3) all_tool_results에만 있고 큐/pending 어디에도 없음 → 재등록
                if cn in all_tool_results:
                    pending_reviews.insert(0, {
                        "case_number": cn, "result": all_tool_results[cn], "found_by": agent_name,
                    })
                    while len(pending_reviews) > 2:
                        review_queue.append(pending_reviews.pop(-1))
                    logger.info("[discussion] Re-added %s to pending (cited by %s)", cn, agent_name)
                    continue
                # 4) 완전히 새로운 판례 → 검색 후 추가
                try:
                    _search_query = f"{team_opinion[:200]} {topic}" if team_opinion else (topic or "법률 쟁점")
                    _auto_result = await _on_demand_precedent_search(
                        cn, _search_query, case_id_map, legal_api,
                    )
                    all_tool_results[cn] = _auto_result
                    pending_reviews.insert(0, {
                        "case_number": cn, "result": _auto_result, "found_by": agent_name,
                    })
                    while len(pending_reviews) > 2:
                        review_queue.append(pending_reviews.pop(-1))
                    logger.info("[discussion] Auto-searched %s (cited by %s)", cn, agent_name)
                except Exception as _auto_exc:
                    logger.debug("[discussion] Auto-search failed for %s: %s", cn, _auto_exc)

        # Parse review tags from response (skip already resolved)
        for _m in _re.findall(r'\[ACCEPT:\s*([^\]]+)\]', response):
            cn = _m.strip()
            if cn in resolved_cases:
                continue
            accept_votes.setdefault(cn, set()).add(agent_id)
            _threshold = max(1, math.ceil(len(members) * accept_ratio))
            if len(accept_votes.get(cn, set())) >= _threshold:
                # accept_ratio (ceil) accepted → remove from pending and mark resolved
                pending_reviews[:] = [pr for pr in pending_reviews if pr.get("case_number") != cn]
                resolved_cases.add(cn)
                # 대기열에서 보충
                while len(pending_reviews) < 2 and review_queue:
                    _next = review_queue.pop(0)
                    if _next.get("case_number") not in resolved_cases:
                        pending_reviews.append(_next)
                        logger.info("[discussion] Queue replenish: %s promoted to pending", _next.get("case_number"))
                logger.info("[discussion] Precedent %s ACCEPTED (%d/%d votes)", cn, len(accept_votes[cn]), len(members))

        for _m in _re.findall(r'\[REJECT:\s*([^\]]+)\]', response):
            cn = _m.strip()
            if cn in resolved_cases:
                continue
            reject_votes.setdefault(cn, set()).add(agent_id)
            pending_reviews[:] = [pr for pr in pending_reviews if pr.get("case_number") != cn]
            resolved_cases.add(cn)
            # 대기열에서 보충
            while len(pending_reviews) < 2 and review_queue:
                _next = review_queue.pop(0)
                if _next.get("case_number") not in resolved_cases:
                    pending_reviews.append(_next)
                    logger.info("[discussion] Queue replenish: %s promoted to pending", _next.get("case_number"))
            logger.info("[discussion] Precedent %s REJECTED by %s", cn, agent_name)

        for _m in _re.findall(r'\[BLACKLIST:\s*([^\]]+)\]', response):
            cn = _m.strip()
            if cn in resolved_cases:
                continue
            pending_reviews[:] = [pr for pr in pending_reviews if pr.get("case_number") != cn]
            resolved_cases.add(cn)
            # 대기열에서 보충
            while len(pending_reviews) < 2 and review_queue:
                _next = review_queue.pop(0)
                if _next.get("case_number") not in resolved_cases:
                    pending_reviews.append(_next)
                    logger.info("[discussion] Queue replenish: %s promoted to pending", _next.get("case_number"))
            logger.info("[discussion] Precedent %s BLACKLISTED by %s", cn, agent_name)

        for _m in _re.findall(r'\[REVIEW_MORE:\s*([^\]]+)\]', response):
            cn = _m.strip()
            # Canonicalize: if a variant of this case is already tracked
            # (e.g. "대법원 2020다12345" vs "2020다12345"), reuse the existing key
            # so distinct-agent counting doesn't get split by format differences.
            _cn_digits = _re.sub(r"[^0-9]", "", cn)
            if _cn_digits and len(_cn_digits) >= 4:
                _canonical = None
                for _existing in list(review_more_agents.keys()):
                    if _re.sub(r"[^0-9]", "", _existing) == _cn_digits:
                        _canonical = _existing
                        break
                if _canonical is None:
                    for _existing in list(accept_votes.keys()):
                        if _re.sub(r"[^0-9]", "", _existing) == _cn_digits:
                            _canonical = _existing
                            break
                if _canonical is None:
                    for _existing in list(resolved_cases):
                        if _re.sub(r"[^0-9]", "", _existing) == _cn_digits:
                            _canonical = _existing
                            break
                if _canonical is not None:
                    cn = _canonical
            if cn in resolved_cases:
                continue

            # 서로 다른 에이전트 집합에 현재 agent_id 추가 (라운드 경계에서도 유지)
            _rm_agent_set = review_more_agents.setdefault(cn, set())
            _rm_agent_set.add(agent_id)

            # --- Fix 3: REVIEW_MORE 시 누적 query 기반 lazy RAG fetch ---
            # agent가 `[REVIEW_MORE]`로 "본문 확인 필요" 의사를 표시하면
            # `search_precedent_detail` tool을 호출하지 않았더라도 system이 대신
            # `_on_demand_precedent_search`를 호출해 판결문 본문을 RAG로 조회한다.
            # query는 세 source를 합성:
            #   1) 현재 speaker의 `_independent_analysis` (Phase 2 법리 관점)
            #   2) `review_history[cn]` (이전 agent들의 REVIEW_MORE 서술 누적)
            #   3) 현재 agent의 response (이번 턴에서 표현한 의심)
            # VectorDB dedup 덕분에 OpenAPI 호출은 첫 REVIEW_MORE 때 1회만,
            # 이후는 가벼운 embedding + RAG search로 query에 따라 다른 chunks 반환.
            # 이렇게 해야 auto-ACCEPT가 발동되더라도 downstream(_produce_statement)이
            # 참고할 수 있는 실체적 본문이 남아 hallucination을 방지한다.
            if case_id_map is not None and legal_api is not None:
                try:
                    _query_parts: list[str] = []
                    _own_analysis_text = str(
                        speaker.get("_independent_analysis", "") or ""
                    ).strip()
                    if _own_analysis_text:
                        _query_parts.append(_own_analysis_text[:500])
                    else:
                        _fb = f"{(team_opinion or '')[:300]} {topic or ''}".strip()
                        if _fb:
                            _query_parts.append(_fb)
                    for _prev in review_history.get(cn, []):
                        _query_parts.append(str(_prev)[:300])
                    _query_parts.append(str(response or "")[:400])
                    _rm_query = "\n".join(p for p in _query_parts if p).strip() or "법률 쟁점"

                    _rm_fetch_result = await _on_demand_precedent_search(
                        cn, _rm_query, case_id_map, legal_api,
                    )
                    if _rm_fetch_result and not _rm_fetch_result.startswith("["):
                        all_tool_results[cn] = _rm_fetch_result
                        # 이미 pending/queue에 있는 엔트리의 result 덮어쓰기
                        for _pr in pending_reviews:
                            if _pr.get("case_number") == cn:
                                _pr["result"] = _rm_fetch_result
                        for _rq in review_queue:
                            if _rq.get("case_number") == cn:
                                _rq["result"] = _rm_fetch_result
                        # search_results 원본에도 back-fill →
                        # _produce_statement 의 effective_body (수정 1) 가 이 값을 사용
                        for _sr in search_results:
                            if (
                                isinstance(_sr, dict)
                                and str(_sr.get("case_number", "")) == cn
                            ):
                                _sr["full_text_excerpt"] = _rm_fetch_result
                                break
                        logger.info(
                            "[discussion] REVIEW_MORE lazy-RAG for %s (by %s, query_parts=%d, result_len=%d)",
                            cn, agent_name, len(_query_parts), len(_rm_fetch_result),
                        )
                    else:
                        logger.info(
                            "[discussion] REVIEW_MORE lazy-RAG stub for %s: %s",
                            cn, (_rm_fetch_result or "")[:80],
                        )
                except Exception as _rm_auto_exc:
                    logger.debug(
                        "[discussion] REVIEW_MORE lazy-RAG failed for %s: %s",
                        cn, _rm_auto_exc,
                    )
            # --- /Fix 3 ---

            # (A) ceil(members*review_more_ratio) 이상의 서로 다른 에이전트가 REVIEW_MORE 요청 → auto-ACCEPT
            _rm_distinct_threshold = max(1, math.ceil(len(members) * review_more_ratio))
            if len(_rm_agent_set) >= _rm_distinct_threshold:
                logger.info(
                    "[discussion] %s: REVIEW_MORE reached by %d/%d distinct agents → auto-ACCEPT",
                    cn, len(_rm_agent_set), _rm_distinct_threshold,
                )
                pending_reviews[:] = [pr for pr in pending_reviews if pr.get("case_number") != cn]
                # 모든 REVIEW_MORE 요청자를 accept_votes에 기록 → 임계치 자연 통과
                accept_votes.setdefault(cn, set()).update(_rm_agent_set)
                resolved_cases.add(cn)
                while len(pending_reviews) < 2 and review_queue:
                    _next = review_queue.pop(0)
                    if _next.get("case_number") not in resolved_cases:
                        pending_reviews.append(_next)
                        logger.info("[discussion] Queue replenish: %s promoted to pending", _next.get("case_number"))
                continue

            # (B) 기존 라운드 한도 로직 — next_round_priority로 이관
            _case_rm_count = review_more_counts.get(cn, 0)
            if _case_rm_count >= max_review_more:
                logger.info("[discussion] REVIEW_MORE limit for %s reached, moving to next round priority", cn)
                # pending에서 제거 → 다음 라운드 우선 논의로 이관
                pending_reviews[:] = [pr for pr in pending_reviews if pr.get("case_number") != cn]
                next_round_priority.append({
                    "case_number": cn,
                    "result": all_tool_results.get(cn, ""),
                    "found_by": agent_name,
                    "review_history": review_history.get(cn, []),
                })
                # queue에서 보충
                while len(pending_reviews) < 2 and review_queue:
                    _next = review_queue.pop(0)
                    if _next.get("case_number") not in resolved_cases:
                        pending_reviews.append(_next)
                        logger.info("[discussion] Queue replenish: %s promoted to pending", _next.get("case_number"))
                continue

            # (C) 정상 REVIEW_MORE — 카운터 증가 + 재등록
            review_more_counts[cn] = _case_rm_count + 1
            # 리뷰 이력 기록
            _review_snippet = response[:500]
            review_history.setdefault(cn, []).append(f"{agent_name}: {_review_snippet}")
            _existing_result = all_tool_results.get(cn, "")
            pending_reviews.append({
                "case_number": cn, "result": _existing_result,
                "found_by": agent_name, "review_request": True,
            })
            while len(pending_reviews) > 2:
                review_queue.append(pending_reviews.pop(0))
            logger.info(
                "[discussion] %s requested REVIEW_MORE for %s (distinct=%d/%d, count=%d/%d)",
                agent_name, cn,
                len(_rm_agent_set), _rm_distinct_threshold,
                review_more_counts[cn], max_review_more,
            )

        # --- Hallucinated citation detection ---
        # Check if agent cited precedents not in the search results pool
        if case_id_map is not None:
            _all_cited = _re.findall(r'\[판례:\s*([^\]]+)\]', response)
            _known_pool = set(case_id_map.keys()) | set(all_tool_results.keys())
            _hallucinated: list[str] = []
            for _cite in _all_cited:
                _cn = _cite.strip()
                _found = _cn in _known_pool
                if not _found:
                    _cn_digits = _re.sub(r"[^0-9]", "", _cn)
                    if len(_cn_digits) >= 4:
                        _found = any(
                            _cn_digits == _re.sub(r"[^0-9]", "", k)
                            for k in _known_pool
                        )
                if not _found:
                    _hallucinated.append(_cn)
            if _hallucinated:
                _warn_cases = ", ".join(_hallucinated)
                _hallucination_warning = (
                    f"\n⚠ WARNING: {agent_name} cited [{_warn_cases}] but these cases do NOT exist "
                    f"in the search results. These are likely fabricated (hallucinated) citations. "
                    f"You MUST immediately [REJECT: case_number] for each of these cases.\n"
                )
                logger.warning(
                    "[discussion] Hallucinated citation by %s: %s",
                    agent_name, _warn_cases,
                )

        # Track consensus — ignore if the same response contains review tags
        has_consensus = "[CONSENSUS]" in response
        _has_review_tags = any(
            f"[{_rtag}:" in response
            for _rtag in ("ACCEPT", "REJECT", "BLACKLIST", "REVIEW_MORE")
        )
        if has_consensus:
            if _has_review_tags:
                logger.info("[discussion] %s voted [CONSENSUS] alongside review tags — vote ignored (review ≠ end of discussion)", agent_name)
            elif pending_reviews:
                logger.info("[discussion] %s voted [CONSENSUS] but %d pending review(s) — vote ignored", agent_name, len(pending_reviews))
            else:
                consensus_votes[agent_id] = True

        # Check consensus conditions
        all_spoke = len({d["agent_id"] for d in discussion_log}) >= len(members)

        # Round boundary detection and pending_reviews cleanup
        member_speak_counts = {
            m.get("agent_id"): sum(1 for d in discussion_log if d["agent_id"] == m.get("agent_id"))
            for m in members
        }
        _min_speaks = min(member_speak_counts.values()) if member_speak_counts else 0
        # Round boundary: clear current reviews and load priority items from REVIEW_MORE limit
        if _min_speaks > 0 and all(c == _min_speaks for c in member_speak_counts.values()):
            if pending_reviews or review_queue:
                # 부분 accept된 항목 보존 (투표가 1개 이상 있지만 threshold 미달)
                _partial_accept = [
                    pr for pr in pending_reviews
                    if pr.get("case_number") in accept_votes
                    and pr.get("case_number") not in resolved_cases
                ]
                logger.info("[discussion] Round %d complete, clearing reviews. Preserving %d partial-accept items.",
                            _min_speaks, len(_partial_accept))
                pending_reviews.clear()
                review_queue.clear()
                # 부분 accept 항목을 다시 pending에 로드
                for item in _partial_accept:
                    if len(pending_reviews) < 2:
                        pending_reviews.append(item)
                    else:
                        review_queue.append(item)
            # Load next round priority items (REVIEW_MORE 제한 도달 판례)
            if next_round_priority:
                for item in next_round_priority:
                    if item.get("case_number") not in resolved_cases:
                        if len(pending_reviews) < 2:
                            pending_reviews.append(item)
                        else:
                            review_queue.append(item)
                logger.info("[discussion] Round %d: loaded %d priority items for review", _min_speaks, len(next_round_priority))
                next_round_priority.clear()

        all_spoke_twice = all(c >= 2 for c in member_speak_counts.values()) if member_speak_counts else False

        if all_spoke_twice:
            consensus_count = sum(1 for v in consensus_votes.values() if v)
            if consensus_count > len(members) // 2:
                logger.info("[discussion] Majority consensus at message %d/%d (%d/%d agreed)",
                            msg_idx+1, max_messages, consensus_count, len(members))
                break

        # Full consensus: all agreed
        if all_spoke and all(consensus_votes.get(m.get("agent_id"), False) for m in members):
            logger.info("[discussion] Full consensus at message %d/%d", msg_idx+1, max_messages)
            break

        # Check for pause/stop and user extension request
        if debate_id and all_spoke and msg_idx >= len(members):
            try:
                from app.api.debate import DebateStore
                state = await DebateStore.aload(debate_id)

                # Pause/stop check
                if state.get("status") in ("paused", "stopped"):
                    logger.info("[discussion] %s detected — stopping at message %d/%d",
                                state["status"], msg_idx + 1, max_messages)
                    break

            except Exception:
                pass

    logger.info("[discussion] Completed with %d messages", len(discussion_log))

    # Build set of accepted case numbers (accept_ratio team agreement, ceil)
    _accept_threshold = max(1, math.ceil(len(members) * accept_ratio))
    accepted_cases: set[str] = {
        cn for cn, voters in accept_votes.items()
        if len(voters) >= _accept_threshold
    }
    if accepted_cases:
        logger.info("[discussion] Accepted precedents: %s", accepted_cases)

    return discussion_log, accepted_cases


# ------------------------------------------------------------------
# Statement production: select speaker and generate final statement
# ------------------------------------------------------------------

async def _produce_statement(
    state: TeamState,
    search_results: list[dict],
    discussion_log: list[dict],
    agreed_strategy: str,
    llm_client: LLMClient,
    language: str = "ko",
    team_display_name: str = "",
    blacklisted_items: list[str] | None = None,
    discussed_cases: set[str] | None = None,
    accepted_cases: set[str] | None = None,
) -> tuple[str, str, list[dict]]:
    """
    Select the representative speaker and produce the final debate statement.

    The speaker is selected from members with "primary" priority in their
    role assignment, or falls back to the first member.

    Args:
        state: Current TeamState.
        search_results: All collected search results.
        discussion_log: Internal discussion log.
        agreed_strategy: The consensus strategy from discussion.
        llm_client: LLM client for statement generation.

    Returns:
        Tuple of (selected_speaker_id, statement_text, evidence_list).
    """
    members = state.get("members", [])
    assignments = state.get("role_assignments", [])

    # Select speaker: prefer the agent with "primary" priority
    selected_speaker_id = ""
    selected_member = None
    for assignment in assignments:
        if assignment.get("priority") == "primary":
            selected_speaker_id = assignment.get("agent_id", "")
            break

    if selected_speaker_id:
        for m in members:
            if m.get("agent_id") == selected_speaker_id:
                selected_member = m
                break

    # Fallback to first member
    if not selected_member and members:
        selected_member = members[0]
        selected_speaker_id = selected_member.get("agent_id", "agent_0")

    if not selected_member:
        return "", "(No team members available to produce a statement)", []

    agent_name = selected_member.get("name", "Speaker")

    # Build evidence list from search results
    evidence_list: list[dict] = []
    evidence_citations: list[str] = []

    # Build mapping tables from search results for reference resolution
    _law_name_map: dict[str, dict] = {}   # {법령명: {seq_id, url}}
    _case_number_map: dict[str, dict] = {} # {사건번호: {seq_id, url}}
    for _r in search_results:
        if not isinstance(_r, dict):
            continue
        _entry = {
            "seq_id": _r.get("법령일련번호", "") or _r.get("판례일련번호", "") or _r.get("id", "") or _r.get("_item_id", _r.get("item_id", "")),
            "url": _r.get("url", ""),
        }
        _ln = _r.get("법령명한글", "") or _r.get("law_name", "")
        _cn = _r.get("사건번호", "") or _r.get("case_number", "")
        if _ln:
            _law_name_map[_ln] = _entry
        if _cn:
            _case_number_map[_cn] = _entry

    # Defensive normalizer for source type. Some saved/legacy search_results
    # carry `source_type: None` and `type: None` (or `type: "vector"/"graph"`
    # which are storage backends, not legal categories). The only reliable
    # signal in those cases is `source`, which may hold "court_precedent",
    # "prec", "legal_statute", etc. Without this helper, precedent entries
    # with null source_type get labeled "- [None] ..." and the evidence mode
    # detector below fails to count them, incorrectly switching the LLM into
    # statute-only mode.
    def _normalize_src_type(_r: dict) -> str:
        _st = _r.get("source_type")
        if _st:
            return str(_st)
        _t = _r.get("type")
        if _t and _t not in ("vector", "graph"):
            return str(_t)
        _src = str(_r.get("source", "") or "")
        return _src if _src and _src != "unknown" else ""

    # Build available sources whitelist for anti-hallucination
    # Only include items that were discussed/reviewed during team discussion
    _available_sources: list[str] = []
    _seen_labels: set[str] = set()
    _excluded_undiscussed = 0
    for _r in search_results:
        if not isinstance(_r, dict):
            continue
        _src_type = _normalize_src_type(_r)
        _case_num = _r.get("case_number", _r.get("사건번호", ""))
        _law_nm = _r.get("law_name", _r.get("법령명한글", ""))
        _title = _r.get("title", "")
        _doc_name = ""
        _meta = _r.get("metadata", {})
        if isinstance(_meta, dict):
            _doc_name = _meta.get("doc_name", "")
        _label = _case_num or _law_nm or _title or _doc_name or ""
        # Precedents require ACCEPT vote (60%+) to be whitelisted;
        # statutes pass through without voting.
        _is_precedent = _src_type in ("court_precedent", "prec",
                                       "constitutional_decision",
                                       "admin_tribunal_decision")
        if accepted_cases is not None and _case_num and _is_precedent:
            _in_accepted = any(
                _case_num in ac or ac in _case_num
                for ac in accepted_cases
            )
            if not _in_accepted:
                _excluded_undiscussed += 1
                continue
        if _label and _label not in _seen_labels:
            _seen_labels.add(_label)
            # Keep the label prefix well-formed even when normalization
            # returns an empty string (e.g., totally unknown source).
            _label_st = _src_type or "unknown"
            _available_sources.append(f"- [{_label_st}] {_label}")
    if _excluded_undiscussed:
        logger.info("[evidence] Excluded %d undiscussed items from statement whitelist", _excluded_undiscussed)

    _sources_text = "\n".join(_available_sources) if _available_sources else (
        "(No search results found for this round. "
        "See CITATION REQUIREMENT section above for the required fallback protocol.)"
    )

    # --- Evidence mode detection for citation requirement branching ---
    _precedent_prefixes = (
        "- [court_precedent]",
        "- [constitutional_decision]",
        "- [admin_tribunal_decision]",
        "- [prec]",
    )
    _statute_prefixes = ("- [legal_statute]",)
    _precedent_count = sum(
        1 for s in _available_sources if s.startswith(_precedent_prefixes)
    )
    _statute_count = sum(
        1 for s in _available_sources if s.startswith(_statute_prefixes)
    )
    if _precedent_count == 0 and _statute_count == 0:
        _evidence_mode = "none"
    elif _precedent_count == 0:
        _evidence_mode = "statute_only"
    else:
        _evidence_mode = "normal"
    logger.info(
        "[team_speak] evidence mode: %s (precedents=%d, statutes=%d)",
        _evidence_mode, _precedent_count, _statute_count,
    )

    if _evidence_mode == "normal":
        _citation_requirement_block = (
            "## MINIMUM CITATION REQUIREMENT\n"
            "The evidence list below contains usable precedents.\n"
            "Your statement MUST cite at least 1 precedent with [판례: ID] tag and verbatim quote.\n\n"
        )
    elif _evidence_mode == "statute_only":
        _citation_requirement_block = (
            "## NO PRECEDENT AVAILABLE — STATUTE-ONLY MODE\n"
            "No directly applicable court precedent was found in the legal database for this issue.\n"
            "You MUST follow this protocol:\n"
            "1. Explicit acknowledgment: Your statement MUST contain a single sentence that\n"
            "   explicitly states, in the debate's output language, that no directly applicable\n"
            "   Supreme Court or Constitutional Court precedent was identified for this issue.\n"
            "   Do not hide this fact. Transparency preserves credibility.\n"
            "2. Statute-only reasoning: Build your argument from the available statutes using\n"
            "   [법령: ID] tags. Quote the exact article text in 「」 brackets, then apply\n"
            "   textual, systematic, and teleological interpretation.\n"
            "3. FORBIDDEN in this mode:\n"
            "   - Inventing a case number. There is no precedent — do NOT make one up.\n"
            "   - Any phrase that refers to 'the Supreme Court held', 'case law provides', or\n"
            "     similar, WITHOUT a matching [판례: ID] tag. Such phrases will be\n"
            "     treated as fabricated citations.\n"
            "   - Using [판례: ...] tags at all unless a case number appears in the whitelist below.\n"
            "4. Allowed: General legal principles (good faith, proportionality, public welfare,\n"
            "   prohibition of excess, etc.) may be invoked WITHOUT a citation tag, provided\n"
            "   they are clearly labeled as a principle and not as a court ruling.\n\n"
        )
    else:  # "none"
        _citation_requirement_block = (
            "## NO LEGAL AUTHORITY AVAILABLE — PRINCIPLE-ONLY MODE\n"
            "Neither precedent nor statute was retrieved for this issue.\n"
            "You MUST follow this protocol:\n"
            "1. Explicit acknowledgment: Your statement MUST begin with a sentence, in the\n"
            "   debate's output language, stating that no directly applicable precedent or\n"
            "   explicit statute was identified for this issue and that the argument will\n"
            "   therefore proceed on the basis of general legal principles.\n"
            "2. Principle-based reasoning only: You may invoke general legal principles\n"
            "   (good faith, proportionality, public welfare, prohibition of excess, etc.).\n"
            "   Clearly label each as a principle, not as case law.\n"
            "3. NEVER emit [판례: ...], [법령: ...], [헌재: ...], [행심: ...], or [문서: ...]\n"
            "   in this mode. Any such tag will be flagged as [⚠ 미확인 인용].\n"
            "4. Do not pretend to know a ruling or statute that isn't in the whitelist.\n"
            "   The team's credibility depends on acknowledging the absence honestly.\n\n"
        )

    # Iterate the FULL search_results list (no top-N slice). The filters
    # below (blacklist / non-accepted precedent / empty / ghost) already
    # narrow the set down to usable entries, and ACCEPT-voted precedents
    # must be guaranteed to reach the evidence builder regardless of their
    # original position in search_results. Previously this was `[:8]`, which
    # silently dropped ACCEPT-voted precedents that happened to sit beyond
    # index 8 — most notably cases that accumulated REVIEW_MORE votes
    # through review_queue promotions (typically tail-positioned entries).
    for i, result in enumerate(search_results):
        # Skip blacklisted items (from discussion phase)
        if blacklisted_items:
            _bl_label = result.get("case_number", "") or result.get("law_name", "")
            if _bl_label and any(
                bl in _bl_label or _bl_label in bl
                for bl in blacklisted_items
            ):
                logger.info("[evidence] Skipping blacklisted: %s", _bl_label)
                continue
        # Skip precedents not accepted by team vote (statutes pass through)
        if accepted_cases is not None:
            _acc_cn = result.get("case_number", "") or result.get("사건번호", "")
            _acc_src = result.get("source_type", result.get("type", ""))
            _acc_is_prec = _acc_src in ("court_precedent", "prec",
                                         "constitutional_decision",
                                         "admin_tribunal_decision")
            if _acc_cn and _acc_is_prec:
                if not any(_acc_cn in ac or ac in _acc_cn for ac in accepted_cases):
                    logger.info("[evidence] Skipping non-accepted precedent: %s", _acc_cn)
                    continue
        # Skip empty results (API returned metadata only, no actual content)
        _content_check = str(result.get("content", "")).strip()
        _title_check = str(result.get("title", "")).strip()
        _has_id = bool(
            result.get("case_number") or result.get("law_name")
            or result.get("title") or result.get("사건번호")
            or result.get("법령명한글")
        )
        if not _content_check and not _has_id:
            logger.info("[evidence] Skipping empty result[%d]: keys=%s", i, list(result.keys())[:5])
            continue
        # Compute "effective body" across parallel fields that may hold the
        # real precedent text. Phase 3 deep-dive stores RAG originals in
        # `full_text_excerpt`, and teams may attach their own `_phase3_analysis`
        # summary — but the upstream `content` field often carries only the
        # case title. We therefore pick the first *substantive* candidate
        # (length > 40 AND different from the title). If every candidate is
        # thin, fall back to the longest one so ghost detection still has
        # something to measure.
        _full_excerpt = str(result.get("full_text_excerpt", "") or "").strip()
        _phase3_text = str(result.get("_phase3_analysis", "") or "").strip()
        _body_candidates = [_content_check, _full_excerpt, _phase3_text]
        _effective_body = ""
        for _cand in _body_candidates:
            if _cand and len(_cand) > 40 and _cand != _title_check:
                _effective_body = _cand
                break
        if not _effective_body:
            _effective_body = max(_body_candidates, key=len) if any(_body_candidates) else ""
        # Skip ghost results: effective body is just the title or too short for precedents.
        # NOTE: we check `_effective_body` (not `_content_check`) because the
        # real holding may live in `full_text_excerpt` even when `content`
        # degenerates to just the title.
        _is_precedent = "prec" in str(result.get("source", "")) or result.get("source_type") == "court_precedent"
        if _is_precedent and _effective_body and (
            _effective_body == _title_check or len(_effective_body) < 20
        ):
            logger.info("[evidence] Skipping ghost precedent result[%d]: %s (effective_body='%s')",
                        i, result.get("case_number", ""), _effective_body[:50])
            continue

        logger.debug(
            "[evidence] result[%d] keys=%s case=%s law=%s title=%s url=%s",
            i, list(result.keys())[:6],
            result.get("case_number", ""), result.get("law_name", ""),
            result.get("title", "")[:30], result.get("url", "")[:40],
        )
        source = result.get("source", "unknown")
        # Use the effective body (full_text_excerpt / _phase3_analysis) when
        # `content` is thin. The downstream holding citation block (line
        # ~2810) reads this `content` variable to build the [DECISION SUMMARY]
        # block, so promoting the effective body here is what gets real
        # precedent text into the LLM prompt.
        content = _effective_body or result.get("content", "")
        result_type = result.get("type", "vector")

        # Map search result type to evidence source_type
        source_type_map = {
            "vector": "uploaded_document",
            "graph": "graph_relation",
            "legal_statute": "legal_statute",
            "court_precedent": "court_precedent",
            "constitutional_decision": "constitutional_decision",
            "legal_interpretation": "legal_interpretation",
            "admin_tribunal_decision": "admin_tribunal_decision",
            "local_ordinance": "local_ordinance",
            "administrative_rule": "administrative_rule",
            "treaty": "treaty",
            "legal_term": "legal_term",
            "special_admin_decision": "special_admin_decision",
            "consulting_opinion": "consulting_opinion",
            "ministry_interpretation": "ministry_interpretation",
            "legal_table": "legal_table",
            "committee_decision": "committee_decision",
        }
        evidence_source = result.get("source", "")
        if evidence_source == "legal_statute":
            src_type = "legal_statute"
        elif evidence_source == "court_precedent":
            src_type = "court_precedent"
        else:
            src_type = source_type_map.get(result_type, "uploaded_document")

        # Names that should NOT be used as evidence IDs or labels
        _type_names = {"court_precedent", "legal_statute", "uploaded_document",
                       "user_injected", "unknown", "graph_relation",
                       "constitutional_decision", "admin_tribunal_decision",
                       "local_ordinance", "administrative_rule", "treaty",
                       "legal_term", "special_admin_decision", "consulting_opinion",
                       "ministry_interpretation", "legal_table", "committee_decision"}

        # Extract real identifiers from search result
        case_number = result.get("사건번호", "") or result.get("case_number", "") or result.get("case_id", "")
        law_name = result.get("법령명한글", "") or result.get("law_name", "")
        seq_id = result.get("법령일련번호", "") or result.get("판례일련번호", "") or result.get("id", "") or result.get("_item_id", result.get("item_id", ""))

        # Fallback for vector/graph search results that lack legal identifiers
        title = result.get("title", "")
        if not case_number and not law_name:
            source_field = result.get("source", "")
            meta = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
            doc_name = meta.get("doc_name", "") or meta.get("original_text", "")[:50]

            if doc_name and doc_name not in _type_names:
                law_name = doc_name
            elif source_field and source_field not in _type_names and not source_field.startswith("graph:"):
                law_name = source_field
            elif isinstance(source_field, str) and source_field.startswith("graph:"):
                label = meta.get("label", "")
                law_name = label or source_field.replace("graph:", "")
            elif not title:
                # Last resort: first meaningful line of content
                content_text = str(result.get("content", ""))
                first_line = content_text.split('\n')[0].strip()[:50]
                if first_line and first_line not in _type_names:
                    title = first_line

        # Generate proper URL to law.go.kr using mapping tables
        url = result.get("url", "") if isinstance(result, dict) else ""
        if not url:
            if seq_id:
                if src_type in ("court_precedent", "prec"):
                    url = f"https://www.law.go.kr/LSW/precInfoP.do?precSeq={seq_id}"
                elif src_type in ("legal_statute", "law"):
                    url = f"https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq={seq_id}"
                elif src_type in ("constitutional_decision", "const", "dethc"):
                    url = f"https://www.law.go.kr/LSW/detcInfoP.do?detcSeq={seq_id}"
            elif case_number:
                # Try mapping table for seq_id-based URL
                if case_number in _case_number_map and _case_number_map[case_number].get("seq_id"):
                    url = f"https://www.law.go.kr/LSW/precInfoP.do?precSeq={_case_number_map[case_number]['seq_id']}"
                else:
                    # Strip court name prefix if present (e.g. "대구고등법원 2025나10683" → "2025나10683")
                    import re as _re_cn
                    _cn_m = _re_cn.search(r'(\d{2,4}\s*[가-힣]+\s*\d+)', case_number)
                    _pure_cn = _cn_m.group(1).strip() if _cn_m else case_number
                    url = f"https://www.law.go.kr/precSc.do?tabMenuId=465&query={_pure_cn}"
            elif law_name:
                # Clean law name (remove parenthetical suffixes, article numbers)
                _clean = re.sub(r'\s*[\(\[（【].*$', '', law_name)
                _clean = re.sub(r'\s+제\d+조.*$', '', _clean).strip()
                # Try mapping table for seq_id-based URL
                matched_name = _clean
                for _mn, _me in _law_name_map.items():
                    if _clean in _mn or _mn in _clean:
                        matched_name = _mn
                        if _me.get("seq_id"):
                            url = f"https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq={_me['seq_id']}"
                        break
                if not url:
                    url = f"https://www.law.go.kr/법령/{matched_name}"

        # Use real identifier — NO UUID fallback
        # Priority: case_number > law_name > title > source_detail > content summary > "Evidence #N"
        if not title:
            title = result.get("title", "")
        raw_sd = result.get("source_detail", "")[:50]
        source_detail_fallback = "" if raw_sd in _type_names else raw_sd

        # Content-based label as last resort before "Evidence #N"
        content_label = ""
        if not (case_number or law_name or title or source_detail_fallback):
            content_text = str(result.get("content", ""))[:100]
            first_sentence = content_text.split('.')[0].strip()[:40]
            if first_sentence and first_sentence not in _type_names:
                content_label = first_sentence

        real_id = (case_number or law_name or title
                   or source_detail_fallback or content_label or f"Evidence #{i+1}")
        # Safety: ensure real_id is a clean string (not dict/list)
        if isinstance(real_id, (dict, list)):
            import hashlib as _hl
            real_id = _hl.md5(json.dumps(real_id, sort_keys=True, default=str).encode()).hexdigest()[:12]
        real_id = str(real_id).strip()
        if len(real_id) > 80:
            import hashlib as _hl
            real_id = _hl.md5(real_id.encode()).hexdigest()[:12]
        if not real_id or real_id in _type_names:
            real_id = f"Evidence #{i+1}"

        # Log to DB for debugging UUID issues
        try:
            from app.db.database import _connection as _db_conn
            if _db_conn:
                _is_uuid = 1 if _re.match(r'^[0-9a-f]{8}-', real_id) else 0
                await _db_conn.execute(
                    "INSERT INTO evidence_debug_log (debate_id, round, speaker, result_keys, case_number, law_name, title, source, real_id, is_uuid) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (state.get("debate_id", ""), state.get("round", 0), selected_speaker_id,
                     '|'.join(list(result.keys())[:8]),
                     case_number[:30], law_name[:30], title[:30], source[:20], real_id[:30], _is_uuid)
                )
                await _db_conn.commit()
        except Exception:
            pass

        # Build rich source_detail with actual law name / case number from API
        case_name = result.get("사건명", "") or result.get("case_name", "")
        title = result.get("title", "")
        court = result.get("court", "")
        date = result.get("date", "")

        if case_number and law_name:
            rich_detail = f"{law_name} ({case_number})"
        elif case_number:
            detail_parts = [p for p in [court, case_number, case_name, date] if p]
            rich_detail = ", ".join(detail_parts)
        elif law_name:
            detail_parts = [p for p in [law_name, date] if p]
            rich_detail = ", ".join(detail_parts)
        else:
            detail_parts = [p for p in [title, court, date] if p]
            fallback_source = source if source not in _type_names else ""
            rich_detail = ", ".join(detail_parts) if detail_parts else (fallback_source or f"Evidence #{i+1}")

        evidence_entry = {
            "evidence_id": real_id,
            "content": content,
            "source_type": src_type,
            "source_detail": rich_detail,
            "url": url,
            "submitted_by": state.get("team_id", "unknown"),
            "round": state.get("round", 0),
            "speaker": selected_speaker_id,
            "relevance": "",
        }

        evidence_list.append(evidence_entry)
        # Map source type to Korean citation format
        _cite_prefix_map = {
            "legal_statute": "법령",
            "court_precedent": "판례",
            "constitutional_decision": "헌재",
            "admin_tribunal_decision": "행심",
            "uploaded_document": "문서",
        }
        _cite_prefix = _cite_prefix_map.get(src_type, "판례")
        if src_type in ("court_precedent", "prec"):
            # Content now contains only 판결요지 (decision summary)
            _holding = content
            if "[판결요지]" in _holding:
                _holding = _holding.split("[판결요지]", 1)[1].strip()
            # Ghost check: if holding is just the title or < 20 chars, block citation
            if len(_holding.strip()) < 20 or _holding.strip() == title.strip():
                evidence_citations.append(
                    f"[{_cite_prefix}: {real_id}] ({rich_detail})\n"
                    f"  ⚠ NO DECISION SUMMARY AVAILABLE — DO NOT quote or cite this precedent.\n"
                    f"  The search result only contains the case title, not the actual holding."
                )
            else:
                evidence_citations.append(
                    f"[{_cite_prefix}: {real_id}] ({rich_detail})\n"
                    f"  [DECISION SUMMARY — you MUST quote from this text verbatim in 「」 when citing]:\n"
                    f"  {_holding}"
                )
        else:
            evidence_citations.append(
                f"[{_cite_prefix}: {real_id}] ({rich_detail}) {content}"
            )

    logger.info(
        "[evidence] built %d entries from %d search_results (accepted_cases=%d)",
        len(evidence_list), len(search_results),
        len(accepted_cases) if accepted_cases is not None else -1,
    )

    # Include extra evidence in citations
    extra_evidence = state.get("extra_evidence", [])
    for e in extra_evidence:
        eid = (
            e.get("사건번호", "") or e.get("case_number", "")
            or e.get("법령명한글", "") or e.get("law_name", "")
            or e.get("title", "") or e.get("source_detail", "")
            or str(uuid4())
        )
        evidence_list.append({
            "evidence_id": eid,
            "content": e.get("content", ""),
            "source_type": "user_injected",
            "source_detail": e.get("source_detail", "User intervention"),
            "submitted_by": state.get("team_id", "unknown"),
            "round": state.get("round", 0),
            "speaker": selected_speaker_id,
            "relevance": "",
        })
        evidence_citations.append(
            f"[문서: user_evidence_{eid}] {e.get('content', '')}"
        )

    # NOTE: discussion_summary deliberately omitted from the speaker's input.
    # Raw discussion log may contain unreviewed precedent mentions; passing it
    # to the speaker LLM is the primary leak path. The speaker now sees only
    # agreed_strategy (built from accepted_cases) + evidence_citations
    # (filtered to accepted_cases) + situation/opponent context.

    # Build the statement generation prompt
    system_msg = (
        f"You are {agent_name}, representing your debate team as the "
        f"official speaker.\n"
        f"Specialty: {selected_member.get('specialty', 'general')}.\n"
        f"Debate style: {selected_member.get('debate_style', 'balanced')}.\n"
        f"Personality: {selected_member.get('personality', 'professional')}.\n\n"
        f"Produce a formal debate statement for your team. "
        f"The statement should be well-structured, persuasive, and "
        f"directly address the opponent's arguments.\n\n"
        f"## Diversity Requirement\n"
        f"Your statement MUST integrate ALL distinct perspectives from your team's discussion.\n"
        f"Do NOT collapse them into a single argument. Structure your statement to present\n"
        f"MULTIPLE independent arguments, each from a different legal angle.\n"
        f"Present at least 3 distinct arguments, each grounded in different evidence.\n\n"
        f"## Citation Format\n"
        f"ONLY these citation tag formats are allowed:\n"
        f"  [판례: ID], [법령: ID], [헌재: ID], [행심: ID], [문서: ID]\n"
        f"Copy the EXACT tag shown in the evidence list below. Any other format will be IGNORED.\n\n"
        f"{_citation_requirement_block}"
        + (
            "## PRECEDENT CITATION RULE (MANDATORY — VIOLATION = INVALID STATEMENT)\n"
            "When citing ANY court precedent, you MUST follow this EXACT structure:\n"
            "1. Quote the court's actual holding VERBATIM from the evidence text (in 「」 brackets)\n"
            "2. State the RESULT clearly: who won, what the court decided (upheld/denied/dismissed/granted)\n"
            "   The RESULT is the most persuasive part of a precedent — it shows what a court actually did, not just what was argued.\n"
            "3. THEN connect it to the current case\n\n"
            "REQUIRED STRUCTURE for every precedent citation:\n"
            "  [판례: ID] 「verbatim quote from the court's holding text」 followed by the court's decision and your analysis connecting it to this case.\n"
            "  EVERY 「」 quote MUST be preceded by a [판례: ID] tag. A quote without a tag is INVALID.\n\n"
            "VIOLATION EXAMPLES (these will INVALIDATE your entire statement):\n"
            "  ❌ Saying 'the court held X' without a verbatim quote from the holding text\n"
            "  ❌ Paraphrasing or summarizing the holding instead of quoting it\n"
            "  ❌ Selectively quoting to reverse the meaning of the holding\n\n"
            "If the holding CONTRADICTS your position, you may:\n"
            "  - Distinguish it: 「verbatim quote」 → However, this case is distinguishable because [factual difference]\n"
            "  - Acknowledge and counter: 「verbatim quote」 → While this precedent favors the opponent, [counter-argument]\n"
            "  - But you MUST NEVER misrepresent what the court actually decided.\n\n"
            if _evidence_mode == "normal" else ""
        )
        +
        f"## Available Evidence (WHITELIST — STRICT)\n"
        f"You may ONLY cite sources from this list. Do NOT invent or fabricate any document, "
        f"case number, law name, or reference that is NOT in this list.\n"
        f"{_sources_text}\n\n"
        + (
            f"## BLACKLISTED EVIDENCE (DO NOT CITE THESE)\n"
            f"Your team identified these as IRRELEVANT during internal discussion.\n"
            f"You MUST NOT cite any of these items:\n"
            + "\n".join(f"- {item}" for item in blacklisted_items) + "\n\n"
            if blacklisted_items else ""
        )
        + f"CRITICAL RULES:\n"
        f"- Any [문서: ...] reference MUST match an actually uploaded document from this list.\n"
        f"- Any [판례: ...] MUST match a case number from this list.\n"
        f"- Any [법령: ...] MUST match a law name from this list.\n"
        f"- NEVER create fictional document names (e.g., '보증보험 계약서 사본', '내부 매뉴얼').\n"
        f"- If you need more evidence, state '추가 검색이 필요합니다' instead of fabricating.\n\n"
    )

    # Add opponent cited evidence to system prompt
    opp_summary = state.get("_opponent_cited_summary", "")
    if opp_summary:
        system_msg += (
            f"## Evidence Attribution\n"
            f"- YOUR TEAM evidence above: you MAY cite using [판례: ...] or [법령: ...] tags.\n"
            f"- OPPONENT evidence below: you may REFERENCE or REBUT, but do NOT re-cite as your own.\n\n"
            f"## Opponent's Cited Evidence (for reference/rebuttal ONLY)\n"
            f"{opp_summary}\n\n"
        )

    system_msg += (
        _build_qa_history_summary(state.get("judge_qa_log", []))
        + get_language_instruction(language)
    )

    citations_text = "\n".join(evidence_citations) if evidence_citations else "(no evidence available)"

    team_id = state.get("team_id", "unknown")
    _display = team_display_name or team_id
    situation = state.get("situation_brief", "")
    if team_id != "team_a":
        _situation_label = "Situation (opposing party's account — argue AGAINST this)"
    else:
        _situation_label = "Situation (your client's account — argue FOR this)"
    user_msg = (
        f"CRITICAL: You are producing a statement for {_display}.\n"
        f"{_situation_label}: {situation}\n\n"
        f"## YOUR CORE POSITION (ANCHOR — NEVER ABANDON)\n"
        f"{state.get('team_opinion', '')}\n\n"
        f"Your statement MUST argue FOR this position.\n"
        f"Your statement MUST argue AGAINST: {state.get('opponent_opinion', '')}\n"
        f"DO NOT produce a balanced or neutral statement. You are an advocate, not a judge.\n\n"
        f"Round: {state.get('round', 1)}\n\n"
        f"=== OPPONENT'S LATEST ARGUMENT (counter this) ===\n"
        f"{state.get('opponent_statement', '(none)')}\n"
        f"=== END ===\n\n"
        f"Team's agreed strategy:\n{agreed_strategy}\n\n"
        f"Available evidence with citation tags:\n{citations_text}\n\n"
        f"Produce the formal debate statement now. Use [판례: ID] "
        f"and [법령: ID] format when referencing evidence."
    )

    _stmt_model = selected_member.get("llm_override") or None if selected_member else None
    try:
        statement = await llm_client.achat(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.6,
            max_tokens=8000,
            model_override=_stmt_model,
        )
    except Exception as exc:
        logger.error("[team_speak] Statement generation error: %s", exc)
        statement = (
            f"(Error generating statement for {agent_name}. "
            f"Team strategy: {agreed_strategy})"
        )

    # Post-processing: verify all CITE tags reference actual retrieval + prior citations.
    # Uses normalized comparison (NFC + middle-dot unification + whitespace collapse)
    # so that minor spacing/punctuation variants don't cause false "미확인 인용" flags.
    _valid_norm: set[str] = set()
    # 1. Current team's search results
    for _sr in search_results:
        if isinstance(_sr, dict):
            for _key in ('case_number', 'case_id', 'law_name', 'title', 'evidence_id'):
                _val = _sr.get(_key, '')
                if _val and not (len(str(_val)) == 36 and '-' in str(_val)):
                    _valid_norm.add(_norm_cite(_val))
    # 2. All accumulated evidence from previous rounds (both teams)
    for _ev in state.get("all_evidences", []):
        if isinstance(_ev, dict):
            for _key in ('case_number', 'case_id', 'law_name', 'title',
                          'evidence_id', 'source_detail'):
                _val = _ev.get(_key, '')
                if _val and not (len(str(_val)) == 36 and '-' in str(_val)):
                    _valid_norm.add(_norm_cite(_val))

    _cite_extract = _re.compile(r'\[(판례|법령|헌재|행심):\s*([^\]]+)\]')

    # 3a. Citations from previous representative statements — cross-team SHARED.
    #     debate_log / debate_context is public and visible to both teams.
    for _entry in state.get("debate_context", state.get("debate_log", [])):
        if not isinstance(_entry, dict):
            continue
        _stmt_prev = _entry.get("statement", "")
        for _cm in _cite_extract.finditer(_stmt_prev):
            _valid_norm.add(_norm_cite(_cm.group(2)))

    # 3b. Citations from THIS TEAM's internal discussion ONLY — same-team scoped.
    #     TeamState structure ensures state.discussion_log / state.internal_discussion
    #     contain only the current team's internal messages (see main_graph.py:191).
    #     The opposing team's internal discussion is never injected into this subgraph,
    #     so cross-team leak is structurally impossible.
    _internal = state.get("discussion_log", state.get("internal_discussion", []))
    for _entry in _internal:
        if not isinstance(_entry, dict):
            continue
        _txt = _entry.get("content", "") or _entry.get("message", "") or _entry.get("statement", "")
        for _cm in _cite_extract.finditer(_txt):
            _valid_norm.add(_norm_cite(_cm.group(2)))

    logger.info(
        "[team_speak] citation whitelist built (retrieval + shared debate_log + own internal): %d ids",
        len(_valid_norm),
    )

    _cite_pat = _re.compile(r'\[(판례|법령|헌재|행심):\s*([^\]]+)\]')
    for _m in list(_cite_pat.finditer(statement)):
        _raw = _m.group(2).strip()
        if _norm_cite(_raw) not in _valid_norm:
            logger.warning(
                "[team_speak] Unverified citation flagged: '%s' (normalized '%s' not in %d valid ids)",
                _raw, _norm_cite(_raw), len(_valid_norm),
            )
            statement = statement.replace(
                _m.group(0),
                f"[⚠ 미확인 인용: {_raw}]",
            )

    # Detect and remove fictional/hypothetical case references
    _fake_pat = _re.compile(
        r'[^.]*(?:가상|hypothetical|fictional|예시|illustrative)\s*(?:의\s*)?(?:판례|사건|case)[^.]*\.?',
        _re.IGNORECASE,
    )
    for _fm in list(_fake_pat.finditer(statement)):
        logger.warning("[team_speak] Fictional case reference detected: '%s'", _fm.group(0)[:80])

    # Strip citations of blacklisted precedents from final statement
    if blacklisted_items:
        _bl_case_nums: set[str] = {bl.strip() for bl in blacklisted_items if bl.strip()}
        _cite_check_pat = _re.compile(r'\[판례:\s*([^\]]+)\]')
        for _cm in list(_cite_check_pat.finditer(statement)):
            _cited_case = _cm.group(1).strip()
            for _bc in _bl_case_nums:
                if _bc in _cited_case or _cited_case in _bc:
                    logger.warning(
                        "[team_speak] Stripped blacklisted citation: '%s'",
                        _cited_case,
                    )
                    statement = statement.replace(_cm.group(0), "")
                    break

    # --- Auto-add evidence from statement citations ---
    _existing_ev_ids = {e.get("evidence_id", "") for e in evidence_list}
    _cite_ev_pat = _re.compile(r'\[(판례|법령|헌재|행심):\s*([^\]]+)\]')
    _src_type_map = {"판례": "court_precedent", "법령": "legal_statute",
                     "헌재": "constitutional_decision", "행심": "admin_tribunal_decision"}
    _prec_cite_types = {"판례", "헌재", "행심"}

    for _m in _cite_ev_pat.finditer(statement):
        _cite_type = _m.group(1)
        _cite_id = _m.group(2).strip()
        if _cite_id in _existing_ev_ids or _cite_id.startswith("⚠"):
            continue

        # Block non-accepted precedents (statutes pass through)
        if accepted_cases is not None and _cite_type in _prec_cite_types:
            if not any(_cite_id in ac or ac in _cite_id for ac in accepted_cases):
                logger.warning("[evidence] Stripped non-accepted citation from statement: [%s: %s]",
                               _cite_type, _cite_id)
                statement = statement.replace(_m.group(0), "")
                continue

        # Find matching item in search_results (partial match)
        _matched_sr = None
        for _sr in search_results:
            _sr_cn = _sr.get("case_number", "") or _sr.get("case_id", "")
            _sr_ln = _sr.get("law_name", "")
            _sr_title = _sr.get("title", "")
            if ((_sr_cn and (_cite_id in _sr_cn or _sr_cn in _cite_id))
                    or (_sr_ln and (_cite_id in _sr_ln or _sr_ln in _cite_id))
                    or (_sr_title and (_cite_id in _sr_title or _sr_title in _cite_id))):
                _matched_sr = _sr
                break

        _ev_content = _matched_sr.get("content", "") if _matched_sr else ""
        _ev_url = _matched_sr.get("url", "") if _matched_sr else ""

        evidence_list.append({
            "evidence_id": _cite_id,
            "content": _ev_content,
            "source_type": _src_type_map.get(_cite_type, "court_precedent"),
            "source_detail": _cite_id,
            "url": _ev_url,
            "submitted_by": state.get("team_id", "unknown"),
            "round": state.get("round", 0),
            "speaker": selected_speaker_id,
            "relevance": "",
        })
        _existing_ev_ids.add(_cite_id)
        logger.info("[evidence] Auto-added from statement: [%s: %s]", _cite_type, _cite_id)

    # --- Narrow evidence_list to only items actually cited in the final statement ---
    # Rationale: earlier in _produce_statement we pre-append the top-N search results
    # into evidence_list (to preserve rich metadata). Without this filter the UI would
    # render uncited retrieval hits as if they were evidence chips. User rule:
    # "원문(발언 본문)에서 나온 것만 evidence 에 넣어라".
    #
    # Uses _norm_cite() so that minor formatting differences between the LLM's tag
    # text and the evidence dict's field values don't drop legitimate matches.
    # Unverified citations (already replaced with "[⚠ 미확인 인용: ...]" above) are
    # not in the _cited_norms set, so uncited *and* hallucinated items are both removed.
    _cited_norms: set[str] = {_norm_cite(_m.group(2)) for _m in _cite_pat.finditer(statement)}

    def _ev_is_cited(ev: dict) -> bool:
        for _key in ('case_number', 'case_id', 'law_name', 'title',
                      'evidence_id', 'source_detail'):
            _val = ev.get(_key, '')
            if _val and _norm_cite(_val) in _cited_norms:
                return True
        return False

    _before_narrow = len(evidence_list)
    evidence_list = [ev for ev in evidence_list if isinstance(ev, dict) and _ev_is_cited(ev)]
    logger.info(
        "[team_speak] evidence_list narrowed by statement citations: %d → %d",
        _before_narrow, len(evidence_list),
    )

    logger.info(
        "[team_speak] Speaker %s produced statement (%d chars, %d evidence items).",
        agent_name, len(statement), len(evidence_list),
    )

    return selected_speaker_id, statement, evidence_list


# ------------------------------------------------------------------
# Main node entry point
# ------------------------------------------------------------------

async def team_speak_node(
    state: TeamState,
    llm_client: LLMClient,
    searcher: Searcher | None = None,
    legal_api: LegalAPIClient | None = None,
    language: str = "ko",
) -> dict:
    """
    Main team subgraph orchestrator node.

    Coordinates the full processing pipeline for a single team's turn:
    1. Execute evidence searches based on role assignments (parallel)
    2. Conduct internal team discussion
    3. Produce the final representative statement with citations

    Args:
        state: Current TeamState containing members, assignments,
            opponent statement, and debate context.
        llm_client: LLM client for all LLM operations.
        searcher: Optional Searcher instance for document/graph search.
        legal_api: Optional LegalAPIClient for statute/precedent search.

    Returns:
        Partial state update with ``output_statement``, ``output_evidence``,
        ``internal_discussion``, ``search_results``, and
        ``selected_speaker``.
    """
    team_id = state.get("team_id", "unknown")
    current_round = state.get("round", 0)
    members = state.get("members", [])
    assignments = state.get("role_assignments", [])

    # Load configurable team display names from state
    team_a_name = state.get("team_a_name", "Team A")
    team_b_name = state.get("team_b_name", "Team B")
    team_display_name = team_a_name if team_id == "team_a" else team_b_name

    # Load max_tool_rounds and language from settings
    from app.api.settings import settings_mgr
    try:
        _settings = settings_mgr.load()
        _debate_cfg = _settings.get("debate", {})
        max_tool_rounds = _debate_cfg.get("max_api_calls_per_round", 10)
        language = _debate_cfg.get("language", language)
    except Exception:
        max_tool_rounds = 10

    logger.info(
        "[team_speak] Starting team processing for %s (round %d, %d members, max_tool_rounds=%d).",
        team_id, current_round, len(members), max_tool_rounds,
    )

    # Re-verify previous unverified citations against accumulated evidence
    import re as _re
    _unverified_pat = _re.compile(r'\[⚠ 미확인 인용:\s*([^\]]+)\]')
    _all_evidence = state.get("all_evidences", [])
    _verified_ids: set[str] = set()
    for _ev in _all_evidence:
        if isinstance(_ev, dict):
            for _key in ('case_number', 'case_id', 'law_name', 'title', 'evidence_id'):
                _val = _ev.get(_key, '')
                if _val and not (len(str(_val)) == 36 and '-' in str(_val)):
                    _verified_ids.add(str(_val).strip())

    _debate_log = state.get("debate_context", state.get("debate_log", []))
    _log_updated = False
    for _entry in _debate_log:
        _stmt = _entry.get("statement", "")
        for _m in list(_unverified_pat.finditer(_stmt)):
            _cited = _m.group(1).strip()
            if _cited in _verified_ids:
                _ctype = "판례" if any(c in _cited for c in "다가나합") else "법령"
                _entry["statement"] = _stmt.replace(_m.group(0), f"[{_ctype}: {_cited}]")
                _log_updated = True
                logger.info("[team_speak] Previously unverified '%s' now verified", _cited)

    # Build unverified list for agent search prompts
    _unverified_for_prompt: list[str] = []
    for _entry in _debate_log:
        _stmt = _entry.get("statement", "")
        for _m in _unverified_pat.finditer(_stmt):
            _cited = _m.group(1).strip()
            _spk = _entry.get("speaker", "?")
            _rnd = _entry.get("round", "?")
            _unverified_for_prompt.append(f"- {_cited} (Round {_rnd}, {_spk})")

    if _log_updated:
        _debate_id = state.get("debate_id", "")
        if _debate_id:
            try:
                from app.api.debate import DebateStore
                await DebateStore.asave(_debate_id, state)
            except Exception:
                pass

    # Phase 1: Execute searches in parallel for all agents
    search_tasks = []
    for member in members:
        agent_id = member.get("agent_id", "")
        # Find this agent's assignment
        agent_assignment = None
        for a in assignments:
            if a.get("agent_id") == agent_id:
                agent_assignment = a
                break
        if agent_assignment is None:
            agent_assignment = {
                "agent_id": agent_id,
                "task": "Search for general supporting evidence",
                "search_type": "document",
                "priority": "supporting",
            }

        search_tasks.append(
            _execute_agent_search(
                agent=member,
                assignment=agent_assignment,
                state=state,
                llm_client=llm_client,
                searcher=searcher,
                legal_api=legal_api,
                max_tool_rounds=max_tool_rounds,
                language=language,
                unverified_citations=_unverified_for_prompt,
            )
        )

    # Carry over previous round results (exclude blacklisted)
    _prev_search = state.get("search_results", [])
    _prev_blacklist = set(state.get("blacklisted_evidence", []))
    carried_results: list[dict] = []
    existing_keys: set[str] = set()
    for sr in _prev_search:
        if not isinstance(sr, dict):
            continue
        cn = sr.get("case_number", sr.get("case_id", ""))
        ln = sr.get("law_name", "")
        key = cn or ln
        if key and key in _prev_blacklist:
            continue
        carried_results.append(sr)
        if key:
            existing_keys.add(key)

    # Run agent searches in batches (max 3 concurrent to avoid rate limiting)
    _SEARCH_BATCH_SIZE = 3
    if search_tasks:
        search_results_nested: list = []
        for _batch_start in range(0, len(search_tasks), _SEARCH_BATCH_SIZE):
            _batch = search_tasks[_batch_start:_batch_start + _SEARCH_BATCH_SIZE]
            _batch_results = await asyncio.gather(*_batch, return_exceptions=True)
            search_results_nested.extend(_batch_results)
        new_results: list[dict] = []
        for result in search_results_nested:
            if isinstance(result, list):
                for sr in result:
                    if not isinstance(sr, dict):
                        continue
                    cn = sr.get("case_number", sr.get("case_id", ""))
                    ln = sr.get("law_name", "")
                    key = cn or ln
                    if key and (key in _prev_blacklist or key in existing_keys):
                        continue  # blacklisted or already carried over
                    if key:
                        existing_keys.add(key)
                    new_results.append(sr)
            elif isinstance(result, Exception):
                logger.error("[team_speak] Agent search exception: %s", result)
    else:
        new_results = []

    all_search_results = carried_results + new_results
    logger.info(
        "[team_speak] Search: %d carried + %d new = %d total for %s.",
        len(carried_results), len(new_results), len(all_search_results), team_id,
    )

    # LLM relevance filter: remove items with no direct legal relevance to topic
    if all_search_results:
        all_search_results = await _llm_relevance_filter(
            items=all_search_results,
            topic=state.get("topic", ""),
            team_opinion=state.get("team_opinion", ""),
            llm_client=llm_client,
            language=language,
        )

    # Collect used search queries for next round duplicate avoidance
    _used_queries = set(state.get("used_search_queries", []))
    for sr in new_results:
        _q = sr.get("_search_query", "")
        if _q:
            _used_queries.add(_q)

    # Build case_number → item_id map for on-demand precedent lookup
    case_id_map = _build_case_id_map(all_search_results) if all_search_results else {}
    if case_id_map:
        logger.info("[team_speak] Built case_id_map with %d precedents for on-demand lookup.", len(case_id_map))

    # Phase 2: Internal team discussion
    # Build analysis summary for discussion
    _analysis_parts = [f"Topic: {state.get('topic', '')}"]
    _ki = state.get("key_issues", [])
    if _ki:
        _analysis_parts.append("Key issues: " + ", ".join(str(i) for i in _ki[:5]))
    _analysis_summary = "\n".join(_analysis_parts)

    # Extract opponent's CITED evidence (used in discussion + statement)
    _opp_cited = []
    for _entry in state.get("debate_log", []):
        if _entry.get("team") != team_id:
            for _ev in _entry.get("evidence", []):
                if isinstance(_ev, dict):
                    _opp_cited.append(_ev)
    _opp_lines: list[str] = []
    _opp_seen: set[str] = set()
    for _ev in _opp_cited:
        _cn = _ev.get("case_number", _ev.get("case_id", ""))
        _ln = _ev.get("law_name", "")
        _det = _ev.get("source_detail", "")[:80]
        _key = _cn or _ln or _det
        if _key and _key not in _opp_seen:
            _opp_seen.add(_key)
            if _cn:
                _opp_lines.append(f"- [판례] {_cn}: {_det}")
            elif _ln:
                _opp_lines.append(f"- [법령] {_ln}: {_det}")
    _opp_summary = "\n".join(_opp_lines) if _opp_lines else ""

    # Extract our team's latest representative statement
    _our_prev_stmt = ""
    for _entry in reversed(state.get("debate_log", [])):
        if (_entry.get("team") == team_id
                and _entry.get("entry_type") not in ("judge_question", "qa_answer")):
            _our_prev_stmt = _entry.get("statement", _entry.get("content", ""))[:1500]
            break

    discussion_log, accepted_cases = await _conduct_discussion(
        members=members,
        search_results=all_search_results,
        team_opinion=state.get("team_opinion", ""),
        opponent_stmt=state.get("opponent_statement", "(none)"),
        llm_client=llm_client,
        language=language,
        debate_id=state.get("debate_id", ""),
        team_id=state.get("team_id", "unknown"),
        judge_qa_log=state.get("judge_qa_log", []),
        assignments=assignments,
        situation_brief=state.get("situation_brief", ""),
        analysis_summary=_analysis_summary,
        opponent_cited_summary=_opp_summary,
        team_display_name=team_display_name,
        case_id_map=case_id_map,
        legal_api=legal_api,
        topic=state.get("topic", ""),
        our_prev_stmt=_our_prev_stmt,
        judge_improvement_feedback=state.get("judge_improvement_feedback", ""),
    )

    # Parse blacklist from discussion
    blacklisted_items: list[str] = []
    for entry in discussion_log:
        content = entry.get("content", "")
        if "FINAL_BLACKLIST:" in content:
            bl_line = content.split("FINAL_BLACKLIST:")[-1].strip()
            blacklisted_items.extend(
                item.strip().strip("[]") for item in bl_line.split(",") if item.strip()
            )
        elif "BLACKLIST:" in content and "FINAL_BLACKLIST" not in content:
            bl_line = content.split("BLACKLIST:")[-1].split("—")[0].strip()
            if bl_line:
                blacklisted_items.append(bl_line.strip().strip("[]"))
    blacklisted_items = list(set(blacklisted_items))
    if blacklisted_items:
        logger.info("[team_speak] Blacklisted evidence: %s", blacklisted_items)

    # Parse rejected precedents from discussion
    rejected_cases: set[str] = set()
    for entry in discussion_log:
        content = entry.get("content", "")
        for _m in _re.findall(r'\[REJECT:\s*([^\]]+)\]', content):
            rejected_cases.add(_m.strip())
    if rejected_cases:
        logger.info("[team_speak] Rejected precedents: %s", rejected_cases)
        # Add rejected precedents to blacklist
        blacklisted_items.extend(rejected_cases)
        blacklisted_items = list(set(blacklisted_items))

    # Parse discussed/reviewed cases from discussion log (for whitelist filtering)
    discussed_cases: set[str] = set()
    for entry in discussion_log:
        _dc_content = entry.get("content", "")
        # [판례: ...] and [법령: ...] tags
        for _m in _re.findall(r'\[판례:\s*([^\]]+)\]', _dc_content):
            discussed_cases.add(_m.strip())
        for _m in _re.findall(r'\[법령:\s*([^\]]+)\]', _dc_content):
            discussed_cases.add(_m.strip())
        # ACCEPT/REJECT/BLACKLIST/REVIEW_MORE tags
        for _tag in ['ACCEPT', 'REJECT', 'BLACKLIST', 'REVIEW_MORE']:
            for _m in _re.findall(rf'\[{_tag}:\s*([^\]]+)\]', _dc_content):
                discussed_cases.add(_m.strip())
    if discussed_cases:
        logger.info("[team_speak] Discussed cases: %d items — %s", len(discussed_cases),
                     list(discussed_cases)[:10])

    # Generate consensus strategy — built ONLY from ACCEPT-approved precedents
    # and statutes, NOT from raw discussion_log. This prevents unreviewed
    # precedent mentions in the discussion from leaking into the strategy text
    # and then into the representative speaker's input.
    _accepted_brief_lines: list[str] = []
    for _sr in all_search_results:
        if not isinstance(_sr, dict):
            continue
        _cn = _sr.get("case_number", "") or _sr.get("사건번호", "")
        _src_t = _sr.get("source_type", _sr.get("type", ""))
        _is_prec = _src_t in ("court_precedent", "prec",
                              "constitutional_decision",
                              "admin_tribunal_decision")
        if not _is_prec or not _cn:
            continue
        if not any(_cn in ac or ac in _cn for ac in accepted_cases):
            continue
        _title = _sr.get("title", "")
        _holding = str(_sr.get("content", ""))
        if "[판결요지]" in _holding:
            _holding = _holding.split("[판결요지]", 1)[1].strip()
        _accepted_brief_lines.append(
            f"- [{_cn}] {_title}\n  요지: {_holding[:300]}"
        )

    _statute_brief_lines: list[str] = []
    for _sr in all_search_results:
        if not isinstance(_sr, dict):
            continue
        _ln = _sr.get("law_name", "") or _sr.get("법령명한글", "")
        if not _ln:
            continue
        _statute_brief_lines.append(
            f"- [{_ln}] {str(_sr.get('content', ''))[:200]}"
        )

    _accepted_block = "\n".join(_accepted_brief_lines) or "(no accepted precedent)"
    _statute_block = "\n".join(_statute_brief_lines[:8]) or "(no statute)"

    consensus_prompt = (
        f"You are summarizing a debate team's strategy.\n"
        f"Topic / position: {state.get('team_opinion', '')}\n\n"
        f"## APPROVED Precedents (the only ones the team may rely on)\n"
        f"{_accepted_block}\n\n"
        f"## Available Statutes\n"
        f"{_statute_block}\n\n"
        f"Write 2-3 sentences describing the team's main argument grounded "
        f"ONLY in the approved precedents and statutes above. Do NOT mention "
        f"any other case number or holding."
    )

    try:
        agreed_strategy = await llm_client.achat(
            messages=[
                {"role": "system", "content": "You are a debate team coordinator. Summarize the consensus." + get_language_instruction(language)},
                {"role": "user", "content": consensus_prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
    except Exception as exc:
        logger.error("[team_speak] Consensus generation error: %s", exc)
        agreed_strategy = state.get("agreed_strategy", "") or "Proceed with available evidence."

    logger.info(
        "[team_speak] Discussion phase complete for %s: %d messages, strategy: %s",
        team_id, len(discussion_log), agreed_strategy[:80],
    )

    # Phase 3: Produce final statement
    # Pause check before expensive statement generation
    _debate_id = state.get("debate_id", "")
    if _debate_id:
        try:
            from app.api.debate import DebateStore
            _ps = await DebateStore.aload(_debate_id)
            if _ps.get("status") in ("paused", "stopped"):
                logger.info("[team_speak] %s detected — skipping statement generation", _ps["status"])
                return {
                    "output_statement": "",
                    "output_evidence": [],
                    "internal_discussion": discussion_log,
                    "search_results": all_search_results,
                    "selected_speaker": "",
                    "blacklisted_evidence": blacklisted_items,
                }
        except Exception:
            pass

    # Update state with latest strategy before statement production
    updated_state = dict(state)
    updated_state["agreed_strategy"] = agreed_strategy

    # Reuse opponent evidence summary computed before discussion
    updated_state["_opponent_cited_summary"] = _opp_summary

    selected_speaker, statement, evidence_list = await _produce_statement(
        state=updated_state,
        search_results=all_search_results,
        discussion_log=discussion_log,
        agreed_strategy=agreed_strategy,
        llm_client=llm_client,
        language=language,
        team_display_name=team_display_name,
        blacklisted_items=blacklisted_items,
        discussed_cases=discussed_cases,
        accepted_cases=accepted_cases,
    )

    logger.info(
        "[team_speak] Team %s complete. Speaker: %s, Statement: %d chars, Evidence: %d.",
        team_id, selected_speaker, len(statement), len(evidence_list),
    )

    return {
        "output_statement": statement,
        "output_evidence": evidence_list,
        "internal_discussion": discussion_log,
        "search_results": all_search_results,
        "selected_speaker": selected_speaker,
        "agreed_strategy": agreed_strategy,
        "blacklisted_evidence": blacklisted_items,
        "used_search_queries": list(_used_queries),
    }

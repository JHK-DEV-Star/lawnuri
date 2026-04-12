"""
External legal API client for Korean government legal databases.

Provides async access to the Korean National Law Information Center
(국가법령정보센터) for searching statutes, court precedents, and 12 additional
legal-data categories.  Results are cached locally to reduce redundant API
calls.

Supports dual caching: SQLite (primary) with local file fallback.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import httpx

from app.utils.logger import logger

_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    """Return a shared ``httpx.AsyncClient`` with connection pooling."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _http_client

_CASE_NUM_RE = re.compile(r'(\d{2,4}[-\s]*[가-힣]+[-\s]*\d+)')


def _extract_pure_case_number(raw: str) -> str:
    """Extract pure case number from a string that may include court name.

    '대구고등법원 2025나10683' → '2025나10683'
    '2023다12345' → '2023다12345'
    """
    m = _CASE_NUM_RE.search(raw)
    return m.group(1).strip() if m else raw


async def _try_get_db():
    """Return the active SQLite connection, or None if unavailable."""
    try:
        from app.db.database import get_db_connection
        return await get_db_connection()
    except Exception:
        return None

LEGAL_CATEGORIES: dict[str, dict[str, Any]] = {
    "law": {
        "target": "law",
        "name": "법령",
        "desc": "법률/시행령/시행규칙 (국가 법령)",
        "id_field": "법령일련번호",
        "detail_param": "MST",
        "source": "legal_statute",
        "title_fields": ["법령명한글", "법령명"],
        "content_fields": ["법령명약칭"],
    },
    "prec": {
        "target": "prec",
        "name": "판례",
        "desc": "대법원/고등법원/지방법원 판례",
        "id_field": "판례일련번호",
        "detail_param": "ID",
        "source": "court_precedent",
        "title_fields": ["사건명", "판례명"],
        "content_fields": ["판시사항"],
        "extra_fields": {"court": "법원명", "date": "선고일자", "case_number": "사건번호"},
    },
    "const": {
        "target": "const",
        "name": "헌재결정례",
        "desc": "헌법재판소 결정례",
        "id_field": "헌재결정례일련번호",
        "detail_param": "ID",
        "source": "constitutional_decision",
        "title_fields": ["사건명"],
        "content_fields": ["판시사항"],
    },
    "interp": {
        "target": "interp",
        "name": "법령해석례",
        "desc": "법제처 법령 유권해석",
        "id_field": "법령해석례일련번호",
        "detail_param": "ID",
        "source": "legal_interpretation",
        "title_fields": ["안건명"],
        "content_fields": ["해석내용"],
    },
    "detc": {
        "target": "detc",
        "name": "행정심판재결례",
        "desc": "행정심판위원회 재결",
        "id_field": "행정심판재결례일련번호",
        "detail_param": "ID",
        "source": "admin_tribunal_decision",
        "title_fields": ["사건명"],
        "content_fields": ["재결요지"],
    },
    "ordin": {
        "target": "ordin",
        "name": "자치법규",
        "desc": "지방자치단체 조례/규칙",
        "id_field": "자치법규일련번호",
        "detail_param": "ID",
        "source": "local_ordinance",
        "title_fields": ["자치법규명"],
        "content_fields": ["자치법규명약칭"],
    },
    "admrul": {
        "target": "admrul",
        "name": "행정규칙",
        "desc": "훈령/예규/고시/공고",
        "id_field": "행정규칙일련번호",
        "detail_param": "ID",
        "source": "administrative_rule",
        "title_fields": ["행정규칙명"],
        "content_fields": ["행정규칙명약칭"],
    },
    "treaty": {
        "target": "treaty",
        "name": "조약",
        "desc": "국제 조약 및 협정",
        "id_field": "조약일련번호",
        "detail_param": "ID",
        "source": "treaty",
        "title_fields": ["조약명"],
        "content_fields": ["조약명약칭"],
    },
    "term": {
        "target": "term",
        "name": "법령용어",
        "desc": "법률 용어 정의 검색",
        "id_field": "용어일련번호",
        "detail_param": "ID",
        "source": "legal_term",
        "title_fields": ["용어명"],
        "content_fields": ["정의"],
    },
    "special": {
        "target": "special",
        "name": "특별행정심판",
        "desc": "특별행정심판 재결례",
        "id_field": "사건번호",
        "detail_param": "ID",
        "source": "special_admin_decision",
        "title_fields": ["사건명"],
        "content_fields": ["재결요지"],
    },
    "consulting": {
        "target": "consulting",
        "name": "사전컨설팅",
        "desc": "감사원 사전컨설팅 의견",
        "id_field": "의견번호",
        "detail_param": "ID",
        "source": "consulting_opinion",
        "title_fields": ["안건명"],
        "content_fields": ["의견요지"],
    },
    "ministry": {
        "target": "ministry",
        "name": "중앙부처해석",
        "desc": "중앙부처 1차 법령해석",
        "id_field": "해석번호",
        "detail_param": "ID",
        "source": "ministry_interpretation",
        "title_fields": ["안건명"],
        "content_fields": ["해석내용"],
    },
    "table": {
        "target": "table",
        "name": "별표서식",
        "desc": "법령 별표 및 서식",
        "id_field": "별표서식일련번호",
        "detail_param": "ID",
        "source": "legal_table",
        "title_fields": ["별표서식명"],
        "content_fields": ["별표서식명"],
    },
    "committee": {
        "target": "committee",
        "name": "위원회결정문",
        "desc": "각종 위원회 결정문 (org 파라미터로 위원회 지정)",
        "id_field": "사건번호",
        "detail_param": "ID",
        "source": "committee_decision",
        "title_fields": ["사건명"],
        "content_fields": ["결정요지"],
    },
}

CATEGORY_HINTS: dict[str, str] = {
    "law": (
        "추가 필터: sort(정렬), efYd(시행일범위 YYYYMMDD~YYYYMMDD), "
        "knd(법률종류코드). get_legal_detail로 조문 전문 조회 가능."
    ),
    "prec": (
        "추가 필터: court(법원명), prncYd(선고일 YYYYMMDD~YYYYMMDD), "
        "nb(사건번호), jo(참조법령), search=2(본문검색). "
        "get_legal_detail로 판례 전문."
    ),
    "const": "추가 필터: sort, prncYd. get_legal_detail로 헌재결정 전문 조회.",
    "interp": "법제처 유권해석. get_legal_detail로 해석 전문 조회 가능.",
    "detc": "행정심판 재결례. get_legal_detail로 재결문 전문 조회.",
    "ordin": "추가 필터: org(지방자치단체코드). get_legal_detail로 조례/규칙 전문.",
    "admrul": "훈령/예규/고시. get_legal_detail로 전문 조회 가능.",
    "treaty": "국제조약/협정. get_legal_detail로 조약 전문 조회.",
    "term": "법령용어 정의. 특정 용어의 법적 정의를 찾을 때 유용.",
    "special": "특별행정심판 재결. get_legal_detail로 재결문 전문.",
    "consulting": "감사원 사전컨설팅 의견. get_legal_detail로 의견서 전문.",
    "ministry": "중앙부처 1차 해석. get_legal_detail로 해석 전문.",
    "table": "법령 별표/서식. get_legal_detail로 전문 조회.",
    "committee": (
        "위원회 결정문. org 파라미터 필수 "
        "(ftc=공정거래, pipc=개인정보보호, acrc=국민권익, fsc=금융, "
        "nlrc=노동, kcc=방송통신, nhrc=인권위, eir=고용보험심사, "
        "iacr=산업재해보상, clec=토지수용, edrc=환경분쟁, sfc=증권선물)."
    ),
}

COMMITTEE_ORGS: dict[str, str] = {
    "pipc": "개인정보보호위원회",
    "eir": "고용보험심사위원회",
    "ftc": "공정거래위원회",
    "acrc": "국민권익위원회",
    "fsc": "금융위원회",
    "nlrc": "노동위원회",
    "kcc": "방송미디어통신위원회",
    "iacr": "산업재해보상보험재심사위원회",
    "clec": "중앙토지수용위원회",
    "edrc": "중앙환경분쟁조정위원회",
    "sfc": "증권선물위원회",
    "nhrc": "국가인권위원회",
}


# ======================================================================
# Client
# ======================================================================

class LegalAPIClient:
    """
    Async client for Korean legal APIs.

    Connects to the 국가법령정보센터 DRF API for statute, precedent, and
    additional legal-data category search/detail operations.  Provides
    local file-based caching and graceful degradation when API keys are
    not configured.
    """

    # Base URLs for the Korean National Law Information Center DRF API
    _SEARCH_BASE = "https://www.law.go.kr/DRF/lawSearch.do"
    _DETAIL_BASE = "https://www.law.go.kr/DRF/lawService.do"

    def __init__(
        self,
        api_key: str = "",
        cache_dir: str = "",
        law_api_key: str = "",
        precedent_api_key: str = "",
        debate_id: str = "",
    ) -> None:
        """
        Initialize the legal API client.

        Args:
            api_key: Unified API key (OC) used for all categories.  Falls
                back to *law_api_key* / *precedent_api_key* for backward
                compatibility.
            cache_dir: Directory for local response caching.  If empty,
                caching is disabled.
            law_api_key: Legacy API key for statute searches.
            precedent_api_key: Legacy API key for precedent searches.
            debate_id: Debate identifier used for SQLite cache scoping.
        """
        self._api_key = api_key or law_api_key or precedent_api_key
        self._law_api_key = law_api_key or api_key
        self._precedent_api_key = precedent_api_key or api_key
        self._cache_dir = cache_dir
        self._debate_id = debate_id

        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            "LegalAPIClient initialized (api_key=%s, law_key=%s, prec_key=%s, cache=%s).",
            "set" if self._api_key else "empty",
            "set" if self._law_api_key else "empty",
            "set" if self._precedent_api_key else "empty",
            cache_dir or "disabled",
        )

    # ------------------------------------------------------------------
    # Helper: resolve the appropriate OC key for a given category
    # ------------------------------------------------------------------

    def _resolve_api_key(self, category: str) -> str:
        """Return the best available API key for *category*."""
        if category == "law":
            return self._law_api_key or self._api_key
        if category == "prec":
            return self._precedent_api_key or self._api_key
        return self._api_key or self._law_api_key or self._precedent_api_key

    # ------------------------------------------------------------------
    # Public API - Generic search / detail
    # ------------------------------------------------------------------

    async def search_legal(
        self,
        category: str,
        query: str,
        max_results: int = 5,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Search any supported legal category.

        Args:
            category: One of the keys in ``LEGAL_CATEGORIES``.
            query: Search keyword (Korean or English).
            max_results: Maximum number of results to return.
            **kwargs: Optional API filters.  Recognised keys:
                court, sort, prncYd, nb, jo, org, search, efYd, date, knd.

        Returns:
            A dict with keys ``results`` (list[dict]), ``total_count``
            (int), and ``hint`` (str).
        """
        if category not in LEGAL_CATEGORIES:
            return {
                "results": [{"error": f"지원하지 않는 카테고리: {category}. "
                             f"사용 가능: {', '.join(LEGAL_CATEGORIES)}"}],
                "total_count": 0,
                "hint": "",
            }

        cat_cfg = LEGAL_CATEGORIES[category]
        api_key = self._resolve_api_key(category)

        if not api_key:
            logger.warning("%s search blocked: no API key configured.", category)
            return {
                "results": [{"error": "법령/판례 API key(OC)가 설정되지 않았습니다. Settings에서 입력해주세요."}],
                "total_count": 0,
                "hint": "",
            }

        # Committee searches require an org parameter
        if category == "committee" and not kwargs.get("org"):
            return {
                "results": [{"error": "위원회 검색에는 org 파라미터가 필수입니다. "
                             f"사용 가능: {', '.join(f'{k}({v})' for k, v in COMMITTEE_ORGS.items())}"}],
                "total_count": 0,
                "hint": CATEGORY_HINTS.get(category, ""),
            }

        # Build base params
        params: dict[str, str] = {
            "OC": api_key,
            "target": cat_cfg["target"],
            "type": "JSON",
            "query": query,
            "display": str(max_results),
        }

        # Map optional kwargs to API parameter names
        _kwarg_map: dict[str, str] = {
            "court": "curt",
            "sort": "sort",
            "prncYd": "prncYd",
            "nb": "nb",
            "jo": "JO",
            "org": "org",
            "search": "search",
            "efYd": "efYd",
            "date": "date",
            "knd": "knd",
        }

        for kw_key, param_name in _kwarg_map.items():
            value = kwargs.get(kw_key)
            if value is not None and str(value):
                # Special handling: court="all" is a no-op
                if kw_key == "court" and str(value).lower() == "all":
                    continue
                params[param_name] = str(value)

        # Cache lookup
        cache_key = self._cache_key(f"{category}_search", params)
        cached = await self._get_cached(cache_key)
        if cached is not None:
            logger.debug("%s search cache hit for query='%s'.", category, query)
            return cached

        # Remote request
        try:
            data = await self._request(params, base_url=self._SEARCH_BASE)
        except Exception as exc:
            logger.warning("%s search failed for query='%s': %s", category, query, exc)
            return {"results": [], "total_count": 0, "hint": CATEGORY_HINTS.get(category, "")}

        results = self._parse_search_results(data, category, max_results)
        results = self._rerank_results(results, query)
        total_count = self._extract_total_count(data)

        payload: dict[str, Any] = {
            "results": results,
            "total_count": total_count,
            "hint": CATEGORY_HINTS.get(category, ""),
        }
        await self._set_cache(cache_key, payload)

        logger.info(
            "%s search for '%s' returned %d results (total=%d).",
            category, query, len(results), total_count,
        )
        return payload

    async def get_legal_detail(
        self,
        category: str,
        item_id: str,
        summary_only: bool = False,
    ) -> dict[str, Any] | None:
        """
        Retrieve the full detail of a single legal item.

        Args:
            category: One of the keys in ``LEGAL_CATEGORIES``.
            item_id: The unique identifier of the item (value depends on
                category; for ``law`` it is the MST/법령일련번호, for others
                it is the ID field).

        Returns:
            A detail dict, or ``None`` on failure.
        """
        if category not in LEGAL_CATEGORIES:
            logger.warning("get_legal_detail called with unknown category '%s'.", category)
            return None

        cat_cfg = LEGAL_CATEGORIES[category]
        api_key = self._resolve_api_key(category)

        if not api_key:
            logger.debug("%s detail skipped: no API key configured.", category)
            return None

        detail_param = cat_cfg["detail_param"]

        params: dict[str, str] = {
            "OC": api_key,
            "target": cat_cfg["target"],
            "type": "JSON",
            detail_param: item_id,
        }

        cache_key = self._cache_key(f"{category}_detail{'_summary' if summary_only else ''}", params)
        # Detail lookups use global cache so results are shared across debates
        _cache_scope = "__global__"
        cached = await self._get_cached(cache_key, scope=_cache_scope)
        if cached is not None:
            logger.debug("%s detail cache hit for id='%s'.", category, item_id)
            return cached

        try:
            data = await self._request(params, base_url=self._DETAIL_BASE)
        except Exception as exc:
            logger.warning("%s detail failed for id='%s': %s", category, item_id, exc)
            return None

        result = self._parse_detail_result(data, category, item_id, summary_only=summary_only)
        if result:
            await self._set_cache(cache_key, result, scope=_cache_scope)
        return result

    # ------------------------------------------------------------------
    # Public API - Legacy wrappers (backward compatibility)
    # ------------------------------------------------------------------

    async def search_statutes(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict]:
        """
        Search statutes (법령) from the national law database.

        This is a backward-compatible wrapper around :pymeth:`search_legal`.

        Args:
            query: Search keyword (Korean or English).
            max_results: Maximum number of results to return.

        Returns:
            List of statute result dicts with keys:
                title, content, source, law_id, url.
            Returns an empty list if the API key is not set or on error.
        """
        payload = await self.search_legal("law", query, max_results=max_results)
        return payload.get("results", [])

    async def search_precedents(
        self,
        query: str,
        court: str = "all",
        max_results: int = 5,
        search: int = 0,
        org: str = "",
        jo: str = "",
        sort: str = "",
        date: int = 0,
        prncYd: str = "",
        nb: str = "",
    ) -> list[dict]:
        """
        Search court precedents (판례) from the national law database.

        This is a backward-compatible wrapper around :pymeth:`search_legal`.

        Args:
            query: Search keyword (Korean or English).
            court: Court filter (e.g. "대법원", "서울고등법원").
            max_results: Maximum number of results to return.
            search: Search scope (1=판례명, 2=본문검색).
            org: Court type code (대법원:400201, 하위법원:400202).
            jo: Reference statute name (참조법령명, e.g. "형법", "민법").
            sort: Sort order (ddes=선고일 내림, dasc=선고일 오름, lasc/ldes=사건명).
            date: Specific ruling date (선고일자).
            prncYd: Ruling date range (e.g. "20090101~20090130").
            nb: Case number (사건번호).

        Returns:
            List of precedent result dicts.
            Returns error dict if API key is not configured.
        """
        kwargs: dict[str, Any] = {}
        if court and court != "all":
            kwargs["court"] = court
        if search:
            kwargs["search"] = search
        if org:
            kwargs["org"] = org
        if jo:
            kwargs["jo"] = jo
        if sort:
            kwargs["sort"] = sort
        if date:
            kwargs["date"] = date
        if prncYd:
            kwargs["prncYd"] = prncYd
        if nb:
            kwargs["nb"] = nb

        payload = await self.search_legal("prec", query, max_results=max_results, **kwargs)
        return payload.get("results", [])

    # ------------------------------------------------------------------
    # Public API - Legacy detail wrappers
    # ------------------------------------------------------------------

    async def get_statute_detail(self, law_id: str) -> dict | None:
        """
        Retrieve the full text of a specific statute by its law ID.

        Backward-compatible wrapper around :pymeth:`get_legal_detail`.
        """
        return await self.get_legal_detail("law", law_id)

    async def get_precedent_detail(self, case_id: str) -> dict | None:
        """
        Retrieve the full text of a specific court precedent by its case ID.

        Backward-compatible wrapper around :pymeth:`get_legal_detail`.
        """
        return await self.get_legal_detail("prec", case_id)

    # ------------------------------------------------------------------
    # HTTP request helper
    # ------------------------------------------------------------------

    async def _request(
        self,
        params: dict[str, str],
        base_url: str = "",
    ) -> Any:
        """
        Execute an async HTTP GET request to the legal API.

        Args:
            params: Query parameters for the API call.
            base_url: The base URL to send the request to.  Defaults to
                ``_SEARCH_BASE`` when not supplied.

        Returns:
            Parsed JSON response data.

        Raises:
            httpx.HTTPStatusError: On non-2xx responses.
            Exception: On network or parsing errors.
        """
        url = base_url or self._SEARCH_BASE
        client = _get_http_client()
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Result reranking
    # ------------------------------------------------------------------

    @staticmethod
    def _rerank_results(results: list[dict], query: str) -> list[dict]:
        """Rerank results: exact title match > starts-with > contains."""
        q = query.strip()
        exact, starts, rest = [], [], []
        for r in results:
            title = r.get("title", "")
            if title == q:
                exact.append(r)
            elif title.startswith(q):
                starts.append(r)
            else:
                rest.append(r)
        return exact + starts + rest

    # ------------------------------------------------------------------
    # Generic response parsers
    # ------------------------------------------------------------------

    def _parse_search_results(
        self,
        data: Any,
        category: str,
        max_results: int,
    ) -> list[dict]:
        """
        Parse the JSON response from any category search into a list of
        standardised result dicts.

        Uses the field mappings defined in ``LEGAL_CATEGORIES[category]``.
        """
        cat_cfg = LEGAL_CATEGORIES[category]
        target = cat_cfg["target"]
        source = cat_cfg["source"]
        id_field = cat_cfg["id_field"]
        title_fields = cat_cfg["title_fields"]
        content_fields = cat_cfg["content_fields"]
        extra_fields: dict[str, str] = cat_cfg.get("extra_fields", {})

        items = self._extract_items(data, target)
        results: list[dict] = []

        for item in items[:max_results]:
            # Resolve title: try each candidate field in order
            title = ""
            for tf in title_fields:
                title = item.get(tf, "")
                if title:
                    break

            # Resolve content: try each candidate field, fall back to title
            # For precedents, combine 판시사항 + 판결요지 for better context
            if category == "prec":
                # Only include 판결요지 (decision summary) — NOT 판시사항 (issue summary)
                # 판시사항 is just a list of legal issues; agents misinterpret it as holdings
                pangyeol = item.get("판결요지", "")
                content = f"[판결요지]\n{pangyeol}" if pangyeol else title
            else:
                content = ""
                for cf in content_fields:
                    content = item.get(cf, "")
                    if content:
                        break
                if not content:
                    content = title

            # Skip items with no meaningful content
            if not title and not content:
                continue

            # Resolve unique ID
            item_id = str(item.get(id_field, item.get("ID", item.get("MST", ""))))

            # Build URL (use search URL for prec, 한글주소 for law)
            if category == "prec":
                case_num = item.get("사건번호", "")
                url = f"https://www.law.go.kr/precSc.do?tabMenuId=465&query={_extract_pure_case_number(case_num)}" if case_num else ""
            elif category == "law":
                url = f"https://www.law.go.kr/법령/{title}" if title else ""
            else:
                url = f"https://www.law.go.kr/{cat_cfg['name']}/{title}" if title else ""

            result: dict[str, Any] = {
                "title": title,
                "content": content,
                "source": source,
                "_item_id": item_id,  # internal only; not exposed to LLM
                "url": url,
            }

            # Normalized human-readable identifiers for all categories
            result["law_name"] = ""
            result["case_number"] = ""

            if category == "law":
                result["law_name"] = title  # 법령명한글 (e.g., "민법")
                result["law_id"] = title
            elif category == "prec":
                case_num = item.get("사건번호", "")
                result["case_number"] = case_num
                result["case_id"] = case_num or title
            elif category == "const":
                result["case_number"] = item.get("사건번호", "")
                result["law_name"] = title  # 사건명
            elif category in ("interp", "detc", "ordin", "admrul", "treaty",
                               "term", "special", "consulting", "ministry",
                               "table", "committee"):
                result["law_name"] = title  # 안건명/법규명/조약명 등

            # Extra fields (e.g. court, date for prec)
            for result_key, api_key_name in extra_fields.items():
                result[result_key] = item.get(api_key_name, "")

            # Tip for LLM: how to get full text (reference by case_number/law_name)
            _ref_id = result.get("case_number", "") or result.get("law_name", "") or title
            if _ref_id:
                result["tip"] = (
                    f"Call get_legal_detail(category='{category}', reference='{_ref_id}') "
                    f"to read the full text (본문)"
                )

            results.append(result)

        return results

    def _parse_detail_result(
        self,
        data: Any,
        category: str,
        item_id: str,
        summary_only: bool = False,
    ) -> dict[str, Any] | None:
        """
        Parse the JSON response from a detail request for any category.
        """
        if not data:
            return None

        cat_cfg = LEGAL_CATEGORIES[category]
        target = cat_cfg["target"]
        source = cat_cfg["source"]
        title_fields = cat_cfg["title_fields"]
        content_fields = cat_cfg["content_fields"]
        extra_fields: dict[str, str] = cat_cfg.get("extra_fields", {})

        # Unwrap common envelope patterns
        item: dict[str, Any]
        if isinstance(data, dict):
            item = data
            # Try envelope key by Korean name or target
            for envelope_key in [cat_cfg["name"], target]:
                if envelope_key in item and isinstance(item[envelope_key], dict):
                    item = item[envelope_key]
                    break
        elif isinstance(data, list) and data:
            item = data[0] if isinstance(data[0], dict) else {}
        else:
            return None

        # Resolve title
        title = ""
        for tf in title_fields:
            title = item.get(tf, "")
            if title:
                break

        # Resolve content
        content = ""
        if summary_only:
            # Phase 2: summary only — skip full-text keys, use 판결요지/조문 요약
            if category == "prec":
                pangyeol = item.get("판결요지", "")
                if pangyeol:
                    content = f"[판결요지]\n{pangyeol}"
            elif category == "law":
                content = item.get("조문내용", "")  # statute text is short enough
            if not content:
                for cf in content_fields:
                    content = item.get(cf, "")
                    if content:
                        content = content[:500]  # cap at 500 chars for summary
                        break
        else:
            # Phase 3+: full text — try full-text keys first
            _detail_content_keys = [
                "조문내용", "판례내용", "내용", "결정내용", "해석내용전문",
                "재결내용", "의견내용", "본문",
            ]
            for ck in _detail_content_keys:
                content = item.get(ck, "")
                if content:
                    break

            # For precedents: use only 판결요지 (decision summary), not 판시사항
            if not content and category == "prec":
                pangyeol = item.get("판결요지", "")
                if pangyeol:
                    content = f"[판결요지]\n{pangyeol}"

            if not content:
                for cf in content_fields:
                    content = item.get(cf, "")
                    if content:
                        break
        if not content:
            content = str(item)

        # Build category-aware URL
        if category == "prec":
            case_num = item.get("사건번호", "")
            url = f"https://www.law.go.kr/precSc.do?tabMenuId=465&query={_extract_pure_case_number(case_num)}" if case_num else ""
        elif category == "law":
            url = f"https://www.law.go.kr/법령/{title}" if title else ""
        else:
            url = f"https://www.law.go.kr/{cat_cfg['name']}/{title}" if title else ""

        result: dict[str, Any] = {
            "title": title,
            "content": content,
            "source": source,
            "item_id": item_id,
            "url": url,
        }

        # Normalized identifier fields (same as _parse_search_results)
        result["law_name"] = ""
        result["case_number"] = ""

        if category == "law":
            result["law_name"] = title
            result["law_id"] = item_id
        elif category == "prec":
            case_num = item.get("사건번호", "")
            result["case_number"] = case_num
            result["case_id"] = case_num if case_num else item_id
            result["law_name"] = title  # 사건명
        elif category == "const":
            result["case_number"] = item.get("사건번호", "")
            result["law_name"] = title
        else:
            result["law_name"] = title

        for result_key, api_key_name in extra_fields.items():
            result[result_key] = item.get(api_key_name, "")

        # Verify the result has meaningful content (not just metadata)
        if not title and not content:
            logger.warning(
                "_parse_detail_result: empty result for %s/%s (no title or content)",
                category, item_id,
            )
            return None

        return result

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_total_count(data: Any) -> int:
        """
        Attempt to pull a total-result count from the API response.

        The DRF API may include totalCnt / totalcount / 전체건수 depending
        on the category.
        """
        if not isinstance(data, dict):
            return 0
        for key in ("totalCnt", "totalcount", "전체건수", "totalCount"):
            val = data.get(key)
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
            # Also look one level deeper (some envelopes nest it)
            for inner in data.values():
                if isinstance(inner, dict):
                    val = inner.get(key)
                    if val is not None:
                        try:
                            return int(val)
                        except (ValueError, TypeError):
                            pass
        return 0

    @staticmethod
    def _extract_items(data: Any, target: str) -> list[dict]:
        """
        Extract the list of items from an API response.

        The API response structure varies; this method handles the most
        common patterns observed in the 국가법령정보센터 DRF responses.

        Args:
            data: Raw parsed JSON response.
            target: The target type (e.g. "law", "prec", "const", ...).

        Returns:
            A list of item dicts extracted from the response.
        """
        if isinstance(data, list):
            return data

        if not isinstance(data, dict):
            return []

        # Try common response envelope keys
        _envelope_keys = [
            target,
            f"{target}Search",
            "PrecSearch",
            "LawSearch",
            "list",
            # Additional envelopes for new categories
            "ConstSearch",
            "InterpSearch",
            "DetcSearch",
            "OrdinSearch",
            "AdmrulSearch",
            "TreatySearch",
            "TermSearch",
            "SpecialSearch",
            "ConsultingSearch",
            "MinistrySearch",
            "TableSearch",
            "CommitteeSearch",
        ]

        for key in _envelope_keys:
            if key in data:
                value = data[key]
                if isinstance(value, list):
                    return value
                if isinstance(value, dict):
                    # Sometimes the items are nested one more level
                    for inner_key in ["row", "list", target]:
                        if inner_key in value:
                            inner = value[inner_key]
                            if isinstance(inner, list):
                                return inner
                            if isinstance(inner, dict):
                                return [inner]
                    return [value]

        return []

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, endpoint: str, params: dict) -> str:
        """
        Generate a deterministic cache key from the endpoint and parameters.

        The API key is excluded from the hash to avoid invalidating cache
        when keys are rotated.

        Args:
            endpoint: Logical endpoint name (e.g. "law_search").
            params: The query parameters dict.

        Returns:
            A hex digest string usable as a filename.
        """
        # Build a stable representation excluding the API key
        filtered = {k: v for k, v in sorted(params.items()) if k != "OC"}
        raw = f"{endpoint}:{json.dumps(filtered, sort_keys=True, ensure_ascii=False)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def _get_cached(self, cache_key: str, scope: str = "") -> dict | None:
        """
        Look up a cached response by its cache key.

        Tries SQLite first, then falls back to local file cache.

        Args:
            cache_key: The cache key (hex digest).
            scope: Cache scope (debate_id). Use ``"__global__"`` for
                cross-debate shared cache (file-only, no SQLite).
                Defaults to ``self._debate_id``.

        Returns:
            Parsed cached data dict, or None if not found.
        """
        effective_scope = scope or self._debate_id

        # --- Global scope: file-only (SQLite has FK on debate_id) ---
        if effective_scope == "__global__":
            global_dir = Path("data") / "shared" / "cache" / "legal"
            global_dir.mkdir(parents=True, exist_ok=True)
            cache_file = global_dir / f"{cache_key}.json"
            if cache_file.is_file():
                try:
                    return json.loads(cache_file.read_text(encoding="utf-8"))
                except Exception:
                    pass
            return None

        # --- SQLite primary ---
        if effective_scope:
            db = await _try_get_db()
            if db is not None:
                try:
                    from app.db.sqlite.cache import SQLiteCacheRepo
                    repo = SQLiteCacheRepo()
                    result = await repo.get(db, effective_scope, cache_key)
                    if result is not None:
                        return result
                except Exception:
                    logger.warning("SQLite cache get failed; falling back to file.", exc_info=True)

        # --- File fallback ---
        if not self._cache_dir:
            return None

        cache_file = Path(self._cache_dir) / f"{cache_key}.json"
        if not cache_file.is_file():
            return None

        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read cache file %s: %s", cache_file, exc)
            return None

    async def _set_cache(self, cache_key: str, data: dict, scope: str = "") -> None:
        """
        Write a response to the cache.

        Writes to SQLite first (if available), then to the local file
        for backward compatibility.

        Args:
            cache_key: The cache key (hex digest).
            data: The data dict to cache.
            scope: Cache scope (debate_id). Use ``"__global__"`` for
                cross-debate shared cache (file-only, no SQLite).
                Defaults to ``self._debate_id``.
        """
        effective_scope = scope or self._debate_id

        # --- Global scope: file-only (SQLite has FK on debate_id) ---
        if effective_scope == "__global__":
            global_dir = Path("data") / "shared" / "cache" / "legal"
            global_dir.mkdir(parents=True, exist_ok=True)
            cache_file = global_dir / f"{cache_key}.json"
            try:
                cache_file.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logger.debug("Global cache written to %s.", cache_file)
            except Exception as exc:
                logger.warning("Failed to write global cache %s: %s", cache_file, exc)
            return

        # --- SQLite primary ---
        if effective_scope:
            db = await _try_get_db()
            if db is not None:
                try:
                    from app.db.sqlite.cache import SQLiteCacheRepo
                    repo = SQLiteCacheRepo()
                    await repo.set(db, effective_scope, cache_key, data)
                except Exception:
                    logger.warning("SQLite cache set failed; file write will still proceed.", exc_info=True)

        # --- File fallback ---
        if not self._cache_dir:
            return

        cache_file = Path(self._cache_dir) / f"{cache_key}.json"
        try:
            cache_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.debug("Cached response to %s.", cache_file)
        except Exception as exc:
            logger.warning("Failed to write cache file %s: %s", cache_file, exc)

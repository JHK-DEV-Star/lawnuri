"""
PII anonymizer for legal texts.

Detects and masks sensitive information (names, phone numbers, ID numbers,
case numbers, emails, company names) before sending text to LLM APIs.
Supports round-trip de-anonymization via a mapping dict.

Mapping persistence supports dual-write: SQLite (primary) with JSON file fallback.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from app.utils.logger import logger


# ---------------------------------------------------------------------------
# Helper: obtain the SQLite connection outside of FastAPI DI
# ---------------------------------------------------------------------------

async def _try_get_db():
    """Return the active SQLite connection, or None if unavailable."""
    try:
        from app.db.database import get_db_connection
        return await get_db_connection()
    except Exception:
        return None


class Anonymizer:
    """Detect and replace PII tokens in Korean legal texts."""

    def __init__(self) -> None:
        """Initialize regex patterns and thread-safe counters."""

        # Thread-safe counters for generating unique tokens
        self._lock = threading.Lock()
        self._person_counter = 0
        self._company_counter = 0
        self._case_counter = 0

        # ------------------------------------------------------------------
        # Regex patterns (order matters -- more specific patterns first)
        # ------------------------------------------------------------------

        # Korean resident registration number: 6 digits - 7 digits
        self._re_id_number = re.compile(
            r"\b(\d{6})\s*[-]\s*(\d{7})\b"
        )

        # Phone numbers: 010-1234-5678, 02-123-4567, (02)123-4567, etc.
        self._re_phone = re.compile(
            r"(?:\(?\d{2,3}\)?\s*[-.]?\s*\d{3,4}\s*[-.]?\s*\d{4})"
        )

        # Email addresses
        self._re_email = re.compile(
            r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
        )

        # Case numbers (사건번호):
        # Patterns like 2024가합12345, 2023나4567, 2024고단123,
        # or with spaces/dashes: 2024 가합 12345
        self._re_case_number = re.compile(
            r"\d{4}\s*[가-힣]{1,4}\s*\d{1,6}"
        )

        # Company names: ending with 주식회사, (주), ㈜, or prefixed with 주식회사
        # e.g. 삼성전자 주식회사, (주)카카오, ㈜네이버, 주식회사 하이브
        self._re_company = re.compile(
            r"(?:"
            r"[가-힣A-Za-z0-9]+\s*(?:주식회사|유한회사|유한책임회사)"  # name + suffix
            r"|(?:주식회사|유한회사|유한책임회사)\s*[가-힣A-Za-z0-9]+"  # prefix + name
            r"|[(\(]주[)\)]\s*[가-힣A-Za-z0-9]+"                      # (주)name
            r"|㈜\s*[가-힣A-Za-z0-9]+"                                # ㈜name
            r"|[가-힣A-Za-z0-9]+\s*[(\(]주[)\)]"                      # name(주)
            r"|[가-힣A-Za-z0-9]+\s*㈜"                                # name㈜
            r")"
        )

        # Korean names: 2-4 Korean characters (applied last to reduce false positives).
        # We use a simple heuristic -- sequences of 2-4 Hangul syllables
        # preceded by a role indicator or surrounded by certain patterns.
        # To avoid matching every 2-char Korean word, we only match names
        # that appear near contextual cues (e.g. after role words).
        self._name_context_prefixes = [
            "원고", "피고", "피의자", "피해자", "고소인", "고발인",
            "증인", "참고인", "변호사", "판사", "검사", "대리인",
            "채권자", "채무자", "신청인", "피신청인", "상고인", "피상고인",
            "항소인", "피항소인", "선정자", "이해관계인",
        ]
        self._re_korean_name_contextual = re.compile(
            r"(?:"
            + "|".join(re.escape(p) for p in self._name_context_prefixes)
            + r")\s*([가-힣]{2,4})"
        )

        # Standalone Korean name pattern (fallback, less precise).
        # Matches common Korean surname + given name structures.
        _common_surnames = (
            "김|이|박|최|정|강|조|윤|장|임|한|오|서|신|권|황|안|송|류|유|"
            "전|홍|고|문|양|손|배|백|허|남|심|노|하|곽|성|차|주|우|구|"
            "민|진|나|지|엄|채|원|천|방|공|현|함|변|염|석|선|설|마|길|"
            "연|위|표|명|기|반|왕|금|옥|육|인|맹|제|탁|남궁|사공|독고|"
            "동방|황보|제갈|선우|사|소|시|범|어|둥|경|봉|비|용|예|편|"
            "도|수|온|랑|피|감|태|추|팽|섭"
        )
        self._re_korean_name_standalone = re.compile(
            r"(?<![가-힣])(" + f"(?:{_common_surnames})" + r"[가-힣]{1,3})(?![가-힣])"
        )

        logger.info("Anonymizer initialized with PII detection patterns.")

    # ------------------------------------------------------------------
    # Token generation (thread-safe)
    # ------------------------------------------------------------------

    def _next_person_token(self) -> str:
        with self._lock:
            self._person_counter += 1
            idx = self._person_counter
        if idx <= 26:
            return f"Person_{chr(64 + idx)}"  # Person_A, Person_B, ...
        return f"Person_{idx}"

    def _next_company_token(self) -> str:
        with self._lock:
            self._company_counter += 1
            idx = self._company_counter
        if idx <= 26:
            return f"Company_{chr(64 + idx)}"  # Company_A, Company_B, ...
        return f"Company_{idx}"

    def _next_case_token(self) -> str:
        with self._lock:
            self._case_counter += 1
            idx = self._case_counter
        return f"Case_{idx:03d}"

    # ------------------------------------------------------------------
    # Core anonymization
    # ------------------------------------------------------------------

    def anonymize(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Anonymize PII in *text* and return the cleaned text plus a mapping.

        The mapping allows later de-anonymization via :meth:`deanonymize`.

        Detected PII categories:
            - Korean names (contextual and surname-based)
            - Company names (주식회사 / ㈜ / (주) patterns)
            - Phone numbers
            - Resident registration numbers (주민등록번호)
            - Case numbers (사건번호, e.g. 2024가합12345)
            - Email addresses

        Args:
            text: Raw input text, potentially containing PII.

        Returns:
            A tuple of ``(anonymized_text, mapping)`` where *mapping* is
            ``{token: original_value}``.
        """
        # We build a *reverse* lookup (original -> token) to ensure the
        # same original value always gets the same token within one call.
        original_to_token: dict[str, str] = {}
        mapping: dict[str, str] = {}  # token -> original

        def _get_or_create_token(original: str, token_factory) -> str:
            """Return existing token for *original* or create a new one."""
            if original in original_to_token:
                return original_to_token[original]
            token = token_factory()
            original_to_token[original] = token
            mapping[token] = original
            return token

        result = text

        # 1. Resident registration numbers -> [ID_NUM]
        def _replace_id(m: re.Match) -> str:
            return "[ID_NUM]"

        result = self._re_id_number.sub(_replace_id, result)
        # Track ID_NUM in mapping as a fixed key (not unique per value for privacy)
        if "[ID_NUM]" in result and "[ID_NUM]" not in mapping:
            for m in self._re_id_number.finditer(text):
                mapping.setdefault("[ID_NUM]", m.group(0))
                break

        # 2. Email addresses -> [EMAIL]
        emails_found: list[str] = self._re_email.findall(text)
        for email in emails_found:
            if email in original_to_token:
                continue
            token = "[EMAIL]"
            original_to_token[email] = token
            mapping.setdefault(token, email)
        result = self._re_email.sub("[EMAIL]", result)

        # 3. Phone numbers -> [PHONE]
        phones_found: list[str] = self._re_phone.findall(text)
        for phone in phones_found:
            if phone in original_to_token:
                continue
            token = "[PHONE]"
            original_to_token[phone] = token
            mapping.setdefault(token, phone)
        result = self._re_phone.sub("[PHONE]", result)

        # 4. Case numbers -> Case_001, Case_002, ...
        def _replace_case(m: re.Match) -> str:
            original = m.group(0)
            return _get_or_create_token(original, self._next_case_token)

        result = self._re_case_number.sub(_replace_case, result)

        # 5. Company names -> Company_A, Company_B, ...
        def _replace_company(m: re.Match) -> str:
            original = m.group(0).strip()
            return _get_or_create_token(original, self._next_company_token)

        result = self._re_company.sub(_replace_company, result)

        # 6. Korean names (contextual -- after role keywords)
        def _replace_name_ctx(m: re.Match) -> str:
            name = m.group(1)
            token = _get_or_create_token(name, self._next_person_token)
            # Keep the prefix (role keyword), only replace the name part
            prefix = m.group(0)[: m.start(1) - m.start(0)]
            return prefix + token

        result = self._re_korean_name_contextual.sub(_replace_name_ctx, result)

        # 7. Korean names (standalone surname-based, fallback)
        def _replace_name_standalone(m: re.Match) -> str:
            name = m.group(1)
            return _get_or_create_token(name, self._next_person_token)

        result = self._re_korean_name_standalone.sub(_replace_name_standalone, result)

        logger.debug(
            "Anonymized text: %d PII tokens replaced (%d unique mappings).",
            sum(result.count(t) for t in mapping),
            len(mapping),
        )

        return result, mapping

    # ------------------------------------------------------------------
    # De-anonymization
    # ------------------------------------------------------------------

    def deanonymize(self, text: str, mapping: dict[str, str]) -> str:
        """
        Restore original values from an anonymized text using *mapping*.

        Args:
            text: Anonymized text containing placeholder tokens.
            mapping: ``{token: original_value}`` dict produced by :meth:`anonymize`.

        Returns:
            Text with all known tokens replaced by their original values.
        """
        result = text
        # Sort tokens by length descending to avoid partial replacements
        # (e.g. replacing "Person_A" before "Person_AB").
        for token in sorted(mapping, key=len, reverse=True):
            result = result.replace(token, mapping[token])

        logger.debug("De-anonymized text using %d mapping entries.", len(mapping))
        return result

    # ------------------------------------------------------------------
    # Mapping persistence
    # ------------------------------------------------------------------

    async def save_mapping(
        self,
        mapping: dict[str, str],
        path: str,
        debate_id: str = "",
    ) -> None:
        """
        Save a mapping dict.

        Writes to SQLite first (if available), then to the JSON file
        for backward compatibility.

        Args:
            mapping: Token-to-original mapping to persist.
            path: Destination file path.
            debate_id: Debate identifier for SQLite scoping.
        """
        # --- SQLite primary ---
        if debate_id:
            db = await _try_get_db()
            if db is not None:
                try:
                    from app.db.sqlite.cache import SQLiteAnonymizationRepo
                    repo = SQLiteAnonymizationRepo()
                    await repo.save(db, debate_id, mapping)
                    logger.info("Anonymization mapping saved to SQLite (debate=%s, %d entries).", debate_id, len(mapping))
                except Exception:
                    logger.warning("SQLite anonymization save failed; falling back to file.", exc_info=True)

        # --- File fallback ---
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(mapping, fh, ensure_ascii=False, indent=2)
        logger.info("Anonymization mapping saved to %s (%d entries).", path, len(mapping))

    async def load_mapping(
        self,
        path: str,
        debate_id: str = "",
    ) -> dict[str, str]:
        """
        Load a mapping dict.

        Tries SQLite first, then falls back to the JSON file.

        Args:
            path: Path to the JSON mapping file.
            debate_id: Debate identifier for SQLite scoping.

        Returns:
            The loaded ``{token: original_value}`` dict.
        """
        # --- SQLite primary ---
        if debate_id:
            db = await _try_get_db()
            if db is not None:
                try:
                    from app.db.sqlite.cache import SQLiteAnonymizationRepo
                    repo = SQLiteAnonymizationRepo()
                    result = await repo.load(db, debate_id)
                    if result:
                        logger.info("Anonymization mapping loaded from SQLite (debate=%s, %d entries).", debate_id, len(result))
                        return result
                except Exception:
                    logger.warning("SQLite anonymization load failed; falling back to file.", exc_info=True)

        # --- File fallback ---
        with open(path, "r", encoding="utf-8") as fh:
            mapping: dict[str, str] = json.load(fh)
        logger.info("Anonymization mapping loaded from %s (%d entries).", path, len(mapping))
        return mapping

    # ------------------------------------------------------------------
    # Counter reset (useful for tests)
    # ------------------------------------------------------------------

    def reset_counters(self) -> None:
        """Reset all token counters to zero."""
        with self._lock:
            self._person_counter = 0
            self._company_counter = 0
            self._case_counter = 0
        logger.debug("Anonymizer counters reset.")

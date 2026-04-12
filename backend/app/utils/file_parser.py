"""
File parser for extracting text from PDF, Markdown, and plain text files.
Supports automatic encoding detection and smart text chunking at sentence boundaries.
"""

import re
from pathlib import Path
from typing import List, Optional

from app.utils.logger import logger


class FileParser:
    """
    Utility class for extracting text content from various file formats.
    Supports PDF, Markdown (.md), and plain text (.txt) files.
    """

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}

    @classmethod
    def extract_text(cls, file_path: str | Path) -> str:
        """
        Extract text content from a file.

        Automatically detects the file type by extension and uses the
        appropriate extraction method.

        Args:
            file_path: Path to the file to parse.

        Returns:
            Extracted text content as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: '{ext}'. "
                f"Supported: {cls.SUPPORTED_EXTENSIONS}"
            )

        logger.info("[FileParser] Extracting text from: %s", path.name)

        if ext == ".pdf":
            return cls._extract_pdf(path)
        else:
            # .md and .txt are both plain text
            return cls._extract_text_file(path)

    @classmethod
    def read_pdf_bytes(cls, file_path: str | Path) -> bytes:
        """Read PDF as raw bytes for multimodal LLM pass-through."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        data = path.read_bytes()
        logger.info("[FileParser] Read PDF as bytes: %s (%d bytes)", path.name, len(data))
        return data

    @classmethod
    def _extract_pdf(cls, path: Path) -> str:
        """
        Extract text from a PDF file using PyMuPDF (fitz).

        Args:
            path: Path to the PDF file.

        Returns:
            Extracted text from all pages.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF parsing. "
                "Install it with: pip install PyMuPDF"
            )

        text_parts: List[str] = []

        with fitz.open(str(path)) as doc:
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text("text")
                if page_text.strip():
                    text_parts.append(page_text)
                    logger.debug(
                        "[FileParser] PDF page %d: %d chars",
                        page_num,
                        len(page_text),
                    )

        result = "\n".join(text_parts)
        logger.info(
            "[FileParser] PDF extraction complete: %d pages, %d chars",
            len(text_parts),
            len(result),
        )
        return result

    @classmethod
    def _extract_text_file(cls, path: Path) -> str:
        """
        Extract text from a plain text or markdown file.
        Uses charset_normalizer for encoding detection with chardet as fallback.

        Args:
            path: Path to the text file.

        Returns:
            File contents as a string.
        """
        raw_bytes = path.read_bytes()

        # Try UTF-8 first (most common)
        try:
            text = raw_bytes.decode("utf-8")
            logger.debug("[FileParser] Decoded %s as UTF-8", path.name)
            return text
        except UnicodeDecodeError:
            pass

        # Try charset_normalizer for automatic detection
        encoding = cls._detect_encoding(raw_bytes)
        if encoding:
            try:
                text = raw_bytes.decode(encoding)
                logger.debug(
                    "[FileParser] Decoded %s as %s (auto-detected)",
                    path.name,
                    encoding,
                )
                return text
            except (UnicodeDecodeError, LookupError):
                pass

        # Last resort: latin-1 (never fails)
        logger.warning(
            "[FileParser] Falling back to latin-1 for %s", path.name
        )
        return raw_bytes.decode("latin-1")

    @staticmethod
    def _detect_encoding(raw_bytes: bytes) -> Optional[str]:
        """
        Detect the encoding of raw bytes using charset_normalizer,
        falling back to chardet if unavailable.

        Args:
            raw_bytes: The raw file bytes.

        Returns:
            Detected encoding name or None.
        """
        # Try charset_normalizer first
        try:
            from charset_normalizer import from_bytes

            result = from_bytes(raw_bytes).best()
            if result is not None:
                return result.encoding
        except ImportError:
            pass

        # Fallback to chardet
        try:
            import chardet

            detection = chardet.detect(raw_bytes)
            if detection and detection.get("encoding"):
                return detection["encoding"]
        except ImportError:
            logger.warning(
                "[FileParser] Neither charset_normalizer nor chardet is installed. "
                "Install one for better encoding detection: "
                "pip install charset-normalizer"
            )

        return None


# Sentence-ending pattern for smart splitting
_SENTENCE_END_RE = re.compile(r"[.!?。！？]\s+")


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[str]:
    """
    Split text into overlapping chunks with smart sentence boundary splitting.

    Tries to break chunks at sentence boundaries to avoid cutting mid-sentence.
    Falls back to character-level splitting if no sentence boundary is found
    within the chunk.

    Args:
        text: The text to split.
        chunk_size: Target maximum size of each chunk in characters.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            # Last chunk: take everything remaining
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Try to find a sentence boundary near the end of the chunk
        # Search in the last 20% of the chunk for a good break point
        search_start = start + int(chunk_size * 0.8)
        search_region = text[search_start:end]

        # Find the last sentence-ending punctuation in the search region
        best_break = None
        for match in _SENTENCE_END_RE.finditer(search_region):
            best_break = search_start + match.end()

        if best_break is not None:
            chunk = text[start:best_break].strip()
        else:
            # No sentence boundary found; try to break at a space
            space_pos = text.rfind(" ", start + int(chunk_size * 0.8), end)
            if space_pos > start:
                chunk = text[start:space_pos].strip()
                best_break = space_pos + 1
            else:
                # No space found; hard break at chunk_size
                chunk = text[start:end].strip()
                best_break = end

        if chunk:
            chunks.append(chunk)

        # Move start forward, accounting for overlap
        start = best_break - overlap if best_break else end - overlap
        # Ensure we always move forward
        if start <= (best_break - chunk_size if best_break else end - chunk_size):
            start = best_break if best_break else end

    logger.debug(
        "[split_text] Split %d chars into %d chunks (chunk_size=%d, overlap=%d)",
        len(text),
        len(chunks),
        chunk_size,
        overlap,
    )

    return chunks

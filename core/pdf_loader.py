"""PDF text extraction module for Athena AI Lite.

This module provides safe, page-aware text extraction from PDF files using PyMuPDF (fitz).
Key educational aspects:
- Extracts text page-by-page to maintain precise document provenance.
- Preserves 1-based page numbers for human-friendly citations.
- Identifies unreadable/scanned PDFs where text extraction yields no content,
  explicitly notifying the user that OCR is not supported in this lightweight version.
- Handles corrupted files, password-protected documents, and empty pages cleanly.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union
import pymupdf as fitz


class PDFExtractionError(Exception):
    """Custom exception raised when PDF loading or extraction fails."""
    pass


@dataclass
class ExtractedPage:
    """Represents text and metadata extracted from an individual PDF page."""
    page_number: int  # 1-indexed for human readability
    text: str
    char_count: int
    is_empty: bool = False


@dataclass
class PDFDocument:
    """Represents a fully parsed PDF with page-level extractions and summary stats."""
    filename: str
    total_pages: int
    pages: List[ExtractedPage] = field(default_factory=list)
    total_characters: int = 0
    has_extractable_text: bool = False
    error_message: Optional[str] = None

    def get_text_preview(self, max_chars: int = 500) -> str:
        """Returns a trimmed preview of the extracted text across pages."""
        for page in self.pages:
            stripped = page.text.strip()
            if stripped:
                preview = stripped[:max_chars]
                if len(stripped) > max_chars:
                    preview += "..."
                return preview
        return "No text available."

    def get_full_text(self) -> str:
        """Concatenates all pages with clear page separation headers."""
        return "\n\n".join(
            f"--- Page {p.page_number} ---\n{p.text}"
            for p in self.pages
            if not p.is_empty
        )


def extract_text_from_pdf(
    file_bytes: bytes,
    filename: str = "document.pdf"
) -> PDFDocument:
    """Extracts text page-by-page from raw PDF bytes.

    Args:
        file_bytes: Raw binary content of the PDF file.
        filename: Original name of the uploaded file for attribution.

    Returns:
        PDFDocument containing page text, stats, and extractability status.

    Raises:
        PDFExtractionError: If the file is invalid, encrypted, or corrupted.
    """
    if not file_bytes:
        raise PDFExtractionError(f"Cannot extract text: '{filename}' is completely empty (0 bytes).")

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise PDFExtractionError(
            f"Failed to open '{filename}'. The file may be corrupted or not a valid PDF: {e}"
        )

    try:
        if doc.is_encrypted:
            raise PDFExtractionError(
                f"Document '{filename}' is password-protected or encrypted. "
                "Please decrypt the PDF before uploading."
            )

        total_pages = len(doc)
        if total_pages == 0:
            raise PDFExtractionError(f"Document '{filename}' contains 0 pages.")

        pages: List[ExtractedPage] = []
        total_chars = 0

        for page_idx in range(total_pages):
            page_num = page_idx + 1  # 1-indexed
            page = doc[page_idx]
            # "text" flag extracts plain textual characters layout-order
            text = page.get_text("text") or ""
            clean_text = text.strip()
            char_count = len(clean_text)
            total_chars += char_count

            pages.append(
                ExtractedPage(
                    page_number=page_num,
                    text=clean_text,
                    char_count=char_count,
                    is_empty=(char_count == 0)
                )
            )

        has_text = total_chars > 0

        pdf_doc = PDFDocument(
            filename=filename,
            total_pages=total_pages,
            pages=pages,
            total_characters=total_chars,
            has_extractable_text=has_text,
            error_message=None if has_text else (
                f"No extractable text found in '{filename}' across {total_pages} page(s). "
                "This document appears to be a scanned image or raster PDF. "
                "Athena AI Lite v1 does not include OCR (Optical Character Recognition). "
                "Please upload a digital PDF with selectable text."
            )
        )

        return pdf_doc

    finally:
        doc.close()


def extract_text_from_pdf_path(file_path: Union[str, Path]) -> PDFDocument:
    """Extracts text page-by-page from a file on disk."""
    path = Path(file_path)
    if not path.exists():
        raise PDFExtractionError(f"File not found: {path}")
    if not path.is_file():
        raise PDFExtractionError(f"Path is not a regular file: {path}")

    try:
        file_bytes = path.read_bytes()
    except Exception as e:
        raise PDFExtractionError(f"Error reading file '{path.name}': {e}")

    return extract_text_from_pdf(file_bytes=file_bytes, filename=path.name)

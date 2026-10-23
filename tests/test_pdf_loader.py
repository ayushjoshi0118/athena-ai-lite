"""Unit tests for the PDF loader module in Athena AI Lite.

Verifies:
- Safe extraction of text page by page from synthetic PDF fixtures.
- 1-based page numbering preservation.
- Identification of empty pages.
- Flagging scanned / raster documents lacking extractable text.
- Clean error reporting on corrupted or 0-byte input.
"""

import pymupdf as fitz
import pytest
from core.pdf_loader import PDFDocument, PDFExtractionError, extract_text_from_pdf


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Creates an in-memory 2-page synthetic PDF with selectable text."""
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((50, 72), "Athena University Department of Computer Science.")
    page1.insert_text((50, 100), "Course: CS101 Introduction to Artificial Intelligence.")

    page2 = doc.new_page()
    page2.insert_text((50, 72), "Grading Policy: Midterm 30%, Final Project 40%, Homework 30%.")

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def empty_pdf_bytes() -> bytes:
    """Creates a synthetic PDF with 1 page containing no text (simulating a scanned doc)."""
    doc = fitz.open()
    doc.new_page()  # Blank page without text
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_extract_text_success(sample_pdf_bytes):
    """Should correctly extract pages, character counts, and text content."""
    doc: PDFDocument = extract_text_from_pdf(sample_pdf_bytes, filename="cs101_syllabus.pdf")

    assert doc.filename == "cs101_syllabus.pdf"
    assert doc.total_pages == 2
    assert doc.has_extractable_text is True
    assert doc.total_characters > 0
    assert doc.error_message is None

    # Verify Page 1
    p1 = doc.pages[0]
    assert p1.page_number == 1
    assert "Athena University" in p1.text
    assert "CS101" in p1.text
    assert p1.is_empty is False

    # Verify Page 2
    p2 = doc.pages[1]
    assert p2.page_number == 2
    assert "Grading Policy" in p2.text
    assert p2.is_empty is False


def test_extract_text_scanned_or_empty(empty_pdf_bytes):
    """Scanned or empty PDFs should flag has_extractable_text as False with helpful message."""
    doc: PDFDocument = extract_text_from_pdf(empty_pdf_bytes, filename="scanned_notes.pdf")

    assert doc.filename == "scanned_notes.pdf"
    assert doc.total_pages == 1
    assert doc.has_extractable_text is False
    assert doc.total_characters == 0
    assert doc.error_message is not None
    assert "OCR" in doc.error_message
    assert "scanned image" in doc.error_message


def test_extract_text_zero_bytes():
    """Empty byte array must raise PDFExtractionError."""
    with pytest.raises(PDFExtractionError, match="completely empty"):
        extract_text_from_pdf(b"", filename="empty.pdf")


def test_extract_text_corrupt_bytes():
    """Corrupted non-PDF bytes must raise PDFExtractionError."""
    with pytest.raises(PDFExtractionError, match="Failed to open"):
        extract_text_from_pdf(b"This is definitely not a valid PDF file stream", filename="corrupt.pdf")

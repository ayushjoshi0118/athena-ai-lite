"""Unit tests for the chunker module in Athena AI Lite.

Verifies:
- Configurable chunk size and overlap boundaries.
- Empty text handling.
- Short document handling (no unnecessary splits).
- Page number and document metadata preservation.
- Error checking for invalid chunking hyperparameters.
"""

import pytest
from core.chunker import TextChunk, chunk_document, chunk_text
from core.pdf_loader import ExtractedPage, PDFDocument


def test_chunk_text_empty():
    """Empty or whitespace-only text should return an empty list of chunks."""
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_chunk_text_short_document():
    """Text shorter than chunk_size should produce exactly one chunk."""
    text = "Artificial Intelligence at Athena University covers algorithms and ethics."
    chunks = chunk_text(text, doc_name="syllabus.pdf", page_number=1, chunk_size=200, chunk_overlap=20)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.doc_name == "syllabus.pdf"
    assert chunk.page_number == 1
    assert chunk.text == text
    assert chunk.char_count == len(text)
    assert chunk.start_char == 0
    assert chunk.end_char == len(text)


def test_chunk_text_overlap():
    """Longer text should produce overlapping chunks that preserve boundary content."""
    # A text of 60 words
    words = [f"word{i}" for i in range(1, 61)]
    text = " ".join(words)

    chunk_size = 100
    chunk_overlap = 20

    chunks = chunk_text(text, doc_name="lecture.pdf", page_number=2, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    assert len(chunks) > 1

    # Check that consecutive chunks have overlapping text
    for i in range(len(chunks) - 1):
        c1 = chunks[i]
        c2 = chunks[i + 1]

        # The end of c1 should share words with the beginning of c2
        assert c2.start_char < c1.end_char
        overlap_slice = text[c2.start_char:c1.end_char]
        assert len(overlap_slice) > 0


def test_chunk_text_invalid_parameters():
    """Invalid chunk size or overlap parameters must raise ValueError."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("test", chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        chunk_text("test", chunk_size=100, chunk_overlap=-10)

    with pytest.raises(ValueError, match="must be strictly less than"):
        chunk_text("test", chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="must be strictly less than"):
        chunk_text("test", chunk_size=100, chunk_overlap=120)


def test_chunk_document_page_provenance():
    """Document chunking must preserve page numbers and document names across pages."""
    pages = [
        ExtractedPage(page_number=1, text="Page one introduces RAG concepts.", char_count=32),
        ExtractedPage(page_number=2, text="", char_count=0, is_empty=True),  # empty page
        ExtractedPage(page_number=3, text="Page three discusses vector similarity metrics.", char_count=48),
    ]
    doc = PDFDocument(
        filename="rag_lecture.pdf",
        total_pages=3,
        pages=pages,
        total_characters=80,
        has_extractable_text=True
    )

    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)

    # Empty page 2 should be skipped
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[0].doc_name == "rag_lecture.pdf"
    assert "Page one introduces" in chunks[0].text

    assert chunks[1].page_number == 3
    assert chunks[1].doc_name == "rag_lecture.pdf"
    assert "Page three discusses" in chunks[1].text


def test_chunk_document_unextractable():
    """A document marked without extractable text must produce 0 chunks."""
    doc = PDFDocument(
        filename="scanned.pdf",
        total_pages=2,
        pages=[],
        total_characters=0,
        has_extractable_text=False
    )
    assert chunk_document(doc) == []

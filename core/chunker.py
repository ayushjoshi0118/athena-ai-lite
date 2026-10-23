"""Text chunking module for Athena AI Lite.

EDUCATIONAL CONCEPT: Why do we chunk documents?
1. Embedding Specificity:
   If you embed an entire 20-page document into a single vector, all individual
   facts blur together into a generic average. A query about "final exam dates"
   will get lost in the noise of administrative boilerplate.
2. Context Window & Relevance:
   Vector retrieval fetches the top-k most semantically relevant passages. Smaller,
   coherent chunks allow the retriever to return only the specific paragraphs needed
   to answer the question, keeping prompt costs low and LLM focus high.
3. The Role of Overlap:
   If a vital definition or fact happens to straddle the boundary between chunk N
   and chunk N+1, splitting cleanly without overlap cuts the idea in half. Overlapping
   a fixed number of characters (e.g., 50-100 chars) ensures phrases bridging boundaries
   remain intact in at least one chunk.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from core.pdf_loader import PDFDocument, ExtractedPage


@dataclass
class TextChunk:
    """Represents an atomic, searchable fragment of document text with provenance."""
    chunk_id: str
    doc_name: str
    page_number: int  # 1-indexed source page
    text: str
    start_char: int
    end_char: int
    char_count: int = field(init=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.char_count = len(self.text)
        if not self.metadata:
            self.metadata = {
                "chunk_id": self.chunk_id,
                "doc_name": self.doc_name,
                "page_number": self.page_number,
                "char_count": self.char_count,
            }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def chunk_text(
    text: str,
    doc_name: str = "document",
    page_number: int = 1,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    id_prefix: Optional[str] = None
) -> List[TextChunk]:
    """Splits raw text into overlapping chunks while preserving word boundaries.

    Args:
        text: Input string to chunk.
        doc_name: Name of source document for citations.
        page_number: Source page number for citations.
        chunk_size: Target character length per chunk (must be > 0).
        chunk_overlap: Number of overlapping characters between consecutive chunks.
        id_prefix: Optional string prefix for chunk IDs.

    Returns:
        List of TextChunk instances.

    Raises:
        ValueError: If chunk_size <= 0 or chunk_overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    clean_text = text.strip()
    if not clean_text:
        return []

    prefix = id_prefix or f"{doc_name}_p{page_number}"
    chunks: List[TextChunk] = []

    # If the text is shorter than or equal to the target size, keep it as a single chunk
    if len(clean_text) <= chunk_size:
        chunks.append(
            TextChunk(
                chunk_id=f"{prefix}_c0",
                doc_name=doc_name,
                page_number=page_number,
                text=clean_text,
                start_char=0,
                end_char=len(clean_text)
            )
        )
        return chunks

    step = chunk_size - chunk_overlap
    start = 0
    text_len = len(clean_text)
    chunk_idx = 0

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Snap to word boundary if not at the very end of the text
        if end < text_len:
            # Look backwards up to 40 characters for a whitespace or newline
            boundary = clean_text.rfind(" ", start, end)
            newline_boundary = clean_text.rfind("\n", start, end)
            split_point = max(boundary, newline_boundary)

            # Only snap if the split point isn't too close to start
            if split_point > start + (chunk_size // 3):
                end = split_point

        chunk_slice = clean_text[start:end].strip()
        if chunk_slice:
            chunks.append(
                TextChunk(
                    chunk_id=f"{prefix}_c{chunk_idx}",
                    doc_name=doc_name,
                    page_number=page_number,
                    text=chunk_slice,
                    start_char=start,
                    end_char=end
                )
            )
            chunk_idx += 1

        # Advance start position by step, or break if we reached the end
        if end >= text_len:
            break
        start += step

    return chunks


def chunk_document(
    document: PDFDocument,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[TextChunk]:
    """Splits all pages of a PDFDocument into chunks, preserving page provenance.

    Args:
        document: PDFDocument instance produced by pdf_loader.
        chunk_size: Target character length per chunk.
        chunk_overlap: Overlapping characters between adjacent chunks.

    Returns:
        List of TextChunk instances across all pages.
    """
    if not document.has_extractable_text or not document.pages:
        return []

    all_chunks: List[TextChunk] = []
    global_chunk_idx = 0

    for page in document.pages:
        if page.is_empty or not page.text:
            continue

        page_chunks = chunk_text(
            text=page.text,
            doc_name=document.filename,
            page_number=page.page_number,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            id_prefix=f"{document.filename}_p{page.page_number}"
        )

        for chunk in page_chunks:
            # Assign clean sequential IDs while preserving page metadata
            chunk.chunk_id = f"chunk_{global_chunk_idx:03d}_p{chunk.page_number}"
            chunk.metadata["chunk_id"] = chunk.chunk_id
            all_chunks.append(chunk)
            global_chunk_idx += 1

    return all_chunks

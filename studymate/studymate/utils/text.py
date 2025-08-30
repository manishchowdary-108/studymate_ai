import re
from typing import List


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks by characters, attempting to break on sentence boundaries.

    This is a lightweight approach that avoids extra heavy dependencies while
    producing high-quality chunks for semantic search.
    """
    if not text:
        return []

    text = normalize_whitespace(text)

    # Prefer splitting on sentences, then re-joining to reach desired sizes
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []

    current: List[str] = []
    current_len = 0
    for sent in sentences:
        if not sent:
            continue
        if current_len + len(sent) + 1 <= chunk_size:
            current.append(sent)
            current_len += len(sent) + 1
        else:
            if current:
                chunks.append(" ".join(current))

            # Start a new chunk with overlap from previous
            if overlap > 0 and chunks:
                overlap_source = chunks[-1]
                overlap_text = overlap_source[-overlap:]
                current = [overlap_text, sent]
                current_len = len(overlap_text) + len(sent) + 1
            else:
                current = [sent]
                current_len = len(sent) + 1

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]

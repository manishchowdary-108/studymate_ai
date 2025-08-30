from __future__ import annotations

import os
from typing import Dict, Iterable, List

import fitz  # PyMuPDF

from studymate.utils.text import normalize_whitespace, chunk_text


def extract_pages_from_pdf(pdf_path: str) -> List[Dict]:
    """Extract text by page from a single PDF.

    Returns a list of dicts with keys: text, metadata.
    metadata includes: doc_path, doc_name, page_number (1-based)
    """
    pages: List[Dict] = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = normalize_whitespace(text)
            pages.append(
                {
                    "text": text,
                    "metadata": {
                        "doc_path": pdf_path,
                        "doc_name": os.path.basename(pdf_path),
                        "page_number": page_index,
                    },
                }
            )
    return pages


def extract_and_chunk_pdfs(
    pdf_paths: Iterable[str],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[Dict]:
    """Extract all text from a collection of PDFs and split into chunks.

    Returns a list of dicts with keys: text, metadata
    metadata extends with chunk_id per page.
    """
    all_chunks: List[Dict] = []
    for path in pdf_paths:
        for page in extract_pages_from_pdf(path):
            chunks = chunk_text(page["text"], chunk_size=chunk_size, overlap=overlap)
            for chunk_idx, chunk in enumerate(chunks):
                meta = dict(page["metadata"])  # copy
                meta["chunk_id"] = chunk_idx
                all_chunks.append({"text": chunk, "metadata": meta})
    return all_chunks

import os
import json
from pathlib import Path
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv

from studymate.core.pdf import extract_and_chunk_pdfs
from studymate.core.embeddings import build_faiss_index, FaissStore, search
from studymate.models.watsonx import generate_answer


load_dotenv()


def get_data_dir() -> Path:
    return Path(os.getenv("STUDYMATE_DATA_DIR", "studymate/data")).resolve()


def save_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[str]:
    target_dir = get_data_dir() / "uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []
    for f in uploaded_files:
        out_path = target_dir / f.name
        with open(out_path, "wb") as out:
            out.write(f.read())
        saved_paths.append(str(out_path))
    return saved_paths


def build_index_view():
    st.subheader("Build Knowledge Index")
    uploaded = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        chunk_size = st.number_input("Chunk size", min_value=256, max_value=4000, value=int(os.getenv("CHUNK_SIZE", "1000")), step=64)
    with col2:
        overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=int(os.getenv("CHUNK_OVERLAP", "200")), step=16)
    with col3:
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=int(os.getenv("TOP_K", "5")), step=1)

    if st.button("Build Index", type="primary"):
        if not uploaded:
            st.warning("Please upload at least one PDF.")
            return
        saved_paths = save_uploaded_files(uploaded)
        with st.status("Extracting and chunking PDFs...", expanded=True):
            chunks = extract_and_chunk_pdfs(saved_paths, chunk_size=chunk_size, overlap=overlap)
            st.write(f"Extracted {len(chunks)} chunks")

        texts = [c["text"] for c in chunks]
        metas: List[Dict] = [c["metadata"] for c in chunks]

        with st.status("Building embeddings and FAISS index...", expanded=True):
            store = build_faiss_index(texts, metas)
            index_dir = get_data_dir() / "index"
            store.save(str(index_dir))
            st.write(f"Saved index to {index_dir}")

        st.success("Index built successfully. You can now ask questions.")

    st.session_state["top_k"] = int(top_k)


def qa_view():
    st.subheader("Ask Questions")
    index_dir = get_data_dir() / "index"
    if not (index_dir / "index.faiss").exists():
        st.info("No index found. Build the index first in the 'Build Index' tab.")
        return

    # Lazy import to keep startup fast
    from studymate.core.embeddings import FaissStore

    store = FaissStore.load(str(index_dir))
    question = st.text_input("Your question", placeholder="e.g., What is gradient descent?")
    if st.button("Search", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            return
        results = search(store, question, top_k=int(st.session_state.get("top_k", 5)))
        retrieved = []
        for idx, score in results:
            retrieved.append({
                "text": store.texts[idx],
                "metadata": store.metadatas[idx],
                "score": float(score),
            })

        st.write("Retrieved context:")
        for i, r in enumerate(retrieved, start=1):
            meta = r["metadata"]
            label = f"{meta['doc_name']} p.{meta['page_number']} (chunk {meta['chunk_id']}) — score {r['score']:.3f}"
            with st.expander(label):
                st.write(r["text"])

        # Compose LLM prompt
        context_lines = []
        for r in retrieved:
            m = r["metadata"]
            context_lines.append(
                f"Source: {m['doc_name']} p.{m['page_number']} chunk {m['chunk_id']}\n{r['text']}"
            )
        context = "\n\n".join(context_lines)
        prompt = (
            "You are StudyMate, a helpful academic assistant. "
            "Answer the question using ONLY the provided sources. "
            "Cite sources inline as (doc p.page).\n\n"
            f"Question: {question}\n\n"
            f"Sources:\n{context}\n\n"
            "Answer:"
        )

        with st.status("Generating answer with IBM watsonx...", expanded=True):
            answer = generate_answer(prompt)
            st.write("Model response ready.")

        st.markdown("**Answer**")
        st.write(answer)


def main():
    st.set_page_config(page_title="StudyMate", layout="wide")
    st.title("StudyMate — PDF Q&A")
    st.caption("Upload PDFs, build an index, and ask grounded questions.")

    tab1, tab2 = st.tabs(["Build Index", "Ask Questions"])
    with tab1:
        build_index_view()
    with tab2:
        qa_view()


if __name__ == "__main__":
    main()

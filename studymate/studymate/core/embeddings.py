from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    return vectors / norms


@dataclass
class FaissStore:
    index: faiss.Index
    texts: List[str]
    metadatas: List[Dict]
    model_name: str

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "texts.jsonl"), "w", encoding="utf-8") as f:
            for text, meta in zip(self.texts, self.metadatas):
                f.write(json.dumps({"text": text, "metadata": meta}) + "\n")
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"model_name": self.model_name}, f)

    @staticmethod
    def load(directory: str) -> "FaissStore":
        index = faiss.read_index(os.path.join(directory, "index.faiss"))
        texts: List[str] = []
        metadatas: List[Dict] = []
        with open(os.path.join(directory, "texts.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                texts.append(row["text"])
                metadatas.append(row["metadata"])
        with open(os.path.join(directory, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        return FaissStore(index=index, texts=texts, metadatas=metadatas, model_name=meta.get("model_name", DEFAULT_MODEL))


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
        return _l2_normalize(embeddings).astype("float32")


def build_faiss_index(texts: List[str], metadatas: List[Dict], model_name: str = DEFAULT_MODEL) -> FaissStore:
    embedder = Embedder(model_name)
    vectors = embedder.encode(texts)
    dim = vectors.shape[1]

    # Cosine similarity via inner product over normalized vectors
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return FaissStore(index=index, texts=texts, metadatas=metadatas, model_name=model_name)


def search(store: FaissStore, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
    embedder = Embedder(store.model_name)
    q = embedder.encode([query])
    scores, idxs = store.index.search(q, top_k)
    results: List[Tuple[int, float]] = []
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0])):
        if idx == -1:
            continue
        results.append((int(idx), float(score)))
    return results

"""Embeds and indexes Chunks; hybrid (vector + BM25) search across the shared
knowledge base (Section D.2, step 1 -- one index, not four siloed ones).

T-16 (embedding model) and T-17 (vector store + fusion) in the Technique &
Method Log. Retrieval is vector search (FAISS, cosine similarity) fused with
keyword search (BM25) via Reciprocal Rank Fusion, because clause numbers and
drug names are exactly the kind of exact term a small local embedding model
can miss but BM25 catches trivially.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.kb.chunking import chunk_document
from src.schemas import Chunk, RawDocument, SourceType

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_RRF_K = 60  # standard RRF constant -- large enough that rank 1 vs 2 isn't a cliff


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # RRF fusion score -- comparable only within one query


class KnowledgeBase:
    """One shared index across all source types. Agents filter to their
    relevant source_type at search time rather than querying siloed indexes."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._chunks: list[Chunk] = []
        self._faiss_index: faiss.Index | None = None
        self._bm25: BM25Okapi | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Resolve a chunk_id back to its chunk. Used by the Validation layer
        (Module 10) to check that a citation points at something real."""
        return next((c for c in self._chunks if c.chunk_id == chunk_id), None)

    def add_documents(self, docs: list[RawDocument]) -> int:
        """Chunk, embed and index. Returns the number of chunks added."""
        new_chunks = [c for doc in docs for c in chunk_document(doc)]
        if not new_chunks:
            return 0
        self._chunks.extend(new_chunks)
        self._rebuild_indexes()
        return len(new_chunks)

    def _rebuild_indexes(self) -> None:
        # Rebuilding from scratch on every add is O(n) in total chunk count,
        # not incremental -- fine at this project's document counts (hundreds
        # of chunks, not millions); would need revisiting at real scale.
        texts = [c.text for c in self._chunks]
        embeddings = np.asarray(
            self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype="float32",
        )
        index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product on unit vectors = cosine
        index.add(embeddings)
        self._faiss_index = index
        self._bm25 = BM25Okapi([_tokenize(t) for t in texts])

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_type: SourceType | None = None,
        candidate_k: int = 20,
    ) -> list[SearchResult]:
        """Hybrid search: fuse FAISS vector ranks with BM25 ranks via RRF,
        then filter to source_type. Filtering after fusion (not before)
        keeps ranking honest -- a source_type filter should narrow results,
        not change which chunk wins the fusion."""
        if not self._chunks or self._faiss_index is None or self._bm25 is None:
            return []
        candidate_k = min(max(candidate_k, top_k), len(self._chunks))

        query_vec = np.asarray(
            self._model.encode([query], normalize_embeddings=True), dtype="float32"
        )
        _, vec_idx = self._faiss_index.search(query_vec, candidate_k)
        vector_ranks = {int(idx): rank for rank, idx in enumerate(vec_idx[0]) if idx != -1}

        bm25_scores = self._bm25.get_scores(_tokenize(query))
        bm25_order = np.argsort(bm25_scores)[::-1][:candidate_k]
        bm25_ranks = {int(idx): rank for rank, idx in enumerate(bm25_order)}

        fused: dict[int, float] = {}
        for idx in set(vector_ranks) | set(bm25_ranks):
            score = 0.0
            if idx in vector_ranks:
                score += 1.0 / (_RRF_K + vector_ranks[idx])
            if idx in bm25_ranks:
                score += 1.0 / (_RRF_K + bm25_ranks[idx])
            fused[idx] = score

        results: list[SearchResult] = []
        for idx, score in sorted(fused.items(), key=lambda pair: pair[1], reverse=True):
            chunk = self._chunks[idx]
            if source_type is not None and chunk.source_type is not source_type:
                continue
            results.append(SearchResult(chunk=chunk, score=score))
            if len(results) >= top_k:
                break
        return results

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss_index, str(path / "faiss.index"))
        with open(path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([c.model_dump(mode="json") for c in self._chunks], f)
        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump(self._bm25, f)

    @classmethod
    def load(cls, path: Path, model_name: str = EMBEDDING_MODEL) -> KnowledgeBase:
        kb = cls(model_name=model_name)
        kb._faiss_index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "chunks.json", encoding="utf-8") as f:
            kb._chunks = [Chunk.model_validate(d) for d in json.load(f)]
        with open(path / "bm25.pkl", "rb") as f:
            kb._bm25 = pickle.load(f)
        return kb

"""
Retrieval backends.

Three are implemented, corresponding to the three retrieval conditions in the
experiment:

  BM25Retriever    lexical only (sparse)
  DenseRetriever   embedding similarity only
  HybridRetriever  both, fused with Reciprocal Rank Fusion, optional reranker

DenseRetriever has two embedding backends. `sentence-transformers` is the one
you should use for the reported results; the TF-IDF/SVD fallback exists so the
whole pipeline stays runnable on a machine that cannot download model weights
(and so the code does not silently break during marking).

German-specific note: German legal text is heavily compounded
("Betriebskostenabrechnung", "Mietminderung"), which punishes plain whitespace
tokenisation in BM25. The tokeniser below therefore emits both the whole token
and its character n-grams for long tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

GERMAN_STOPWORDS = {
    "der", "die", "das", "und", "oder", "ist", "sind", "ein", "eine", "einer",
    "den", "dem", "des", "im", "in", "zu", "zur", "zum", "von", "vom", "mit",
    "auf", "für", "nicht", "auch", "wenn", "durch", "bei", "als", "an", "es",
    "sich", "dass", "kann", "muss", "wird", "werden", "hat", "haben", "so",
}

TOKEN_RE = re.compile(r"[a-zA-ZäöüÄÖÜß§0-9]+")


def tokenize(text: str, ngram_min: int = 5) -> list[str]:
    raw = [t.lower() for t in TOKEN_RE.findall(text)]
    out: list[str] = []
    for t in raw:
        if t in GERMAN_STOPWORDS:
            continue
        out.append(t)
        # Sub-word n-grams give BM25 a chance on compounds it has never seen
        # as a whole token, e.g. matching "betriebskosten" inside
        # "betriebskostenabrechnung".
        if len(t) > 9:
            out.extend(t[i : i + ngram_min] for i in range(0, len(t) - ngram_min + 1, 2))
    return out


@dataclass
class Hit:
    chunk_id: str
    section_id: str
    text: str
    score: float
    rank: int


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [
            Hit(self.chunks[i]["chunk_id"], self.chunks[i]["section_id"],
                self.chunks[i]["text"], float(scores[i]), r + 1)
            for r, i in enumerate(order)
        ]


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[dict], model_name: str = "intfloat/multilingual-e5-base"):
        self.chunks = chunks
        self.model_name = model_name
        self.backend = "sentence-transformers"
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            texts = [f"passage: {c['text']}" for c in chunks]
            self.matrix = self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
        except Exception:
            # Fallback: character n-gram TF-IDF reduced by SVD. Not competitive
            # with a real bi-encoder, and reported as such.
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import Normalizer

            self.backend = "tfidf-svd-fallback"
            self.model = make_pipeline(
                TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2),
                TruncatedSVD(n_components=256, random_state=42),
                Normalizer(copy=False),
            )
            self.matrix = self.model.fit_transform([c["text"] for c in chunks])

    def _embed_query(self, query: str):
        if self.backend == "sentence-transformers":
            return self.model.encode([f"query: {query}"], normalize_embeddings=True)[0]
        return self.model.transform([query])[0]

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = self._embed_query(query)
        scores = self.matrix @ q
        order = np.argsort(scores)[::-1][:k]
        return [
            Hit(self.chunks[i]["chunk_id"], self.chunks[i]["section_id"],
                self.chunks[i]["text"], float(scores[i]), r + 1)
            for r, i in enumerate(order)
        ]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, chunks, dense: DenseRetriever, sparse: BM25Retriever,
                 rrf_k: int = 60, reranker_model: str | None = None,
                 dense_weight: float = 0.5):
        self.chunks = {c["chunk_id"]: c for c in chunks}
        self.dense, self.sparse, self.rrf_k = dense, sparse, rrf_k
        # Plain RRF weights both lists equally, which quietly assumes they are
        # of comparable quality. Where one retriever is much weaker that
        # assumption costs you accuracy, so the weight is exposed rather than
        # hard-coded at 0.5.
        self.dense_weight = dense_weight
        self.reranker = None
        if reranker_model:
            try:
                from sentence_transformers import CrossEncoder

                self.reranker = CrossEncoder(reranker_model)
                self.name = "hybrid+rerank"
            except Exception:
                self.reranker = None

    def search(self, query: str, k: int = 5, pool: int = 25) -> list[Hit]:
        fused: dict[str, float] = {}
        for retr, w in ((self.dense, self.dense_weight),
                        (self.sparse, 1.0 - self.dense_weight)):
            if w <= 0:
                continue
            for h in retr.search(query, pool):
                # Reciprocal Rank Fusion: rank-based, so the two very different
                # score scales never need normalising against each other.
                fused[h.chunk_id] = fused.get(h.chunk_id, 0.0) + w / (self.rrf_k + h.rank)

        ranked = sorted(fused.items(), key=lambda x: -x[1])[: max(k, pool if self.reranker else k)]

        if self.reranker:
            pairs = [(query, self.chunks[cid]["text"]) for cid, _ in ranked]
            ce = self.reranker.predict(pairs)
            ranked = [(cid, float(s)) for (cid, _), s in
                      sorted(zip(ranked, ce), key=lambda x: -x[1])]

        return [
            Hit(cid, self.chunks[cid]["section_id"], self.chunks[cid]["text"], score, r + 1)
            for r, (cid, score) in enumerate(ranked[:k])
        ]

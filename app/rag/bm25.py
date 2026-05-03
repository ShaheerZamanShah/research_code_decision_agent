"""BM25 sparse retrieval using rank_bm25."""

from typing import Any, Dict, List
from rank_bm25 import BM25Okapi
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BM25Retriever:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.documents: List[Dict[str, Any]] = []
        self._tokenized: List[List[str]] = []

    def index(self, documents: List[Dict[str, Any]]):
        """Index documents. Each doc must have 'content' and 'source' keys."""
        self.documents = documents
        self._tokenized = [doc["content"].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self._tokenized)
        logger.info(f"BM25 indexed {len(documents)} documents")

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.bm25:
            logger.warning("BM25 index not built — skipping")
            return []

        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked_indices[:top_k]:
            doc = self.documents[idx].copy()
            doc["bm25_score"] = float(scores[idx])
            results.append(doc)

        return results

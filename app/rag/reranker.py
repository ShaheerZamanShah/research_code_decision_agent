"""Cross-encoder reranking for precision retrieval."""

from typing import Any, Dict, List
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model = None
        self._model_name = model_name

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
            logger.info(f"Loaded reranker: {self._model_name}")

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not documents:
            return []
        self._load()

        pairs = [(query, doc["content"]) for doc in documents]
        scores = self._model.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        result = []
        for doc, score in ranked[:top_k]:
            d = doc.copy()
            d["rerank_score"] = float(score)
            result.append(d)

        logger.info(f"Reranked {len(documents)} → {len(result)} docs")
        return result

"""FAISS-based dense vector retrieval."""

from typing import Any, Dict, List, Optional
import numpy as np
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model = None
        self._model_name = embedding_model_name
        self._index = None
        self.documents: List[Dict[str, Any]] = []
        self._dim: Optional[int] = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info(f"Loaded embedding model: {self._model_name}")

    def _embed(self, texts: List[str]) -> np.ndarray:
        self._load_model()
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def index(self, documents: List[Dict[str, Any]]):
        """Build FAISS index from documents."""
        import faiss
        self.documents = documents
        texts = [doc["content"] for doc in documents]
        embeddings = self._embed(texts).astype("float32")
        self._dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)  # Inner product (cosine on normalized vecs)
        self._index.add(embeddings)
        logger.info(f"FAISS indexed {len(documents)} docs (dim={self._dim})")

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self._index is None:
            logger.warning("FAISS index empty — skipping")
            return []

        q_emb = self._embed([query]).astype("float32")
        scores, indices = self._index.search(q_emb, min(top_k, len(self.documents)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                doc = self.documents[idx].copy()
                doc["dense_score"] = float(score)
                results.append(doc)
        return results

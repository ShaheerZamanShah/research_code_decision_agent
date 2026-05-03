"""Hybrid retriever: FAISS dense + BM25 sparse with RRF fusion."""

from typing import Any, Dict, List
import asyncio
from app.rag.vector_store import VectorStore
from app.rag.bm25 import BM25Retriever
from app.rag.reranker import Reranker
from app.rag.query_rewriter import QueryRewriter
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _rrf_fuse(dense_results: List[Dict], sparse_results: List[Dict], k: int = 60) -> List[Dict]:
    """Reciprocal Rank Fusion."""
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict] = {}

    for rank, doc in enumerate(dense_results):
        key = doc.get("source", "") + doc["content"][:64]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(sparse_results):
        key = doc.get("source", "") + doc["content"][:64]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[k] for k, _ in ranked]


class HybridRetriever:
    def __init__(self):
        self.vector_store = VectorStore(settings.embedding_model)
        self.bm25 = BM25Retriever()
        self.reranker = Reranker(settings.reranker_model)
        self.query_rewriter = QueryRewriter()
        self._indexed = False

    def index_documents(self, documents: List[Dict[str, Any]]):
        self.vector_store.index(documents)
        self.bm25.index(documents)
        self._indexed = True
        logger.info(f"Hybrid retriever indexed {len(documents)} documents")

    async def retrieve(self, query: str, top_k: int = None, rerank: bool = True) -> List[Dict[str, Any]]:
        top_k = top_k or settings.top_k_retrieval

        # Step 1: Query rewriting (multi-hop)
        sub_queries = await self.query_rewriter.rewrite(query)

        all_dense, all_sparse = [], []

        # Step 2: Parallel retrieval for each sub-query
        async def _retrieve_one(q: str):
            loop = asyncio.get_event_loop()
            dense = await loop.run_in_executor(None, self.vector_store.retrieve, q, top_k)
            sparse = await loop.run_in_executor(None, self.bm25.retrieve, q, top_k)
            return dense, sparse

        results = await asyncio.gather(*[_retrieve_one(q) for q in sub_queries])
        for dense, sparse in results:
            all_dense.extend(dense)
            all_sparse.extend(sparse)

        # Step 3: Deduplicate and fuse
        fused = _rrf_fuse(all_dense, all_sparse)
        logger.info(f"Fused {len(fused)} candidate documents")

        # Step 4: Rerank with cross-encoder
        if rerank and fused:
            loop = asyncio.get_event_loop()
            final = await loop.run_in_executor(
                None,
                lambda: self.reranker.rerank(query, fused[:30], settings.top_k_reranked),
            )
            return final

        return fused[:settings.top_k_reranked]

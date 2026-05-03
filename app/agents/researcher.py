"""ResearcherAgent: Multi-source retrieval with hybrid RAG."""

import time
import asyncio
from typing import Any, Dict, List
from app.schemas.state import AgentState, AgentLog
from app.rag.hybrid_retriever import HybridRetriever
from app.tools.web_search import WebSearchTool
from app.tools.arxiv import ArxivTool
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ResearcherAgent:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.web_tool = WebSearchTool(max_results=6)
        self.arxiv_tool = ArxivTool(max_results=3)
        self.name = "ResearcherAgent"

    async def run(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        logger.info(f"[{self.name}] Researching: {state.query[:80]}")

        # 1. Parallel source fetching
        web_docs, arxiv_docs = await asyncio.gather(
            self.web_tool.asearch(state.query),
            self.arxiv_tool.asearch(state.query),
        )

        all_docs: List[Dict[str, Any]] = web_docs + arxiv_docs
        logger.info(f"[{self.name}] Fetched {len(all_docs)} raw documents")

        # 2. Index and retrieve
        if all_docs:
            self.retriever.index_documents(all_docs)
            ranked_docs = await self.retriever.retrieve(state.query, rerank=True)
        else:
            ranked_docs = []

        elapsed = time.perf_counter() - start
        state.retrieved_docs = all_docs
        state.ranked_docs = ranked_docs
        state.sub_queries = getattr(self.retriever.query_rewriter, "_last_queries", [state.query])
        state.logs.append(AgentLog(
            agent=self.name,
            input_summary=state.query[:120],
            output_summary=f"Retrieved {len(ranked_docs)} ranked docs from {len(all_docs)} raw",
            latency_s=round(elapsed, 3),
            iteration=state.iteration,
        ))
        logger.info(f"[{self.name}] Research complete: {len(ranked_docs)} docs in {elapsed:.2f}s")
        return state

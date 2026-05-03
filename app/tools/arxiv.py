"""ArXiv paper search tool for academic research."""

from typing import Any, Dict, List
import arxiv
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _chunk_text(text: str, chunk_size: int = 400) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


class ArxivTool:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.client = arxiv.Client()

    def search(self, query: str) -> List[Dict[str, Any]]:
        docs = []
        try:
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            for paper in self.client.results(search):
                text = f"Title: {paper.title}. Authors: {', '.join(str(a) for a in paper.authors[:3])}. Abstract: {paper.summary}"
                for chunk in _chunk_text(text):
                    docs.append({
                        "content": chunk,
                        "source": str(paper.entry_id),
                        "title": paper.title,
                        "type": "arxiv",
                    })
            logger.info(f"ArXiv: {len(docs)} chunks from {self.max_results} papers")
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
        return docs

    async def asearch(self, query: str) -> List[Dict[str, Any]]:
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(None, self.search, query)

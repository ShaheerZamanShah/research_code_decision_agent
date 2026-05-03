"""Web search tool using DuckDuckGo (no API key needed)."""

from typing import Any, Dict, List
from duckduckgo_search import DDGS
from tenacity import retry, stop_after_attempt, wait_exponential
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


class WebSearchTool:
    def __init__(self, max_results: int = 8):
        self.max_results = max_results

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
    def search(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search and return chunked documents."""
        docs = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))

            for r in results:
                content = f"{r.get('title', '')}. {r.get('body', '')}"
                source = r.get("href", "web")
                for chunk in _chunk_text(content):
                    docs.append({
                        "content": chunk,
                        "source": source,
                        "title": r.get("title", ""),
                        "type": "web",
                    })

            logger.info(f"Web search: {len(docs)} chunks from {len(results)} results")
        except Exception as e:
            logger.error(f"Web search failed: {e}")

        return docs

    async def asearch(self, query: str) -> List[Dict[str, Any]]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search, query)

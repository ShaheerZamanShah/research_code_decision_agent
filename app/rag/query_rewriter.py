"""LLM-powered query expansion into sub-queries."""

from typing import List
import json
from langchain_openai import ChatOpenAI
try:
    from langchain.schema import HumanMessage, SystemMessage
except Exception:
    from langchain_core.messages import HumanMessage, SystemMessage
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a query expansion assistant. Given a user query, generate 3-5 diverse sub-queries
that capture different aspects and angles of the original question. These will be used for retrieval.
Respond with a JSON array of strings ONLY. No explanation."""


class QueryRewriter:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_model,
            temperature=0.3,
            max_tokens=512,
        )

    async def rewrite(self, query: str) -> List[str]:
        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Query: {query}"),
            ]
            response = await self.llm.ainvoke(messages)
            text = response.content.strip()
            # Strip markdown fences
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            sub_queries = json.loads(text)
            if isinstance(sub_queries, list):
                logger.info(f"Expanded into {len(sub_queries)} sub-queries")
                return [query] + sub_queries  # Include original
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")

        return [query]

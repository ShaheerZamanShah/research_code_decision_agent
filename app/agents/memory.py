"""MemoryAgent: Stores and retrieves past task solutions."""

import json
import time
import os
from typing import Any, Dict, List, Optional
from app.schemas.state import AgentState, AgentLog
from app.utils.logger import get_logger

logger = get_logger(__name__)

MEMORY_FILE = "agent_memory.json"


class MemoryAgent:
    """Simple file-based persistent memory for past tasks."""

    def __init__(self):
        self.name = "MemoryAgent"
        self._store: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save(self):
        try:
            with open(MEMORY_FILE, "w") as f:
                json.dump(self._store[-100:], f, indent=2)  # Keep last 100
        except Exception as e:
            logger.warning(f"Memory save failed: {e}")

    def store(self, state: AgentState):
        if not state.evaluation:
            return
        record = {
            "query": state.query,
            "plan": state.plan,
            "decision": state.decision,
            "code": state.code,
            "score": state.evaluation.score if state.evaluation else 0,
            "iterations": state.iteration,
        }
        self._store.append(record)
        self._save()
        logger.info(f"[{self.name}] Stored task (total={len(self._store)})")

    def retrieve_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Simple keyword-based retrieval from memory."""
        query_words = set(query.lower().split())
        scored = []
        for record in self._store:
            rec_words = set(record["query"].lower().split())
            overlap = len(query_words & rec_words) / max(len(query_words), 1)
            scored.append((overlap, record))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k] if _ > 0.2]

    async def run(self, state: AgentState) -> AgentState:
        """Store completed task in memory."""
        start = time.perf_counter()
        self.store(state)
        elapsed = time.perf_counter() - start
        state.logs.append(AgentLog(
            agent=self.name,
            input_summary=f"Task: {state.query[:80]}",
            output_summary=f"Stored in memory (total={len(self._store)})",
            latency_s=round(elapsed, 3),
            iteration=state.iteration,
        ))
        return state

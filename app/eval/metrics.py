"""Metrics tracking for agent system."""

import time
from typing import Any, Dict, List
from dataclasses import dataclass, field


@dataclass
class RunMetrics:
    query: str
    total_latency_s: float = 0.0
    iterations: int = 0
    execution_success: bool = False
    evaluation_score: float = 0.0
    total_docs_retrieved: int = 0
    total_docs_ranked: int = 0
    agent_latencies: Dict[str, float] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_latency_s": round(self.total_latency_s, 3),
            "iterations": self.iterations,
            "execution_success": self.execution_success,
            "evaluation_score": round(self.evaluation_score, 3),
            "total_docs_retrieved": self.total_docs_retrieved,
            "total_docs_ranked": self.total_docs_ranked,
            "agent_latencies": {k: round(v, 3) for k, v in self.agent_latencies.items()},
        }


def collect_metrics(state, total_latency: float) -> RunMetrics:
    """Collect metrics from completed state."""
    agent_latencies = {}
    for log in state.logs:
        agent_latencies[log.agent] = agent_latencies.get(log.agent, 0) + log.latency_s

    return RunMetrics(
        query=state.query,
        total_latency_s=total_latency,
        iterations=state.iteration,
        execution_success=state.execution_result.success if state.execution_result else False,
        evaluation_score=state.evaluation.score if state.evaluation else 0.0,
        total_docs_retrieved=len(state.retrieved_docs),
        total_docs_ranked=len(state.ranked_docs),
        agent_latencies=agent_latencies,
    )

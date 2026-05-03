"""LangGraph state schema and supporting types."""

from typing import Any, Dict, List, Optional, Annotated
from pydantic import BaseModel, Field
from operator import add


class AgentLog(BaseModel):
    agent: str
    input_summary: str
    output_summary: str
    latency_s: float
    iteration: int


class ExecutionResult(BaseModel):
    success: bool
    output: str
    error: str = ""
    duration_s: float = 0.0


class EvaluationResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    feedback: str
    passed: bool
    details: Dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    """Full state passed through LangGraph."""
    # Core
    query: str = ""
    plan: List[str] = Field(default_factory=list)

    # Research
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list)
    ranked_docs: List[Dict[str, Any]] = Field(default_factory=list)
    sub_queries: List[str] = Field(default_factory=list)

    # Decision & Code
    decision: str = ""
    code: str = ""

    # Execution
    execution_result: Optional[ExecutionResult] = None

    # Evaluation
    evaluation: Optional[EvaluationResult] = None

    # Final output
    final_report: str = ""

    # Control flow
    iteration: int = 0
    max_iterations: int = 3
    should_retry: bool = False

    # Observability
    logs: List[AgentLog] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    token_usage: Dict[str, int] = Field(default_factory=dict)


def make_state(query: str, max_iterations: int = 3) -> AgentState:
    return AgentState(query=query, max_iterations=max_iterations)

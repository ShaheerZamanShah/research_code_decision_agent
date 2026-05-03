"""API-level input/output schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="Research query")
    max_iterations: int = Field(default=3, ge=1, le=5)


class AgentLogOut(BaseModel):
    agent: str
    input_summary: str
    output_summary: str
    latency_s: float
    iteration: int


class RunResponse(BaseModel):
    report: str
    code: str
    metrics: Dict[str, Any]
    logs: List[AgentLogOut]
    success: bool
    error: Optional[str] = None

"""FastAPI route definitions."""

import time
from fastapi import APIRouter, HTTPException
from app.schemas.inputs import RunRequest, RunResponse, AgentLogOut
from app.schemas.state import make_state, AgentLog
from app.graph.agent_graph import get_graph
from app.eval.metrics import collect_metrics
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/run", response_model=RunResponse)
async def run_agent(request: RunRequest):
    """Run the full autonomous agent pipeline."""
    logger.info(f"POST /run query={request.query[:80]}")
    start = time.perf_counter()

    try:
        state = make_state(request.query, request.max_iterations)
        graph = get_graph()

        # Run the graph
        final_state_dict = await graph.ainvoke(state.model_dump())

        from app.schemas.state import AgentState
        final_state = AgentState(**final_state_dict)

        total_latency = time.perf_counter() - start
        metrics = collect_metrics(final_state, total_latency)

        return RunResponse(
            report=final_state.final_report,
            code=final_state.code,
            metrics=metrics.to_dict(),
            logs=[
                AgentLogOut(
                    agent=log.agent,
                    input_summary=log.input_summary,
                    output_summary=log.output_summary,
                    latency_s=log.latency_s,
                    iteration=log.iteration,
                )
                for log in final_state.logs
            ],
            success=True,
        )

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return RunResponse(
            report="",
            code="",
            metrics={},
            logs=[],
            success=False,
            error=str(e),
        )


@router.get("/health")
async def health():
    return {"status": "ok"}

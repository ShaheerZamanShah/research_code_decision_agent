"""ReflectorAgent: Decides whether to retry, improve, or finalize."""

import time
from app.schemas.state import AgentState, AgentLog
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ReflectorAgent:
    def __init__(self):
        self.name = "ReflectorAgent"

    async def run(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        state.iteration += 1

        eval_result = state.evaluation
        should_retry = False
        reason = ""

        if not eval_result:
            should_retry = True
            reason = "No evaluation result"
        elif not eval_result.passed:
            if state.iteration < state.max_iterations:
                should_retry = True
                reason = f"Score {eval_result.score:.2f} below threshold {settings.critic_threshold}"
            else:
                reason = f"Max iterations ({state.max_iterations}) reached — finalizing despite low score"
        else:
            reason = f"Score {eval_result.score:.2f} ≥ threshold — finalizing"

        state.should_retry = should_retry

        elapsed = time.perf_counter() - start
        state.logs.append(AgentLog(
            agent=self.name,
            input_summary=f"Iteration {state.iteration}, score={eval_result.score if eval_result else 'N/A'}",
            output_summary=f"{'Retry' if should_retry else 'Finalize'}: {reason}",
            latency_s=round(elapsed, 3),
            iteration=state.iteration,
        ))
        logger.info(f"[{self.name}] {'Retrying' if should_retry else 'Finalizing'}: {reason}")
        return state

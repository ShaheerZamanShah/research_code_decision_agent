"""ExecutorAgent: Safely runs generated Python code."""

import time
from app.schemas.state import AgentState, AgentLog
from app.tools.python_exec import PythonExecutor
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ExecutorAgent:
    def __init__(self):
        self.executor = PythonExecutor()
        self.name = "ExecutorAgent"

    async def run(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        logger.info(f"[{self.name}] Executing code (iteration={state.iteration})")

        if not state.code:
            logger.warning(f"[{self.name}] No code to execute")
            from app.schemas.state import ExecutionResult
            state.execution_result = ExecutionResult(
                success=False, output="", error="No code provided"
            )
            return state

        result = await self.executor.aexecute(state.code)

        elapsed = time.perf_counter() - start
        state.execution_result = result
        state.logs.append(AgentLog(
            agent=self.name,
            input_summary=f"Code: {len(state.code.splitlines())} lines",
            output_summary=(
                f"✅ Success: {result.output[:120]}"
                if result.success
                else f"❌ Error: {result.error[:120]}"
            ),
            latency_s=round(elapsed, 3),
            iteration=state.iteration,
        ))
        logger.info(
            f"[{self.name}] Execution {'succeeded' if result.success else 'failed'} in {elapsed:.2f}s"
        )
        return state

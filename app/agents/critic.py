"""CriticAgent: Evaluates code quality, output, and correctness."""

import time
import json
from langchain_openai import ChatOpenAI
try:
    from langchain.schema import HumanMessage, SystemMessage
except Exception:
    from langchain_core.messages import HumanMessage, SystemMessage
from app.schemas.state import AgentState, AgentLog, EvaluationResult
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM = """You are a rigorous code and output evaluator. Given a query, generated code, and execution result, score the solution.

Return ONLY a JSON object with these fields:
{
  "score": <float 0.0-1.0>,
  "passed": <bool>,
  "feedback": "<specific actionable feedback>",
  "details": {
    "correctness": <0-1>,
    "completeness": <0-1>,
    "code_quality": <0-1>,
    "output_clarity": <0-1>
  }
}

Scoring guide:
- 0.9-1.0: Excellent, correct, complete
- 0.7-0.9: Good but minor issues
- 0.5-0.7: Partially correct, needs improvement
- 0.0-0.5: Major issues or failure
"""


class CriticAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_model,
            temperature=0.1,
            max_tokens=768,
        )
        self.name = "CriticAgent"

    async def run(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        logger.info(f"[{self.name}] Evaluating solution (iteration={state.iteration})")

        exec_result = state.execution_result
        if not exec_result:
            state.evaluation = EvaluationResult(
                score=0.0, feedback="No execution result.", passed=False
            )
            return state

        # Build prompt
        code_snippet = state.code[:1500] if state.code else "No code"
        output_snippet = exec_result.output[:500] if exec_result.output else ""
        error_snippet = exec_result.error[:300] if exec_result.error else ""

        messages = [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=(
                f"Query: {state.query}\n\n"
                f"Code:\n```python\n{code_snippet}\n```\n\n"
                f"Execution Success: {exec_result.success}\n"
                f"Output:\n{output_snippet}\n"
                f"Error:\n{error_snippet}"
            )),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            text = response.content.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            evaluation = EvaluationResult(
                score=float(data.get("score", 0.5)),
                passed=bool(data.get("passed", False)),
                feedback=data.get("feedback", ""),
                details=data.get("details", {}),
            )
        except Exception as e:
            logger.warning(f"[{self.name}] Parsing failed: {e} — using heuristic score")
            # Heuristic fallback
            score = 0.8 if exec_result.success else 0.2
            evaluation = EvaluationResult(
                score=score,
                passed=score >= settings.critic_threshold,
                feedback=f"Auto-scored based on execution: {'success' if exec_result.success else 'failed'}",
            )

        elapsed = time.perf_counter() - start
        state.evaluation = evaluation
        state.logs.append(AgentLog(
            agent=self.name,
            input_summary=f"Code + execution result",
            output_summary=f"Score: {evaluation.score:.2f} | {'✅ Pass' if evaluation.passed else '❌ Fail'} | {evaluation.feedback[:100]}",
            latency_s=round(elapsed, 3),
            iteration=state.iteration,
        ))
        logger.info(f"[{self.name}] Score={evaluation.score:.2f} passed={evaluation.passed} in {elapsed:.2f}s")
        return state

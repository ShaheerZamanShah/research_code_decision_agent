"""CoderAgent: Generates executable Python code."""

import time
from langchain_openai import ChatOpenAI
try:
    from langchain.schema import HumanMessage, SystemMessage
except Exception:
    from langchain_core.messages import HumanMessage, SystemMessage
from app.schemas.state import AgentState, AgentLog
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM = """You are an expert Python engineer. Generate a complete, runnable Python script that:
1. Solves the given query
2. Follows the decided approach
3. Includes data loading, processing, modeling/computation, and output
4. Prints results clearly to stdout
5. Uses only standard library + common packages that are already available in this app (numpy, pandas, sklearn, scipy etc.)
6. Handles errors gracefully

RULES:
- Output ONLY the Python code block — no explanation outside it
- Code must be complete and self-contained
- Use print() to show results and metrics
- Include a main() function and `if __name__ == '__main__': main()`
- Do NOT use internet access inside the code
- Do NOT import optional visualization packages such as matplotlib or seaborn
- Prefer textual summaries over plots so execution stays reliable in a minimal environment
- For ML tasks, generate synthetic data if no dataset is provided"""


def _extract_code(text: str) -> str:
    """Extract Python code from markdown or raw response."""
    if "```python" in text:
        parts = text.split("```python")
        if len(parts) > 1:
            return parts[1].split("```")[0].strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) > 1:
            return parts[1].strip()
    return text.strip()


class CoderAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_model,
            temperature=0.1,
            max_tokens=settings.llm_max_tokens,
        )
        self.name = "CoderAgent"

    def _format_context(self, state: AgentState) -> str:
        docs = state.ranked_docs[:3]
        snippets = []
        for doc in docs:
            snippets.append(f"# From {doc.get('source','?')[:60]}\n# {doc['content'][:200]}")
        return "\n".join(snippets) if snippets else ""

    async def run(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        logger.info(f"[{self.name}] Generating code (iteration={state.iteration})")

        context = self._format_context(state)
        feedback_section = ""
        if state.evaluation and state.iteration > 0:
            feedback_section = (
                f"\n\nPREVIOUS CODE FAILED OR SCORED LOW ({state.evaluation.score:.2f}).\n"
                f"Error/Feedback:\n{state.evaluation.feedback}\n"
                f"Previous execution output:\n{state.execution_result.output if state.execution_result else 'N/A'}\n"
                f"Fix the issues and improve the code."
            )

        messages = [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=(
                f"Query: {state.query}\n\n"
                f"Decision/Approach:\n{state.decision}\n\n"
                f"Relevant context (for reference):\n{context}"
                f"{feedback_section}"
            )),
        ]

        response = await self.llm.ainvoke(messages)
        code = _extract_code(response.content)

        elapsed = time.perf_counter() - start
        state.code = code
        state.logs.append(AgentLog(
            agent=self.name,
            input_summary=f"Query + decision + {len(state.ranked_docs)} docs",
            output_summary=f"Generated {len(code.splitlines())} lines of code",
            latency_s=round(elapsed, 3),
            iteration=state.iteration,
        ))
        logger.info(f"[{self.name}] Code generated ({len(code.splitlines())} lines) in {elapsed:.2f}s")
        return state

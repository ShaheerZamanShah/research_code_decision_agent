"""DecisionAgent: Selects approach based on research context."""

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

SYSTEM = """You are a technical decision-maker. Given:
1. A user query
2. A research plan
3. Retrieved context documents

Decide the best approach:
- Which algorithm/technique to use
- What Python libraries are needed
- High-level implementation strategy
- Key considerations

Be specific and actionable. Your output will guide code generation.
Write 2-4 paragraphs. Do NOT write code yet."""


class DecisionAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_model,
            temperature=0.2,
            max_tokens=1024,
        )
        self.name = "DecisionAgent"

    def _format_context(self, state: AgentState) -> str:
        docs = state.ranked_docs[:5]
        parts = []
        for i, doc in enumerate(docs, 1):
            src = doc.get("source", "unknown")
            content = doc.get("content", "")[:300]
            parts.append(f"[Doc {i}] ({src})\n{content}")
        return "\n\n".join(parts) if parts else "No context available."

    async def run(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        logger.info(f"[{self.name}] Making decision (iteration={state.iteration})")

        context = self._format_context(state)
        plan_str = "\n".join(f"- {s}" for s in state.plan)

        # Include critic feedback on retry
        feedback_section = ""
        if state.evaluation and state.iteration > 0:
            feedback_section = f"\n\nPREVIOUS FEEDBACK:\n{state.evaluation.feedback}\nScore: {state.evaluation.score:.2f}"

        messages = [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=(
                f"Query: {state.query}\n\n"
                f"Plan:\n{plan_str}\n\n"
                f"Context:\n{context}"
                f"{feedback_section}"
            )),
        ]

        response = await self.llm.ainvoke(messages)
        decision = response.content.strip()

        elapsed = time.perf_counter() - start
        state.decision = decision
        state.logs.append(AgentLog(
            agent=self.name,
            input_summary=f"Query + {len(state.ranked_docs)} docs",
            output_summary=decision[:150],
            latency_s=round(elapsed, 3),
            iteration=state.iteration,
        ))
        logger.info(f"[{self.name}] Decision made in {elapsed:.2f}s")
        return state

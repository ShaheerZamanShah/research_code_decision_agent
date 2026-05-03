"""WriterAgent: Produces the final research report with citations."""

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

SYSTEM = """You are a technical report writer. Produce a comprehensive final report in Markdown.

Structure:
# Executive Summary
Brief 2-3 sentence overview

# Problem Statement
What was asked and why it matters

# Research Findings
Key insights from retrieved sources (with inline citations as [Source: URL])

# Technical Approach
The chosen approach and rationale

# Implementation
Overview of the generated solution

# Results & Evaluation
Execution results and quality score

# Conclusion
Key takeaways and potential improvements

Use clear headers, bullet points where appropriate. Be specific and cite sources."""


class WriterAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_model,
            temperature=0.3,
            max_tokens=2048,
        )
        self.name = "WriterAgent"

    def _build_citations(self, state: AgentState) -> str:
        seen = set()
        lines = []
        for doc in state.ranked_docs[:8]:
            src = doc.get("source", "")
            if src and src not in seen:
                seen.add(src)
                title = doc.get("title", src[:60])
                doc_type = doc.get("type", "web")
                lines.append(f"- [{title[:60]}]({src}) [{doc_type}]")
        return "\n".join(lines) if lines else "No external sources"

    async def run(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        logger.info(f"[{self.name}] Writing final report")

        citations = self._build_citations(state)
        exec_out = state.execution_result
        eval_res = state.evaluation

        messages = [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=(
                f"Query: {state.query}\n\n"
                f"Plan:\n" + "\n".join(f"- {s}" for s in state.plan) + "\n\n"
                f"Decision:\n{state.decision}\n\n"
                f"Execution Success: {exec_out.success if exec_out else 'N/A'}\n"
                f"Execution Output:\n{exec_out.output[:600] if exec_out else 'N/A'}\n\n"
                f"Evaluation Score: {eval_res.score:.2f} | Feedback: {eval_res.feedback if eval_res else 'N/A'}\n\n"
                f"Iterations: {state.iteration}\n\n"
                f"Sources:\n{citations}"
            )),
        ]

        response = await self.llm.ainvoke(messages)
        report = response.content.strip()

        # Append citations section
        report += f"\n\n---\n## Sources\n{citations}"

        elapsed = time.perf_counter() - start
        state.final_report = report
        state.logs.append(AgentLog(
            agent=self.name,
            input_summary="All state data",
            output_summary=f"Report: {len(report)} chars",
            latency_s=round(elapsed, 3),
            iteration=state.iteration,
        ))
        logger.info(f"[{self.name}] Report written in {elapsed:.2f}s")
        return state

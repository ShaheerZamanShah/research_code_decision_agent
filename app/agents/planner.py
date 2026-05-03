"""PlannerAgent: Breaks a query into actionable steps."""

import time
import json
from typing import List
from langchain_openai import ChatOpenAI
try:
    from langchain.schema import HumanMessage, SystemMessage
except Exception:
    from langchain_core.messages import HumanMessage, SystemMessage
from app.schemas.state import AgentState, AgentLog
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM = """You are a strategic planner. Given a research/coding query, produce a concise plan as a JSON array of 4-6 action steps.
Each step is a clear, actionable string. Example: ["Step 1: ...", "Step 2: ..."]
Respond with JSON array ONLY."""


class PlannerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_model,
            temperature=0.2,
            max_tokens=512,
        )
        self.name = "PlannerAgent"

    async def run(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        logger.info(f"[{self.name}] Planning for: {state.query[:80]}")

        try:
            messages = [
                SystemMessage(content=SYSTEM),
                HumanMessage(content=f"Query: {state.query}"),
            ]
            response = await self.llm.ainvoke(messages)
            text = response.content.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            plan = json.loads(text)
            if not isinstance(plan, list):
                plan = [str(plan)]
        except Exception as e:
            logger.warning(f"[{self.name}] Falling back to default plan: {e}")
            plan = [
                "Step 1: Research the problem domain and gather relevant information",
                "Step 2: Decide on the best algorithmic/technical approach",
                "Step 3: Write executable Python code to solve the problem",
                "Step 4: Execute and test the code",
                "Step 5: Evaluate results and refine if needed",
                "Step 6: Produce a final report with findings",
            ]

        elapsed = time.perf_counter() - start
        state.plan = plan
        state.logs.append(AgentLog(
            agent=self.name,
            input_summary=state.query[:120],
            output_summary=f"Plan with {len(plan)} steps",
            latency_s=round(elapsed, 3),
            iteration=state.iteration,
        ))
        logger.info(f"[{self.name}] Created plan with {len(plan)} steps in {elapsed:.2f}s")
        return state

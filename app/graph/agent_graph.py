"""LangGraph stateful agent orchestration graph."""

from typing import Any, Dict
from langgraph.graph import StateGraph, START, END
from app.schemas.state import AgentState
from app.agents.planner import PlannerAgent
from app.agents.researcher import ResearcherAgent
from app.agents.decision import DecisionAgent
from app.agents.coder import CoderAgent
from app.agents.executor import ExecutorAgent
from app.agents.critic import CriticAgent
from app.agents.reflector import ReflectorAgent
from app.agents.memory import MemoryAgent
from app.agents.writer import WriterAgent
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Instantiate agents (singleton-like)
_planner = PlannerAgent()
_researcher = ResearcherAgent()
_decision = DecisionAgent()
_coder = CoderAgent()
_executor = ExecutorAgent()
_critic = CriticAgent()
_reflector = ReflectorAgent()
_memory = MemoryAgent()
_writer = WriterAgent()


# Node wrappers: LangGraph passes state as dict
async def planner_node(state: Dict) -> Dict:
    s = AgentState(**state)
    s = await _planner.run(s)
    return s.model_dump()


async def researcher_node(state: Dict) -> Dict:
    s = AgentState(**state)
    s = await _researcher.run(s)
    return s.model_dump()


async def decision_node(state: Dict) -> Dict:
    s = AgentState(**state)
    s = await _decision.run(s)
    return s.model_dump()


async def coder_node(state: Dict) -> Dict:
    s = AgentState(**state)
    s = await _coder.run(s)
    return s.model_dump()


async def executor_node(state: Dict) -> Dict:
    s = AgentState(**state)
    s = await _executor.run(s)
    return s.model_dump()


async def critic_node(state: Dict) -> Dict:
    s = AgentState(**state)
    s = await _critic.run(s)
    return s.model_dump()


async def reflector_node(state: Dict) -> Dict:
    s = AgentState(**state)
    s = await _reflector.run(s)
    return s.model_dump()


async def memory_node(state: Dict) -> Dict:
    s = AgentState(**state)
    s = await _memory.run(s)
    return s.model_dump()


async def writer_node(state: Dict) -> Dict:
    s = AgentState(**state)
    s = await _writer.run(s)
    return s.model_dump()


def should_retry(state: Dict) -> str:
    """Conditional edge: retry or finalize."""
    if state.get("should_retry", False):
        logger.info("Graph: routing → Decision (retry)")
        return "retry"
    logger.info("Graph: routing → Writer (finalize)")
    return "finalize"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph agent graph."""
    graph = StateGraph(dict)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("decision", decision_node)
    graph.add_node("coder", coder_node)
    graph.add_node("executor", executor_node)
    graph.add_node("critic", critic_node)
    graph.add_node("reflector", reflector_node)
    graph.add_node("memory", memory_node)
    graph.add_node("writer", writer_node)

    # Linear flow
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "decision")
    graph.add_edge("decision", "coder")
    graph.add_edge("coder", "executor")
    graph.add_edge("executor", "critic")
    graph.add_edge("critic", "reflector")

    # Conditional retry loop
    graph.add_conditional_edges(
        "reflector",
        should_retry,
        {
            "retry": "decision",    # Back to decision with feedback
            "finalize": "memory",   # Forward to memory + writer
        },
    )

    graph.add_edge("memory", "writer")
    graph.add_edge("writer", END)

    return graph.compile()


# Global compiled graph
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
        logger.info("Agent graph compiled")
    return _graph

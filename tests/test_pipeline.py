"""Unit and integration tests for the agent pipeline."""

import asyncio
import types
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.schemas.state import AgentState, ExecutionResult, EvaluationResult, make_state
from app.tools.python_exec import PythonExecutor, _compose_script
from app.agents.reflector import ReflectorAgent
from app.rag.bm25 import BM25Retriever


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_state():
    return make_state("Test query: sort a list in Python", max_iterations=2)


# ── BM25 ──────────────────────────────────────────────────────────────────────

def test_bm25_index_and_retrieve():
    bm25 = BM25Retriever()
    docs = [
        {"content": "Python is a high-level programming language", "source": "wiki"},
        {"content": "Sorting algorithms include quicksort and mergesort", "source": "cs"},
        {"content": "Machine learning uses neural networks", "source": "ml"},
    ]
    bm25.index(docs)
    results = bm25.retrieve("Python sorting", top_k=2)
    assert len(results) <= 2
    assert all("bm25_score" in r for r in results)


def test_bm25_empty_index():
    bm25 = BM25Retriever()
    results = bm25.retrieve("test query")
    assert results == []


# ── Code Executor ─────────────────────────────────────────────────────────────

def test_executor_success():
    exe = PythonExecutor(timeout=10)
    result = exe.execute("print('hello world')")
    assert result.success
    assert "hello world" in result.output


def test_executor_error():
    exe = PythonExecutor(timeout=10)
    result = exe.execute("raise ValueError('test error')")
    assert not result.success
    assert "ValueError" in result.error


def test_executor_timeout():
    exe = PythonExecutor(timeout=2)
    result = exe.execute("import time; time.sleep(10)")
    assert not result.success
    assert "timed out" in result.error.lower()


def test_executor_syntax_error():
    exe = PythonExecutor(timeout=10)
    result = exe.execute("def broken(:\n    pass")
    assert not result.success


def test_executor_handles_future_imports():
    script = _compose_script("from __future__ import annotations\nprint('ok')")
    lines = script.splitlines()
    assert lines[0] == "from __future__ import annotations"
    assert any("warnings.filterwarnings('ignore')" in line for line in lines)


def test_executor_handles_missing_stdio(monkeypatch):
    exe = PythonExecutor(timeout=10)

    fake_proc = types.SimpleNamespace(returncode=1, stdout=None, stderr=None)

    def fake_run(*args, **kwargs):
        return fake_proc

    monkeypatch.setattr("app.tools.python_exec.subprocess.run", fake_run)

    result = exe.execute("raise RuntimeError('boom')")
    assert not result.success
    assert result.output == ""
    assert result.error == ""


# ── Reflector ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reflector_retry_on_fail(base_state):
    base_state.evaluation = EvaluationResult(
        score=0.3, feedback="Too slow", passed=False
    )
    base_state.iteration = 0
    reflector = ReflectorAgent()
    result = await reflector.run(base_state)
    assert result.should_retry is True
    assert result.iteration == 1


@pytest.mark.asyncio
async def test_reflector_finalize_on_pass(base_state):
    base_state.evaluation = EvaluationResult(
        score=0.9, feedback="Excellent", passed=True
    )
    base_state.iteration = 0
    reflector = ReflectorAgent()
    result = await reflector.run(base_state)
    assert result.should_retry is False


@pytest.mark.asyncio
async def test_reflector_finalize_at_max_iter(base_state):
    base_state.evaluation = EvaluationResult(
        score=0.2, feedback="Poor", passed=False
    )
    base_state.iteration = base_state.max_iterations - 1  # Will hit max after increment
    reflector = ReflectorAgent()
    result = await reflector.run(base_state)
    # At max iterations, should NOT retry
    assert result.should_retry is False


# ── State schema ──────────────────────────────────────────────────────────────

def test_state_creation():
    state = make_state("test", max_iterations=3)
    assert state.query == "test"
    assert state.max_iterations == 3
    assert state.iteration == 0
    assert state.plan == []
    assert state.logs == []


def test_state_serialization():
    state = make_state("query")
    data = state.model_dump()
    restored = AgentState(**data)
    assert restored.query == "query"


# ── Integration (mock LLM) ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_planner_mock():
    """Test planner agent with mocked LLM."""
    from app.agents.planner import PlannerAgent

    mock_response = MagicMock()
    mock_response.content = '["Step 1: Research", "Step 2: Code", "Step 3: Evaluate"]'

    with patch("app.agents.planner.ChatOpenAI") as MockLLM:
        instance = MockLLM.return_value
        instance.ainvoke = AsyncMock(return_value=mock_response)

        agent = PlannerAgent()
        agent.llm = instance
        state = make_state("test query")
        result = await agent.run(state)

        assert len(result.plan) == 3
        assert len(result.logs) == 1
        assert result.logs[0].agent == "PlannerAgent"


@pytest.mark.asyncio
async def test_executor_agent():
    """Integration test: ExecutorAgent runs real code."""
    from app.agents.executor import ExecutorAgent

    state = make_state("test")
    state.code = "x = 2 + 2\nprint(f'Result: {x}')"

    agent = ExecutorAgent()
    result = await agent.run(state)

    assert result.execution_result is not None
    assert result.execution_result.success
    assert "Result: 4" in result.execution_result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

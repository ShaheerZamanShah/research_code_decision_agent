# ⚡ Autonomous Research + Code + Decision Agent System

A production-grade multi-agent system with LangGraph orchestration, hybrid RAG, code execution, and reflection loops.

## Architecture

```
START → Planner → Researcher (RAG) → Decision → Coder → Executor → Critic → Reflector
                                         ↑____________________________|  (retry loop)
                                                                        ↓
                                                              Memory → Writer → END
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph |
| LLM | OpenAI-compatible (gpt-4o-mini default) |
| Dense Retrieval | FAISS + sentence-transformers |
| Sparse Retrieval | BM25 (rank_bm25) |
| Reranking | Cross-encoder (ms-marco-MiniLM-L-6-v2) |
| Web Search | DuckDuckGo (no API key) |
| Academic Search | ArXiv |
| Code Execution | Subprocess sandbox |
| API | FastAPI |
| Frontend | Streamlit |
| Logging | Loguru |
| Validation | Pydantic v2 |

## Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 3. Start backend
cd project_root
uvicorn app.main:app --reload --port 8000

# 4. Start frontend (new terminal)
streamlit run frontend/app.py
```

## Environment Variables

```env
OPENAI_API_KEY=your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1   # or any OpenAI-compatible endpoint
LLM_MODEL=gpt-4o-mini
MAX_ITERATIONS=3
CRITIC_THRESHOLD=0.7
EXECUTION_TIMEOUT=30
```

## API

### POST /api/v1/run

```json
{
  "query": "Build a logistic regression classifier and evaluate it",
  "max_iterations": 3
}
```

Response:
```json
{
  "report": "# Executive Summary\n...",
  "code": "import numpy as np\n...",
  "metrics": {
    "total_latency_s": 18.4,
    "iterations": 2,
    "execution_success": true,
    "evaluation_score": 0.87
  },
  "logs": [...]
}
```

## Agent Responsibilities

| Agent | Role |
|-------|------|
| **PlannerAgent** | Breaks query into structured steps |
| **ResearcherAgent** | Hybrid RAG (web + arxiv + FAISS + BM25 + rerank) |
| **DecisionAgent** | Chooses algorithm and approach |
| **CoderAgent** | Generates complete, runnable Python |
| **ExecutorAgent** | Sandboxed subprocess execution |
| **CriticAgent** | Scores 0-1, provides feedback |
| **ReflectorAgent** | Decides retry vs finalize |
| **MemoryAgent** | Persistent task storage |
| **WriterAgent** | Final Markdown report with citations |

## RAG Pipeline

1. **Query rewriting** → 3-5 sub-queries via LLM
2. **Parallel retrieval** → Web search + ArXiv simultaneously
3. **Hybrid fusion** → FAISS dense + BM25 sparse with RRF
4. **Cross-encoder reranking** → Top-5 most relevant chunks

## Reflection Loop

```
Critic score < 0.7 AND iteration < max_iterations → retry Decision + Coder
Critic score ≥ 0.7 OR iteration ≥ max_iterations → finalize
```

## Tests

```bash
pytest tests/ -v
```

## Project Structure

```
agent_system/
├── app/
│   ├── agents/          # All agent classes
│   ├── graph/           # LangGraph orchestration
│   ├── rag/             # Hybrid retrieval pipeline
│   ├── tools/           # Web, ArXiv, code execution
│   ├── schemas/         # Pydantic state + I/O schemas
│   ├── eval/            # Metrics tracking
│   ├── config/          # Settings
│   └── utils/           # Logger, timing
├── frontend/
│   └── app.py           # Streamlit UI
├── tests/
│   └── test_pipeline.py
└── requirements.txt
```

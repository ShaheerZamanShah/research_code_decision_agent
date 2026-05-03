"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Autonomous Research + Code Agent System",
    description="Production-grade multi-agent system with RAG, code execution, and reflection loops",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    logger.info("Agent System API starting up...")
    # Pre-warm graph
    from app.graph.agent_graph import get_graph
    get_graph()
    logger.info("Agent graph compiled and ready")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="info",
    )

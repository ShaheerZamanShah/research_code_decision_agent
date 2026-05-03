"""Structured logging with loguru."""

import sys
import time
from loguru import logger
from functools import wraps
from app.config.settings import settings

# Remove default handler
logger.remove()

# Console handler
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level=settings.log_level,
    colorize=True,
)

# File handler (structured JSON)
logger.add(
    settings.log_file,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message} | {extra}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    serialize=True,
)


def log_agent_step(agent_name: str):
    """Decorator to log agent execution with timing."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            logger.info(f"[{agent_name}] Starting", agent=agent_name)
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info(f"[{agent_name}] Completed in {elapsed:.2f}s", agent=agent_name, latency=elapsed)
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(f"[{agent_name}] Failed: {e}", agent=agent_name, error=str(e), latency=elapsed)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            logger.info(f"[{agent_name}] Starting", agent=agent_name)
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info(f"[{agent_name}] Completed in {elapsed:.2f}s", agent=agent_name, latency=elapsed)
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(f"[{agent_name}] Failed: {e}", agent=agent_name, error=str(e), latency=elapsed)
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def get_logger(name: str):
    return logger.bind(module=name)

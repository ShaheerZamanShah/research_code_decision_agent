"""Safe Python code execution with subprocess sandbox."""

import subprocess
import sys
import tempfile
import os
import time
from typing import Dict, Any
from app.schemas.state import ExecutionResult
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_PREAMBLE = """
import sys, os, json, math, traceback
import warnings
warnings.filterwarnings('ignore')
"""


def _compose_script(code: str) -> str:
    """Insert the safety preamble without breaking leading future imports."""
    lines = code.splitlines()
    insert_at = 0

    while insert_at < len(lines) and lines[insert_at].startswith("from __future__ import"):
        insert_at += 1

    return "\n".join(lines[:insert_at] + [_PREAMBLE.strip()] + ["", *lines[insert_at:]])


def _normalize_text(value: str | None) -> str:
    return (value or "").strip()


class PythonExecutor:
    def __init__(self, timeout: int = None):
        self.timeout = timeout or settings.execution_timeout

    def execute(self, code: str) -> ExecutionResult:
        """Execute Python code in a subprocess with timeout."""
        full_code = _compose_script(code)
        start = time.perf_counter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(full_code)
            tmp_path = f.name

        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=self.timeout,
                cwd=tempfile.gettempdir(),
                env=env,
            )
            duration = time.perf_counter() - start

            if proc.returncode == 0:
                output = _normalize_text(proc.stdout)
                logger.info(f"Code executed successfully in {duration:.2f}s")
                return ExecutionResult(
                    success=True,
                    output=output or "(no output)",
                    duration_s=duration,
                )
            else:
                error = _normalize_text(proc.stderr)
                # Truncate long errors
                if len(error) > 2000:
                    error = error[-2000:]
                logger.warning(f"Code execution error: {error[:200]}")
                return ExecutionResult(
                    success=False,
                    output=_normalize_text(proc.stdout),
                    error=error,
                    duration_s=duration,
                )

        except subprocess.TimeoutExpired:
            duration = time.perf_counter() - start
            logger.error(f"Code execution timed out after {self.timeout}s")
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timed out after {self.timeout} seconds.",
                duration_s=duration,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                duration_s=time.perf_counter() - start,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    async def aexecute(self, code: str) -> ExecutionResult:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, code)

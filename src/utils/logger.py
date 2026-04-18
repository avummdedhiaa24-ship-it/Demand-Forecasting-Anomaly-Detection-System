"""
Structured logging module using loguru.
Provides consistent, production-grade logging across all system components.
"""

from __future__ import annotations

import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from src.utils.config import cfg

# ── Setup ─────────────────────────────────────────────────────────────

def setup_logger() -> None:
    """Configure loguru logger with file + console sinks."""
    log_path = Path(cfg.logging.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console sink (human-readable)
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        level=cfg.logging.level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File sink (JSON-friendly for log aggregation)
    logger.add(
        str(log_path),
        format=cfg.logging.format,
        level=cfg.logging.level,
        rotation=cfg.logging.rotation,
        retention=cfg.logging.retention,
        serialize=False,
        backtrace=True,
        diagnose=False,
        enqueue=True,          # thread-safe async writes
    )

    logger.info("Logger initialised. Level={}", cfg.logging.level)


def get_logger(name: str):
    """Return a named logger instance."""
    return logger.bind(module=name)


# ── Decorators ────────────────────────────────────────────────────────

def log_execution_time(func: Callable) -> Callable:
    """Decorator: log function execution time."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                "⏱ {}.{} completed in {:.2f}ms",
                func.__module__,
                func.__qualname__,
                elapsed_ms,
            )
            return result
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "✗ {}.{} failed after {:.2f}ms — {}",
                func.__module__,
                func.__qualname__,
                elapsed_ms,
                exc,
            )
            raise
    return wrapper


def log_async_execution_time(func: Callable) -> Callable:
    """Decorator: log async function execution time."""
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                "⏱ async {}.{} completed in {:.2f}ms",
                func.__module__,
                func.__qualname__,
                elapsed_ms,
            )
            return result
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "✗ async {}.{} failed after {:.2f}ms — {}",
                func.__module__,
                func.__qualname__,
                elapsed_ms,
                exc,
            )
            raise
    return wrapper


# Initialise on import
setup_logger()

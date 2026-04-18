# ─────────────────────────────────────────────────────────────────────
# Intelligent Demand Forecasting & Anomaly Detection System
# Multi-stage Dockerfile
# ─────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────
FROM python:3.10-slim-bullseye AS builder

WORKDIR /build

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip wheel

# Install Python dependencies into a prefix
COPY requirements.txt .
RUN pip install --user --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────
FROM python:3.10-slim-bullseye AS runtime

LABEL maintainer="demand-forecast-system"
LABEL description="Demand Forecasting & Anomaly Detection API"
LABEL version="1.0.0"

# Non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# System runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application source
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup config/ ./config/
COPY --chown=appuser:appgroup dashboard/ ./dashboard/
COPY --chown=appuser:appgroup .env.example ./.env

# Create required directories
RUN mkdir -p data/raw data/processed artifacts/models artifacts/plots \
             artifacts/metrics logs && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Add local packages to PATH
ENV PATH="/home/appuser/.local/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose ports
EXPOSE 8000  
# API
EXPOSE 8501  
# Dashboard

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: start API
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]

# ─────────────────────────────────────────────────────────────────────
# Demand Forecasting System — Makefile
# ─────────────────────────────────────────────────────────────────────

.PHONY: help install setup train api dashboard test docker-up docker-down clean lint

PYTHON := python3
PIP := pip3

# Default target
help:
	@echo ""
	@echo "⚡ Demand Forecasting System — Available Commands"
	@echo "─────────────────────────────────────────────────"
	@echo "  make install        Install Python dependencies"
	@echo "  make setup          Create .env from template"
	@echo "  make train          Run full training pipeline"
	@echo "  make train-lstm     Run pipeline with LSTM"
	@echo "  make api            Start FastAPI server"
	@echo "  make dashboard      Start Streamlit dashboard"
	@echo "  make test           Run all unit + integration tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make docker-up      Start all services with Docker Compose"
	@echo "  make docker-down    Stop all Docker services"
	@echo "  make clean          Remove generated artifacts and cache"
	@echo "  make lint           Run code linting (flake8/ruff)"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────

install:
	@echo "📦 Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Done"

setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅ Created .env from .env.example — edit with your credentials"; \
	else \
		echo "⚠️  .env already exists"; \
	fi
	@mkdir -p data/raw data/processed artifacts/models artifacts/plots artifacts/metrics logs

# ── Training ──────────────────────────────────────────────────────────

train:
	@echo "🚀 Running training pipeline..."
	PYTHONPATH=. $(PYTHON) -m src.pipeline

train-lstm:
	@echo "🧠 Running pipeline with LSTM..."
	PYTHONPATH=. $(PYTHON) -m src.pipeline --lstm

train-no-eda:
	@echo "⚡ Fast training (no EDA plots)..."
	PYTHONPATH=. $(PYTHON) -m src.pipeline --no-eda

# ── Services ──────────────────────────────────────────────────────────

api:
	@echo "🌐 Starting API on http://localhost:8000"
	PYTHONPATH=. uvicorn src.api.main:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload \
		--log-level info

dashboard:
	@echo "📊 Starting Dashboard on http://localhost:8501"
	PYTHONPATH=. streamlit run dashboard/app.py \
		--server.port 8501 \
		--server.address 0.0.0.0

# ── Testing ───────────────────────────────────────────────────────────

test:
	@echo "🧪 Running all tests..."
	PYTHONPATH=. $(PYTHON) -m pytest tests/ -v

test-unit:
	PYTHONPATH=. $(PYTHON) -m pytest tests/unit/ -v -m "not slow"

test-integration:
	PYTHONPATH=. $(PYTHON) -m pytest tests/integration/ -v

test-coverage:
	PYTHONPATH=. $(PYTHON) -m pytest tests/ --cov=src --cov-report=html
	@echo "Coverage report: artifacts/coverage/index.html"

# ── Docker ────────────────────────────────────────────────────────────

docker-build:
	docker compose build

docker-up:
	@echo "🐳 Starting services..."
	docker compose up -d db api dashboard
	@echo "API:       http://localhost:8000/docs"
	@echo "Dashboard: http://localhost:8501"

docker-train:
	docker compose --profile train run --rm trainer

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api

docker-reset:
	docker compose down -v --remove-orphans

# ── Utilities ─────────────────────────────────────────────────────────

lint:
	@echo "🔍 Linting..."
	$(PYTHON) -m flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503 || true
	@echo "Done"

clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov artifacts/coverage
	@echo "✅ Clean"

# Example API requests
example-predict:
	@curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"timestamp":"2023-06-15T14:00:00","features":{"demand_lag_1h":850.0,"demand_lag_24h":820.0,"demand_lag_168h":830.0,"demand_roll_mean_24h":840.0,"demand_roll_std_24h":45.0,"hour":14,"day_of_week":3,"month":6,"is_weekend":0,"is_peak_hour":0,"hour_sin":0.866,"hour_cos":-0.5}}' \
		| python3 -m json.tool

example-anomaly:
	@curl -s -X POST http://localhost:8000/detect-anomaly \
		-H "Content-Type: application/json" \
		-d '{"timestamp":"2023-06-15T14:00:00","actual_demand":2500.0,"predicted_demand":850.0,"method":"ensemble"}' \
		| python3 -m json.tool

example-health:
	@curl -s http://localhost:8000/health | python3 -m json.tool

.PHONY: help install install-dev test lint format type-check clean build docs serve-docs
.DEFAULT_GOAL := help

# Colors for output
BLUE = \033[0;34m
GREEN = \033[0;32m
RED = \033[0;31m
YELLOW = \033[0;33m
NC = \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)v10r Development Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# Environment Setup
setup: ## Initial project setup with uv
	@echo "$(BLUE)Setting up v10r development environment...$(NC)"
	@if ! command -v uv &> /dev/null; then \
		echo "$(RED)uv is not installed. Please install it first:$(NC)"; \
		echo "curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	fi
	uv venv
	@echo "$(YELLOW)Activate the virtual environment with:$(NC)"
	@echo "source .venv/bin/activate"

install: ## Install the package in development mode
	uv pip install -e .

install-dev: ## Install development dependencies
	uv pip install -e ".[dev,test,docs]"

# ============================================================================
# Testing Targets
# ============================================================================

.PHONY: test test-unit test-integration test-all test-coverage test-fast

test: test-unit  ## Run unit tests only (fast)
	@echo "✅ Unit tests completed"

test-unit:  ## Run unit tests
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "$(RED)ERROR: Virtual environment not activated!$(NC)"; \
		echo "$(YELLOW)Please run: source .venv/bin/activate$(NC)"; \
		exit 1; \
	fi
	pytest tests/unit/ -v

test-integration:  ## Run integration tests (requires Docker Compose)
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "$(RED)ERROR: Virtual environment not activated!$(NC)"; \
		echo "$(YELLOW)Please run: source .venv/bin/activate$(NC)"; \
		exit 1; \
	fi
	@echo "🚀 Starting integration tests with Docker Compose..."
	pytest tests/integration/ -v -m integration

test-all:  ## Run all tests (unit + integration)
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "$(RED)ERROR: Virtual environment not activated!$(NC)"; \
		echo "$(YELLOW)Please run: source .venv/bin/activate$(NC)"; \
		exit 1; \
	fi
	@echo "🧪 Running all tests..."
	pytest tests/ -v

test-fast:  ## Run only fast tests (exclude slow/integration)
	pytest tests/ -v -m "not slow and not integration"

test-coverage:  ## Run tests with coverage report
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "$(RED)ERROR: Virtual environment not activated!$(NC)"; \
		echo "$(YELLOW)Please run: source .venv/bin/activate$(NC)"; \
		exit 1; \
	fi
	pytest tests/ --cov=src/v10r --cov-report=html --cov-report=term-missing

test-watch:  ## Run tests in watch mode (requires pytest-watch)
	ptw tests/ -- -v

# Individual test running
test-embedding:  ## Run embedding-related tests only
	pytest tests/ -v -m embedding

test-database:  ## Run database-related tests only
	pytest tests/ -v -m database

# ============================================================================
# Docker and Environment Management
# ============================================================================

.PHONY: docker-up docker-down docker-logs docker-clean

docker-up:  ## Start test environment (Docker Compose)
	docker-compose -f docker-compose.test.yml up -d --build

docker-down:  ## Stop test environment
	docker-compose -f docker-compose.test.yml down -v

docker-logs:  ## Show Docker Compose logs
	docker-compose -f docker-compose.test.yml logs -f

docker-clean:  ## Clean Docker resources
	docker-compose -f docker-compose.test.yml down -v --remove-orphans
	docker system prune -f

# ============================================================================
# Development Environment
# ============================================================================

.PHONY: dev-setup dev-test dev-check

dev-setup: install  ## Set up development environment
	@echo "🔧 Setting up development environment..."
	@echo "✅ Development environment ready!"
	@echo ""
	@echo "Quick test commands:"
	@echo "  make test          - Run unit tests"
	@echo "  make test-integration - Run integration tests" 
	@echo "  make test-all      - Run all tests"

dev-test: test-fast  ## Quick development test (fast tests only)

dev-check: lint test-fast  ## Full development check (lint + fast tests)

# Code Quality
lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	ruff check .
	black --check .

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	ruff format .
	black .

type-check: ## Run type checking
	@echo "$(BLUE)Running type checks...$(NC)"
	mypy src/

check-all: lint type-check test ## Run all quality checks

# Cleaning
clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build and Release
build: clean ## Build the package
	@echo "$(BLUE)Building package...$(NC)"
	uv build

release-test: build ## Release to test PyPI
	@echo "$(BLUE)Uploading to test PyPI...$(NC)"
	uv publish --repository testpypi

release: build ## Release to PyPI
	@echo "$(BLUE)Uploading to PyPI...$(NC)"
	uv publish

# Documentation
docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	mkdocs build

serve-docs: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	mkdocs serve

# Development Server
run-listener: ## Run the v10r listener service
	@echo "$(BLUE)Starting v10r listener service...$(NC)"
	python -m v10r.cli listen --config examples/v10r-config.yaml

run-worker: ## Run the v10r worker service
	@echo "$(BLUE)Starting v10r worker service...$(NC)"
	python -m v10r.cli worker --config examples/v10r-config.yaml

# Database Setup
db-setup: ## Set up local PostgreSQL with pgvector
	@echo "$(BLUE)Setting up local PostgreSQL with pgvector...$(NC)"
	@echo "$(YELLOW)This requires Docker to be running$(NC)"
	docker run --name v10r-postgres -e POSTGRES_PASSWORD=v10r -e POSTGRES_DB=v10r -p 5432:5432 -d pgvector/pgvector:pg16

db-stop: ## Stop local PostgreSQL
	@echo "$(BLUE)Stopping local PostgreSQL...$(NC)"
	docker stop v10r-postgres

db-start: ## Start local PostgreSQL
	@echo "$(BLUE)Starting local PostgreSQL...$(NC)"
	docker start v10r-postgres

db-remove: ## Remove local PostgreSQL container
	@echo "$(BLUE)Removing local PostgreSQL container...$(NC)"
	docker rm -f v10r-postgres

# Redis Setup
redis-setup: ## Set up local Redis
	@echo "$(BLUE)Setting up local Redis...$(NC)"
	docker run --name v10r-redis -p 6379:6379 -d redis:7-alpine

redis-stop: ## Stop local Redis
	@echo "$(BLUE)Stopping local Redis...$(NC)"
	docker stop v10r-redis

redis-start: ## Start local Redis
	@echo "$(BLUE)Starting local Redis...$(NC)"
	docker start v10r-redis

redis-remove: ## Remove local Redis container
	@echo "$(BLUE)Removing local Redis container...$(NC)"
	docker rm -f v10r-redis

# Complete local environment
dev-stack-up: db-setup redis-setup ## Start complete development stack
	@echo "$(GREEN)Development stack is ready!$(NC)"
	@echo "PostgreSQL: postgresql://postgres:v10r@localhost:5432/v10r"
	@echo "Redis: redis://localhost:6379"

dev-stack-down: db-remove redis-remove ## Stop and remove development stack
	@echo "$(GREEN)Development stack removed$(NC)"

# Pre-commit hooks
pre-commit-install: ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	pre-commit install

pre-commit-run: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

# Project Info
info: ## Show project information
	@echo "$(BLUE)v10r Project Information$(NC)"
	@echo "Repository: https://github.com/yourusername/v10r"
	@echo "Python version: $(shell python --version)"
	@echo "uv version: $(shell uv --version)"
	@echo "Virtual environment: $(VIRTUAL_ENV)" 
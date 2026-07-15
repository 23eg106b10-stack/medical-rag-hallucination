# ── Development convenience wrapper (Windows-safe) ────────────────────────
# Usage:  make <target>  (from bash / WSL / Git Bash)
#         use "nmake" or just run the commands manually on Windows cmd

.PHONY: install install-dev test lint format clean help

help:
	@echo Available targets:
	@echo   install      Install production dependencies
	@echo   install-dev  Install all dependencies (incl. dev)
	@echo   test         Run tests with pytest
	@echo   lint         Run ruff and mypy
	@echo   format       Format code with black
	@echo   clean        Remove __pycache__ and .pyc files

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest --cov=.

lint:
	ruff check .
	mypy .

format:
	black .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -type f -name "*.pyc" -delete 2>/dev/null; \
	echo "Cleaned."
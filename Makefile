.PHONY: setup test lint fmt figures

setup:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

# regenerates every figure/table in the readme. empty until phase 1 has something to plot
figures:
	@echo "no figures yet"

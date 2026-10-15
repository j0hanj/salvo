.PHONY: setup test lint fmt suites figures

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

# verify every eval suite still expands to exactly what its .lock says
suites:
	uv run python -m scripts.build_suites

# regenerates every figure/table in the readme from scratch
figures: suites
	uv run python -m scripts.render_engagement

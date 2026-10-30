.PHONY: setup test lint fmt suites compare train scaling figures

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

# every baseline on the same suite: logged run + readme table + figure
compare:
	uv run python -m scripts.compare

# imitation learning: bc on milp's episodes + 3 dagger rounds. a few minutes on a laptop cpu
train:
	uv run python -m scripts.train

# zero-shot scaling study, uses the newest model in results/
scaling:
	uv run python -m scripts.scaling

# regenerates every figure/table in the readme from scratch
figures: suites compare train scaling
	uv run python -m scripts.render_engagement

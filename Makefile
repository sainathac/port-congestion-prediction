.PHONY: test lint build train evaluate explain all

test:
	uv run pytest

lint:
	uv run ruff check .

build:
	uv run python -m src.build_dataset

train:
	uv run python -m src.train

evaluate:
	uv run python -m src.evaluate

explain:
	uv run python -m src.explain --report

all: build train evaluate explain test lint

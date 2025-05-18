all: format lint typecheck

format:
	ruff format models

lint:
	ruff check --fix models

typecheck:
	pyright models

install:
	uv sync
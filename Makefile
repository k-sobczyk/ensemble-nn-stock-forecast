all: format lint typecheck

format:
	ruff format src

lint:
	ruff check --fix src

typecheck:
	pyright models

install:
	uv sync
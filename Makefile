all: format lint

format:
	ruff format src

lint:
	ruff check --fix src

install:
	uv sync

.PHONY: evaluate

CONFIG_FILE=cats_vs_dogs_v1.yaml

evaluate:
	uv run modal run -m src.image_to_json.evaluate --config-file-name $(CONFIG_FILE)

local-run:
	uv run -m src.image_to_json.evaluate

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

format:
	uv run ruff format .

lint-fix-and-format: lint-fix format
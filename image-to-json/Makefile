.PHONY: evaluate

evaluate:
	uv run modal run -m src.image_to_json.evaluate --config-file-name $(CONFIG_FILE)

evaluate-cats-vs-dogs:
	uv run modal run -m src.image_to_json.evaluate --config-file-name cats_vs_dogs_v1.yaml

evaluate-standford-cars:
	uv run modal run -m src.image_to_json.evaluate --config-file-name standford_cars_v0.yaml

local-run:
	uv run -m src.image_to_json.evaluate

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

format:
	uv run ruff format .

lint-fix-and-format: lint-fix format
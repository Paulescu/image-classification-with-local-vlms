.PHONY: evaluate

# example usage: make evaluate CONFIG_FILE=cats_vs_dogs_v1.yaml
evaluate:
	uv run modal run -m src.image_to_json.evaluate --config-file-name $(CONFIG_FILE)

lm-server:
	llama-server -hf LiquidAI/LFM2-VL-450M-GGUF

train-test-split:
	uv run python scripts/generate_train_test_split_datasets.py \
		--input-dataset-name $(INPUT_DATASET_NAME) \
		--output-dir $(OUTPUT_DIR) \
		--train-size $(TRAIN_SIZE) \
		--seed $(SEED)
	


# Linting and formatting
lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

format:
	uv run ruff format .

lint-fix-and-format: lint-fix format
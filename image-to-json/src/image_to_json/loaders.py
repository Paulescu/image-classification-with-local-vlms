from typing import Optional

import datasets
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_dataset(
    dataset_name: str,
    split: str,
    n_samples: Optional[int] = None,
    seed: Optional[int] = 42,
) -> datasets.Dataset:
    """
    Loads a dataset from the Hugging Face dataset hub.
    """
    print(f"Loading dataset {dataset_name}")
    dataset = datasets.load_dataset(dataset_name, split=split, num_proc=1)

    # Shuffle the dataset
    dataset = dataset.shuffle(seed=seed)

    # Select a subset of the dataset
    if n_samples is not None:
        n_samples = min(n_samples, dataset.num_rows)
        dataset = dataset.select(range(n_samples))

    print(f"Dataset {dataset_name} loaded successfully: {dataset.num_rows} rows")

    return dataset


def load_model_and_processor(
    model_id: str,
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """
    Loads a model and processor from the Hugging Face model hub.
    """
    print("📚 Loading processor...")
    processor_source = model_id
    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=True,
        max_image_tokens=256,
    )

    print("🧠 Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype="bfloat16",
        trust_remote_code=True,
        device_map="auto",
    )

    print("\n✅ Local model loaded successfully!")
    print(f"📖 Vocab size: {len(processor.tokenizer)}")
    print(
        f"🖼️ Image processed in up to {processor.max_tiles} patches of size {processor.tile_size}"
    )
    print(f"🔢 Parameters: {model.num_parameters():,}")
    print(f"💾 Model size: ~{model.num_parameters() * 2 / 1e9:.1f} GB (bfloat16)")

    return model, processor
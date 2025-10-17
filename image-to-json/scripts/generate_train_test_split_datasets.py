
from datasets import Dataset, load_dataset
from PIL import Image


def generate_train_test_split_datasets(
    input_dataset_hf_id: str,
    output_dataset_hf_id: str,
    train_size: float = 0.8,
    seed: int = 42,
):
    """
    Loads the `input_dataset_hf_id` dataset from Hugging Face, splits it into training and testing sets
    based on the `train_size` ratio, and saves the resulting datasets to `output_dataset_hf_id` on Hugging Face Hub.

    Args:
        input_dataset_hf_id (str): The Hugging Face dataset identifier to load.
        output_dataset_hf_id (str): The Hugging Face dataset identifier to save the split
        train_size (float): The proportion of the dataset to include in the training set. Defaults to 0.8.
        seed (int): Random seed for reproducibility. Defaults to 42.
    """
    print(f"Loading dataset: {input_dataset_hf_id}")
    dataset = load_dataset(input_dataset_hf_id)

    if "train" in dataset:
        original_dataset = dataset["train"]
    else:
        raise ValueError("The input dataset must have a 'train' split.")

    if not isinstance(original_dataset, Dataset):
        raise ValueError("Expected a Dataset object")

    print(f"Original dataset size: {len(original_dataset)}")

    # Filter out images that are not in RGB mode
    def is_rgb_image(example):
        """Check if the image is in RGB mode."""
        try:
            image = example["image"]  # Assuming the image column is named "image"
            if isinstance(image, Image.Image):
                return image.mode == "RGB"
            else:
                # If it's not a PIL Image, try to convert and check
                if hasattr(image, 'mode'):
                    return image.mode == "RGB"
                return True  # Default to keeping if we can't determine mode
        except Exception:
            return False  # Filter out if there's any error processing the image

    print("Filtering images to keep only RGB mode...")
    original_dataset = original_dataset.filter(is_rgb_image)
    print(f"Dataset size after RGB filtering: {len(original_dataset)}")

    stratify_column = "labels" if "labels" in original_dataset.column_names else None
    split_dataset = original_dataset.train_test_split(
        train_size=train_size,
        seed=seed,
        stratify_by_column=stratify_column
    )

    print(f"Train set size: {len(split_dataset['train'])}")
    print(f"Test set size: {len(split_dataset['test'])}")

    print(f"Pushing dataset to: {output_dataset_hf_id}")
    split_dataset.push_to_hub(output_dataset_hf_id)


if __name__ == "__main__":

    generate_train_test_split_datasets(
        input_dataset_hf_id="microsoft/cats_vs_dogs",
        output_dataset_hf_id="Paulescu/cats_vs_dogs",
        train_size=0.9,
        seed=42,
    )

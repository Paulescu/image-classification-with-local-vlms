
from datasets import load_dataset


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
    
    print(f"Original dataset size: {len(original_dataset)}")
    
    split_dataset = original_dataset.train_test_split(
        train_size=train_size, 
        seed=seed,
        stratify_by_column="labels" if "labels" in original_dataset.column_names else None
    )
    
    print(f"Train set size: {len(split_dataset['train'])}")
    print(f"Test set size: {len(split_dataset['test'])}")
    
    print(f"Pushing dataset to: {output_dataset_hf_id}")
    split_dataset.push_to_hub(output_dataset_hf_id)


if __name__ == "__main__":
    
    generate_train_test_split_datasets(
        input_dataset_hf_id="microsoft/cats_vs_dogs",
        output_dataset_hf_id="Paulescu/cats_vs_dogs",
        train_size=0.8,
        seed=42,
    )
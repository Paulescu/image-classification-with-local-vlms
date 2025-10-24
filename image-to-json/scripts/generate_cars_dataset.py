#!/usr/bin/env python3
"""
Script to generate a filtered Stanford Cars dataset.
- Loads the original tanganke/stanford_cars dataset
- Keeps only train and test splits
- Filters to keep only JPEG images (JpegImageFile type)
- Transforms labels into separate maker, model, year columns
- Pushes to HF under Paulescu/stanford_cars
"""

import datasets
import json
from tqdm import tqdm
from PIL.JpegImagePlugin import JpegImageFile


def load_car_class_mapping():
    """Load the car class mapping from JSON file."""
    with open('stanford_cars_classes.json', 'r') as f:
        classes = json.load(f)
    
    # Create mapping from class_id to car info
    class_mapping = {}
    for car_class in classes:
        class_mapping[car_class['class_id']] = {
            'maker': car_class['maker'],
            'model': car_class['model'],
            'year': car_class['year']
        }
    
    return class_mapping


def filter_jpeg_images(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Filter dataset to keep only JPEG images.
    
    Args:
        dataset: Input dataset
        
    Returns:
        Filtered dataset with only JPEG images
    """
    print(f"Filtering {len(dataset)} samples to keep only JPEG images...")
    
    def is_jpeg_image(example):
        image = example['image']
        return isinstance(image, JpegImageFile)
    
    filtered_dataset = dataset.filter(is_jpeg_image)
    print(f"Kept {len(filtered_dataset)} JPEG images out of {len(dataset)} total")
    
    return filtered_dataset


def transform_dataset_structure(dataset: datasets.Dataset, class_mapping: dict) -> datasets.Dataset:
    """
    Transform dataset to have image, maker, model, year columns.
    
    Args:
        dataset: Input dataset with image and label columns
        class_mapping: Mapping from class_id to car info
        
    Returns:
        Transformed dataset with 4 columns: image, maker, model, year
    """
    print(f"Transforming dataset structure for {len(dataset)} samples...")
    
    def transform_example(example):
        label_id = example['label']
        car_info = class_mapping[label_id]
        
        return {
            'image': example['image'],
            'maker': car_info['maker'],
            'model': car_info['model'],
            'year': car_info['year']
        }
    
    transformed_dataset = dataset.map(transform_example)
    print(f"Transformed {len(transformed_dataset)} samples")
    
    return transformed_dataset


def main(
    input_dataset_name: str,
    output_dataset_name: str,
):
    """Load, filter, and transform the Stanford Cars dataset."""
    
    print(f"Loading dataset: {input_dataset_name}")
    
    # Load car class mapping
    print("Loading car class mapping...")
    class_mapping = load_car_class_mapping()
    
    # Load train and test splits
    train_dataset = datasets.load_dataset(input_dataset_name, split="train")
    test_dataset = datasets.load_dataset(input_dataset_name, split="test")
    
    # Ensure we have Dataset objects (not DatasetDict)
    if isinstance(train_dataset, datasets.DatasetDict):
        train_dataset = train_dataset['train']
    if isinstance(test_dataset, datasets.DatasetDict):
        test_dataset = test_dataset['test']
    
    print(f"Original train dataset: {len(train_dataset)} samples")
    print(f"Original test dataset: {len(test_dataset)} samples")
    
    # Filter to keep only JPEG images
    train_filtered = filter_jpeg_images(train_dataset)
    test_filtered = filter_jpeg_images(test_dataset)
    
    # Transform dataset structure to have 4 columns: image, maker, model, year
    train_transformed = transform_dataset_structure(train_filtered, class_mapping)
    test_transformed = transform_dataset_structure(test_filtered, class_mapping)
    
    # Combine into DatasetDict
    final_dataset = datasets.DatasetDict({
        'train': train_transformed,
        'test': test_transformed
    })
    
    # Push to Hugging Face Hub
    print(f"Pushing transformed dataset to: {output_dataset_name}")
    final_dataset.push_to_hub(output_dataset_name)
    
    print("Dataset generation completed successfully!")
    print(f"Train samples: {len(train_transformed)}")
    print(f"Test samples: {len(test_transformed)}")
    print(f"Dataset columns: {list(train_transformed.column_names)}")
    print(f"Dataset available at: https://huggingface.co/datasets/{output_dataset_name}")


if __name__ == "__main__":
    main(
        input_dataset_name="tanganke/stanford_cars",
        output_dataset_name="Paulescu/stanford_cars",
    )
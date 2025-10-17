"""Debug script to investigate image formats in the dataset."""

import datasets
from PIL import Image
import numpy as np
from collections import Counter


def debug_image_formats():
    """Load the cats vs dogs dataset and analyze image formats."""
    print("Loading Paulescu/cats_vs_dogs dataset...")
    
    # Load the dataset
    dataset = datasets.load_dataset("Paulescu/cats_vs_dogs", split="train")
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Track image formats
    image_types = []
    image_modes = []
    image_shapes = []
    errors = []
    
    print("\nAnalyzing image formats...")
    
    for i, sample in enumerate(dataset):
        try:
            image = sample["image"]
            
            # Record the type
            image_type = type(image).__name__
            image_types.append(image_type)
            
            # If it's a PIL Image, get the mode
            if isinstance(image, Image.Image):
                image_modes.append(image.mode)
                image_shapes.append(image.size)  # (width, height)
            elif isinstance(image, np.ndarray):
                image_modes.append("numpy_array")
                image_shapes.append(image.shape)
            else:
                image_modes.append("unknown")
                image_shapes.append("unknown")
            
            # Print details for first 10 samples
            if i < 10:
                print(f"Sample {i}:")
                print(f"  Type: {image_type}")
                if isinstance(image, Image.Image):
                    print(f"  Mode: {image.mode}")
                    print(f"  Size: {image.size}")
                elif isinstance(image, np.ndarray):
                    print(f"  Shape: {image.shape}")
                    print(f"  Dtype: {image.dtype}")
                print(f"  Label: {sample['labels']}")
                print()
                
        except Exception as e:
            errors.append(f"Sample {i}: {str(e)}")
            if len(errors) < 5:  # Print first 5 errors
                print(f"ERROR in sample {i}: {e}")
    
    # Summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    print(f"\nTotal samples: {len(dataset)}")
    print(f"Errors encountered: {len(errors)}")
    
    print(f"\nImage types:")
    type_counts = Counter(image_types)
    for img_type, count in type_counts.items():
        print(f"  {img_type}: {count}")
    
    print(f"\nImage modes:")
    mode_counts = Counter(image_modes)
    for mode, count in mode_counts.items():
        print(f"  {mode}: {count}")
    
    print(f"\nUnique shapes (first 20):")
    shape_counts = Counter(str(shape) for shape in image_shapes)
    for shape, count in list(shape_counts.items())[:20]:
        print(f"  {shape}: {count}")
    
    if errors:
        print(f"\nFirst few errors:")
        for error in errors[:5]:
            print(f"  {error}")


if __name__ == "__main__":
    debug_image_formats()
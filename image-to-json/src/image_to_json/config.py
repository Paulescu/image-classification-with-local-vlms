""" """

from pydantic_settings import BaseSettings
import yaml
from pathlib import Path

from .paths import get_path_to_configs

class EvaluationConfig(BaseSettings):
    seed: int = 23

    # Model parameters
    model: str = "LiquidAI/LFM2-VL-450M"

    structured_generation: bool = False

    # Dataset parameters
    dataset: str = "microsoft/cats_vs_dogs"
    n_samples: int = 100

    system_prompt: str = """
    You are a veterinarian specialized in analyzing pictures of cats and dogs
    You excel at identifying the type of animal from a picture and its breed.
    """

    user_prompt: str = """
    What animal and breed do you see in this image?

    Provide your final JSON response without any additional text or formatting.

    Example output 1:
    {
      "animal": "dog",
      "breed": "Labrador"
    }

    Example output 2:
    {
      "animal": "cat",
      "breed": "Siamese"
    }

    Example output 3:
    {
      "animal": "other",
    }

    """
    image_column: str = "image"
    label_column: str = "labels"
    label_mapping: dict = {
        0: "cat",
        1: "dog",
    }

    @classmethod
    def from_yaml(cls, file_name: str) -> "EvaluationConfig":
        """
        Loads configuration from a YAML file located in the configs directory.
        """
        file_path = str(Path(get_path_to_configs()) / file_name)
        print(f"Loading config from {file_path}")
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls(**data)


# from pydantic import BaseModel


# class CatsVsDogsClassificationOutputType(BaseModel):
#     animal: str
#     breed: str


# eval_config = EvaluationConfig()

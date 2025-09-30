""" """

from pathlib import Path

import yaml
from pydantic_settings import BaseSettings

from .paths import get_path_to_configs


class EvaluationConfig(BaseSettings):
    seed: int = 23

    # Model parameters
    model: str
    structured_generation: bool

    # Dataset parameters
    dataset: str
    split: str
    n_samples: int
    system_prompt: str
    user_prompt: str
    image_column: str
    label_column: str
    label_mapping: dict

    @classmethod
    def from_yaml(cls, file_name: str) -> "EvaluationConfig":
        """
        Loads configuration from a YAML file located in the configs directory.
        """
        file_path = str(Path(get_path_to_configs()) / file_name)
        print(f"Loading config from {file_path}")
        with open(file_path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

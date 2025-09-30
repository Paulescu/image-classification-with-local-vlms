from typing import Literal, TypeVar

from pydantic import BaseModel

ModelOutputType = TypeVar('ModelOutputType', bound=BaseModel)

class CatsVsDogsClassificationOutputType(BaseModel):
    pred_class: Literal["cat", "dog"]

def get_model_output_schema(dataset_name: str) -> BaseModel:
    if dataset_name == "microsoft/cats_vs_dogs":
        return CatsVsDogsClassificationOutputType
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")



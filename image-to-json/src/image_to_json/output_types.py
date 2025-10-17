from typing import Literal, TypeVar

from pydantic import BaseModel

ModelOutputType = TypeVar('ModelOutputType', bound=BaseModel)

class CatsVsDogsClassificationOutputType(BaseModel):
    pred_class: Literal["cat", "dog"]
    
    @classmethod
    def from_pred_class(cls, pred_class: str) -> str:
        """Create instance from pred_class and return as JSON string."""
        instance = cls(pred_class=pred_class)
        return instance.model_dump_json()

def get_model_output_schema(dataset_name: str) -> BaseModel:
    if dataset_name == "microsoft/cats_vs_dogs":
        return CatsVsDogsClassificationOutputType
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")



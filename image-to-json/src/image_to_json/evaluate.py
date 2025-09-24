"""
Evaluates a VL model on a given dataset

Steps:
1. Download the dataset
2. Load the model
3. Loop through the dataset and compute model outputs
4. Compute accuracy as a binary score: 1 if the model output matches the ground truth, 0 otherwise
"""
from tqdm import tqdm

from .modal_infra import (
    get_modal_app,
    get_docker_image,
    get_volume,
    get_retries,
)
from .config import EvaluationConfig
from .inference import get_model_output, get_structured_model_output
from .report import save_predictions_to_disk
from .loaders import load_dataset, load_model_and_processor

app = get_modal_app("vlm-model-evaluation")
image = get_docker_image()
volume = get_volume("vlm-model-evaluation-datasets")


@app.function(
    image=image,
    gpu="L40S",
    volumes={
        "/datasets": volume,
    },
    # secrets=get_secrets(),
    timeout=1 * 60 * 60,
    retries=get_retries(max_retries=1),
    max_inputs=1,  # Ensure we get a fresh container on retry
)
def evaluate(
    config: EvaluationConfig,
 ) -> list[dict]:
    """
    """
    print(f"Starting evaluation of {config.model} on {config.dataset}")

    dataset = load_dataset(
        dataset_name=config.dataset,
        split=config.split,
        n_samples=config.n_samples,
        seed=config.seed
    )

    model, processor = load_model_and_processor(model_id=config.model)
    
    # Prepare CSV file
    predictions_data = []

    # Naive evaluation loop without batching
    accurate_predictions: int = 0
    for sample in tqdm(dataset):
        
        # Extracts sample image and normalized label
        image = sample[config.image_column]
        label = config.label_mapping[sample[config.label_column]]

        # create the conversation
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": config.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": config.user_prompt},
                ],
            },
        ]

        if config.structured_generation:
            output: dict = get_structured_model_output(
                model, processor, config.user_prompt, image
            )
        else:
            output: str = get_model_output(model, processor, conversation)

        # Compare predicton vs ground truth.
        accurate_predictions += 1 if output == label else 0

        # Save prediction to list
        predictions_data.append({
            "ground_truth": label,
            "predicted": output,
            "correct": output == label
        })

        print("--------------------------------")
    
    print(f"Accuracy: {accurate_predictions / len(dataset):.2f}")

    print("Evaluation completed successfully")

    return predictions_data



@app.local_entrypoint()
def main(
    # model: str,
    # dataset: str,
    config_file_name: str,
):
    # config = EvaluationConfig()
    config = EvaluationConfig.from_yaml(config_file_name)

    predictions_data = evaluate.remote(config)

    # print('Predictions: ', predictions_data)
    output_path = save_predictions_to_disk(predictions_data)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    config = EvaluationConfig()

    print(f"Loading dataset {config.dataset}")
    dataset = load_dataset(
        dataset_name=config.dataset, n_samples=config.n_samples, seed=config.seed
    )
    print(f"Dataset loaded successfully: {dataset.num_rows} rows")

    # Naive evaluation loop without batching
    accurate_predictions: int = 0
    for sample in dataset:
        print("Extracting sample image and normalized label")
        image = sample[config.image_column]

        # breakpoint()

        try:
            label = config.label_mapping[sample[config.label_column]]
        except KeyError:
            print("Error mapping label: ", sample[config.label_column])
            breakpoint()

        print("--------------------------------")

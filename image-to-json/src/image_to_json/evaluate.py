"""
Evaluates a VL model on a given dataset

Steps:
1. Download the dataset
2. Load the model
3. Loop through the dataset and compute model outputs
4. Compute accuracy as a binary score: 1 if the model output matches the ground truth, 0 otherwise
"""

import json
from typing import Optional

from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
import datasets

from .modal_infra import (
    get_modal_app,
    get_docker_image,
    get_volume,
    get_retries,
)
from .config import EvaluationConfig
from .inference import get_model_output, get_structured_model_output

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
):
    """ """
    print(f"Starting evaluation of {config.model} on {config.dataset}")

    dataset = load_dataset(
        dataset_name=config.dataset, n_samples=config.n_samples, seed=config.seed
    )

    model, processor = load_model_and_processor(model_id=config.model)

    # TODO
    # test_one_sample(model, processor)

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

        # # Parse output as dict
        # try:
        #     response = json.loads(output)
        # except json.JSONDecodeError:
        #     print("Error parsing model output: ", output)
        #     continue
        # print("Parsed response:", response)

        # Compare predicton vs ground truth.
        accurate_predictions += 1 if output == label else 0

        print("--------------------------------")

    print(f"Accuracy: {accurate_predictions / len(dataset):.2f}")

    print("Evaluation completed successfully")


def load_dataset(
    dataset_name: str,
    n_samples: Optional[int] = None,
    seed: Optional[int] = 42,
) -> datasets.Dataset:
    """
    Loads a dataset from the Hugging Face dataset hub.
    """
    print(f"Loading dataset {dataset_name}")
    dataset = datasets.load_dataset(dataset_name, split="train", num_proc=1)

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


def test_one_sample(model, processor):
    """ """
    from transformers.image_utils import load_image

    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = load_image(url)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": config.prompt},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(model.device)

    print("type(inputs)", type(inputs))
    # print("inputs.shape", inputs.shape)

    outputs = model.generate(**inputs, max_new_tokens=64)
    output = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    print("Model prediction: ", output)


@app.local_entrypoint()
def main(
    # model: str,
    # dataset: str,
    config_file_name: str,
):
    # config = EvaluationConfig()
    config = EvaluationConfig.from_yaml(config_file_name)

    evaluate.remote(config)


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

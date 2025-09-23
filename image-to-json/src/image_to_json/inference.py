from typing import Optional
import re

import outlines
from outlines.inputs import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def get_structured_model_output(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    user_prompt: str,
    image,
    # conversation: list[dict],
    max_new_tokens: Optional[int] = 64,
) -> str:
    """ """
    model = outlines.from_transformers(model, processor)

    # Wrap image
    # prompt = Chat(conversation)
    # TODO: quick hack
    # from .config import EvaluationConfig
    # config = EvaluationConfig()
    # prompt = Chat([
    #     {
    #         "role": "system",
    #         "content": [{"type": "text", "text": config.system_prompt}],
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": Image(image)},
    #             {"type": "text", "text": config.user_prompt},
    #         ],
    #     },
    # ])

    # image = image.convert("RGB")
    image.format = "JPEG"
    prompt = [f"<image>{user_prompt}", Image(image)]

    from .config import CatsVsDogsClassificationOutputType
    response = model(
        prompt,
        output_type=CatsVsDogsClassificationOutputType,
        max_new_tokens=max_new_tokens,
    )
    return response


def get_model_output(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    conversation: list[dict],
    max_new_tokens: Optional[int] = 64,
) -> str:
    """
    Gets the model output for a given conversation
    """
    # Generate Answer
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(model.device)

    # print('type(inputs)', type(inputs))
    # print('dir(inputs)', dir(inputs))

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    outputs_wout_input_tokens = outputs[:, inputs["input_ids"].shape[1] :]

    output = processor.batch_decode(
        outputs_wout_input_tokens, skip_special_tokens=True
    )[0]

    # Find first complete JSON object
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output)
    if match:
        output = match.group()

    return output

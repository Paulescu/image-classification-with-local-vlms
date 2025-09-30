from .modal_infra import (
    get_docker_image,
    get_modal_app,
    get_retries,
    get_volume,
)

app = get_modal_app("leap-sft-with-modal")
image = get_docker_image()
volume = get_volume("leap-sft-with-modal-model-cache")


@app.function(
    image=image,
    gpu="L40S",
    volumes={
        "/model_checkpoints": volume,
    },
    # secrets=get_secrets(),
    timeout=1 * 60 * 60,
    retries=get_retries(max_retries=1),
    max_inputs=1,  # Ensure we get a fresh container on retry
)
def fine_tune(
    run_name: str,
):
    """Fine-tune an Image-text-to-Text model using LoRA and SFT."""
    print("Run name:", run_name)

    import os

    import torch
    import transformers
    import trl

    os.environ["WANDB_DISABLED"] = "true"

    print(f"📦 PyTorch version: {torch.__version__}")
    print(f"🤗 Transformers version: {transformers.__version__}")
    print(f"📊 TRL version: {trl.__version__}")

    # Load the model and processor
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = "LiquidAI/LFM2-VL-450M"  # or LiquidAI/LFM2-VL-1.6B

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

    from datasets import load_dataset

    raw_ds = load_dataset("simwit/omni-med-vqa-mini")
    full_dataset = raw_ds["test"]
    split = full_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print("✅ SFT Dataset loaded:")
    print(f"   📚 Train samples: {len(train_dataset)}")
    print(f"   🧪 Eval samples: {len(eval_dataset)}")
    print(
        f"\n📝 Single Sample: [IMAGE] {train_dataset[0]['question']} {train_dataset[0]['gt_answer']}"
    )

    system_message = (
        "You are a medical Vision Language Model specialized in analyzing medical images and providing clinical insights. "
        "Provide concise, clinically relevant answers based on the image and question."
    )

    def format_medical_sample(sample):
        return [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": sample["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["gt_answer"]}],
            },
        ]

    train_dataset = [format_medical_sample(s) for s in train_dataset]
    eval_dataset = [format_medical_sample(s) for s in eval_dataset]

    print("✅ SFT Dataset formatted:")
    print(f"   📚 Train samples: {len(train_dataset)}")
    print(f"   🧪 Eval samples: {len(eval_dataset)}")

    def create_collate_fn(processor):
        """Create a collate function that prepares batch inputs for the processor."""

        def collate_fn(sample):
            batch = processor.apply_chat_template(
                sample, tokenize=True, return_dict=True, return_tensors="pt"
            )
            labels = batch["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
            return batch

        return collate_fn

    collate_fn = create_collate_fn(processor)

    # Optional: model = get_peft_model(model, peft_config)

    from trl import SFTConfig, SFTTrainer

    sft_config = SFTConfig(
        output_dir="lfm2-vl-med",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        optim="adamw_torch_8bit",
        gradient_checkpointing=True,
        max_length=512,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to=None,
    )

    print("🏗️  Creating SFT trainer...")
    sft_trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,
    )


@app.local_entrypoint()
def main(run_name: str):
    fine_tune.remote(run_name=run_name)

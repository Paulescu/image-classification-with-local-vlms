## Evaluating and fine-tuning VL models for text-to-json tasks

- Evaluating LFM2-VL-450M on [Cats vs Dogs classification task](https://huggingface.co/datasets/microsoft/cats_vs_dogs).
    - Post-process to extract first complete JSON object

- Ditch the JSON formatting and just output classes as strings.
    - [x] Extract configs to YAML files.
    - [x] Solve cats and dogs
    - [x] Solve Human_Action_Recognition -> a hug counter.
    - [ ] Generate train/test split of the standford-cars-dataset and push to HF
    - [ ] Fine-tune the VL to the train split, evaluate on the test one.
        - [ ] Use as much code as possible from leap-finetune
    
    - [ ] Fine-tune VL for image classification

- Evaluating LFM2-VL-450M on [Car model classification task](https://huggingface.co/datasets/Multimodal-Fatima/StanfordCars_test).

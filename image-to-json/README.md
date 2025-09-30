# Evaluating and fine-tuning VL models for Image classification tasks

## Task 1. Cats vs Dogs

```sh
# Using LFM2-VL-450M
$ make evaluate CONFIG_FILE=cats_vs_dogs_v1.yaml

Accuracy: 0.97
```

```sh
# Using LFM2-VL-1.6B
$ make evaluate CONFIG_FILE=cats_vs_dogs_v1.yaml

Accuracy: 0.99
```

## Task 2. Human Action Recognition

```sh
make evaluate CONFIG_FILE=human_action_recognition_v1.yaml
```

## Task 3. Car brand, model and year identification

```sh
make evaluate CONFIG_FILE=car_brand_model_year_v1.yaml
```

## TODOs

- [ ] Automatic prompt optimization to get LFM2-VL-450M to 99% accuracy.
    - [x] Start llama.cpp server
    - [ ] Send a curl request with a text and an image in the user prompt
    - [ ] Build DSPy program and optimize the instruction prompt.
    
- [ ] Cache base models downloaded from HF using Modal volumes.

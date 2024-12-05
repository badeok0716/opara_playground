#!/bin/bash

# Define the options for model and batch size
nbatchs=("1" "4" "16")
models=("deepfm" "googlenet" "nasnet")
conda activate nimble

# Loop through each combination of model and batch size
for nbatch in "${nbatchs[@]}"; do
    for model in "${models[@]}"; do
        echo "Running eval_nimble.py with model=${model} and nbatch=${nbatch}"
        python examples/eval_nimble.py --model "$model" --batch_size "$nbatch"
    done
done
#!/bin/bash

# Define the options for model and method
nbatchs=("1" "4" "16")
models=("deepfm" "googlenet" "nasnet")
methods=("torch" "torch_compile" "sequence" "opara")

conda activate stream
# Loop through each combination of model and method
for nbatch in "${nbatchs[@]}"; do
    for model in "${models[@]}"; do
        for method in "${methods[@]}"; do
            echo "Running eval_opara.py with model=${model} and method=${method} and nbatch=${nbatch}"
            python examples/eval_opara.py --model "$model" --method "$method" --batch_size "$nbatch"
        done
    done
done
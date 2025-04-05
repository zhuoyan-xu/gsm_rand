#!/bin/bash

# Array of seed values
seeds=(37 42 134 1567 8787)

# Loop through each seed
for seed in "${seeds[@]}"
do
    echo "Running with seed: $seed"
    python ground_truth.py --variable_seed $seed
    echo "----------------------------------------"
done
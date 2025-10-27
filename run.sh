#!/bin/bash

set -e

echo "Starting WCST Transformer v2 Curriculum Learning..."

echo "Running: Train Rule 0"
python train_v2.py --force_rule 0 --max_lr 5e-5 --n_steps 30000 --save_suffix "rule_0"

echo "Running: Fine-tune on Rule 1"
python train_v2.py --force_rule 1 --load_model wcst_transformer_rule_0.pth --max_lr 1e-5 --n_steps 30000 --save_suffix "rule_0_1"

echo "Running: Fine-tune on Rule 2"
python train_v2.py --force_rule 2 --load_model wcst_transformer_rule_0_1.pth --max_lr 1e-5 --n_steps 30000 --save_suffix "rule_0_1_2"

echo "Running: Fine-tune on Random Switching"
python train_v2.py --force_rule -1 --load_model wcst_transformer_rule_0_1_2.pth --max_lr 1e-5 --n_steps 50000 --save_suffix "final_v2"

echo "Running: Evaluate Final Model"
python evaluate_v2.py --model_path wcst_transformer_final_v2.pth

echo "Script Finished."

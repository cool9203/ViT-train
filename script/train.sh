# /bin/bash

python -m src.train \
    --output_dir "ViT-classification" \
    --learning_rate 2e-4 \
    --train_epochs 4 \
    --batch_size 16 \
    --dataset_path ./data/dataset

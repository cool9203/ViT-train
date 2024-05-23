# /bin/bash

python -m src.make_dataset.py \
    --image_folder ./data/component \
    --output_folder ./data/dataset \
    --image_count 100

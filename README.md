# ViT-train

## How to run

### Install require package

```shell
pip install -r requirements.txt

# install torch-cpu/cuda 12.1
pip install torch

# install torch-cuda 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Make dataset

`sh ./script/make_dataset.sh`

Change your image data path in line 4

### Train ViT model

`sh ./script/train.sh`

# coding: utf-8

from pathlib import Path

import torch
import tqdm
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


def predict(
    processor: ViTImageProcessor,
    model: ViTForImageClassification,
    image,
) -> torch.Tensor:
    with torch.no_grad():
        logits = model(**processor(image, return_tensors="pt")).logits
        return logits.argmax(axis=-1)


if __name__ == "__main__":
    model_path = "./model/checkpoint-400"
    dataset_path = "./data/dataset/eval"

    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path)

    accuracy = 0
    all_count = 0

    folder_iter = [folder for folder in Path(dataset_path).iterdir()]
    for i in tqdm.trange(len(folder_iter)):
        folder = folder_iter[i]
        file_iter = [file for file in Path(folder).iterdir()]
        for j in tqdm.trange(len(file_iter), desc="Folder progress", leave=False):
            filename = file_iter[j]
            if filename.suffix in [".jsonl", ".csv", ".json"]:
                continue

            all_count += 1
            result = predict(
                processor=processor,
                model=model,
                image=Image.open(
                    str(filename),
                ),
            )
            result = result.cpu().detach().numpy()
            accuracy += 1 if result[0] == i else 0

    print(f"accuracy: {accuracy/all_count}")

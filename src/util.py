# coding: utf-8

from pathlib import Path

import datasets
import pandas as pd
from PIL import Image


class MyDataset(datasets.GeneratorBasedBuilder):
    """My image dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "labels": datasets.Value("int64"),
                    "label_text": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": Path(self.config_kwargs["data_dir"], "train"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": Path(self.config_kwargs["data_dir"], "validation"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": Path(self.config_kwargs["data_dir"], "eval"),
                },
            ),
        ]

    def _generate_examples(self, filepath: str):
        for label_text in Path(filepath).iterdir():
            metadata = pd.read_json(path_or_buf=str(Path(label_text, "metadata.jsonl")), lines=True)
            for image_path in label_text.iterdir():
                if image_path.suffix in [".png", ".jpg", ".jpeg"]:
                    label = list(metadata[metadata["file_name"] == image_path.name].label.values)[0]
                    yield (
                        str(image_path),
                        {
                            "image": Image.open(str(image_path)),
                            "labels": label,
                            "label_text": label_text.name,
                        },
                    )

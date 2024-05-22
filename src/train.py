# coding: utf-8
# Reference: https://huggingface.co/blog/fine-tune-vit
# wandb reference: https://docs.wandb.ai/guides/integrations/huggingface

from pathlib import Path

import datasets
import evaluate
import numpy as np
import pandas as pd
import shortuuid
import torch
import wandb
from PIL import Image
from transformers import Trainer, TrainingArguments, ViTForImageClassification, ViTImageProcessor

model_name_or_path = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(model_name_or_path)
metric = evaluate.load("accuracy")
_learning_rate = 2e-4
_train_epochs = 1
_model_name = "ViT-classification"
_run_id = shortuuid.uuid()

wandb.login()


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


def process_example(example):
    inputs = processor(example["image"], return_tensors="pt")
    inputs["labels"] = example["labels"]
    return inputs


def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x for x in example_batch["image"]], return_tensors="pt")

    # Don't forget to include the labels!
    inputs["labels"] = example_batch["labels"]
    return inputs


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1),
        references=p.label_ids,
    )


print("Start load dataset")
ds_builder = MyDataset(
    data_dir="./data/dataset",
)
ds_builder.download_and_prepare()
ds = ds_builder.as_dataset()
prepared_ds = ds.with_transform(transform)
print("End load dataset")

train_labels = {data["labels"]: data["label_text"] for data in ds["train"]}
validation_labels = {data["labels"]: data["label_text"] for data in ds["validation"]}
label_to_text = {**train_labels, **validation_labels}
labels = list(label_to_text.keys())

# Create `.cache` dir
Path(".cache").mkdir(exist_ok=True)

with wandb.init(
    project="ViT-train",
    id=_run_id,
    dir=".cache",
) as run:
    # Connect an Artifact to the run
    # my_checkpoint_name = f"checkpoint-{_run_id}:latest"
    # my_checkpoint_artifact = run.use_artifact(_model_name)

    # Download checkpoint to a folder and return the path
    # checkpoint_dir = my_checkpoint_artifact.download()

    print("Start load model")
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label=label_to_text,
        label2id={v: k for k, v in label_to_text.items()},
    )
    print("End load model")

    training_args = TrainingArguments(
        output_dir=f"./model/{_run_id}",
        per_device_train_batch_size=16,
        eval_strategy="steps",
        num_train_epochs=_train_epochs,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=_learning_rate,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["validation"],
        tokenizer=processor,
    )

    train_results = trainer.train(
        # resume_from_checkpoint=checkpoint_dir,
    )
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate(prepared_ds["validation"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

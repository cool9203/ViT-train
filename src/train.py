# coding: utf-8
# Reference: https://huggingface.co/blog/fine-tune-vit
# wandb reference: https://docs.wandb.ai/guides/integrations/huggingface

import argparse
from pathlib import Path

import evaluate
import numpy as np
import torch
import wandb
from transformers import Trainer, TrainingArguments, ViTForImageClassification, ViTImageProcessor

import utils

from .util import MyDataset


def train(
    model_name_or_path: str,
    output_dir: str,
    learning_rate: float,
    train_epochs: int,
    batch_size: int,
    checkpoint_dir: str,
    cache_dir: str,
    **kwds,
) -> None:
    processor = ViTImageProcessor.from_pretrained(model_name_or_path)
    metric = evaluate.load("accuracy")

    run_id = Path(output_dir).stem.replace("/", "_")
    print(f"run_id: {run_id}")

    output_dir = str(Path("./model", (output_dir if output_dir else run_id)))
    print(f"output_dir: {output_dir}")

    # Create cache dir
    Path(cache_dir).mkdir(exist_ok=True)

    # wandb login
    wandb_config = utils.load_config("./.wandb-config.toml")
    wandb.login(
        key=wandb_config.get("api-key", None),
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
        cache_dir=cache_dir,
    )
    ds_builder.download_and_prepare()
    ds = ds_builder.as_dataset()
    prepared_ds = ds.with_transform(transform)
    print("End load dataset")

    train_labels = {data["labels"]: data["label_text"] for data in ds["train"]}
    validation_labels = {data["labels"]: data["label_text"] for data in ds["validation"]}
    label_to_text = {**train_labels, **validation_labels}
    labels = list(label_to_text.keys())

    with wandb.init(
        project="ViT-train",
        id=run_id,
        dir=cache_dir,
    ) as run:
        # Connect an Artifact to the run
        # my_checkpoint_name = f"checkpoint-{_run_id}:latest"
        # my_checkpoint_artifact = run.use_artifact(_output_model_name)

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
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            eval_strategy="steps",
            num_train_epochs=train_epochs,
            fp16=True,
            save_steps=100,
            eval_steps=100,
            logging_steps=10,
            learning_rate=learning_rate,
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
            resume_from_checkpoint=checkpoint_dir if checkpoint_dir else None,
        )
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

        metrics = trainer.evaluate(prepared_ds["validation"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def arg_parser() -> argparse.Namespace:
    """取得執行程式時傳遞的參數

    tutorial: https://docs.python.org/zh-tw/3/howto/argparse.html#
    reference: https://docs.python.org/zh-tw/3/library/argparse.html#nargs

    Returns:
        argparse.Namespace: 使用args.name取得傳遞的參數
    """

    parser = argparse.ArgumentParser(description="Train ViT model")

    parser.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        default="google/vit-base-patch16-224-in21k",
        help="Model name or path for ViT model",
    )
    parser.add_argument("-o", "--output_dir", type=str, default="ViT-classification", help="Model output folder")
    parser.add_argument("-l", "--learning_rate", type=float, default=2e-4, help="Training learning rate")
    parser.add_argument("-e", "--train_epochs", type=int, default=4, help="Training train epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Continue training from checkpoint")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="Data cache dir")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = arg_parser()
    train(**vars(args))

# coding: utf-8

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Union

import imgaug.augmenters as iaa
import numpy as np
import tqdm
from PIL import Image

# Monkey patch imgaug error
np.bool = np.bool_

# Define type
ImageType = Image.Image
DataValueType = Dict[str, Union[ImageType, str]]
DatasetType = List[DataValueType]
LabelBaseDatasetType = Dict[str, List[DataValueType]]
PathType = Union[str, Path]

# Default value
_image_size = [224, 224]
_output_folder = "./data/dataset"
_image_count = 100

augmentation_seq = iaa.Sequential(
    [
        iaa.RandAugment(n=2, m=9),
    ]
)

rng = None


def load_origin_data(
    path: PathType,
) -> DatasetType:
    dataset = list()
    labels = dict()
    for filename in Path(path).iterdir():
        image = Image.open(filename)
        label_text = filename.stem.lower()
        for suffix in filename.suffixes:
            label_text = label_text.replace(suffix, "")

        if label_text not in labels:
            labels[label_text] = len(labels)
        dataset.append(
            {
                "image": image,
                "label": labels[label_text],
                "label_text": label_text,
            }
        )
    return dataset


def process_image(
    dataset: DatasetType,
    image_size: Sequence = _image_size,
    **kwds,
) -> DatasetType:
    _dataset: DatasetType = list()
    for data in dataset:
        image = data["image"]
        image = image.resize(image_size)

        _dataset.append(
            {
                "image": image,
                "label": data["label"],
                "label_text": data["label_text"],
            }
        )
    return _dataset


def _convert_to_label_base_dataset(
    dataset: DatasetType,
) -> LabelBaseDatasetType:
    label_base_dataset: LabelBaseDatasetType = dict()
    for data in dataset:
        label = data["label"]
        if label not in label_base_dataset:
            label_base_dataset[label] = list()
        label_base_dataset[label].append(data)
    return label_base_dataset


def _make_data(
    dataset: DatasetType,
    output_path: PathType,
) -> None:
    label_count = dict()
    _dataset = _convert_to_label_base_dataset(dataset=dataset)
    with tqdm.tqdm(total=len(dataset)) as progress_bar:
        for label, all_data in _dataset.items():
            if not all_data:
                continue

            label_text = all_data[0]["label_text"]
            label_count[label_text] = 0
            output_folder = Path(output_path, label_text)
            output_folder.mkdir(parents=True, exist_ok=True)

            with Path(output_folder, "metadata.jsonl").open("w", encoding="utf-8") as metadata_fp:
                for data in all_data:
                    filename = f"{label_count[label_text]}.png"
                    data["image"].save(str(Path(output_folder, filename)))

                    # Write to metadata
                    metadata_fp.write(
                        json.dumps(
                            {
                                "file_name": filename,
                                "label_text": label_text,
                                "label": label,
                            }
                        )
                        + "\n"
                    )
                    label_count[label_text] += 1
                    progress_bar.update()


def make_eval_data(
    dataset: DatasetType,
    output_path: PathType,
) -> None:
    output_path = Path(output_path)
    output_path = output_path if output_path.name == "eval" else Path(output_path, "eval")
    shutil.rmtree(output_path)
    output_path.mkdir()

    _make_data(
        dataset=dataset,
        output_path=output_path,
    )


def make_train_and_validation_data(
    dataset: DatasetType,
    output_path: PathType,
    train_size: float = 0.8,
    shuffle: bool = True,
) -> None:
    output_path = Path(output_path)
    if output_path.name in ["train", "validation", "val", "eval"]:
        output_path = output_path.parent

    train_output_path = Path(output_path, "train")
    shutil.rmtree(train_output_path)
    train_output_path.mkdir()

    validation_output_path = Path(output_path, "validation")
    shutil.rmtree(validation_output_path)
    validation_output_path.mkdir()

    _dataset = _convert_to_label_base_dataset(dataset=dataset)

    train_dataset = list()
    validation_dataset = list()
    for i in _dataset.keys():
        if shuffle:
            np.random.shuffle(_dataset[i])
        train_dataset += _dataset[i][: int(len(_dataset[i]) * train_size)]
        validation_dataset += _dataset[i][int(len(_dataset[i]) * train_size) :]

    _make_data(
        dataset=train_dataset,
        output_path=train_output_path,
    )

    _make_data(
        dataset=validation_dataset,
        output_path=validation_output_path,
    )


def make_image_augmentation(
    dataset: DatasetType,
    image_count: int = 100,
) -> DatasetType:
    label_base_dataset = _convert_to_label_base_dataset(dataset)
    _dataset: DatasetType = list()
    labels = list(label_base_dataset.keys())
    label_count = {label: 0 for label in labels}
    total_count = len(labels) * image_count

    with tqdm.tqdm(total=total_count) as progress_bar:
        for label, all_data in label_base_dataset.items():
            if not all_data:
                continue

            while True:
                index = rng.randint(0, len(all_data) - 1)
                data = all_data[index]

                if label_count[label] == image_count:
                    break
                label_count[label] += 1

                augmented_image = augmentation_seq(image=np.asarray(data["image"]))
                _dataset.append(
                    {
                        "image": Image.fromarray(augmented_image),
                        "label": label,
                        "label_text": data["label_text"],
                    }
                )
                progress_bar.update()
    return _dataset


def make(
    image_folder: str,
    output_folder: str,
    seed: int,
    image_count: int,
    **kwds,
) -> None:
    global rng
    if seed and seed >= 0:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    dataset = load_origin_data(
        image_folder,
    )

    print("Process image data")
    processed_dataset = process_image(
        dataset=dataset,
        **kwds,
    )

    print("Make eval data")
    make_eval_data(
        dataset=dataset,
        output_path=output_folder,
    )

    print("Make augmentation image data")
    augmented_data = make_image_augmentation(
        dataset=processed_dataset,
        image_count=image_count,
    )

    print("Make train and validation data")
    make_train_and_validation_data(
        dataset=augmented_data,
        output_path=output_folder,
    )


def arg_parser() -> argparse.Namespace:
    """取得執行程式時傳遞的參數

    tutorial: https://docs.python.org/zh-tw/3/howto/argparse.html#
    reference: https://docs.python.org/zh-tw/3/library/argparse.html#nargs

    Returns:
        argparse.Namespace: 使用args.name取得傳遞的參數
    """

    parser = argparse.ArgumentParser(description="Make image dataset with image augmentation")
    parser.add_argument("-i", "--image_folder", type=str, required=True, help="Origin image data folder")
    parser.add_argument("-o", "--output_folder", type=str, default=_output_folder, help="Output dataset path")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed")
    parser.add_argument("-c", "--image_count", type=int, default=_image_count, help="Ever image augmentation count")
    parser.add_argument("--image_size", type=int, nargs="+", default=_image_size, help="Image size")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = arg_parser()
    make(**vars(args))

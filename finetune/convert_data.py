# -*- coding: utf-8 -*-
# @Date    : 2024-04-21 15:22:29
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, json, re
from typing import Any
from dataclasses import dataclass, field
from transformers import HfArgumentParser


@dataclass
class DataArgs:
    action: str
    origin_path: str = field(
        default="dataset/original",
        metadata={"help": "path to original data, containing a .tsv file and multiple directories"},
    )
    common_path: str = field(
        default="dataset/common/index.json",
        metadata={"help": "path to the data converted into a common form (a .json file)"},
    )
    common_image_dir: str = field(
        default="dataset/common/images",
        metadata={"help": "path to the data converted into a common form (the image dir)"},
    )


def origin2common(args: DataArgs):
    """
    Convert the given dataset to a more universal format.
    This will only do the conversion and filter out missing samples, without any processing.
    """
    os.makedirs(args.common_image_dir, exist_ok=True)

    # dump all files into one same dir
    for desc_name in os.listdir(args.origin_path):
        path = os.path.join(args.origin_path, desc_name)
        if not os.path.isdir(path):
            continue
        path = os.path.join(path, "files")
        for desc_name in os.listdir(path):
            for file in os.listdir(os.path.join(path, desc_name)):
                dst = os.path.join(args.common_image_dir, file)
                if not os.path.exists(dst):
                    os.link(os.path.join(path, desc_name, file), dst)

    # convert the table to json
    image_k_names = ["image", "figure_type", "specimen_type", "notes", "pixel/mm"]
    image_v_cols = [1, 3, 4, 10, 13]
    data: dict[str, dict] = {}  # desc_name -> {specimen_attr, list[image_attr]}
    for i, line in enumerate(open(os.path.join(args.origin_path, "index.tsv"))):
        if i == 0:
            continue
        items = line.strip("\n \r").split("\t")
        image_attr: dict[str, Any] = {v_name: items[v_col] for v_name, v_col in zip(image_k_names, image_v_cols)}
        try:
            image_attr["pixel/mm"] = float(image_attr["pixel/mm"]) / 0.11
        except ValueError:
            image_attr["pixel/mm"] = 0.0

        desc_name = items[9].split(".")[0]
        if desc_name in data:
            if os.path.exists(os.path.join(args.common_image_dir, image_attr["image"])):
                data[desc_name]["images"].append(image_attr)
            else:
                print(f"Warning: image {image_attr['image']} not found.")
        else:
            desc_file_path = os.path.join(args.common_image_dir, desc_name + ".txt")
            if os.path.exists(desc_file_path):
                data[desc_name] = {
                    "name": items[2].strip('"'),
                    "desc_lang": items[11],
                    "desc": open(desc_file_path).read(),
                    "images": [image_attr],
                }
            else:
                print(f"Warning: desc {desc_name} not found.")
    json.dump(data, open(args.common_path, "w"), indent=2)


def desc_filter(desc: str) -> str:
    "Find the part we interested in from the full description."
    # now our impl is to find the beginning of the description that ends with numbers
    # typically this is the coarse description of the overall shape and pattern
    prefix = "TYPE DESCRIPTION: "
    if desc.startswith(prefix):
        desc = desc[len(prefix) :]
    desc = re.sub(r"(\[.*\])|(\(.*\))|\"", "", desc)
    sentences = desc.split(".")
    for i, sent in enumerate(sentences):
        if re.search(r"\d", sent):
            break
    return ".".join(sentences[:i])


def common2llava(args: DataArgs):
    pass


def main():
    args: DataArgs = HfArgumentParser(DataArgs).parse_args_into_dataclasses()[0]  # type:ignore
    globals()[args.action](args)


if __name__ == "__main__":
    main()

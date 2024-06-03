# -*- coding: utf-8 -*-
# @Date    : 2024-05-20 16:43:29
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"data processing for [1 description: 1 images] format"

from __future__ import annotations
import os, json, re
from typing import Any, Callable
from tqdm import tqdm
from common.args import DataArgs, RunArgs, PromptArgs, data_args, run_args, prompts, logger
from data.utils import dump_files
from models.llm import get_generator


def origin2common(args: DataArgs, _, __):
    """
    Convert the given dataset to a more universal format:
    `identifier: [file_name, section_type, specimen_type, figure_type, specimen_status, notes, lang, pixel/mm, desc]`.
    This will only do the conversion and filter out missing samples, without any processing.
    """
    dump_files(args)

    # convert the table to json
    image_k_names = [
        "image",
        "section_type",
        "specimen_type",
        "figure_type",
        "specimen_status",
        "notes",
        "pixel/mm",
    ]
    image_v_cols = [1, 4, 5, 11, 12, 13, 17]
    data: dict[str, Any] = {}  # desc_name -> {lang, desc, list[image_attr]}
    for i, line in enumerate(open(os.path.join(args.origin_path, "index.tsv"))):
        if i == 0:
            continue
        items = line.strip("\n \r").split("\t")
        image_attr: dict[str, Any] = {v_name: items[v_col] for v_name, v_col in zip(image_k_names, image_v_cols)}
        try:
            image_attr["pixel/mm"] = float(image_attr["pixel/mm"])
        except ValueError:
            image_attr["pixel/mm"] = 0.0

        if not os.path.exists(os.path.join(args.common_image_dir, image_attr["image"])):
            logger.warn(f"In line {i+1}: image {image_attr['image']} not found.")
            continue

        desc_name = items[10].split(".")[0]
        if desc_name in data:
            data[desc_name]["images"].append(image_attr)
        else:
            desc_file_path = os.path.join(args.common_image_dir, desc_name + ".txt")
            if os.path.exists(desc_file_path):
                data[desc_name] = {
                    "lang": items[14],
                    "desc": open(desc_file_path).read(),
                    "images": [image_attr],
                }
            else:
                logger.warn(f"In line {i+1}: description {desc_name} not found.")
    json.dump(data, open(args.common_data_path, "w"), indent=2)


def sample_filter(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Filter out samples that are not useful for training.
    """
    return samples


def desc_processor(desc: str, prompts: PromptArgs, llm_generate: Callable[[str], str]) -> tuple[str, str]:
    """
    Do some preprocessing on the description and return the processed description and numerical info.
    """
    desc = llm_generate(prompts.desc_processing.format(desc=desc))
    num_info = llm_generate(prompts.desc_extraction.format(desc=desc))
    return desc, num_info


def common2intern(data_args: DataArgs, run_args: RunArgs, prompts: PromptArgs):
    "convert to the format of InternLM-XComposer model, doing any preprocessing required and ready for training"
    generator = get_generator()
    common: dict[str, dict] = json.load(open(data_args.common_data_path))
    data, total = [], 0
    for i, specie in tqdm(enumerate(common.values()), total=len(common)):
        desc, info = desc_processor(specie["desc"], prompts, generator)
        inputs = prompts.desc_single.format(info=info)
        for sample in sample_filter(specie["images"]):
            data.append(
                {
                    "id": str(i),
                    "image": os.path.join(data_args.common_image_dir, sample["image"]),
                    "conversations": [
                        {"from": "user", "value": inputs},
                        {"from": "assistant", "value": desc},
                    ],
                }
            )
        total += len(specie["images"])
    json.dump({"vl_data": data}, open(os.path.join(data_args.intern_path, "data.json"), "w"), indent=2)
    logger.info(f"{len(data)} / {total} samples converted.")


def main():
    globals()[run_args.action](data_args, run_args, prompts)


if __name__ == "__main__":
    main()

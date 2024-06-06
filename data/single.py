# -*- coding: utf-8 -*-
# @Date    : 2024-05-20 16:43:29
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"data processing for [1 description: 1 images] format"

from __future__ import annotations
import os, json
from typing import Any
from common.args import data_args, run_args, prompts, logger
from data.utils import dump_files
from models.llm import LLMGenerator


def origin2common():
    """
    Convert the given dataset to a more universal format.
    This will only do the conversion and filter out missing samples, without any processing.
    """
    dump_files(data_args)

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
    data: dict[str, Any] = {}  # desc_name -> {name, desc, list[image_attr]}
    for i, line in enumerate(open(os.path.join(data_args.origin_path, "index.tsv"))):
        if i == 0:
            continue
        items = line.strip("\n \r").split("\t")
        image_attr: dict[str, Any] = {v_name: items[v_col] for v_name, v_col in zip(image_k_names, image_v_cols)}
        try:
            image_attr["pixel/mm"] = float(image_attr["pixel/mm"])
        except ValueError:
            image_attr["pixel/mm"] = 0.0

        if not os.path.exists(os.path.join(data_args.common_image_dir, image_attr["image"])):
            logger.warn(f"In line {i+1}: image {image_attr['image']} not found.")
            continue

        desc_name = items[10].split(".")[0]
        if desc_name in data:
            data[desc_name]["images"].append(image_attr)
        else:
            desc_file_path = os.path.join(data_args.common_image_dir, desc_name + ".txt")
            if os.path.exists(desc_file_path):
                data[desc_name] = {
                    "name": items[2],
                    "desc": open(desc_file_path).read(),
                    "images": [image_attr],
                }
            else:
                logger.warn(f"In line {i+1}: description {desc_name} not found.")
    json.dump(data, open(data_args.common_data_path, "w"), indent=2)


def sample_filter(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Filter out samples that are not useful for training.
    """
    return samples


def desc_processor(descs: list[str]) -> tuple[list[str], list[str]]:
    "Do some preprocessing on the description and return the processed description and numerical info."
    llm_generator = LLMGenerator()
    prompt_processing_tmpl = open(prompts.desc_processing_tmpl).read()
    for rule_file in prompts.desc_processing_rules:
        rule = open(rule_file).read().strip()
        descs = llm_generator(
            [prompt_processing_tmpl.format(rule=rule, desc=desc) for desc in descs],
            batch_size=data_args.desc_infer_bs,
        )

    prompt_extraction = open(prompts.desc_extraction).read()
    num_info = llm_generator(
        [prompt_extraction.format(desc=desc) for desc in descs],
        batch_size=data_args.info_infer_bs,
    )
    return descs, num_info


def common2intern():
    "convert to the format of InternLM-XComposer model, doing any preprocessing required and ready for training"
    common: dict[str, dict] = json.load(open(data_args.common_data_path))
    # sort by length for efficient batching in llm inference, longest first for detecting OOM
    species = sorted(common.values(), key=lambda x: len(x["desc"]), reverse=True)[: data_args.end_pos]
    descs = [specie["desc"] for specie in species]
    descs, numerics = desc_processor(descs)
    data, total = [], 0
    for i, (specie, desc, numeric) in enumerate(zip(species, descs, numerics)):
        inputs = open(prompts.generation_single).read().format(info=numeric, name=specie["name"])
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

    os.makedirs(data_args.intern_path, exist_ok=True)
    json.dump({"vl_data": data}, open(os.path.join(data_args.intern_path, "data.json"), "w"), indent=2)
    logger.info(f"{len(data)} / {total} samples converted.")


def main():
    globals()[run_args.action]()


if __name__ == "__main__":
    main()

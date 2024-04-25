# -*- coding: utf-8 -*-
# @Date    : 2024-04-21 15:22:29
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, json, re
from typing import Any
from common.args import DataArgs, RunArgs, PromptArgs, data_args, run_args, prompts, logger


def origin2common(args: DataArgs, _, __):
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
                logger.warn(f"image {image_attr['image']} not found.")
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
                logger.warn(f"desc {desc_name} not found.")
    json.dump(data, open(args.common_path, "w"), indent=2)


def desc_filter(desc: str) -> str:
    "Find the part we interested in from the full description."
    # now our impl is to find the beginning of the description that ends with numbers, and at least 10 words
    # typically this is the coarse description of the overall shape and pattern
    # TODO complete in the future

    # first split out the "TYPE DESCRIPTION" part, assuming it's the first part
    desc_list = re.split(r"[A-Z ]{6,}:", desc)
    desc = desc_list[1] if len(desc_list) > 1 else desc
    desc = desc.strip(" \u201c\n")

    # then delete the "Translation from" part
    pos = desc.find("Translation from")
    if pos != -1:
        desc = desc[pos:]
        desc = desc[min(desc.find(":"), desc.find("-")) :].strip()

    # strip brackets and leading punctuations
    desc = re.sub(r"(\[.*\])|(\(.*\))|\"", "", desc)
    desc = re.sub(r"^[^a-zA-Z]+", "", desc)

    # find the leading sentences until digit
    sentences = re.split(r"[.;]", desc)
    for i, sent in enumerate(sentences):
        if re.search(r"\d", sent):
            break
    desc = ".".join(sentences[:i]) + "."

    # at least 5 words is regraded meaningful
    return "" if len(desc.split(" ")) < 5 else desc


def image_filter(images_attrs: list[dict]) -> dict[str, str]:
    """
    Select the images that are most likely to be useful for the task.
    Returns: image type -> image path, e.g., Axial: aaa.png, Equatorial: bbb.png
    """
    # For now, just select each type of image for the specimen. Prioritize holotype and empty notes.
    # TODO complete in the future

    # sort by specimen type and notes so that holotype and empty notes is automatically prioritized
    attrs = sorted(images_attrs, key=lambda x: (x["specimen_type"], x["notes"]))

    # calculate image type -> list[index]
    type_index: dict[str, list[int]] = {}
    for i, image_attr in enumerate(attrs):
        if image_attr["figure_type"] in type_index:
            type_index[image_attr["figure_type"]].append(i)
        else:
            type_index[image_attr["figure_type"]] = [i]

    # select the first image for each type and append info to the return key
    selected_images: dict[str, str] = {}
    for indices in type_index.values():
        attr = attrs[indices[0]]
        keys = []
        for name in ["figure_type", "specimen_type", "notes"]:
            if attr[name]:
                keys.append(attr[name].lower())
        selected_images[", ".join(keys)] = attr["image"]
    return selected_images


def common2intern(data_args: DataArgs, run_args: RunArgs, prompts: PromptArgs):
    "convert to the format of InternLM-XComposer model, doing any preprocessing required and ready for training"
    data = []
    common: dict[str, dict] = json.load(open(data_args.common_path))
    for i, sample in enumerate(common.values()):
        output_text = desc_filter(sample["desc"])
        if output_text == "":
            logger.warn(f'desc {sample["name"]}: {sample["desc"]} formatting failed, skipping.')
            continue
        images = image_filter(sample["images"])
        image_repr = "; ".join([f"This is a {k} image of the specimen: <ImageHere>" for k in images.keys()]) + "."
        input_text = prompts.task_desc.format(images=image_repr)
        data.append(
            {
                "id": str(i),
                "image": list(images.values()),
                "conversations": [{"from": "user", "value": input_text}, {"from": "assistant", "value": output_text}],
            }
        )
    json.dump(data, open(data_args.intern_path, "w"), indent=2)
    logger.info(f"{len(data)} / {len(common)} samples converted.")


def main():
    globals()[run_args.run_name](data_args, run_args, prompts)


if __name__ == "__main__":
    main()

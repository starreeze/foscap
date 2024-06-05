# -*- coding: utf-8 -*-
# @Date    : 2024-04-23 20:33:32
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import logging
from rich.logging import RichHandler
from typing import cast

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("rich")
logger.info("Rich logger setup.")


@dataclass
class DataArgs:
    origin_path: str = field(
        default="dataset/original",
        metadata={"help": "path to original data, containing a .tsv file and multiple directories"},
    )
    common_data_path: str = field(
        default="dataset/common/data.json",
        metadata={"help": "path to the data converted into a common form (a .json file)"},
    )
    common_image_dir: str = field(
        default="dataset/common/images",
        metadata={"help": "path to the data converted into a common form (the image dir)"},
    )
    intern_path: str = field(default="dataset/intern")
    desc_infer_bs: int = field(default=2, metadata={"help": "batch size for generating descriptions"})
    info_infer_bs: int = field(default=3, metadata={"help": "batch size for generating numerics"})


@dataclass
class ModelArgs:
    llm_path: str = field(
        default="meta-llama/Meta-Llama-3-8B",
        metadata={"help": "path to the model to use"},
    )


@dataclass
class RunArgs:
    module: str = field(default="")
    action: str = field(default="")


@dataclass
class PromptArgs:
    desc_multi: str = field(
        default="The following paleontological fossil images are from a same specimen. {images} "
        "Please give a brief description, including (but not limited to) the overall shape and pattern of the specimen."
    )
    desc_single: str = field(
        default="The following is an image of a paleontological fossil, belonging to spices {name}. "
        "<ImageHere> Please give a detailed description. "
        "Here is some information on the overall shape that must be included in the description: {info}."
    )
    desc_processing: str = field(default=open("prompts/desc_processing.txt").read())
    desc_extraction: str = field(default=open("prompts/desc_extraction.txt").read())


data_args, model_args, run_args, prompts = HfArgumentParser(
    [DataArgs, ModelArgs, RunArgs, PromptArgs]  # type: ignore
).parse_args_into_dataclasses()
data_args = cast(DataArgs, data_args)
model_args = cast(ModelArgs, model_args)
run_args = cast(RunArgs, run_args)
prompts = cast(PromptArgs, prompts)

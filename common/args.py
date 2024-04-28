# -*- coding: utf-8 -*-
# @Date    : 2024-04-23 20:33:32
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import logging
from rich.logging import RichHandler

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
    qwen_path: str = field(default="dataset/qwen")


@dataclass
class RunArgs:
    action: str = field(default="")
    run_name: str = field(default="")


@dataclass
class PromptArgs:
    task_desc: str = field(
        default="The following paleontological fossil images are from a same specimen. "
        "{images} Please describe the overall shape and pattern of the specimen."
    )


data_args, run_args, prompts = HfArgumentParser(
    [DataArgs, RunArgs, PromptArgs]  # type:ignore
).parse_args_into_dataclasses()

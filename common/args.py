# -*- coding: utf-8 -*-
# @Date    : 2024-04-23 20:33:32
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
from dataclasses import dataclass, field, make_dataclass
from transformers import HfArgumentParser
import logging, os
from rich.logging import RichHandler
from typing import Any, cast

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("rich")
logger.info("Rich logger setup.")


@dataclass
class DataArgs:
    origin_path: str = field(
        default="dataset/original",
        metadata={"help": "path to original data, containing a .tsv file and multiple directories"},
    )
    common_text_dir: str = field(
        default="dataset/common",
        metadata={"help": "path to the data converted into a common form (the text dir)"},
    )
    common_image_dir: str = field(
        default="dataset/common/images",
        metadata={"help": "path to the data converted into a common form (the image dir)"},
    )
    intern_path: str = field(default="dataset/intern")
    desc_infer_bs: int = field(default=2, metadata={"help": "batch size for generating descriptions"})
    info_infer_bs: int = field(default=3, metadata={"help": "batch size for generating numerics"})
    end_pos: int = field(default=int(1e9), metadata={"help": "the end position of the data"})


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


def create_prompt_args():
    prompt_files = ["desc_extraction", "desc_processing_tmpl", "generation_multi", "generation_single"]
    fields: list[tuple[str, Any, Any]] = [(name, str, field(default=f"prompts/{name}.txt")) for name in prompt_files]
    rule_files = [f"prompts/{name}" for name in os.listdir("prompts") if name.startswith("desc_processing_rule")]
    fields.append(("desc_processing_rules", list[str], field(default_factory=lambda: rule_files)))
    return make_dataclass("PromptArgs", fields)


PromptArgs = create_prompt_args()
data_args, model_args, run_args, prompts = HfArgumentParser(
    [DataArgs, ModelArgs, RunArgs, PromptArgs]  # type: ignore
).parse_args_into_dataclasses()
data_args = cast(DataArgs, data_args)
model_args = cast(ModelArgs, model_args)
run_args = cast(RunArgs, run_args)

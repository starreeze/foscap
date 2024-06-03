# -*- coding: utf-8 -*-
# @Date    : 2024-06-03 15:48:48
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import transformers, torch
from typing import Callable
from common.args import model_args


def get_generator() -> Callable[[str], str]:
    pipeline = transformers.pipeline(
        "text-generation", model=model_args.llm_path, model_kwargs={"torch_dtype": torch.float16}, device_map="auto"
    )
    return lambda x: pipeline(x)[0]["generated_text"][len(x) :]  # type: ignore


def main():
    print(get_generator()("Once upon a time, "))


if __name__ == "__main__":
    main()

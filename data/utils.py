# -*- coding: utf-8 -*-
# @Date    : 2024-05-20 16:50:32
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os
from common.args import DataArgs


def ls_visible(path: str):
    return sorted(list(filter(lambda x: not x.startswith("."), os.listdir(path))))


def dump_files(args: DataArgs):
    "dump all files into one same dir"
    os.makedirs(args.common_image_dir, exist_ok=True)
    for desc_name in ls_visible(args.origin_path):
        path = os.path.join(args.origin_path, desc_name)
        if not os.path.isdir(path):
            continue
        path = os.path.join(path, "files")
        for desc_name in ls_visible(path):
            for file in ls_visible(os.path.join(path, desc_name)):
                dst = os.path.join(args.common_image_dir, file)
                if not os.path.exists(dst):
                    os.link(os.path.join(path, desc_name, file), dst)

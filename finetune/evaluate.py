# -*- coding: utf-8 -*-
# @Date    : 2024-05-08 10:33:38
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, torch, json
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from common.utils import to_device


class InternDataset(Dataset):
    def __init__(self, data_path, vis_processor):
        self.data = json.load(open(data_path, "r"))["vl_data"]
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.data[index]
        images = []
        for image_path in sample["image"]:
            image = Image.open(image_path).convert("RGB")
            images.append(self.vis_processor(image).squeeze(0))
        return dict(
            text=sample["conversations"][0]["value"],
            images=torch.stack(images),
            answer=sample["conversations"][1]["value"],
        )


def eval_intern():
    from intern.finetune import parse_args, load_config_tokenizer
    from intern.ixc_utils import HD_transform
    from peft.auto import AutoPeftModelForCausalLM

    model_args, data_args, training_args, lora_args = parse_args()
    config, tokenizer = load_config_tokenizer(model_args, training_args)
    print(f"Load model from: {model_args.model_name_or_path}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    model.eval().cuda()
    processor = lambda x: model.vis_processor(HD_transform(x, data_args.hd_num))
    loader = DataLoader(InternDataset(data_args.data_path, processor))

    torch.set_grad_enabled(False)
    results = []
    for sample in loader:
        sample: dict[str, object] = to_device(sample)  # type: ignore
        with torch.cuda.amp.autocast():  # type: ignore
            response, _ = model.chat(
                tokenizer,
                query=sample["text"],
                image=sample["image"],
                hd_num=data_args.hd_num,
                history=[],
                do_sample=False,
                num_beams=3,
            )
            results.append(sample | {"response": response})
    json.dump(results, open(training_args.output_dir + "/eval_results.json", "w"))


if __name__ == "__main__":
    eval_intern()

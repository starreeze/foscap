# -*- coding: utf-8 -*-
# @Date    : 2024-05-08 10:33:38
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import os, torch, json
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from common.utils import to_device
from transformers import AutoModelForCausalLM
from peft.auto import AutoPeftModelForCausalLM


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
            images.append(self.vis_processor(image))
        return dict(
            text=sample["conversations"][0]["value"],
            images=torch.stack(images),
            answer=sample["conversations"][1]["value"],
        )


def eval_intern():
    from intern.finetune import parse_args, load_config_tokenizer
    from intern.ixc_utils import HD_transform

    model_args, data_args, training_args, lora_args = parse_args()
    config, tokenizer = load_config_tokenizer(model_args, training_args)
    print(f"Load model from: {model_args.model_name_or_path}")
    if training_args.use_lora:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    model.eval().cuda()

    processor = lambda x: model.vis_processor(HD_transform(x, data_args.hd_num))
    eval_dataset = random_split(
        InternDataset(data_args.data_path, processor),
        [data_args.train_test_split, 1 - data_args.train_test_split],
        generator=torch.Generator().manual_seed(data_args.train_test_seed),
    )[1]
    loader = DataLoader(eval_dataset, batch_size=data_args.batch_size)

    torch.set_grad_enabled(False)
    results = []
    for sample in loader:
        sample: dict = to_device(sample)  # type: ignore
        with torch.cuda.amp.autocast():  # type: ignore
            response, _ = model.chat(
                tokenizer,
                query=sample["text"][0],
                image=sample["images"][0],
                hd_num=data_args.hd_num,
                history=[],
                do_sample=False,
                num_beams=3,
            )
            sample.pop("images")
            results.append(sample | {"response": response})
    json.dump(results, open(os.path.join(training_args.output_dir + "eval_results.json"), "w"))


if __name__ == "__main__":
    eval_intern()

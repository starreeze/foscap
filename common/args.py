# -*- coding: utf-8 -*-
# @Date    : 2023-10-26 19:51:58
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

import argparse, torch
from time import time
from itertools import product

parser = argparse.ArgumentParser()
# data
## path

## format
parser.add_argument("--column_splitter", type=str, default=" ### ")
parser.add_argument("--object_splitter", type=str, default=", ")
parser.add_argument("--subsentence_splitter_set", type=str, default=",.;!?:")
parser.add_argument("--clip_prompt", type=str, default="A photo containing ")

# prompts
task_prompts = argparse.Namespace()
task_prompts.train = "Please describe the image showing a paleontological fossil. You should describe its shape, pattern and other features in detail."
task_prompts.eval = task_prompts.train

model_prompts = argparse.Namespace()
model_prompts.llava = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant is able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
    "The visual content will be provided with the following format: <Image>visual content</Image>."
    "USER: <image>\n{prompt}\n ASSISTANT: "
)

# insight
## model


parser.add_argument("--max_new_tokens", type=int, default=200, help="max number of generated tokens")
parser.add_argument("--infer_dataloader_worker", type=int, default=0)
parser.add_argument("--valid_data_split", type=float, default=0.05)
parser.add_argument("--wandb_user", type=str, default="starreeze")
parser.add_argument("--print_per_n_step", type=int, default=5)
parser.add_argument("--eval_per_epoch", type=int, default=4)

## models
### common
parser.add_argument("--model", type=str, default="minigpt", help="model name to train")
parser.add_argument("--infer_bs_multiply", type=int, default=2)
parser.add_argument(
    "--train_bs_pos",
    type=int,
    default=1,
    help="number of positive samples (normal objects predicted by clip) in a batch",
)
parser.add_argument(
    "--train_bs_gold",
    type=int,
    default=1,
    help="number of positive samples (gold caption of COCO) in a batch",
)
parser.add_argument(
    "--train_bs_sent",
    type=int,
    default=1,
    help="number of positive samples (generated complete sentence) in a batch",
)
parser.add_argument("--train_bs_neg", type=int, default=1, help="number of negative samples in a batch")
parser.add_argument("--infer_bs_total", type=int, default=0, help="overwrite infer multiply for generatrion")
parser.add_argument("--train_bs_total", type=int, default=0, help="overwrite train for evaluation")
parser.add_argument("--train_lr", type=float, default=1e-5)
parser.add_argument("--train_wd", type=float, default=0.05)
parser.add_argument("--train_epoch", type=int, default=1)
parser.add_argument("--train_dataloader_worker", type=int, default=0)

### minigpt
parser.add_argument("--infer_retry", type=int, default=3)
parser.add_argument(
    "--minigpt_infer_cfg", default="configs/minigpt4_infer_fp16.yaml", help="path to configuration file."
)
parser.add_argument(
    "--minigpt_train_cfg", default="configs/minigpt4_train_fp16.yaml", help="path to configuration file."
)
parser.add_argument("--minigpt_path", type=str, default="checkpoints/minigpt4_llama2_7b/pretrained.pth")
parser.add_argument("--minigpt_ckpt_load_path", type=str, default="checkpoints/minigpt4_llama2_7b/pretrained.pth")
parser.add_argument("--minigpt_ckpt_save_path", type=str, default="checkpoints/minigpt4_llama2_7b")

### instruct-blip
# parser.add_argument("--blip_train_prompt", type=str, default="Please describe the image.")
# parser.add_argument(
#     "--blip_eval_prompt",
#     type=str,
#     default="Please describe the image in great detail. Your response should have at least 100 words.",
# )
# note that this should be modified in the config file, along with vicuna path
# parser.add_argument("--blip_path", type=str, default="checkpoints/blip_vicuna_7b/pretrained.pth")
# parser.add_argument("--blip_ckpt_load_path", type=str, default="checkpoints/blip_vicuna_7b/pretrained.pth")
# parser.add_argument("--blip_ckpt_save_path", type=str, default="checkpoints/blip_vicuna_7b")

### mplug-owl:
parser.add_argument("--owl_path", type=str, default="/root/.cache/huggingface/hub/models--MAGAer13--mplug-owl-llama-7b")
parser.add_argument(
    "--owl_ckpt_load_path", type=str, default="/root/.cache/huggingface/hub/models--MAGAer13--mplug-owl-llama-7b"
)  # todo tobe checked -> pass
parser.add_argument("--owl_ckpt_save_path", type=str, default="checkpoints/owl-llama-7b")  # todo: tobe checked -> pass

### owllrv:
parser.add_argument(
    "--owllrv_path",
    type=str,
    default="/root/.cache/huggingface/hub/models--MAGAer13--mplug-owl-llama-7b-ft/snapshots/8b08efd90767fda988d69892e02eb4b8c642fafb",
)
parser.add_argument(
    "--owllrv_ckpt_load_path",
    type=str,
    default="/root/.cache/huggingface/hub/models--MAGAer13--mplug-owl-llama-7b-ft/snapshots/8b08efd90767fda988d69892e02eb4b8c642fafb",
)
parser.add_argument("--owllrv_ckpt_save_path", type=str, default="checkpoints/owl-lrv")
parser.add_argument("--owllrv_lora_path", type=str, default="checkpoints/owl-lora-model/pytorch_model.bin")

### llava
parser.add_argument(
    "--llava_ckpt_load_path",
    type=str,
    default="/root/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b/snapshots/12e054b30e8e061f423c7264bc97d4248232e965",
)
parser.add_argument(
    "--llava_path",
    type=str,
    default="/root/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b/snapshots/12e054b30e8e061f423c7264bc97d4248232e965",
)
parser.add_argument(
    "--llava_vit_path",
    type=str,
    default="/root/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
)
parser.add_argument("--llava_ckpt_save_path", type=str, default="checkpoints/llava_vicuna_7b")

### llavarlhf:
parser.add_argument(
    "--llavarlhf_ckpt_load_path",
    type=str,
    default="checkpoints/llava_rlhf_merged",
)
parser.add_argument(
    "--llavarlhf_path",
    type=str,
    default="checkpoints/llava_rlhf_merged",
)
parser.add_argument(
    "--llavarlhf_vit_path",
    type=str,
    default="/root/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41",
)
parser.add_argument("--llavarlhf_ckpt_save_path", type=str, default="checkpoints/llava_rlhf_7b")

### share4v
parser.add_argument(
    "--share4v_ckpt_load_path",
    type=str,
    default="/root/.cache/huggingface/hub/models--Lin-Chen--ShareGPT4V-7B/snapshots/a973da7d8dba5e9ac2817f1c88bf9c8f36004078",
)
parser.add_argument(
    "--share4v_path",
    type=str,
    default="/root/.cache/huggingface/hub/models--Lin-Chen--ShareGPT4V-7B/snapshots/a973da7d8dba5e9ac2817f1c88bf9c8f36004078",
)
parser.add_argument(
    "--share4v_vit_path",
    type=str,
    default="/root/.cache/huggingface/hub/models--Lin-Chen--ShareGPT4V-7B_Pretrained_vit-large336-l12/snapshots/55da275fb4755cc5e5d9c6121aa72adc6de01f55",
)
parser.add_argument("--share4v_ckpt_save_path", type=str, default="checkpoints/share4v_7b")

# eval
parser.add_argument("--pope_result_path", type=str, default="evaluate/pope/result")
parser.add_argument("--vqa_result_path", type=str, default="evaluate/vqa/result")
parser.add_argument("--vqa_question_path", type=str, default="dataset/v2_OpenEnded_mscoco_train2014_questions.json")
parser.add_argument("--vqa_annotation_path", type=str, default="dataset/v2_mscoco_train2014_annotations.json")
parser.add_argument("--mme_result_path", type=str, default="evaluate/mme/result")
parser.add_argument("--mme_text_path", type=str, default="dataset/mme_text")
parser.add_argument("--mme_image_path", type=str, default="dataset/mme_images")
parser.add_argument("--default_eval_samples", type=int, default=1600)
parser.add_argument("--generate_length_penalty", type=float, default=-1)

# common control
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--restart", action="store_true")
parser.add_argument("--seed", type=int, default=10654)
parser.add_argument("--start_pos", type=int, default=0)
parser.add_argument("--end_pos", type=int, default=int(1e10))
parser.add_argument("--proxy", type=str, default="")
parser.add_argument("--train_dtype_str", type=str, default="bfloat16")
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("--no_first_eval", action="store_true")
parser.add_argument("--run_name", type=str, default=str(time()))

args = parser.parse_args()

# batch size
args.infer_bs_pos = args.train_bs_pos * args.infer_bs_multiply
args.infer_bs_sent = args.train_bs_sent * args.infer_bs_multiply
args.infer_bs_neg = args.train_bs_neg * args.infer_bs_multiply
args.infer_bs_gold = args.train_bs_gold * args.infer_bs_multiply
if args.train_bs_total == 0:
    args.train_bs_total = args.train_bs_pos + args.train_bs_sent + args.train_bs_neg + args.train_bs_gold
if args.infer_bs_total == 0:
    args.infer_bs_total = args.infer_bs_pos + args.infer_bs_sent + args.infer_bs_neg + args.infer_bs_gold

# dtype
args.train_dtype = getattr(torch, args.train_dtype_str)

# prompt
for model, task in product(model_prompts._get_kwargs(), task_prompts._get_kwargs()):
    prompt = model[1].format(prompt=task[1])
    setattr(args, f"{model[0]}_{task[0]}_prompt", prompt)


# model provided parser
def minigpt4_finetune_parser():
    parser = argparse.ArgumentParser(description="finetune minigpt4")
    parser.add_argument("--cfg-path", default=args.minigpt_infer_cfg, help="path to configuration file.")
    parser.add_argument("--name", type=str, default="A2", help="evaluation name")
    parser.add_argument("--ckpt", type=str, help="path to configuration file.")
    parser.add_argument("--eval_opt", type=str, default="all", help="path to configuration file.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=args.max_new_tokens, help="max number of generated tokens"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=64, help="lora rank of the model")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    return parser

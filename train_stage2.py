import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import torch
import yaml
import argparse
from datetime import datetime
from transformers import TrainingArguments
from trainer import Stage2Trainer
from models.omnimamba import OmniMamba
from util.data import find_latest_model_bin


def create_training_arguments(args):
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.output_dir,
        disable_tqdm=True,
        tf32=True,
        save_steps=5000,
        logging_steps=args.logging_steps,
        overwrite_output_dir=True,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        weight_decay=args.decay,
        adam_beta1=args.beta1,
        adam_beta2=args.beta2,
        per_device_train_batch_size=args.batch_size_t2i+args.batch_size_mmu,
        per_device_eval_batch_size=args.batch_size_t2i+args.batch_size_mmu,
        dataloader_num_workers=args.num_workers,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        evaluation_strategy=args.eval_strategy,
        bf16=args.bf16,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type=args.scheduler,
        lr_scheduler_kwargs={'min_lr_rate': args.min_lr_rate} if args.scheduler != 'linear' else None,
    )
    return training_args

if __name__ == "__main__":

    os.environ["NCCL_ALGO"] = "Tree"


    parser = argparse.ArgumentParser()
    parser.add_argument("--vq-model", type=str, default="VQ-f16", choices=["VQ-f16"], help="VQ models")
    parser.add_argument("--num-workers", type=int, default=16, help="dataloader num workers")
    parser.add_argument("--min-lr-rate", type=float, default=0.01, help="min learning rate") 
    parser.add_argument("--decay", type=float, default=0, help="weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="adam beta2")
    parser.add_argument("--grad-accum", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="warmup ratio")
    parser.add_argument("--scheduler", type=str, default='cosine_with_min_lr', choices=['linear', 'cosine_with_min_lr','constant_with_warmup', 'constant'], help="lr scheduler")

    parser.add_argument("--save-total-limit", type=int, default=5, help="save total limit")
    parser.add_argument("--save-strategy", type=str, default='steps', choices=['steps', 'epoch'], help="save strategy")
    parser.add_argument("--eval-strategy", type=str, default='no', choices=['no', 'steps', 'epochs'], help="eval strategy")
    parser.add_argument("--config", type=str, default='config/config_stage2.yaml', help="train config")

    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)["train"]
    config["batch_size_t2i"] = config["batch_size_t2i"] if config["t2i_task"] else 0
    config["batch_size_mmu"] = config["batch_size_mmu"] if config["mmu_task"] else 0
    config['lr'] = float(config['lr'])
    config = argparse.Namespace(**config)
    args = argparse.Namespace(**vars(args), **vars(config))
    
    # setting output path
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.output_dir = os.path.join(args.output_dir, f"{args.omnimamba_model}_{current_time}")
    if args.omnimamba_ckpt is not None and '.bin' not in args.omnimamba_ckpt:
        args.omnimamba_ckpt = find_latest_model_bin(args.omnimamba_ckpt)

    # create model and dataset
    model = OmniMamba(args,stage=args.stage)

    # create trainer and run
    trainer = Stage2Trainer(
        model,
        create_training_arguments(args),
        configs=args,
    )
    trainer.train(resume_from_checkpoint=args.resume_dir)




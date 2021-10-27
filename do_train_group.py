# -*- coding: utf-8 -*-

"""
@Author             : huggingface
@Date               : 2020/7/26
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/5/14
"""

import argparse
import logging
import os
import torch
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from transformers import (
set_seed,
AutoConfig,
AutoTokenizer,
AutoModelForSeq2SeqLM,
AdamW,
get_linear_schedule_with_warmup,
)

from src.data_processor import DataProcessor
from src.utils.my_utils import init_logger

logger = logging.getLogger(__name__)


def group(args, data_processors, model, tokenizer):
    train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)
    eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)

    # Prepare data for each task
    train_data_iterators = {}
    train_data_counter = {}
    eval_data_loaders = {}
    for task, processor in data_processors.items():
        # Training data
        _, dataset = processor.load_and_cache_data("train", tokenizer, args.suffix)
        sampler = RandomSampler(dataset)
        loader = DataLoader(dataset, sampler=sampler, batch_size=train_batch_size)
        train_data_iterators[task] = iter(loader)
        train_data_counter[task] = len(loader)
        # Evaluation data
        _, dataset = processor.load_and_cache_data("valid", tokenizer, args.suffix)
        sampler = RandomSampler(dataset)
        loader = DataLoader(dataset, sampler=sampler, batch_size=eval_batch_size)
        eval_data_loaders[task] = loader
    affinities = {src_task: {tgt_task: [] for tgt_task in args.tasks} for src_task in args.tasks}
    total_steps = sum(train_data_counter.values())

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps,
    )

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("Instantaneous train batch size per device = {}".format(args.per_device_train_batch_size))
    logger.info("Instantaneous evaluation batch size per device = {}".format(args.per_device_eval_batch_size))
    logger.info("Total train batch size (w. parallel) = {}".format(train_batch_size))
    logger.info("Total evaluation batch size (w. parallel) = {}".format(eval_batch_size))
    logger.info("Total optimization steps = {}".format(total_steps))

    model.train()
    model.zero_grad()
    set_seed(args.seed)
    step_iterator = tqdm(range(1, total_steps + 1))
    for global_step in step_iterator:
        # Sample batches and source task
        eval_batches = [(task, next(iter(loader))) for task, loader in eval_data_loaders.items()]
        src_task = random.choices(list(train_data_counter.keys()), list(train_data_counter.values()), k=1)[0]

        # Calculate loss before update parameters
        loss_before_update = {}
        for tgt_task, tgt_batch in eval_batches:
            if tgt_task == src_task:
                continue
            with torch.no_grad():
                tgt_inputs = {
                    "input_ids": tgt_batch[0].to(args.device),
                    "attention_mask": tgt_batch[1].to(args.device),
                    "labels": tgt_batch[-1].to(args.device),
                }
                tgt_outputs = model(**tgt_inputs)
                tgt_loss = tgt_outputs[0]
                if args.n_gpu > 1:
                    tgt_loss = tgt_loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                loss_before_update[tgt_task] = tgt_loss.detach().cpu().tolist()

        # Update parameters
        src_batch = next(train_data_iterators[src_task])
        src_inputs = {
            "input_ids": src_batch[0].to(args.device),
            "attention_mask": src_batch[1].to(args.device),
            "labels": src_batch[-1].to(args.device),
        }
        src_outputs = model(**src_inputs)
        src_loss = src_outputs[0]
        if args.n_gpu > 1:
            src_loss = src_loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
        src_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        step_iterator.set_description("Global step: {:>6d}, Loss: {:>.4f}".format(global_step, src_loss.item()))

        # Calculate loss after update parameters
        loss_after_update = {}
        for tgt_task, tgt_batch in eval_batches:
            if tgt_task == src_task:
                continue
            with torch.no_grad():
                tgt_inputs = {
                    "input_ids": tgt_batch[0].to(args.device),
                    "attention_mask": tgt_batch[1].to(args.device),
                    "labels": tgt_batch[-1].to(args.device),
                }
                tgt_outputs = model(**tgt_inputs)
                tgt_loss = tgt_outputs[0]
                if args.n_gpu > 1:
                    tgt_loss = tgt_loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                loss_after_update[tgt_task] = tgt_loss.detach().cpu().tolist()

        # Update task affinities
        for tgt_task in loss_before_update.keys():
            loss_1 = loss_before_update[tgt_task]
            loss_2 = loss_after_update[tgt_task]
            affinities[src_task][tgt_task].append(1 - loss_2 / loss_1)

        # Update data counter
        train_data_counter[src_task] -= 1
        if train_data_counter[src_task] <= 0:
            train_data_iterators.pop(src_task)
            train_data_counter.pop(src_task)

        # Save model checkpoint
        if args.save_steps > 0 and global_step % args.save_steps == 0:
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            os.makedirs(output_dir, exist_ok=True)
            logger.info("Saving model checkpoint to {}".format(output_dir))
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            torch.save(affinities, os.path.join(output_dir, "affinities.bin"))

    # Save model checkpoint
    logger.info("Saving model checkpoint to {}".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    torch.save(affinities, os.path.join(args.output_dir, 'affinities.bin'))


def main():
    parser = argparse.ArgumentParser()

    # Datasets parameters
    parser.add_argument("--tasks", required=True, type=str, help="")
    parser.add_argument("--suffix", default=None, type=str, help="")
    parser.add_argument("--with_prefix", action="store_true", help="")

    # Model hyper parameters
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--pretrained_model",
        default=None,
        type=str,
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_src_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--max_tgt_length",
        default=128,
        type=int,
        help="The maximum total output sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    # Directory parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="The log directory where the running details will be written.",
    )

    # Training parameters
    parser.add_argument(
        "--per_device_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--per_device_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    # Other parameters
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()

    # Setup output dir
    args.output_dir = os.path.join(
            args.output_dir,
            "{}_{}_{}_{}".format(
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_src_length,
                args.max_tgt_length,
                "raw" if not args.with_prefix else "prefix",
            ),
        )
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup log dir
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        args.log_file = os.path.join(
            args.log_dir,
            "{}_{}_{}_{}".format(
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_src_length,
                args.max_tgt_length,
                "raw" if not args.with_prefix else "prefix",
            ),
        )
    else:
        args.log_file = None
    init_logger(logging.INFO, args.log_file)

    # Setup CUDA, GPU training
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    # Set seed
    set_seed(args.seed)

    # Parse tasks
    args.tasks = args.tasks.split(",")

    # Load config, tokenizer and pretrained model
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.pretrained_model if args.pretrained_model else args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    logger.info("Training/evaluation config %s", config)

    # Initialize data processors
    data_processors = {}
    for task in args.tasks:
        data_processors[task] = DataProcessor(
            args.model_name_or_path,
            args.max_src_length,
            args.max_tgt_length,
            [task],
            data_dir=args.data_dir,
            with_prefix=args.with_prefix,
            overwrite_cache=args.overwrite_cache,
        )
    group(args, data_processors, model, tokenizer)


if __name__ == "__main__":
    main()

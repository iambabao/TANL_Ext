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
import higher
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
set_seed,
AutoConfig,
AutoTokenizer,
AutoModelForSeq2SeqLM,
AdamW,
get_linear_schedule_with_warmup,
)

from src.data_processor import DataProcessor
from src.utils.my_utils import (
init_logger,
save_json_lines,
refine_outputs,
compute_metrics,
)

logger = logging.getLogger(__name__)


def train(args, data_processors, model, tokenizer, role):
    train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)

    # Prepare data loader for each task
    data_loaders = {}
    task_sampler = {}
    for task, processor in data_processors.items():
        examples, dataset = processor.load_and_cache_data(role, tokenizer, args.suffix)
        sampler = RandomSampler(dataset)
        loader = DataLoader(dataset, sampler=sampler, batch_size=train_batch_size)
        data_loaders[task] = loader
        task_sampler[task] = len(dataset)

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
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_outer_steps,
    )

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("Instantaneous batch size per device = {}".format(args.per_device_train_batch_size))
    logger.info("Total train batch size (w. parallel) = {}".format(train_batch_size))
    logger.info("Total optimization steps = {}".format(args.max_outer_steps))

    current_score, best_score = 0.0, 0.0
    set_seed(args.seed)  # Added here for reproductibility
    model.zero_grad()
    global_iter = tqdm(range(1, args.max_outer_steps + 1))
    for global_step in global_iter:
        model.train()
        total_loss = []
        inner_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate * 10)
        with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (inner_model, diff_opt):
            # Sample tasks from distribution!
            support_task, query_task = random.choices(list(task_sampler.keys()), list(task_sampler.values()), k=2)

            # Meta train!
            for _ in range(args.gradient_accumulation_steps):
                batch = next(iter(data_loaders[support_task]))
                inputs = {
                    "input_ids": batch[0].to(args.device),
                    "attention_mask": batch[1].to(args.device),
                    "labels": batch[-1].to(args.device),
                }
                outputs = inner_model(**inputs)
                loss = outputs[0]
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
            diff_opt.step(loss)

            # Meta test!
            batch = next(iter(data_loaders[query_task]))
            inputs = {
                "input_ids": batch[0].to(args.device),
                "attention_mask": batch[1].to(args.device),
                "labels": batch[-1].to(args.device),
            }
            outputs = inner_model(**inputs)
            loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            loss.backward()
            total_loss.append(loss.item())

        # Update model parameters!
        optimizer.step()
        scheduler.step()
        model.zero_grad()

        # Log metrics
        global_iter.set_description("Global step: {:>6d}, Loss: {:>.4f}".format(
            global_step, sum(total_loss) / len(total_loss)
        ))
        if args.logging_steps > 0 and global_step % args.logging_steps == 0 and args.evaluate_during_training:
            results = evaluate(args, data_processors, model, tokenizer, "valid", prefix=str(global_step))
            current_score, best_score = results["Bleu_4"], max(best_score, results["Bleu_4"])

        # Save model checkpoint
        if args.save_steps > 0 and global_step % args.save_steps == 0:
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            os.makedirs(output_dir, exist_ok=True)
            logger.info("Saving model checkpoint to {}".format(output_dir))
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))

    # Save model checkpoint
    logger.info("Saving model checkpoint to {}".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


def evaluate(args, data_processors, model, tokenizer, role, prefix=""):
    if prefix == "":
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(prefix))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("Batch size = {}".format(eval_batch_size))

    eval_outputs = []
    for task, processor in data_processors.items():
        examples, dataset = processor.load_and_cache_data(role, tokenizer, args.suffix)
        sampler = SequentialSampler(dataset)
        loader = DataLoader(dataset, sampler=sampler, batch_size=eval_batch_size)

        generated = []
        for batch in tqdm(loader, desc="Evaluating"):
            model.eval()
            with torch.no_grad():
                inputs = {
                        "input_ids": batch[0].to(args.device),
                        "attention_mask": batch[1].to(args.device),
                    }
                outputs = model.generate(**inputs, max_length=args.max_tgt_length)
                for out in outputs.detach().cpu().tolist():
                    generated.append(tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False))
        eval_outputs.extend(refine_outputs(examples, generated))

    eval_outputs_file = os.path.join(output_dir, "{}_outputs.json".format(role))
    save_json_lines(eval_outputs, eval_outputs_file)

    eval_results = compute_metrics(eval_outputs)
    eval_results_file = os.path.join(output_dir, "{}_results.txt".format(role))
    with open(eval_results_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in eval_results.keys():
            logger.info("{} = {}".format(key, str(eval_results[key])))
            writer.write("{} = {}\n".format(key, str(eval_results[key])))

    return eval_results


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
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_outer_steps",
        default=1000,
        type=int,
        help="If > 0: set total number of outer loop steps to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    # Other parameters
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
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
            "{}_{}_{}_{}_{}_{:.1e}".format(
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_src_length,
                args.max_tgt_length,
                args.max_outer_steps,
                args.per_device_train_batch_size,
                args.learning_rate,
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

    # Setup CUDA, GPU training
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    assert args.n_gpu <= 1, "higher do not support multi-gpu training"

    # Setup log dir
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        args.log_file = os.path.join(
            args.log_dir,
            "{}_{}_{}_{:.1e}".format(
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_src_length,
                args.max_tgt_length,
                args.learning_rate,
            ),
        )
    else:
        args.log_file = None
    init_logger(logging.INFO, args.log_file)

    # Set seed
    set_seed(args.seed)

    # Parse tasks
    args.tasks = args.tasks.split(",")

    # Load config, tokenizer and pretrained model
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

    # Training
    train(args, data_processors, model, tokenizer, role="train")


if __name__ == "__main__":
    main()

# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2021/3/30
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/3/30
"""

import argparse
import configparser
import logging
import os
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM

from src.data_processor import load_my_dataset as load_dataset
from src.utils.my_utils import init_logger, parse_data_args, save_json, save_json_lines, format_data

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=str, help="")
    parser.add_argument("--splits", required=True, type=str, help="")
    parser.add_argument("--prefix", action="store_true", help="")
    parser.add_argument("--model_name_or_path", required=True, type=str, help="")
    parser.add_argument("--max_src_length", default=128, type=int, help="")
    parser.add_argument("--max_tgt_length", default=128, type=int, help="")
    parser.add_argument("--do_lower_case", action="store_true", help="")
    parser.add_argument("--data_dir", required=True, type=str, help="")
    parser.add_argument("--output_dir", required=True, type=str, help="")
    parser.add_argument("--log_file", default=None, type=str, help="")
    parser.add_argument("--overwrite_cache", action="store_true", help="")
    parser.add_argument("--batch_size", default=16, type=int, help="")
    parser.add_argument("--no_cuda", action="store_true", help="")
    args = parser.parse_args()

    args.task_list = args.tasks.split(",")
    args.split_list = args.splits.strip().split(",")

    init_logger(logging.INFO, args.log_file)

    logger.info('Loading model from: {}'.format(args.model_name_or_path))
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model.to(device)
    model.eval()

    logger.info("Evaluation parameters %s", args)
    logger.info("Evaluation config %s", config)

    for task in args.task_list:
        for split in args.split_list:
            data_args = parse_data_args(args, task, split)
            dataset  = load_dataset(
                dataset_name=data_args.dataset_name,
                data_args=data_args,
                tokenizer=tokenizer,
                max_input_length=data_args.max_seq_length_eval,
                max_output_length=data_args.max_output_seq_length_eval,
                split=data_args.dataset_split,
            )
            data_args_s1 = parse_data_args(args, "{}_s1".format(task), split)
            dataset_s1 = load_dataset(
                dataset_name=data_args_s1.dataset_name,
                data_args=data_args_s1,
                tokenizer=tokenizer,
                max_input_length=data_args_s1.max_seq_length_eval,
                max_output_length=data_args_s1.max_output_seq_length_eval,
                split=data_args_s1.dataset_split,
            )
            data_args_s2 = parse_data_args(args, "{}_s2".format(task), split)
            dataset_s2 = load_dataset(
                dataset_name=data_args_s2.dataset_name,
                data_args=data_args_s2,
                tokenizer=tokenizer,
                max_input_length=data_args_s2.max_seq_length_eval,
                max_output_length=data_args_s2.max_output_seq_length_eval,
                split=data_args_s2.dataset_split,
            )

            if not args.prefix:
                inputs = [" ".join(_.tokens) for _ in dataset_s1.examples]
            else:
                inputs = ["{} : {}".format(dataset_s1.task_descriptor, " ".join(_.tokens)) for _ in dataset_s1.examples]
            encoded = tokenizer.batch_encode_plus(
                inputs,
                padding="max_length",
                truncation="longest_first",
                max_length=args.max_src_length,
                return_tensors="pt",
            )
            eval_dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"])
            eval_sampler = SequentialSampler(eval_dataset)
            eval_data_loader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

            eval_outputs_s1 = []
            with torch.no_grad():
                for batch in tqdm(eval_data_loader, desc="Evaluating (stage 1)"):
                    inputs = {
                        "input_ids": batch[0].to(device),
                        "attention_mask": batch[1].to(device),
                    }
                    outputs = model.generate(**inputs, max_length=args.max_tgt_length)
                    eval_outputs_s1.extend(outputs.detach().cpu().tolist())
            eval_outputs_s1 = tokenizer.batch_decode(
                eval_outputs_s1,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            if not args.prefix:
                inputs = eval_outputs_s1
            else:
                inputs = ["{} : {}".format(dataset_s2.task_descriptor, _) for _ in eval_outputs_s1]
            encoded = tokenizer.batch_encode_plus(
                inputs,
                padding="max_length",
                truncation="longest_first",
                max_length=args.max_src_length,
                return_tensors="pt",
            )
            eval_dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"])
            eval_sampler = SequentialSampler(eval_dataset)
            eval_data_loader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

            eval_outputs_s2 = []
            with torch.no_grad():
                for batch in tqdm(eval_data_loader, desc="Evaluating (stage 2)"):
                    inputs = {
                        "input_ids": batch[0].to(device),
                        "attention_mask": batch[1].to(device),
                    }
                    outputs = model.generate(**inputs, max_length=args.max_tgt_length)
                    eval_outputs_s2.extend(outputs.detach().cpu().tolist())
            eval_outputs_s2 = tokenizer.batch_decode(
                eval_outputs_s2,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            outputs = []
            for example, o1, o2 in tqdm(zip(dataset.examples, eval_outputs_s1, eval_outputs_s2), desc="Parsing results"):
                generated_output = dataset.output_format.run_inference(
                    example, o2,
                    entity_types=dataset.entity_types,
                    relation_types=dataset.relation_types,
                )
                generated_entities, generated_relations = generated_output[:2]
                generated_entities, generated_relations = format_data(
                    tokens=example.tokens,
                    entities=generated_entities,
                    relations=generated_relations,
                )
                outputs.append({
                    "context": " ".join(example.tokens),
                    "s1": o1, "s2": o2,
                    "entities": generated_entities,
                    "relations": generated_relations,
                })

            results_s1 = dataset_s1.evaluate_generated_outputs(eval_outputs_s1)
            for key, value in results_s1.items():
                logger.info("{}: {}".format(key, value))
            results_s2 = dataset_s2.evaluate_generated_outputs(eval_outputs_s2)
            for key, value in results_s2.items():
                logger.info("{}: {}".format(key, value))
            results = {"s1": results_s1, "s2": results_s2}

            save_json(outputs, os.path.join(args.output_dir, "{}_{}_outputs.pipeline.json"))
            save_json(results, os.path.join(args.output_dir, "{}_{}_results.pipeline.json"))

    logger.info("Done!")


if __name__ == '__main__':
    main()

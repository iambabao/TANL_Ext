# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2021/3/30
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/3/30
"""

import argparse
import logging
import os
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM

from src.utils.tanl_utils import augment_sentence
from src.data_processor import load_my_dataset as load_dataset
from src.utils.my_utils import init_logger, parse_data_args, save_json, format_data

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="")
    parser.add_argument("--split", required=True, type=str, help="")
    parser.add_argument("--prefix", action="store_true", help="")
    parser.add_argument("--pretrained_model_s1", required=True, type=str, help="")
    parser.add_argument("--pretrained_model_s2", default=None, type=str, help="")
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

    init_logger(logging.INFO, args.log_file)
    logger.info("Evaluation parameters %s", args)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    logger.info('Loading model for stage 1 from: {}'.format(args.pretrained_model_s1))
    config_s1 = AutoConfig.from_pretrained(args.pretrained_model_s1)
    tokenizer_s1 = AutoTokenizer.from_pretrained(args.pretrained_model_s1)
    model_s1 = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_s1, config=config_s1)
    model_s1.to(device)
    model_s1.eval()

    data_args_s1 = parse_data_args(args, "{}_s1".format(args.task), args.split)
    dataset_s1 = load_dataset(
        dataset_name=data_args_s1.dataset_name,
        data_args=data_args_s1,
        tokenizer=tokenizer_s1,
        max_input_length=data_args_s1.max_seq_length_eval,
        max_output_length=data_args_s1.max_output_seq_length_eval,
        split=data_args_s1.dataset_split,
    )

    if not args.prefix:
        inputs = [" ".join(_.tokens) for _ in dataset_s1.examples]
    else:
        inputs = ["{} : {}".format(dataset_s1.task_descriptor, " ".join(_.tokens)) for _ in dataset_s1.examples]
    encoded = tokenizer_s1.batch_encode_plus(
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
            outputs = model_s1.generate(**inputs, max_length=args.max_tgt_length)
            eval_outputs_s1.extend(outputs.detach().cpu().tolist())
    eval_outputs_s1 = tokenizer_s1.batch_decode(
        eval_outputs_s1,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    model_s1.to("cpu")

    if args.pretrained_model_s2 is not None:
        logger.info('Loading model for stage 2 from: {}'.format(args.pretrained_model_s2))
        config_s2 = AutoConfig.from_pretrained(args.pretrained_model_s2)
        tokenizer_s2 = AutoTokenizer.from_pretrained(args.pretrained_model_s2)
        model_s2 = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_s2, config=config_s2)
    else:
        logger.info('Model reused for stage 2')
        config_s2, tokenizer_s2, model_s2 = config_s1, tokenizer_s1, model_s1
    model_s2.to(device)
    model_s2.eval()

    data_args_s2 = parse_data_args(args, "{}_s2".format(args.task), args.split)
    dataset_s2 = load_dataset(
        dataset_name=data_args_s2.dataset_name,
        data_args=data_args_s2,
        tokenizer=tokenizer_s2,
        max_input_length=data_args_s2.max_seq_length_eval,
        max_output_length=data_args_s2.max_output_seq_length_eval,
        split=data_args_s2.dataset_split,
    )

    inputs = []
    for example, sentence, in zip(dataset_s1.examples, eval_outputs_s1):
        generated_output = dataset_s1.output_format.run_inference(
            example, sentence,
            entity_types=dataset_s1.entity_types,
            relation_types=dataset_s1.relation_types,
        )
        generated_entities = generated_output[0]
        augmentations = [([], start, end) for _, start, end in generated_entities]
        inputs.append(augment_sentence(
            example.tokens, augmentations,
            dataset_s1.input_format.BEGIN_ENTITY_TOKEN, dataset_s1.input_format.SEPARATOR_TOKEN,
            dataset_s1.input_format.RELATION_SEPARATOR_TOKEN, dataset_s1.input_format.END_ENTITY_TOKEN
        ))
    if args.prefix:
        inputs = ["{} : {}".format(dataset_s2.task_descriptor, _) for _ in inputs]
    encoded = tokenizer_s2.batch_encode_plus(
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
            outputs = model_s2.generate(**inputs, max_length=args.max_tgt_length)
            eval_outputs_s2.extend(outputs.detach().cpu().tolist())
    eval_outputs_s2 = tokenizer_s2.batch_decode(
        eval_outputs_s2,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    model_s2.to("cpu")

    outputs = []
    for example, o1, o2 in zip(dataset_s2.examples, eval_outputs_s1, eval_outputs_s2):
        generated_output = dataset_s2.output_format.run_inference(
            example, o2,
            entity_types=dataset_s2.entity_types,
            relation_types=dataset_s2.relation_types,
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

    save_json(outputs, os.path.join(args.output_dir, "{}_{}_outputs.pipeline.json".format(args.split, args.task)))
    save_json(results, os.path.join(args.output_dir, "{}_{}_results.pipeline.json".format(args.split, args.task)))

    logger.info("Done!")


if __name__ == '__main__':
    main()

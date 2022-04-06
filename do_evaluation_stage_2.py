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
import json
import random
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
    parser.add_argument("--pretrained_model", required=True, type=str, help="")
    parser.add_argument("--max_src_length", default=128, type=int, help="")
    parser.add_argument("--max_tgt_length", default=128, type=int, help="")
    parser.add_argument("--keep_ratio", default=1.0, type=float, help="")
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

    logger.info('Loading model from: {}'.format(args.pretrained_model))
    config = AutoConfig.from_pretrained(args.pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model, config=config)
    model.to(device)
    model.eval()

    data_args = parse_data_args(args, args.task + "_s2", args.split)
    dataset = load_dataset(
        dataset_name=data_args.dataset_name,
        data_args=data_args,
        tokenizer=tokenizer,
        max_input_length=data_args.max_seq_length_eval,
        max_output_length=data_args.max_output_seq_length_eval,
        split=data_args.dataset_split,
    )

    eval_inputs = []
    for example in dataset.examples:
        augmentations = [
            ([], entity.start, entity.end) for entity in example.entities if random.random() < args.keep_ratio
        ]
        eval_inputs.append(augment_sentence(
            example.tokens, augmentations,
            dataset.input_format.BEGIN_ENTITY_TOKEN, dataset.input_format.SEPARATOR_TOKEN,
            dataset.input_format.RELATION_SEPARATOR_TOKEN, dataset.input_format.END_ENTITY_TOKEN,
        ))
    if args.prefix:
        eval_inputs = ["{} : {}".format(dataset.task_descriptor, _) for _ in eval_inputs]
    encoded_inputs = tokenizer.batch_encode_plus(
        eval_inputs,
        padding="max_length",
        truncation="longest_first",
        max_length=args.max_src_length,
        return_tensors="pt",
    )
    eval_dataset = TensorDataset(encoded_inputs["input_ids"], encoded_inputs["attention_mask"])
    eval_sampler = SequentialSampler(eval_dataset)
    eval_data_loader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    eval_outputs = []
    with torch.no_grad():
        for batch in tqdm(eval_data_loader, desc="Evaluating (stage 2)"):
            inputs = {
                "input_ids": batch[0].to(device),
                "attention_mask": batch[1].to(device),
            }
            outputs = model.generate(**inputs, max_length=args.max_tgt_length)
            eval_outputs.extend(outputs.detach().cpu().tolist())
    eval_outputs = tokenizer.batch_decode(
        eval_outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    outputs = []
    for example, eval_input, eval_output in zip(dataset.examples, eval_inputs, eval_outputs):
        golden_entities = [entity.to_tuple() for entity in example.entities]
        golden_relations = [relation.to_tuple() for relation in example.relations]
        golden_entities, golden_relations = format_data(
            tokens=example.tokens,
            entities=golden_entities,
            relations=golden_relations,
        )
        generated_output = dataset.output_format.run_inference(
            example, eval_output,
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
            "input": eval_input,
            "output": eval_output,
            "golden_entities": golden_entities,
            "golden_relations": golden_relations,
            "generated_entities": generated_entities,
            "generated_relations": generated_relations,
        })

    results = dataset.evaluate_generated_outputs(eval_outputs)
    logger.info(json.dumps(results, ensure_ascii=False, indent=4))

    save_json(outputs, os.path.join(
        args.output_dir,
        "{}_{}_outputs.{:.2f}.pipeline.json".format(args.split, args.task, args.keep_ratio)
    ))
    save_json(results, os.path.join(
        args.output_dir,
        "{}_{}_results.{:.2f}.pipeline.json".format(args.split, args.task, args.keep_ratio)
    ))

    logger.info("Done!")


if __name__ == '__main__':
    main()

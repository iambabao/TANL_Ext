# -*- coding: utf-8 -*-

"""
@Author             :
@Date               : 2020/7/26
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/5/14
"""

import argparse
import logging
import os
from transformers import AutoTokenizer

from src.data_processor import load_my_dataset as load_dataset
from src.utils.my_utils import init_logger, save_json, save_json_lines, format_data

logger = logging.getLogger(__name__)


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='', help='')
    parser.add_argument('--data_dir', type=str, required=True, help='')
    parser.add_argument('--dataset_name', type=str, required=True, help='')
    parser.add_argument('--dataset_split', type=str, required=True, help='')
    parser.add_argument('--max_seq_length', type=int, required=True, help='')
    parser.add_argument('--max_output_seq_length', type=int, default=None, help='')
    parser.add_argument('--max_seq_length_eval', type=int, default=None, help='')
    parser.add_argument('--max_output_seq_length_eval', type=int, default=None, help='')
    parser.add_argument('--prefix', action='store_true', help='')
    parser.add_argument('--do_lower_case', action='store_true', help='')
    parser.add_argument('--overwrite_cache', action='store_true', help='')
    parser.add_argument('--output_dir', type=str, default='', help='')
    args = parser.parse_args()

    if args.max_output_seq_length is None:
        args.max_output_seq_length = args.max_seq_length
    if args.max_seq_length_eval is None:
        args.max_seq_length_eval = args.max_seq_length
    if args.max_output_seq_length_eval is None:
        args.max_output_seq_length_eval = args.max_output_seq_length

    # setup logging
    init_logger(logging.INFO)
    logger.info('args: {}'.format(args))

    logger.info('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    logger.info('Loading dataset')
    dataset = load_dataset(
        dataset_name=args.dataset_name,
        data_args=args,
        tokenizer=tokenizer,
        max_input_length=args.max_seq_length_eval,
        max_output_length=args.max_output_seq_length_eval,
        split=args.dataset_split,
    )

    outputs = []
    for source, target in zip(dataset.input_sentences, dataset.output_sentences):
        outputs.append({"source": source, "target": target, 'task_name': args.dataset_name})

    formatted_outputs = []
    for example in dataset.examples:
        tokens = example.tokens
        entities = [(e.type.natural, e.start, e.end) for e in example.entities]
        relations = [
            (
                r.type.natural,
                (r.head.type.natural, r.head.start, r.head.end),
                (r.tail.type.natural, r.tail.start, r.tail.end),
            ) for r in example.relations
        ]
        entities, relations = format_data(tokens, entities, relations)
        formatted_outputs.append({'context': ' '.join(tokens), 'entities': entities, 'relations': relations})

    output_dir = os.path.join(args.output_dir, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    save_json_lines(outputs, os.path.join(output_dir, 'data_{}.json'.format(args.dataset_split)))
    save_json(formatted_outputs, os.path.join(output_dir, 'data_{}.formatted.json'.format(args.dataset_split)))


if __name__ == "__main__":
    main()

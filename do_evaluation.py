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
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data_processor import load_my_dataset as load_dataset
from src.utils.my_utils import init_logger, read_json_lines, save_json, format_data

logger = logging.getLogger(__name__)


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='', help='')
    parser.add_argument('--cache_dir', type=str, default=None, help='')
    parser.add_argument('--data_dir', type=str, required=True, help='')
    parser.add_argument('--dataset_name', type=str, required=True, help='')
    parser.add_argument('--dataset_split', type=str, required=True, help='')
    parser.add_argument('--max_seq_length', type=int, required=True, help='')
    parser.add_argument('--max_output_seq_length', type=int, default=None, help='')
    parser.add_argument('--max_seq_length_eval', type=int, default=None, help='')
    parser.add_argument('--max_output_seq_length_eval', type=int, default=None, help='')
    parser.add_argument('--do_lower_case', action='store_true', help='')
    parser.add_argument('--overwrite_cache', type=bool, default=True, help='')
    parser.add_argument('--prefix', action='store_true', help='')
    parser.add_argument('--generated_outputs', type=str, required=True, help='')
    parser.add_argument('--output_file', type=str, required=True, help='')
    parser.add_argument('--log_file', type=str, default=None, help='')
    args = parser.parse_args()

    # the order is slightly different from original code
    if args.max_output_seq_length is None:
        args.max_output_seq_length = args.max_seq_length
    if args.max_seq_length_eval is None:
        args.max_seq_length_eval = args.max_seq_length
    if args.max_output_seq_length_eval is None:
        args.max_output_seq_length_eval = args.max_output_seq_length

    # setup logging
    init_logger(logging.INFO, args.log_file)
    logger.info('args: {}'.format(args))

    logger.info('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

    logger.info('Loading dataset')
    dataset = load_dataset(
        dataset_name=args.dataset_name,
        data_args=args,
        tokenizer=tokenizer,
        max_input_length=args.max_seq_length_eval,
        max_output_length=args.max_output_seq_length_eval,
        split=args.dataset_split,
    )

    generated_outputs = list(line['generated'] for line in read_json_lines(args.generated_outputs))
    assert len(dataset.examples) == len(generated_outputs)

    results = dataset.evaluate_generated_outputs(generated_outputs)
    for key, value in results.items():
        logger.info('{}: {}'.format(key, value))

    outputs = []
    for example, output_sentence in tqdm(zip(dataset.examples, generated_outputs)):
        tokens = example.tokens
        parsed_outputs = dataset.output_format.run_inference(
            example,
            output_sentence,
            entity_types=dataset.entity_types,
            relation_types=dataset.relation_types,
        )
        parsed_entities, parsed_relations = parsed_outputs[0], parsed_outputs[1]
        parsed_entities, parsed_relations = format_data(tokens, parsed_entities, parsed_relations)
        outputs.append({'context': ' '.join(tokens), 'entities': parsed_entities, 'relations': parsed_relations})
    save_json(outputs, args.output_file)


if __name__ == "__main__":
    main()

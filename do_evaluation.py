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
from transformers import AutoTokenizer

from src.data_processor import load_my_dataset as load_dataset
from src.utils.my_utils import init_logger, read_json_lines

logger = logging.getLogger(__name__)


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, default="", help="")
    parser.add_argument("--cache_dir", type=str, default=None, help="")
    parser.add_argument('--data_dir', type=str, required=True, help='')
    parser.add_argument('--dataset_name', type=str, required=True, help='')
    parser.add_argument('--dataset_split', type=str, required=True, help='')
    parser.add_argument('--max_seq_length', type=int, required=True, help='')
    parser.add_argument('--max_output_seq_length', type=int, default=None, help='')
    parser.add_argument('--max_seq_length_eval', type=int, default=None, help='')
    parser.add_argument('--max_output_seq_length_eval', type=int, default=None, help='')
    parser.add_argument('--eval_nll', type=bool, default=False, help='')
    parser.add_argument('--chunk_size', type=int, default=128, help='')
    parser.add_argument('--chunk_overlap', type=int, default=64, help='')
    parser.add_argument('--chunk_size_eval', type=int, default=None, help='')
    parser.add_argument('--chunk_overlap_eval', type=int, default=None, help='')
    parser.add_argument('--overwrite_cache', type=bool, default=True, help='')
    parser.add_argument('--input_format', type=str, default=None, help='')
    parser.add_argument('--output_format', type=str, default=None, help='')
    parser.add_argument("--multitask", action='store_true', help="")
    parser.add_argument('--generated_outputs', type=str, required=True, help='')
    args = parser.parse_args()

    # the order is slightly different from original code
    if args.max_output_seq_length is None:
        args.max_output_seq_length = args.max_seq_length
    if args.max_seq_length_eval is None:
        args.max_seq_length_eval = args.max_seq_length
    if args.max_output_seq_length_eval is None:
        args.max_output_seq_length_eval = args.max_output_seq_length

    if args.chunk_size_eval is None:
        args.chunk_size_eval = args.chunk_size
    if args.chunk_overlap_eval is None:
        args.chunk_overlap_eval = args.chunk_overlap

    # setup logging
    init_logger(logging.INFO)

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
    )

    logger.info("Loading dataset")
    dataset = load_dataset(
        dataset_name=args.dataset_name,
        data_args=args,
        tokenizer=tokenizer,
        max_input_length=args.max_seq_length_eval,
        max_output_length=args.max_output_seq_length_eval,
        split=args.dataset_split,
        shuffle=False,
        is_eval=True,
    )

    generated_outputs = list(line['generated'] for line in read_json_lines(args.generated_outputs))
    assert len(dataset.examples) == len(generated_outputs)

    results = dataset.evaluate_generated_outputs(generated_outputs)
    for key, value in results.items():
        logger.info("{}: {}".format(key, value))


if __name__ == "__main__":
    main()

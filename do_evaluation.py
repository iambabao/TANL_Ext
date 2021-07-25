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
import numpy as np
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer

from src.data_processor import load_dataset
from src.utils.tanl_utils import get_precision_recall_f1
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
    parser.add_argument('--dataset_episode', type=str, required=True, help='')
    parser.add_argument('--max_seq_length', type=int, required=True, help='')
    parser.add_argument('--max_output_seq_length', type=int, default=None, help='')
    parser.add_argument('--max_seq_length_eval', type=int, default=None, help='')
    parser.add_argument('--max_output_seq_length_eval', type=int, default=None, help='')
    parser.add_argument('--eval_nll', type=bool, default=False, help='')
    parser.add_argument('--chunk_size', type=int, default=128, help='')
    parser.add_argument('--chunk_overlap', type=int, default=64, help='')
    parser.add_argument('--chunk_size_eval', type=int, default=None, help='')
    parser.add_argument('--chunk_overlap_eval', type=int, default=None, help='')
    parser.add_argument('--generated_outputs', type=str, required=True, help='')
    parser.add_argument('--compute_macro', action='store_true', help='')
    parser.add_argument('--overwrite_cache', type=bool, default=True, help='')
    parser.add_argument('--input_format', type=str, default=None, help='')
    parser.add_argument('--output_format', type=str, default=None, help='')
    parser.add_argument("--multitask", action='store_true', help="")
    parser.add_argument("--log_file", type=str, default=None, help="")
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
    init_logger(logging.INFO, filename=args.log_file)

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
        seed=args.dataset_episode,
        shuffle=False,
        is_eval=True,
    )
    generated_outputs = list(line['generated'] for line in read_json_lines(args.generated_outputs))
    assert len(dataset.examples) == len(generated_outputs)

    results = Counter()
    for example, output in tqdm(zip(dataset.examples, generated_outputs)):
        cur_results = dataset.evaluate_example(example=example, output_sentence=output)
        results += cur_results
    entity_precision, entity_recall, entity_f1 = get_precision_recall_f1(
        num_correct=results['correct_entities'],
        num_predicted=results['predicted_entities'],
        num_gt=results['gt_entities'],
    )
    entity_precision_no_type, entity_recall_no_type, entity_f1_no_type = get_precision_recall_f1(
        num_correct=results['correct_entities_no_type'],
        num_predicted=results['predicted_entities_no_type'],
        num_gt=results['gt_entities_no_type'],
    )
    relation_precision, relation_recall, relation_f1 = get_precision_recall_f1(
        num_correct=results['correct_relations'],
        num_predicted=results['predicted_relations'],
        num_gt=results['gt_relations'],
    )

    final_results = {
        'wrong_reconstruction': results['wrong_reconstructions'] / results['num_sentences'],
        'label_error': results['label_error'] / results['num_sentences'],
        'entity_error': results['entity_error'] / results['num_sentences'],
        'format_error': results['format_error'] / results['num_sentences'],
        'entity_precision': entity_precision,
        'entity_recall': entity_recall,
        'entity_f1': entity_f1,
        'relation_precision': relation_precision,
        'relation_recall': relation_recall,
        'relation_f1': relation_f1,
        'entity_precision_no_type': entity_precision_no_type,
        'entity_recall_no_type': entity_recall_no_type,
        'entity_f1_no_type': entity_f1_no_type,
    }

    if args.compute_macro:
        # compute also entity macro scores
        entity_precision_by_type = []
        entity_recall_by_type = []
        entity_f1_by_type = []
        for entity_type in dataset.entity_types.values():
            precision, recall, f1 = get_precision_recall_f1(
                num_correct=results['correct_entities', entity_type.natural],
                num_predicted=results['predicted_entities', entity_type.natural],
                num_gt=results['gt_entities', entity_type.natural],
            )
            entity_precision_by_type.append(precision)
            entity_recall_by_type.append(recall)
            entity_f1_by_type.append(f1)
        final_results.update({
            'entity_macro_precision': np.mean(np.array(entity_precision_by_type)),
            'entity_macro_recall': np.mean(np.array(entity_recall_by_type)),
            'entity_macro_f1': np.mean(np.array(entity_f1_by_type)),
        })

    for key, value in final_results.items():
        logger.info("{}: {}".format(key, value))


if __name__ == "__main__":
    main()

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
from transformers import HfArgumentParser, TrainingArguments, AutoTokenizer
from transformers.trainer_utils import is_main_process

from src.utils.arguments import ModelArguments, DataTrainingArguments
from src.data_processor import load_dataset
from src.utils.tanl_utils import get_episode_indices
from src.utils.my_utils import init_logger, save_file

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job', type=str, required=True, help='')
    parser.add_argument('-c', '--config', type=str, default='data/config.ini', help='')
    args, remaining_args = parser.parse_known_args()

    config = configparser.ConfigParser(allow_no_value=False)
    config.read(args.config)
    assert args.job in config
    all_args = []
    for key, value in config.items(args.job):
        if key in ['do_train', 'do_eval', 'do_predict', 'no_cuda']:
            if value.lower() == 'true':
                all_args += ['--{}'.format(key)]
        else:
            all_args += ['--{}'.format(key), value]
    all_args += remaining_args

    hf_parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = hf_parser.parse_args_into_dataclasses(all_args)

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        "{}_{}_{}".format(
            args.job,
            list(filter(None, model_args.model_name_or_path.split("/"))).pop(),
            data_args.max_seq_length,
        ),
    )

    init_logger(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    logger.info("Training/evaluation parameters {}".format(training_args))

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # get list of dataset names
    dataset_names = data_args.datasets.split(',')

    # construct list of episode indices
    episode_indices = get_episode_indices(data_args.episodes)

    # episode loop (note that the episode index is used as the random seed, so that each episode is reproducible)
    for ep_idx in episode_indices:
        logging.info('Episode {} / {}'.format(ep_idx, len(episode_indices)))
        episode_output_dir = os.path.join(training_args.output_dir, 'episode_{:02d}'.format(ep_idx))
        os.makedirs(episode_output_dir, exist_ok=True)

        for dataset_name in dataset_names:
            logging.info('Processing dataset {}'.format(dataset_name))

            logger.info('Train!')
            dataset = load_dataset(
                dataset_name, data_args, split='train',
                max_input_length=data_args.max_seq_length, max_output_length=data_args.max_output_seq_length,
                tokenizer=tokenizer, seed=ep_idx, train_subset=data_args.train_subset,
            )
            src_file = os.path.join(episode_output_dir, 'train.source')
            save_file(dataset.input_sentences, src_file)
            tgt_file = os.path.join(episode_output_dir, 'train.target')
            save_file(dataset.output_sentences, tgt_file)

            logger.info('Valid!')
            dataset = load_dataset(
                dataset_name, data_args, split='dev',
                max_input_length=data_args.max_seq_length, max_output_length=data_args.max_output_seq_length,
                tokenizer=tokenizer, seed=ep_idx, train_subset=data_args.train_subset,
            )
            src_file = os.path.join(episode_output_dir, 'valid.source')
            save_file(dataset.input_sentences, src_file)
            tgt_file = os.path.join(episode_output_dir, 'valid.target')
            save_file(dataset.output_sentences, tgt_file)

            logger.info('Test!')
            dataset = load_dataset(
                dataset_name, data_args, split='test',
                max_input_length=data_args.max_seq_length, max_output_length=data_args.max_output_seq_length,
                tokenizer=tokenizer, seed=ep_idx, train_subset=data_args.train_subset,
            )
            src_file = os.path.join(episode_output_dir, 'test.source')
            save_file(dataset.input_sentences, src_file)
            tgt_file = os.path.join(episode_output_dir, 'test.target')
            save_file(dataset.output_sentences, tgt_file)


if __name__ == '__main__':
    main()

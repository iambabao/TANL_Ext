# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/26
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2021/5/2
"""

import logging
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss

from src.utils.my_utils import read_json_lines

logger = logging.getLogger(__name__)


def convert_examples_to_features(examples, tokenizer, max_src_length=None, max_tgt_length=None, with_prefix=False):
    if max_src_length is None:
        max_src_length = tokenizer.model_max_length
    if max_tgt_length is None:
        max_tgt_length = max_src_length

    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc="Converting Examples")):
        input_text = "{} : {}".format(example["task_name"], example["source"]) if with_prefix else example["source"]
        src_encoded = tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=max_src_length,
        )
        tgt_encoded = tokenizer(example["target"], padding="max_length", truncation=True, max_length=max_tgt_length)
        encoded = {
            "guid": example["guid"],
            "task_name": example["task_name"],
            "task_id": example["task_id"],
            "input_ids": np.array(src_encoded["input_ids"], dtype=np.int),
            "attention_mask": np.array(src_encoded["attention_mask"], dtype=np.int),
            "labels": np.array(tgt_encoded["input_ids"], dtype=np.int),
        }
        features.append(encoded)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(encoded["guid"]))
            logger.info("task_name: {}".format(encoded["task_name"]))
            logger.info("input_ids: {}".format(encoded["input_ids"]))
            logger.info("attention_mask: {}".format(encoded["attention_mask"]))
            logger.info("labels: {}".format(encoded["labels"]))
            logger.info("source: {}".format(tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)))
            logger.info("target: {}".format(tokenizer.decode(encoded["labels"], skip_special_tokens=True)))

    return features


class DataProcessorForAdapter:
    def __init__(
            self,
            model_name_or_path,
            max_src_length,
            max_tgt_length,
            with_prefix,
            tasks,
            data_dir="",
            cache_dir="cache",
            overwrite_cache=False,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.with_prefix = with_prefix

        self.tasks = tasks
        self.data_dir = data_dir
        self.cache_dir = "{}_{}".format(cache_dir, "discrete" if with_prefix else "continuous")
        self.overwrite_cache = overwrite_cache

        self.id2task = {uid: task for uid, task in enumerate(tasks)}
        self.task2id = {task: uid for uid, task in enumerate(tasks)}
        self.transition_matrix = [[1.0] * len(tasks) for _ in range(len(tasks))]

    def load_and_cache_data(self, role, tokenizer, suffix=None):
        if suffix is not None:
            role = '{}_{}'.format(role, suffix)

        all_examples, all_features, counter = [], [], {}
        for task in self.tasks:
            data_dir = os.path.join(self.data_dir, task)
            cache_dir = os.path.join(data_dir, self.cache_dir)
            os.makedirs(cache_dir, exist_ok=True)

            cached_examples = os.path.join(cache_dir, "cached_example_{}".format(role))
            if os.path.exists(cached_examples) and not self.overwrite_cache:
                logger.info("Loading examples from cached file {}".format(cached_examples))
                examples = torch.load(cached_examples)
            else:
                examples = []
                for line in tqdm(
                    list(read_json_lines(os.path.join(data_dir, "data_{}.json".format(role)))),
                    desc="Loading Examples"
                ):
                    sample = {"guid": "{}-{}".format(task, len(examples))}
                    sample.update(line)
                    sample["task_id"] = self.task2id[line["task_name"]]
                    examples.append(sample)
                logger.info("Saving examples into cached file {}".format(cached_examples))
                torch.save(examples, cached_examples)
            all_examples.extend(examples)

            cached_features = os.path.join(
                cache_dir,
                "cached_feature_{}_{}_{}_{}".format(
                    role,
                    list(filter(None, self.model_name_or_path.split("/"))).pop(),
                    self.max_src_length,
                    self.max_tgt_length,
                ),
            )
            if os.path.exists(cached_features) and not self.overwrite_cache:
                logger.info("Loading features from cached file {}".format(cached_features))
                features = torch.load(cached_features)
            else:
                features = convert_examples_to_features(examples, tokenizer, self.max_src_length, self.max_tgt_length)
                logger.info("Saving features into cached file {}".format(cached_features))
                torch.save(features, cached_features)
            all_features.extend(features)

            counter[task] = len(features)

        for task, cnt in counter.items():
            logger.info("{}: {}".format(task, cnt))

        return all_examples, all_features

    def sample_new_tasks(self, raw_tasks, greedy=False):
        raw_task_ids = [self.task2id[task] for task in raw_tasks]

        if greedy:
            new_task_ids = [np.argmax(self.transition_matrix[task_id]) for task_id in raw_task_ids]
        else:
            new_task_ids = [
                random.choices(list(range(len(self.tasks))), self.transition_matrix[task_id])[0]
                for task_id in raw_task_ids
            ]
        new_tasks = [self.id2task[task_id] for task_id in new_task_ids]

        raw_task_ids = torch.tensor(raw_task_ids, dtype=torch.long)
        new_task_ids = torch.tensor(new_task_ids, dtype=torch.long)

        return raw_tasks, new_tasks, raw_task_ids, new_task_ids

    def update_transition_matrix(self, raw_loss, new_loss, raw_task_id, new_task_id):
        for l1, l2, t1, t2 in zip(raw_loss, new_loss, raw_task_id, new_task_id):
            self.transition_matrix[t1][t2] += 1 if l1 > l2 else -1

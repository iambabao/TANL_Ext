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
import copy
import json
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset

from src.utils.my_utils import read_json_lines

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, source, target, task_name):
        self.guid = guid
        self.source = source
        self.target = target
        self.task_name = task_name

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    def __init__(self, guid, input_ids, attention_mask=None, labels=None):
        self.guid = guid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(examples, tokenizer, max_src_length=None, max_tgt_length=None, with_prefix=False):
    if max_src_length is None:
        max_src_length = tokenizer.model_max_length
    if max_tgt_length is None:
        max_tgt_length = max_src_length

    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc="Converting Examples")):
        src_encoded = tokenizer.encode_plus(
            example.source if not with_prefix else "{}: {}".format(example.task_name, example.source),
            padding="max_length",
            truncation=True,
            max_length=max_src_length,
        )
        tgt_encoded = tokenizer.encode_plus(
            example.target,
            padding="max_length",
            truncation=True,
            max_length=max_tgt_length,
        )
        encoded = {
            "guid": example.guid,
            "input_ids": src_encoded["input_ids"],
            "attention_mask": src_encoded["attention_mask"],
            "labels": tgt_encoded["input_ids"],
        }
        features.append(InputFeatures(**encoded))

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(example.guid))
            logger.info("input_ids: {}".format(encoded["input_ids"]))
            logger.info("attention_mask: {}".format(encoded["attention_mask"]))
            logger.info("labels: {}".format(encoded["labels"]))
            logger.info("source: {}".format(tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)))
            logger.info("target: {}".format(tokenizer.decode(encoded["labels"], skip_special_tokens=True)))

    return features


class DataProcessor:
    def __init__(
            self,
            tasks,
            model_name_or_path,
            max_src_length,
            max_tgt_length,
            data_dir="",
            cache_dir="cache",
            with_prefix=False,
            do_lower_case=False,
            overwrite_cache=False
    ):
        self.tasks = tasks
        self.model_name_or_path = model_name_or_path
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length

        self.data_dir = data_dir
        self.cache_dir = "{}_{}_{}".format(
            cache_dir,
            "raw" if not with_prefix else "prefix",
            "uncased" if do_lower_case else "cased",
        )

        self.do_lower_case = do_lower_case
        self.with_prefix = with_prefix
        self.overwrite_cache = overwrite_cache

    def load_and_cache_data(self, tokenizer, role, suffix=None):
        if suffix is not None:
            role = "{}_{}".format(role, suffix)

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
                    sample = {
                        "guid": "{}-{}".format(task, len(examples)),
                        "source": line["source"],
                        "target": line["target"],
                        "task_name": line["task_name"],
                    }
                    examples.append(InputExample(**sample))
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
                features = convert_examples_to_features(
                    examples,
                    tokenizer,
                    self.max_src_length,
                    self.max_tgt_length,
                    with_prefix=self.with_prefix,
                )
                logger.info("Saving features into cached file {}".format(cached_features))
                torch.save(features, cached_features)
            all_features.extend(features)

            counter[task] = len(features)

        for task, cnt in counter.items():
            logger.info("{}: {}".format(task, cnt))

        return all_examples, self._create_tensor_dataset(all_features)

    def _create_tensor_dataset(self, features):
        all_input_ids = torch.tensor([_.input_ids for _ in features], dtype=torch.long)
        all_attention_mask = torch.tensor([_.attention_mask for _ in features], dtype=torch.long)
        all_labels = torch.tensor([_.labels for _ in features], dtype=torch.long)

        return TensorDataset(all_input_ids, all_attention_mask, all_labels)

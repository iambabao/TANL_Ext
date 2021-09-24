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


def convert_examples_to_features(examples, tokenizer, max_src_length=None, max_tgt_length=None):
    if max_src_length is None:
        max_src_length = tokenizer.model_max_length
    if max_tgt_length is None:
        max_tgt_length = max_src_length

    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc="Converting Examples")):
        src_encoded = tokenizer(
            "{} : {}".format(example["task_name"], example["source"]),
            padding="max_length",
            truncation=True,
            max_length=max_src_length,
        )
        tgt_encoded = tokenizer(example["target"], padding="max_length", truncation=True, max_length=max_tgt_length)
        encoded = {
            "guid": example["guid"],
            "task_name": example["task_name"],
            "source": example["source"],
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


class DataProcessorTop:
    def __init__(
            self,
            model_name_or_path,
            max_src_length,
            max_tgt_length,
            tasks,
            data_dir="",
            cache_dir="cache_top",
            overwrite_cache=False,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length

        self.tasks = tasks
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.overwrite_cache = overwrite_cache

        self.transition_matrix = {src: {dst: 1.0 for dst in tasks} for src in tasks}

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
                    sample = {'guid': '{}-{}'.format(task, len(examples))}
                    sample.update(line)
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

    def generate_augmented_data(self, args, model, tokenizer, batch):
        while hasattr(model, "module"):
            model = model.module

        raw_task_name = batch["task_name"]
        new_task_name = [
            random.choices(list(self.transition_matrix[task].keys()), list(self.transition_matrix[task].values()))
            for task in batch['task_name']
        ]

        source = ["{} : {}".format(task_name, source) for task_name, source in zip(new_task_name, batch["source"])]
        encoded = tokenizer.batch_encode_plus(
            source,
            padding="max_length",
            truncation=True,
            max_length=self.max_src_length,
            return_tensors="pt",
        )
        for key, value in encoded.items():
            encoded[key] = value.to(args.device)
        augmented_outputs = model.generate(**encoded, max_length=self.max_tgt_length).detach().cpu().tolist()

        augmented_source = []
        for task_name, line in zip(raw_task_name, augmented_outputs):
            augmented_source.append("{} : {}".format(
                task_name,
                tokenizer.decode(line, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            ))

        encoded = tokenizer.batch_encode_plus(
            augmented_source,
            padding="max_length",
            truncation=True,
            max_length=self.max_src_length,
            return_tensors="pt",
        )

        return encoded, raw_task_name, new_task_name

    def update_transition_matrix(self, raw_logits, new_logits, labels, raw_tasks, new_tasks):
        loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
        raw_loss = loss_fct(raw_logits.view(-1, raw_logits.size(-1)), labels.view(-1))
        raw_loss = raw_loss.view(labels.shape).mean(dim=-1)
        new_loss = loss_fct(new_logits.view(-1, new_logits.size(-1)), labels.view(-1))
        new_loss = new_loss.view(labels.shape).mean(dim=-1)

        for l1, l2, t1, t2 in zip(raw_loss, new_loss, raw_tasks, new_tasks):
            self.transition_matrix[t1][t2] += 1 if l1 > l2 else -1


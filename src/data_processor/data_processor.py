# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/26
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2021/5/2
"""

import logging
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, datasets, tokenizer):
        self.datasets = datasets
        self.tokenizer = tokenizer

    def create_tensor_dataset(self, tasks, split, keep_entity=1.00):
        input_ids, attention_mask, label_ids = [], [], []
        for task_id, task in enumerate(tasks):
            dataset = self.datasets[task][split]
            if isinstance(keep_entity, float):
                if keep_entity == 1.00:
                    features = dataset.features
                else:
                    features = dataset.compute_features(
                        max_input_length=dataset.max_input_length,
                        max_output_length=dataset.max_output_length,
                        prefix=dataset.data_args.prefix,
                        keep_entity=keep_entity,
                    )
            elif isinstance(keep_entity, list):
                assert len(tasks) == len(keep_entity)
                features = dataset.compute_features(
                    max_input_length=dataset.max_input_length,
                    max_output_length=dataset.max_output_length,
                    prefix=dataset.data_args.prefix,
                    keep_entity=keep_entity[task_id],
                )
            else:
                raise ValueError("except float or list, but '{}' is given".format(keep_entity))
            for feature in features:
                input_ids.append(feature.input_ids)
                attention_mask.append(feature.attention_mask)
                label_ids.append(feature.label_ids)
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        label_ids = torch.tensor(label_ids)
        return TensorDataset(input_ids, attention_mask, label_ids)

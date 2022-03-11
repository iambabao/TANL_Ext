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


def create_tensor_dataset(features):
    input_ids = torch.tensor(features.input_ids, dtype=torch.long)
    attention_mask = torch.tensor(features.attention_mask, dtype=torch.long)
    labels = torch.tensor(features.label_ids, dtype=torch.long)

    return TensorDataset(input_ids, attention_mask, labels)


class DataProcessor:
    def __init__(self, datasets, tokenizer):
        self.datasets = datasets
        self.tokenizer = tokenizer

    def create_tensor_dataset(self, tasks, split):
        input_ids, attention_mask, label_ids = [], [], []
        for task in tasks:
            input_ids.extend(self.datasets[task][split].input_ids)
            attention_mask.extend(self.datasets[task][split].attention_mask)
            label_ids.extend(self.datasets[task][split].label_ids)
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        label_ids = torch.tensor(label_ids)
        return TensorDataset(input_ids, attention_mask, label_ids)

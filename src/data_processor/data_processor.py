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

    def create_tensor_dataset(self, tasks, split):
        input_ids, attention_mask, label_ids = [], [], []
        for task in tasks:
            for feature in self.datasets[task][split].features:
                input_ids.append(feature.input_ids)
                attention_mask.append(feature.attention_mask)
                label_ids.append(feature.label_ids)
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        label_ids = torch.tensor(label_ids)
        return TensorDataset(input_ids, attention_mask, label_ids)

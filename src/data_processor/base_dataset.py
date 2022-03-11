# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import logging
from typing import Dict, Generator, Tuple, List
from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, torch_distributed_zero_first, default_data_collator

from src.data_processor.input_example import InputFeatures, InputExample
from src.data_processor.input_formats import INPUT_FORMATS
from src.data_processor.output_formats import OUTPUT_FORMATS


class BaseDataset(Dataset, ABC):
    """
    Base class for all datasets.
    """
    name = None  # name of the dataset
    data_name = None  # name of the directory, if different from the name of the dataset
    task_descriptor = None  # string to prepend to every input sentence if prefix=True (default is self.name)

    default_input_format = 'plain'
    default_output_format = 'full'
    default_data_dir = 'data'

    input_sentences = None
    output_sentences = None

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            max_input_length: int,
            max_output_length: int,
            mode: str = 'train',
            local_rank: int = -1,
            train_subset: float = 1,  # a number < 1 is to use only a subset of training data (random)
            data_args = None,
    ):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.prefix = data_args.prefix

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.input_format = INPUT_FORMATS[self.default_input_format]()
        self.output_format = OUTPUT_FORMATS[self.default_output_format]()

        self.data_path = data_args.data_dir if data_args.data_dir is not None else self.default_data_dir
        cached_data_file = os.path.join(
            self.data_dir(),
            "cached_{}_{}_{}_{}_{}_{}.bin".format(
                self.name,
                mode,
                list(filter(None, data_args.model_name_or_path.split("/"))).pop(),
                max_input_length,
                max_output_length,
                "raw" if not data_args.prefix else "prefix",
                "uncased" if data_args.do_lower_case else "cased",
            ),
        )

        with torch_distributed_zero_first(local_rank):
            # make sure only the first process in distributed training processes the dataset,
            # and the others can use the cached version

            if os.path.exists(cached_data_file) and not self.data_args.overwrite_cache:
                self.load_cached_data(cached_data_file)
            else:
                self.load_schema()   # here the dataset can load information such as entity/relation types
                self.examples = self.load_data(mode=mode)

                # assign examples to this dataset
                for example in self.examples:
                    example.dataset = self

                self.features = self.compute_features(
                    max_input_length=max_input_length,
                    max_output_length=max_output_length,
                    prefix=data_args.prefix,
                )

                if local_rank in [-1, 0]:
                    # save data
                    self.save_data(cached_data_file)

            self.indices = list(range(len(self.examples)))

            # compute effective size of the dataset
            self.effective_size = round(train_subset * len(self.examples))
            if train_subset != 1:
                logging.info(f"Effective dataset size reduced to {self.effective_size} ({train_subset * 100:.0f}%)")

    def __repr__(self):
        return f'Dataset {self.name}'

    def __len__(self):
        return self.effective_size

    def __getitem__(self, i: int) -> InputFeatures:
        return self.features[self.indices[i]]

    def get_example(self, i: int) -> InputExample:
        return self.examples[self.indices[i]]

    def data_dir(self):
        if self.data_name is not None:
            return os.path.join(self.data_path, self.data_name)
        else:
            return os.path.join(self.data_path, self.name)

    def load_cached_data(self, cached_data_file: str):
        d = torch.load(cached_data_file)
        self.examples, self.features = d['examples'], d['features']

    def save_data(self, cached_data_file: str):
        torch.save({'examples': self.examples, 'features': self.features}, cached_data_file)

    def load_schema(self):
        """
        Load extra dataset information, such as entity/relation types.
        """
        pass

    @abstractmethod
    def load_data_single_split(self, split: str) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        pass

    def load_data(self, mode: str) -> List[InputExample]:
        """
        Load all data, where 'mode' is a list of comma-separated splits to use.
        """
        examples = []

        if isinstance(mode, str):
            splits = mode.split(',')
        else:
            assert isinstance(mode, (list, tuple))
            splits = mode

        for split in splits:
            examples += self.load_data_single_split(split)

        return examples

    def _warn_max_sequence_length(self, max_sequence_length: int, sentences: List[str], name: str):
        max_length_needed = max(len(self.tokenizer.tokenize(x)) for x in sentences)
        if max_length_needed > max_sequence_length:
            logging.warning(
                f'Max sequence length is {max_sequence_length} but the longest {name} sequence is '
                f'{max_length_needed} long'
            )

    def compute_features(self, max_input_length: int, max_output_length: int, prefix: bool = False):
        input_sentences = [self.input_format.format_input(example, prefix=prefix) for example in self.examples]
        output_sentences = [self.output_format.format_output(example) for example in self.examples]
        self.input_sentences = input_sentences
        self.output_sentences = output_sentences

        input_tok = self.tokenizer.batch_encode_plus(
            input_sentences,
            max_length=max_input_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        self._warn_max_sequence_length(max_input_length, input_sentences, "input")

        output_tok = self.tokenizer.batch_encode_plus(
            output_sentences,
            max_length=max_output_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        self._warn_max_sequence_length(max_output_length, output_sentences, "output")

        assert input_tok.input_ids.size(0) == output_tok.input_ids.size(0)
    
        features = []
        for sentence_input_ids, att_mask, label_input_ids in zip(
                input_tok.input_ids, input_tok.attention_mask, output_tok.input_ids
        ):
            features.append(InputFeatures(
                input_ids=sentence_input_ids.tolist(),
                attention_mask=att_mask.tolist(),
                label_ids=label_input_ids.tolist()
            ))
    
        return features

    def generate_output_sentences(
            self,
            data_args,
            model,
            device,
            batch_size: int
    ) -> Generator[Tuple[InputExample, str], None, None]:
        """
        Generate pairs (example, output_sentence) for evaluation.
        """
        test_data_loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )

        for i, inputs in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
            predictions = model.generate(
                inputs['input_ids'].to(device),
                max_length=data_args.max_output_seq_length_eval,
                num_beams=data_args.num_beams,
            )

            for j, (input_ids, label_ids, prediction) in enumerate(
                    zip(inputs['input_ids'], inputs['labels'], predictions)
            ):
                current_id = i * batch_size + j
                example = self.get_example(current_id)
                output_sentence = self.tokenizer.decode(
                    prediction, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                yield example, output_sentence

    @abstractmethod
    def evaluate_dataset(self, data_args, model, device, batch_size: int, macro: bool = False) -> Dict[str, float]:
        """
        Evaluate model on this dataset, returning the task-relevant metrics.
        """
        pass

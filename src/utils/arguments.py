# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Uses some code from
# https://github.com/huggingface/transformers/blob/master/examples/seq2seq/finetune_trainer.py


from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: Optional[str] = field(default=None, metadata={"help": "Path to data directory"})
    datasets: Optional[str] = field(
        default=None,
        metadata={"help": "Comma separated list of dataset names for training."}
    )
    eval_datasets: Optional[str] = field(
        default=None,
        metadata={"help": "Comma separated list of dataset names for evaluation. Defaults to the train datasets."}
    )
    train_split: str = field(
        default='train', metadata={"help": "The data split for training. Can be 'train', 'dev', 'test', etc."}
    )
    train_subset: float = field(default=1, metadata={"help": "The portion of training data to use"})
    episodes: str = field(
        default='0',
        metadata={
            "help": "Episode indices -- a single number such as 3 or an interval such as 1-4. "
                    "The index is also used as random seeds "
                    "and this setting is therefore used to repeat multiple experiments."
        }
    )
    multitask: bool = field(
        default=False, metadata={"help": "If true, each input sentence is prepended with the dataset name"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. "
                    "Sequences longer than this will be truncated, sequences shorter will be padded."
        },
    )
    num_beams: int = field(
        default=None, metadata={"help": "Number of beams for beam search during generation (only affects evaluation)"}
    )
    input_format: str = field(default=None, metadata={"help": "Input format"})
    output_format: str = field(default=None, metadata={"help": "Output format"})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )


    """
    Other parameters
    """
    log_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the log."}
    )

    # length arguments
    max_output_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total output sequence length after tokenization. "
                    "Default is the same as 'max_seq_length'."
        },
    )
    max_seq_length_eval: int = field(
        default=None,
        metadata={
            "help": "Maximum input sequence length at evaluation time (default is equal to max_seq_length)"
        },
    )
    max_output_seq_length_eval: int = field(
        default=None,
        metadata={
            "help": "The maximum output sequence length at evaluation time (default is the same as input)"
        },
    )

    # few-shot arguments
    num_shots: int = field(
        default=None, metadata={"help": "number of shots (few-shot argument for the FewRel dataset)"}
    )
    num_ways: int = field(
        default=None, metadata={"help": "number of ways (few-shot argument for the FewRel dataset)"}
    )
    num_query: int = field(
        default=5, metadata={"help": "number of query examples (few-shot argument for the FewRel dataset)"}
    )

    # chunk arguments (used for the CoNLL2012 coreference resolution dataset)
    chunk_size: int = field(
        default=128, metadata={"help": "Size of document chunks"}
    )
    chunk_overlap: int = field(
        default=64, metadata={"help": "Size of overlap between consecutive chunks"}
    )
    chunk_size_eval: int = field(
        default=None, metadata={"help": "Size of document chunks during evaluation (default is equal to chunk_size)"}
    )
    chunk_overlap_eval: int = field(
        default=None, metadata={"help": "Size of overlap between consecutive chunks during evaluation "
                                        "(default is equal to chunk_overlap)"}
    )
    eval_nll: bool = field(
        default=False, metadata={"help": "Evaluate using NLL (only applicable to certain datasets)"}
    )

    def __post_init__(self):
        if self.eval_datasets is None:
            self.eval_datasets = self.datasets

        if self.max_output_seq_length_eval is None:
            # defaults first to max_output_seq_length, then max_seq_length_eval, then max_seq_length
            self.max_output_seq_length_eval = self.max_output_seq_length \
                                              or self.max_seq_length_eval \
                                              or self.max_seq_length
        if self.max_output_seq_length is None:
            self.max_output_seq_length = self.max_seq_length

        if self.max_seq_length_eval is None:
            self.max_seq_length_eval = self.max_seq_length

        if self.chunk_size_eval is None:
            self.chunk_size_eval = self.chunk_size

        if self.chunk_overlap_eval is None:
            self.chunk_overlap_eval = self.chunk_overlap

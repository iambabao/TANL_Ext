# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from abc import ABC, abstractmethod

from src.data_processor.input_example import InputExample
from src.utils.tanl_utils import augment_sentence
from src.utils.my_utils import is_trigger

INPUT_FORMATS = {}


def register_input_format(format_class):
    INPUT_FORMATS[format_class.name] = format_class
    return format_class


class BaseInputFormat(ABC):
    name = None

    BEGIN_ENTITY_TOKEN = '['
    END_ENTITY_TOKEN = ']'
    SEPARATOR_TOKEN = '|'
    RELATION_SEPARATOR_TOKEN = '='
    QUERY_SEPARATOR_TOKEN = ':'

    def format_input(self, example: InputExample, multitask=False, task_descriptor=None):
        res = self._format_input(example=example)
        if multitask:
            name = task_descriptor or example.dataset.task_descriptor or example.dataset.name
            res = f'{name} {self.QUERY_SEPARATOR_TOKEN} ' + res
        return res

    @abstractmethod
    def _format_input(self, example: InputExample) -> str:
        raise NotImplementedError


@register_input_format
class PlainInputFormat(BaseInputFormat):
    """
    This format uses the plain sentence as input.
    """
    name = 'plain'

    def _format_input(self, example: InputExample) -> str:
        return ' '.join(example.tokens)


@register_input_format
class EntityInputFormat(BaseInputFormat):
    """
    This format uses the sentence with given entities as input.
    """
    name = 'entity'

    def _format_input(self, example: InputExample) -> str:
        augmentations = [
            ([(entity.type.natural,)], entity.start, entity.end)
            for entity in example.entities
        ]

        return augment_sentence(
            example.tokens, augmentations,
            self.BEGIN_ENTITY_TOKEN, self.SEPARATOR_TOKEN,
            self.RELATION_SEPARATOR_TOKEN, self.END_ENTITY_TOKEN
        )


@register_input_format
class TriggerInputFormat(BaseInputFormat):
    """
    This format uses the sentence with given triggers as input..
    """
    name = 'trigger'

    def _format_input(self, example: InputExample) -> str:
        augmentations = [
            ([(entity.type.natural,)], entity.start, entity.end)
            for entity in example.entities if is_trigger(entity)
        ]

        return augment_sentence(
            example.tokens, augmentations,
            self.BEGIN_ENTITY_TOKEN, self.SEPARATOR_TOKEN,
            self.RELATION_SEPARATOR_TOKEN, self.END_ENTITY_TOKEN
        )

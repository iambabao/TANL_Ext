# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, Dict
import numpy as np

from src.data_processor.input_example import InputExample, EntityType, RelationType
from src.utils.tanl_utils import augment_sentence

OUTPUT_FORMATS = {}


def register_output_format(format_class):
    OUTPUT_FORMATS[format_class.name] = format_class
    return format_class


class BaseOutputFormat(ABC):
    name = None

    BEGIN_ENTITY_TOKEN = '['
    END_ENTITY_TOKEN = ']'
    SEPARATOR_TOKEN = '|'
    RELATION_SEPARATOR_TOKEN = '='

    @abstractmethod
    def format_output(self, example: InputExample) -> str:
        """
        Format output in augmented natural language.
        """
        raise NotImplementedError

    def run_inference(
            self,
            example: InputExample,
            output_sentence: str,
            entity_types: Dict[str, EntityType] = None,
            relation_types: Dict[str, RelationType] = None
    ) -> Tuple[set, set, bool, bool, bool, bool]:
        """
        Process an output sentence to extract predicted entities and relations (among the given entity/relation types).

        Return the predicted entities, predicted relations, and four booleans which describe if certain kinds of errors
        occurred (wrong reconstruction of the sentence, label error, entity error, augmented language format error).
        """
        label_error = False     # whether the output sentence has at least one non-existing entity or relation type
        entity_error = False    # whether there is at least one relation pointing to a non-existing head entity
        format_error = False    # whether the augmented language format is invalid

        if output_sentence.count(self.BEGIN_ENTITY_TOKEN) != output_sentence.count(self.END_ENTITY_TOKEN):
            # the parentheses do not match
            format_error = True

        entity_types = set(entity_type.natural for entity_type in entity_types.values())

        if relation_types is not None:
            relation_types = set(relation_type.natural for relation_type in relation_types.values())
        else:
            relation_types = {}

        # parse output sentence
        raw_predicted_entities, wrong_reconstruction = self.parse_output_sentence(example, output_sentence)

        # update predicted entities with the positions in the original sentence
        predicted_entities_by_name = defaultdict(list)
        predicted_entities = set()
        raw_predicted_relations = []

        # process and filter entities
        for entity_name, tags, start, end in raw_predicted_entities:
            if len(tags) == 0 or len(tags[0]) > 1:
                # we do not have a tag for the entity type
                entity_tuple = ('none', start, end)
                predicted_entities.add(entity_tuple)
                predicted_entities_by_name[entity_name].append(entity_tuple)
                continue

            entity_type = tags[0][0]
            if entity_type in entity_types:
                entity_tuple = (entity_type, start, end)
                predicted_entities.add(entity_tuple)
                predicted_entities_by_name[entity_name].append(entity_tuple)

                # process tags to get relations
                for tag in tags[1:]:
                    if len(tag) == 2:
                        relation_type, related_entity = tag
                        if relation_type in relation_types:
                            raw_predicted_relations.append((relation_type, entity_tuple, related_entity))
                        else:
                            label_error = True
                    else:
                        # the relation tag has the wrong length
                        format_error = True
            else:
                # the predicted entity type does not exist
                label_error = True

        predicted_relations = set()

        for relation_type, entity_tuple, related_entity in raw_predicted_relations:
            if related_entity in predicted_entities_by_name:
                # look for the closest instance of the related entity (there could be many of them)
                _, head_start, head_end = entity_tuple
                candidates = sorted(
                    predicted_entities_by_name[related_entity],
                    key=lambda x: min(abs(x[1] - head_end), abs(head_start - x[2])),
                )
                for candidate in candidates:
                    relation = (relation_type, entity_tuple, candidate)
                    if relation not in predicted_relations:
                        predicted_relations.add(relation)
                        break
            else:
                # cannot find the related entity in the sentence
                entity_error = True

        return predicted_entities, predicted_relations, wrong_reconstruction, label_error, entity_error, format_error

    def parse_output_sentence(self, example: InputExample, output_sentence: str) -> Tuple[list, bool]:
        """
        Parse an output sentence in augmented language and extract inferred entities and tags.
        Return a pair (predicted_entities, wrong_reconstruction), where:
        - each element of predicted_entities is a tuple (entity_name, tags, start, end)
            - entity_name (str) is the name as extracted from the output sentence
            - tags is a list of tuples, obtained by |-splitting the part of the entity after the entity name
            - this entity corresponds to the tokens example.tokens[start:end]
            - note that the entity_name could differ from ' '.join(example.tokens[start:end]), if the model was not
              able to exactly reproduce the entity name, or if alignment failed
        - wrong_reconstruction (bool) says whether the output_sentence does not match example.tokens exactly

        An example follows.

        example.tokens:
        ['Tolkien', 'wrote', 'The', 'Lord', 'of', 'the', 'Rings']

        output_sentence:
        [ Tolkien | person ] wrote [ The Lord of the Rings | book | author = Tolkien ]

        output predicted entities:
        [
            ('Tolkien', [('person',)], 0, 1),
            ('The Lord of the Rings', [('book',), ('author', 'Tolkien')], 2, 7)
        ]
        """
        output_tokens = []
        unmatched_predicted_entities = []

        # add spaces around special tokens, so that they are alone when we split
        padded_output_sentence = output_sentence
        for special_token in [
            self.BEGIN_ENTITY_TOKEN, self.END_ENTITY_TOKEN,
            self.SEPARATOR_TOKEN, self.RELATION_SEPARATOR_TOKEN,
        ]:
            padded_output_sentence = padded_output_sentence.replace(special_token, ' ' + special_token + ' ')

        entity_stack = []   # stack of the entities we are extracting from the output sentence
        # this is a list of lists [start, state, entity_name_tokens, entity_other_tokens]
        # where state is "name" (before the first | separator) or "other" (after the first | separator)

        for token in padded_output_sentence.split():
            if len(token) == 0:
                continue

            elif token == self.BEGIN_ENTITY_TOKEN:
                # begin entity
                start = len(output_tokens)
                entity_stack.append([start, "name", [], []])

            elif token == self.END_ENTITY_TOKEN and len(entity_stack) > 0:
                # end entity
                start, state, entity_name_tokens, entity_other_tokens = entity_stack.pop()

                entity_name = ' '.join(entity_name_tokens).strip()
                end = len(output_tokens)

                tags = []

                # split entity_other_tokens by |
                splits = [
                    list(y) for x, y in itertools.groupby(entity_other_tokens, lambda z: z == self.SEPARATOR_TOKEN)
                    if not x
                ]

                if state == "other" and len(splits) > 0:
                    for x in splits:
                        tags.append(tuple(' '.join(x).split(' ' + self.RELATION_SEPARATOR_TOKEN + ' ')))

                unmatched_predicted_entities.append((entity_name, tags, start, end))

            else:
                # a normal token
                if len(entity_stack) > 0:
                    # inside some entities
                    if token == self.SEPARATOR_TOKEN:
                        x = entity_stack[-1]

                        if x[1] == "name":
                            # this token marks the end of name tokens for the current entity
                            x[1] = "other"
                        else:
                            # simply add this token to entity_other_tokens
                            x[3].append(token)

                    else:
                        is_name_token = True

                        for x in reversed(entity_stack):
                            # check state
                            if x[1] == "name":
                                # add this token to entity_name_tokens
                                x[2].append(token)

                            else:
                                # add this token to entity_other tokens and then stop going up in the tree
                                x[3].append(token)
                                is_name_token = False
                                break

                        if is_name_token:
                            output_tokens.append(token)

                else:
                    # outside
                    output_tokens.append(token)

        # check if we reconstructed the original sentence correctly, after removing all spaces
        wrong_reconstruction = (''.join(output_tokens) != ''.join(example.tokens))

        # now we align self.tokens with output_tokens (with dynamic programming)
        cost = np.zeros((len(example.tokens) + 1, len(output_tokens) + 1))  # cost of alignment between tokens[:i]
        # and output_tokens[:j]
        best = np.zeros_like(cost, dtype=int)  # best choice when aligning tokens[:i] and output_tokens[:j]

        for i in range(len(example.tokens) + 1):
            for j in range(len(output_tokens) + 1):
                if i == 0 and j == 0:
                    continue

                candidates = []

                # match
                if i > 0 and j > 0:
                    candidates.append(
                        ((0 if example.tokens[i - 1] == output_tokens[j - 1] else 1) + cost[i - 1, j - 1], 1))

                # skip in the first sequence
                if i > 0:
                    candidates.append((1 + cost[i - 1, j], 2))

                # skip in the second sequence
                if j > 0:
                    candidates.append((1 + cost[i, j - 1], 3))

                chosen_cost, chosen_option = min(candidates)
                cost[i, j] = chosen_cost
                best[i, j] = chosen_option

        # reconstruct best alignment
        matching = {}

        i = len(example.tokens) - 1
        j = len(output_tokens) - 1

        while i >= 0 and j >= 0:
            chosen_option = best[i + 1, j + 1]

            if chosen_option == 1:
                # match
                matching[j] = i
                i, j = i - 1, j - 1

            elif chosen_option == 2:
                # skip in the first sequence
                i -= 1

            else:
                # skip in the second sequence
                j -= 1

        # update predicted entities with the positions in the original sentence
        predicted_entities = []

        for entity_name, entity_tags, start, end in unmatched_predicted_entities:
            new_start = None  # start in the original sequence
            new_end = None  # end in the original sequence

            for j in range(start, end):
                if j in matching:
                    if new_start is None:
                        new_start = matching[j]

                    new_end = matching[j]

            if new_start is not None:
                # predict entity
                entity_tuple = (entity_name, entity_tags, new_start, new_end + 1)
                predicted_entities.append(entity_tuple)

        return predicted_entities, wrong_reconstruction


@register_output_format
class EntityOutputFormat(BaseOutputFormat):
    """
    Output format with only entities.
    """
    name = 'entity'

    def format_output(self, example: InputExample) -> str:
        augmentations = [([(entity.type.natural,)], entity.start, entity.end) for entity in example.entities]

        return augment_sentence(
            example.tokens, augmentations,
            self.BEGIN_ENTITY_TOKEN, self.SEPARATOR_TOKEN,
            self.RELATION_SEPARATOR_TOKEN, self.END_ENTITY_TOKEN
        )


@register_output_format
class EntityBoundaryOutputFormat(BaseOutputFormat):
    """
    Output format with only entities.
    """
    name = 'entity_boundary'

    def format_output(self, example: InputExample) -> str:
        augmentations = [([], entity.start, entity.end) for entity in example.entities]

        return augment_sentence(
            example.tokens, augmentations,
            self.BEGIN_ENTITY_TOKEN, self.SEPARATOR_TOKEN,
            self.RELATION_SEPARATOR_TOKEN, self.END_ENTITY_TOKEN
        )


@register_output_format
class FullOutputFormat(BaseOutputFormat):
    """
    Output format with both entities and relations.
    """
    name = 'full'

    def format_output(self, example: InputExample) -> str:
        """
        Get output in augmented natural language, for example:
        [ Tolkien | person | born in = here ] was born [ here | location ]
        """
        # organize relations by head entity
        relations_by_entity = {entity: [] for entity in example.entities}
        for relation in example.relations:
            relations_by_entity[relation.head].append((relation.type, relation.tail))

        augmentations = []
        for entity in example.entities:
            tags = [(entity.type.natural,)]
            for relation_type, tail in relations_by_entity[entity]:
                tags.append((relation_type.natural, ' '.join(example.tokens[tail.start:tail.end])))

            augmentations.append((tags, entity.start, entity.end))

        return augment_sentence(
            example.tokens, augmentations,
            self.BEGIN_ENTITY_TOKEN, self.SEPARATOR_TOKEN,
            self.RELATION_SEPARATOR_TOKEN, self.END_ENTITY_TOKEN
        )

import logging
import os
import json
import torch
import numpy as np
from typing import Dict, List
from collections import Counter
from transformers import PreTrainedTokenizer

from src.data_processor.input_example import EntityType, RelationType, Entity, Relation, InputExample
from src.data_processor.base_dataset import BaseDataset
from src.utils.arguments import DataTrainingArguments
from src.utils.tanl_utils import get_precision_recall_f1

logger = logging.getLogger(__name__)
DATASETS = {}


def register_dataset(dataset_class):
    DATASETS[dataset_class.name] = dataset_class
    return dataset_class


def load_dataset(
        dataset_name: str,
        data_args,
        tokenizer: PreTrainedTokenizer,
        split: str,
        max_input_length: int,
        max_output_length: int,
        train_subset: float = 1,
        seed: int = None,
        shuffle: bool = True,
        is_eval: bool = False
):
    """
    Load a registered dataset.
    """
    return DATASETS[dataset_name](
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
        mode=split,
        overwrite_cache=data_args.overwrite_cache,
        train_subset=train_subset,
        seed=seed,
        shuffle=shuffle,
        data_args=data_args,
        is_eval=is_eval,
    )


class JointERDataset(BaseDataset):
    """
    Base class for datasets of joint entity and relation extraction.
    """
    entity_types = None
    relation_types = None
    natural_entity_types = None
    natural_relation_types = None

    default_output_format = 'joint_er'

    def load_cached_data(self, cached_features_file):
        d = torch.load(cached_features_file)
        self.entity_types, self.relation_types, self.examples, self.features = \
            d['entity_types'], d['relation_types'], d['examples'], d['features']

    def save_data(self, cached_features_file):
        torch.save({
            'entity_types': self.entity_types,
            'relation_types': self.relation_types,
            'examples': self.examples,
            'features': self.features,
        }, cached_features_file)

    def load_schema(self):
        """
        Load entity and relation types.

        This is the default implementation which uses the dictionaries natural_entity_types and natural_relation_types.
        """
        if self.natural_entity_types is not None:
            self.entity_types = {short: EntityType(
                short=short,
                natural=natural,
            ) for short, natural in self.natural_entity_types.items()}

        if self.natural_relation_types is not None:
            self.relation_types = {short: RelationType(
                short=short,
                natural=natural,
            ) for short, natural in self.natural_relation_types.items()}

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        examples = []
        file_path = os.path.join(self.data_dir(), f'data_{split}.json')

        with open(file_path, 'r') as f:
            data_lines = f.readlines()
            logger.info(f"Loaded {len(data_lines)} sentences for split {split} of {self.name}")

            for i, line in enumerate(data_lines):
                x = json.loads(line)
                entities = [
                    Entity(id=j, type=self.entity_types[y['type']], start=y['start'], end=y['end'])
                    for j, y in enumerate(x['entities'])
                ]

                relations = [
                    Relation(
                        type=self.relation_types[y['type']], head=entities[y['head']], tail=entities[y['tail']]
                    )
                    for y in x['relations']
                ]

                tokens = x['tokens']

                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=tokens,
                    entities=entities,
                    relations=relations,
                )

                examples.append(example)

        return examples

    def evaluate_example(self, example: InputExample, output_sentence: str, model=None, tokenizer=None) -> Counter:
        """
        Evaluate an output sentence on a single example of this dataset.
        """
        # extract entities and relations from output sentence
        res = self.output_format.run_inference(
            example,
            output_sentence,
            entity_types=self.entity_types,
            relation_types=self.relation_types,
        )
        predicted_entities, predicted_relations = res[:2]
        if len(res) == 6:
            # the output format provides information about errors
            wrong_reconstruction, label_error, entity_error, format_error = res[2:]
        else:
            # in case the output format does not provide information about errors
            wrong_reconstruction = label_error = entity_error = format_error = False

        predicted_entities_no_type = set([entity[1:] for entity in predicted_entities])

        # load ground truth entities
        gt_entities = set(entity.to_tuple() for entity in example.entities)
        gt_entities_no_type = set([entity[1:] for entity in gt_entities])

        # compute correct entities
        correct_entities = predicted_entities & gt_entities
        correct_entities_no_type = gt_entities_no_type & predicted_entities_no_type

        # load ground truth relations
        gt_relations = set(relation.to_tuple() for relation in example.relations)

        # compute correct relations
        correct_relations = predicted_relations & gt_relations

        assert len(correct_entities) <= len(predicted_entities)
        assert len(correct_entities) <= len(gt_entities)
        assert len(correct_entities_no_type) <= len(predicted_entities_no_type)
        assert len(correct_entities_no_type) <= len(gt_entities_no_type)

        assert len(correct_relations) <= len(predicted_relations)
        assert len(correct_relations) <= len(gt_relations)

        res = Counter({
            'num_sentences': 1,
            'wrong_reconstructions': 1 if wrong_reconstruction else 0,
            'label_error': 1 if label_error else 0,
            'entity_error': 1 if entity_error else 0,
            'format_error': 1 if format_error else 0,
            'gt_entities': len(gt_entities),
            'predicted_entities': len(predicted_entities),
            'correct_entities': len(correct_entities),
            'gt_entities_no_type': len(gt_entities_no_type),
            'predicted_entities_no_type': len(predicted_entities_no_type),
            'correct_entities_no_type': len(correct_entities_no_type),
            'gt_relations': len(gt_relations),
            'predicted_relations': len(predicted_relations),
            'correct_relations': len(correct_relations),
        })

        # add information about each entity/relation type so that we can compute the macro-F1 scores
        if self.entity_types is not None:
            for entity_type in self.entity_types.values():
                predicted = set(entity for entity in predicted_entities if entity[0] == entity_type.natural)
                gt = set(entity for entity in gt_entities if entity[0] == entity_type.natural)
                correct = predicted & gt
                res['predicted_entities', entity_type.natural] = len(predicted)
                res['gt_entities', entity_type.natural] = len(gt)
                res['correct_entities', entity_type.natural] = len(correct)

        if self.relation_types is not None:
            for relation_type in self.relation_types.values():
                predicted = set(relation for relation in predicted_relations if relation[0] == relation_type.natural)
                gt = set(relation for relation in gt_relations if relation[0] == relation_type.natural)
                correct = predicted & gt
                res['predicted_relations', relation_type.natural] = len(predicted)
                res['gt_relations', relation_type.natural] = len(gt)
                res['correct_relations', relation_type.natural] = len(correct)

        return res

    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset.
        """
        results = Counter()

        for example, output_sentence in self.generate_output_sentences(data_args, model, device, batch_size):
            new_result = self.evaluate_example(
                    example=example,
                    output_sentence=output_sentence,
                    model=model,
                    tokenizer=self.tokenizer,
                )
            results += new_result

        entity_precision, entity_recall, entity_f1 = get_precision_recall_f1(
            num_correct=results['correct_entities'],
            num_predicted=results['predicted_entities'],
            num_gt=results['gt_entities'],
        )

        entity_precision_no_type, entity_recall_no_type, entity_f1_no_type = get_precision_recall_f1(
            num_correct=results['correct_entities_no_type'],
            num_predicted=results['predicted_entities_no_type'],
            num_gt=results['gt_entities_no_type'],
        )

        entity_precision_by_type = []
        entity_recall_by_type = []
        entity_f1_by_type = []

        if macro:
            # compute also entity macro scores
            for entity_type in self.entity_types.values():
                precision, recall, f1 = get_precision_recall_f1(
                    num_correct=results['correct_entities', entity_type.natural],
                    num_predicted=results['predicted_entities', entity_type.natural],
                    num_gt=results['gt_entities', entity_type.natural],
                )
                entity_precision_by_type.append(precision)
                entity_recall_by_type.append(recall)
                entity_f1_by_type.append(f1)

        relation_precision, relation_recall, relation_f1 = get_precision_recall_f1(
            num_correct=results['correct_relations'],
            num_predicted=results['predicted_relations'],
            num_gt=results['gt_relations'],
        )

        res = {
            'wrong_reconstruction': results['wrong_reconstructions'] / results['num_sentences'],
            'label_error': results['label_error'] / results['num_sentences'],
            'entity_error': results['entity_error'] / results['num_sentences'],
            'format_error': results['format_error'] / results['num_sentences'],
            'entity_precision': entity_precision,
            'entity_recall': entity_recall,
            'entity_f1': entity_f1,
            'relation_precision': relation_precision,
            'relation_recall': relation_recall,
            'relation_f1': relation_f1,
            'entity_precision_no_type': entity_precision_no_type,
            'entity_recall_no_type': entity_recall_no_type,
            'entity_f1_no_type': entity_f1_no_type,
        }

        if macro:
            res.update({
                'entity_macro_precision': np.mean(np.array(entity_precision_by_type)),
                'entity_macro_recall': np.mean(np.array(entity_recall_by_type)),
                'entity_macro_f1': np.mean(np.array(entity_f1_by_type)),
            })

        return res


@register_dataset
class Conll04Dataset(JointERDataset):
    name = 'conll04'

    natural_entity_types = {
        'Loc': 'location',
        'Org': 'organization',
        'Peop': 'person',
        'Other': 'other',
    }

    natural_relation_types = {
        'Work_For': 'works for',
        'Kill': 'kills',
        'OrgBased_In': 'organization based in',
        'Live_In': 'lives in',
        'Located_In': 'located in'
    }

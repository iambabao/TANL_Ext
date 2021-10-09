import logging
import os
import json
import torch
import numpy as np
from collections import Counter

from src.data_processor.base_dataset import BaseDataset
from src.data_processor.input_example import EntityType, RelationType, Entity, Relation, InputExample
from src.utils.tanl_utils import get_precision_recall_f1
from src.utils.my_utils import to_natural, is_trigger

logger = logging.getLogger(__name__)
DATASETS = {}


def register_dataset(dataset_class):
    DATASETS[dataset_class.name] = dataset_class
    return dataset_class


def load_dataset(
        dataset_name, data_args, tokenizer, split,
        max_input_length, max_output_length,
        train_subset=1.0, seed=None, shuffle=True, is_eval=False,
):
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


class BasicBaseDataset(BaseDataset):
    entity_types = None
    relation_types = None
    natural_entity_types = None
    natural_relation_types = None

    default_input_format = 'plain'
    default_output_format = 'basic'

    def load_cached_data(self, cached_file):
        d = torch.load(cached_file)
        self.entity_types = d['entity_types']
        self.relation_types = d['relation_types']
        self.examples = d['examples']
        self.features = d['features']

    def save_data(self, cached_file):
        torch.save({
            'entity_types': self.entity_types,
            'relation_types': self.relation_types,
            'examples': self.examples,
            'features': self.features,
        }, cached_file)

    def load_schema(self):
        if self.natural_entity_types is not None:
            self.entity_types = {
                short: EntityType(short=short, natural=natural)
                for short, natural in self.natural_entity_types.items()
            }

        if self.natural_relation_types is not None:
            self.relation_types = {
                short: RelationType(short=short, natural=natural)
                for short, natural in self.natural_relation_types.items()
            }

    def load_data_single_split(self, split, seed=None):
        examples = []
        filename = os.path.join(self.data_dir(), 'data_{}.json'.format(split))

        with open(filename, 'r', encoding='utf-8') as fp:
            dataset = [json.loads(line) for line in fp]
            logger.info('Loading {} samples from split {} of {}'.format(len(dataset), split, self.name))

            for i, entry in enumerate(dataset):
                tokens = entry['tokens']

                entities = [
                    Entity(
                        id=j, type=self.entity_types[ent['type']], start=ent['start'], end=ent['end']
                    )
                    for j, ent in enumerate(entry['entities'])
                ]

                relations = [
                    Relation(
                        type=self.relation_types[rel['type']], head=entities[rel['head']], tail=entities[rel['tail']]
                    )
                    for rel in entry['relations']
                ]

                examples.append(InputExample(
                    id='{}-{}'.format(split, i),
                    tokens=tokens,
                    entities=entities,
                    relations=relations,
                ))

        return examples

    def evaluate_example(self, example, output_sentence):
        # extract entities and relations from output sentence
        results = self.output_format.run_inference(
            example,
            output_sentence,
            entity_types=self.entity_types,
            relation_types=self.relation_types,
        )
        predicted_entities, predicted_relations = results[:2]
        if len(results) == 6:
            # the output format provides information about errors
            wrong_reconstruction, label_error, entity_error, format_error = results[2:]
        else:
            # the output format does not provide information about errors
            wrong_reconstruction = label_error = entity_error = format_error = False

        predicted_entities_no_type = set([entity[1:] for entity in predicted_entities])

        # load ground truth entities
        if self.default_output_format in ['argument']:
            gt_entities = set(entity.to_tuple() for entity in example.entities if not is_trigger(entity))
        else:
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

        results = Counter({
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
                results['predicted_entities', entity_type.natural] = len(predicted)
                results['gt_entities', entity_type.natural] = len(gt)
                results['correct_entities', entity_type.natural] = len(correct)

        if self.relation_types is not None:
            for relation_type in self.relation_types.values():
                predicted = set(relation for relation in predicted_relations if relation[0] == relation_type.natural)
                gt = set(relation for relation in gt_relations if relation[0] == relation_type.natural)
                correct = predicted & gt
                results['predicted_relations', relation_type.natural] = len(predicted)
                results['gt_relations', relation_type.natural] = len(gt)
                results['correct_relations', relation_type.natural] = len(correct)

        return results

    def evaluate_generated_outputs(self, generated_outputs):
        results = Counter()
        for example, output in zip(self.examples, generated_outputs):
            cur_results = self.evaluate_example(example, output)
            results += cur_results

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
        relation_precision, relation_recall, relation_f1 = get_precision_recall_f1(
            num_correct=results['correct_relations'],
            num_predicted=results['predicted_relations'],
            num_gt=results['gt_relations'],
        )

        final_results = {
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

        return final_results

    def evaluate_dataset(self, data_args, model, device, batch_size, macro=False):
        results = Counter()

        for example, output_sentence in self.generate_output_sentences(data_args, model, device, batch_size):
            new_result = self.evaluate_example(
                    example=example,
                    output_sentence=output_sentence,
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

        metrics = {
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
            metrics.update({
                'entity_macro_precision': np.mean(np.array(entity_precision_by_type)),
                'entity_macro_recall': np.mean(np.array(entity_recall_by_type)),
                'entity_macro_f1': np.mean(np.array(entity_f1_by_type)),
            })

        return metrics


# NER
@register_dataset
class CoNLL03NERDataset(BasicBaseDataset):
    name = 'conll03_ner'

    natural_entity_types = {
        'LOC': 'location',
        'MISC': 'miscellaneous',
        'ORG': 'organization',
        'PER': 'person',
    }


# NER
@register_dataset
class OntoNotesNERDataset(BasicBaseDataset):
    name = 'ontonotes_ner'

    natural_entity_types = {
        'CARDINAL': 'cardinal',
        'DATE': 'date',
        'EVENT': 'event',
        'FAC': 'facility',
        'GPE': 'country city state',
        'LANGUAGE': 'language',
        'LAW': 'law',
        'LOC': 'location',
        'MONEY': 'monetary',
        'NORP': 'nationality religious political group',
        'ORDINAL': 'ordinal',
        'ORG': 'organization',
        'PERCENT': 'percent',
        'PERSON': 'person',
        'PRODUCT': 'product',
        'QUANTITY': 'quantity',
        'TIME': 'time',
        'WORK_OF_ART': 'work of art',
    }


# NER
@register_dataset
class GENIANERDataset(BasicBaseDataset):
    name = 'genia_ner'

    natural_entity_types = {
        'G#DNA': 'DNA',
        'G#RNA': 'RNA',
        'G#cell_line': 'cell line',
        'G#cell_type': 'cell type',
        'G#protein': 'protein',
    }


# NER
@register_dataset
class ACE2005NERDataset(BasicBaseDataset):
    name = 'ace2005_ner'

    natural_entity_types = {
        'PER': 'person',
        'LOC': 'location',
        'ORG': 'organization',
        'VEH': 'vehicle',
        'GPE': 'geographical entity',
        'WEA': 'weapon',
        'FAC': 'facility',
    }


# Joint entity relation extraction
@register_dataset
class CoNLL04REDataset(BasicBaseDataset):
    name = 'conll04_re'

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
        'Located_In': 'located in',
    }


# Joint entity relation extraction
@register_dataset
class NYTREDataset(BasicBaseDataset):
    name = 'nyt_re'

    natural_entity_types = {
        'PERSON': 'person',
        'LOCATION': 'location',
        'ORGANIZATION': 'organization',
    }

    natural_relation_types = {
        '/people/person/religion': 'religion',
        '/business/company/founders': 'founders',
        '/people/person/place_lived': 'place lived',
        '/location/country/administrative_divisions': 'administrative divisions',
        '/business/person/company': 'company',
        '/business/company/place_founded': 'place founded',
        '/business/company_shareholder/major_shareholder_of': 'major shareholder of',
        '/business/company/advisors': 'advisors',
        '/people/deceased_person/place_of_death': 'place of death',
        '/people/person/nationality': 'nationality',
        '/location/administrative_division/country': 'country',
        '/people/person/profession': 'profession',
        '/sports/sports_team_location/teams': 'teams',
        '/location/location/contains': 'contains',
        '/location/neighborhood/neighborhood_of': 'neighborhood of',
        '/location/country/capital': 'capital',
        '/business/company/major_shareholders': 'major shareholders',
        '/people/ethnicity/geographic_distribution': 'geographic distribution',
        '/people/person/ethnicity': 'ethnicity',
        '/sports/sports_team/location': 'location',
        '/people/ethnicity/people': 'people',
        '/people/person/place_of_birth': 'place of birth',
        '/business/company/industry': 'industry',
        '/people/person/children': 'children',
    }


# Joint entity relation extraction
@register_dataset
class ADEREDataset(BasicBaseDataset):
    name = 'ade_re'

    natural_entity_types = {
        'Adverse-Effect': 'disease',
        'Drug': 'drug',
    }

    natural_relation_types = {
        'Adverse-Effect': 'effect',
    }


# Joint entity relation extraction
@register_dataset
class ACE2005REDataset(BasicBaseDataset):
    name = 'ace2005_re'

    natural_entity_types = {
        'PER': 'person',
        'LOC': 'location',
        'ORG': 'organization',
        'VEH': 'vehicle',
        'GPE': 'geographical entity',
        'WEA': 'weapon',
        'FAC': 'facility',
    }

    natural_relation_types = {
        'PHYS': 'located in',
        'ART': 'artifact',
        'ORG-AFF': 'employer',
        'GEN-AFF': 'affiliation',
        'PER-SOC': 'social',
        'PART-WHOLE': 'part of',
    }


# Relation classification
@register_dataset
class TacRedRCDataset(BasicBaseDataset):
    name = 'tacred_rc'

    entity_types = {}
    relation_types = {}

    default_input_format = 'entity'

    def load_schema(self):
        filename = os.path.join(self.data_dir(), 'schema_entity.txt')
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp:
                short = line.strip()
                natural = to_natural(short)
                self.entity_types[short] = EntityType(short=short, natural=natural)

        filename = os.path.join(self.data_dir(), 'schema_relation.txt')
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp:
                short = line.strip()
                natural = to_natural(short)
                natural = natural.replace('stateorprovince', 'state or province')
                self.relation_types[short] = RelationType(short=short, natural=natural)


# Event trigger
@register_dataset
class ACE2005EventTriggerDataset(BasicBaseDataset):
    name = 'ace2005_trigger'

    entity_types = {}
    relation_types = {}

    def load_schema(self):
        filename = os.path.join(self.data_dir(), 'schema_entity.txt')
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp:
                short = line.strip()
                natural = to_natural(short)
                self.entity_types[short] = EntityType(short=short, natural=natural)

        filename = os.path.join(self.data_dir(), 'schema_relation.txt')
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp:
                short = line.strip()
                natural = to_natural(short)
                self.relation_types[short] = RelationType(short=short, natural=natural)


# Event argument
@register_dataset
class ACE2005EventArgumentDataset(BasicBaseDataset):
    name = 'ace2005_argument'

    entity_types = {}
    relation_types = {}

    default_input_format = 'trigger'
    default_output_format = 'argument'

    def load_schema(self):
        filename = os.path.join(self.data_dir(), 'schema_entity.txt')
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp:
                short = line.strip()
                natural = to_natural(short)
                self.entity_types[short] = EntityType(short=short, natural=natural)

        filename = os.path.join(self.data_dir(), 'schema_relation.txt')
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp:
                short = line.strip()
                natural = to_natural(short)
                self.relation_types[short] = RelationType(short=short, natural=natural)


@register_dataset
class ACE2005EventTriggerDataset(BasicBaseDataset):
    name = 'ace2005_event'

    entity_types = {}
    relation_types = {}

    def load_schema(self):
        filename = os.path.join(self.data_dir(), 'schema_entity.txt')
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp:
                short = line.strip()
                natural = to_natural(short)
                self.entity_types[short] = EntityType(short=short, natural=natural)

        filename = os.path.join(self.data_dir(), 'schema_relation.txt')
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp:
                short = line.strip()
                natural = to_natural(short)
                self.relation_types[short] = RelationType(short=short, natural=natural)


# SRL
@register_dataset
class CoNLL05SRLDataset(BasicBaseDataset):
    name = 'conll05_srl'

    entity_types = {}
    relation_types = {}

    default_input_format = 'trigger'
    default_output_format = 'argument'

    def load_schema(self):
        filename = os.path.join(self.data_dir(), 'schema_entity.txt')
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp:
                short = line.strip()
                natural = to_natural(short)
                self.entity_types[short] = EntityType(short=short, natural=natural)

        filename = os.path.join(self.data_dir(), 'schema_relation.txt')
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp:
                short = line.strip()
                natural = to_natural(short)
                self.relation_types[short] = RelationType(short=short, natural=natural)


# SRL
@register_dataset
class CoNLL12SRLDataset(BasicBaseDataset):
    name = 'conll12_srl'

    entity_types = {}
    relation_types = {}

    default_input_format = 'trigger'
    default_output_format = 'argument'

    def load_schema(self):
        filename = os.path.join(self.data_dir(), 'schema_entity.txt')
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp:
                short = line.strip()
                natural = to_natural(short)
                self.entity_types[short] = EntityType(short=short, natural=natural)

        filename = os.path.join(self.data_dir(), 'schema_relation.txt')
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp:
                short = line.strip()
                natural = to_natural(short)
                self.relation_types[short] = RelationType(short=short, natural=natural)

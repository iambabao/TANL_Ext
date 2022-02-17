# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2021/7/26
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/7/26
"""

import logging
import os
import re
from tqdm import tqdm

from src.utils.my_utils import init_logger, read_file, save_file, read_json, save_json, read_json_lines, save_json_lines

logger = logging.getLogger(__name__)


def process_ace2005_trigger():
    task_name = 'ace2005_trigger'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/processed/{}'.format(task_name)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    for entry in tqdm(read_json('data/raw/ace2005event/train.json')):
        tokens = entry['words']
        entities = []
        for event in entry['golden-event-mentions']:
            trigger = {
                'text': event['trigger']['text'],
                'start': event['trigger']['start'],
                'end': event['trigger']['end'],
                'type': event['event_type'],
            }
            entities.append(trigger)
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_train.json'))

    outputs = []
    for entry in tqdm(read_json('data/raw/ace2005event/dev.json')):
        tokens = entry['words']
        entities = []
        for event in entry['golden-event-mentions']:
            trigger = {
                'text': event['trigger']['text'],
                'start': event['trigger']['start'],
                'end': event['trigger']['end'],
                'type': event['event_type'],
            }
            entities.append(trigger)
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_valid.json'))

    outputs = []
    for entry in tqdm(read_json('data/raw/ace2005event/test.json')):
        tokens = entry['words']
        entities = []
        for event in entry['golden-event-mentions']:
            trigger = {
                'text': event['trigger']['text'],
                'start': event['trigger']['start'],
                'end': event['trigger']['end'],
                'type': event['event_type'],
            }
            entities.append(trigger)
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_test.json'))


def process_ace2005_argument():
    task_name = 'ace2005_argument'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/processed/{}'.format(task_name)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    for entry in tqdm(read_json('data/raw/ace2005event/train.json')):
        tokens = entry['words']
        id2entity = {}
        for entity in entry['golden-entity-mentions']:
            start, end, e_type = entity['start'], entity['end'], entity['entity-type']
            id2entity[(start, end, e_type)] = entity['head']
        for event in entry['golden-event-mentions']:
            entities = [{
                'text': event['trigger']['text'],
                'start': event['trigger']['start'],
                'end': event['trigger']['end'],
                'type': 'trigger:{}'.format(event['event_type']),
            }]
            for argument_info in event['arguments']:
                head = id2entity[(argument_info['start'], argument_info['end'], argument_info['entity-type'])]
                entities.append({
                    'text': head['text'],
                    'start': head['start'],
                    'end': head['end'],
                    'type': argument_info['role'],
                })
            outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_train.json'))

    outputs = []
    for entry in tqdm(read_json('data/raw/ace2005event/dev.json')):
        tokens = entry['words']
        id2entity = {}
        for entity in entry['golden-entity-mentions']:
            start, end, e_type = entity['start'], entity['end'], entity['entity-type']
            id2entity[(start, end, e_type)] = entity['head']
        for event in entry['golden-event-mentions']:
            entities = [{
                'text': event['trigger']['text'],
                'start': event['trigger']['start'],
                'end': event['trigger']['end'],
                'type': 'trigger:{}'.format(event['event_type']),
            }]
            for argument_info in event['arguments']:
                head = id2entity[(argument_info['start'], argument_info['end'], argument_info['entity-type'])]
                entities.append({
                    'text': head['text'],
                    'start': head['start'],
                    'end': head['end'],
                    'type': argument_info['role'],
                })
            outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_valid.json'))

    outputs = []
    for entry in tqdm(read_json('data/raw/ace2005event/test.json')):
        tokens = entry['words']
        id2entity = {}
        for entity in entry['golden-entity-mentions']:
            start, end, e_type = entity['start'], entity['end'], entity['entity-type']
            id2entity[(start, end, e_type)] = entity['head']
        for event in entry['golden-event-mentions']:
            entities = [{
                'text': event['trigger']['text'],
                'start': event['trigger']['start'],
                'end': event['trigger']['end'],
                'type': 'trigger:{}'.format(event['event_type']),
            }]
            for argument_info in event['arguments']:
                head = id2entity[(argument_info['start'], argument_info['end'], argument_info['entity-type'])]
                entities.append({
                    'text': head['text'],
                    'start': head['start'],
                    'end': head['end'],
                    'type': argument_info['role'],
                })
            outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_test.json'))


def process_ace2005_ner():
    task_name = 'ace2005_ner'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/processed/{}'.format(task_name)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    for entry in tqdm(read_json('data/raw/ace2005_ner/train.ner.json')):
        tokens = entry['context'].strip().split()
        entities = []
        for entity_type, block in entry['label'].items():
            for entity in block:
                start, end = map(int, entity.split(';'))
                entities.append({
                    'text': ' '.join(tokens[start:end + 1]),
                    'start': start,
                    'end': end + 1,
                    'type': entity_type,
                })
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_train.json'))

    outputs = []
    for entry in tqdm(read_json('data/raw/ace2005_ner/dev.ner.json')):
        tokens = entry['context'].strip().split()
        entities = []
        for entity_type, block in entry['label'].items():
            for entity in block:
                start, end = map(int, entity.split(';'))
                entities.append({
                    'text': ' '.join(tokens[start:end + 1]),
                    'start': start,
                    'end': end + 1,
                    'type': entity_type,
                })
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_valid.json'))

    outputs = []
    for entry in tqdm(read_json('data/raw/ace2005_ner/test.ner.json')):
        tokens = entry['context'].strip().split()
        entities = []
        for entity_type, block in entry['label'].items():
            for entity in block:
                start, end = map(int, entity.split(';'))
                entities.append({
                    'text': ' '.join(tokens[start:end + 1]),
                    'start': start,
                    'end': end + 1,
                    'type': entity_type,
                })
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_test.json'))


def process_ace2005_re():
    task_name = 'ace2005_re'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/processed/{}'.format(task_name)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    for entry in tqdm(read_json_lines('data/raw/ace2005_joint_er/train.json')):
        shift = 0
        for tokens, entity_block, relation_block in zip(entry['sentences'], entry['ner'], entry['relations']):
            entities = []
            for start, end, e_type in entity_block:
                start, end = start - shift, end - shift + 1
                entities.append({'text': ' '.join(tokens[start:end]), 'start': start, 'end': end, 'type': e_type,})
            relations = []
            for head_start, head_end, tail_start, tail_end, r_type in relation_block:
                head_start, head_end = head_start - shift, head_end - shift + 1
                tail_start, tail_end = tail_start - shift, tail_end - shift + 1
                head_entity, tail_entity = -1, -1
                for i, entity in enumerate(entities):
                    if entity['start'] == head_start and entity['end'] == head_end:
                        head_entity = i
                    if entity['start'] == tail_start and entity['end'] == tail_end:
                        tail_entity = i
                assert head_entity != -1 and tail_entity != -1
                relations.append({'head': head_entity, 'tail': tail_entity, 'type': r_type})
            shift += len(tokens)
            if len(entities) == 0: continue
            outputs.append({'tokens': tokens, 'entities': entities, 'relations': relations, 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_train.json'))

    outputs = []
    for entry in tqdm(read_json_lines('data/raw/ace2005_joint_er/dev.json')):
        shift = 0
        for tokens, entity_block, relation_block in zip(entry['sentences'], entry['ner'], entry['relations']):
            entities = []
            for start, end, e_type in entity_block:
                start, end = start - shift, end - shift + 1
                entities.append({'text': ' '.join(tokens[start:end]), 'start': start, 'end': end, 'type': e_type,})
            relations = []
            for head_start, head_end, tail_start, tail_end, r_type in relation_block:
                head_start, head_end = head_start - shift, head_end - shift + 1
                tail_start, tail_end = tail_start - shift, tail_end - shift + 1
                head_entity, tail_entity = -1, -1
                for i, entity in enumerate(entities):
                    if entity['start'] == head_start and entity['end'] == head_end:
                        head_entity = i
                    if entity['start'] == tail_start and entity['end'] == tail_end:
                        tail_entity = i
                assert head_entity != -1 and tail_entity != -1
                relations.append({'head': head_entity, 'tail': tail_entity, 'type': r_type})
            shift += len(tokens)
            if len(entities) == 0: continue
            outputs.append({'tokens': tokens, 'entities': entities, 'relations': relations, 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_valid.json'))

    outputs = []
    for entry in tqdm(read_json_lines('data/raw/ace2005_joint_er/test.json')):
        shift = 0
        for tokens, entity_block, relation_block in zip(entry['sentences'], entry['ner'], entry['relations']):
            entities = []
            for start, end, e_type in entity_block:
                start, end = start - shift, end - shift + 1
                entities.append({'text': ' '.join(tokens[start:end]), 'start': start, 'end': end, 'type': e_type,})
            relations = []
            for head_start, head_end, tail_start, tail_end, r_type in relation_block:
                head_start, head_end = head_start - shift, head_end - shift + 1
                tail_start, tail_end = tail_start - shift, tail_end - shift + 1
                head_entity, tail_entity = -1, -1
                for i, entity in enumerate(entities):
                    if entity['start'] == head_start and entity['end'] == head_end:
                        head_entity = i
                    if entity['start'] == tail_start and entity['end'] == tail_end:
                        tail_entity = i
                assert head_entity != -1 and tail_entity != -1
                relations.append({'head': head_entity, 'tail': tail_entity, 'type': r_type})
            shift += len(tokens)
            if len(entities) == 0: continue
            outputs.append({'tokens': tokens, 'entities': entities, 'relations': relations, 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_test.json'))


def process_ade_re():
    task_name = 'ade_re'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/processed/{}'.format(task_name)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    for entry in tqdm(read_json('data/raw/ade/ade_split_0_train.json')):
        tokens = entry['tokens']
        entities = []
        for entity in entry['entities']:
            entities.append({
                'text': ' '.join(tokens[entity['start']:entity['end']]),
                'start': entity['start'],
                'end': entity['end'],
                'type': entity['type'],
            })
        relations = []
        for relation in entry['relations']:
            relations.append({
                'head': relation['head'],
                'tail': relation['tail'],
                'type': relation['type'],
            })
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': relations, 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_train.json'))

    outputs = []
    for entry in tqdm(read_json('data/raw/ade/ade_split_0_test.json')):
        tokens = entry['tokens']
        entities = []
        for entity in entry['entities']:
            entities.append({
                'text': ' '.join(tokens[entity['start']:entity['end']]),
                'start': entity['start'],
                'end': entity['end'],
                'type': entity['type'],
            })
        relations = []
        for relation in entry['relations']:
            relations.append({
                'head': relation['head'],
                'tail': relation['tail'],
                'type': relation['type'],
            })
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': relations, 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_valid.json'))
    save_json_lines(outputs, os.path.join(output_dir, 'data_test.json'))


def process_conll03_ner():
    def _process(_tokens, _tags):
        _entities = []
        _start, _type = -1, None
        for _i, _tag in enumerate(_tags):
            if _tag.startswith('I'):
                _prefix, _suffix = _tag[0], _tag[2:]
                if _type is None:
                    _start, _type = _i, _suffix
                elif _suffix != _type:
                    _entities.append({
                        'text': ' '.join(_tokens[_start:_i]),
                        'start': _start,
                        'end': _i,
                        'type': _type,
                    })
                    _start, _type = _i, _suffix
            elif _type is not None:
                _entities.append({
                    'text': ' '.join(_tokens[_start:_i]),
                    'start': _start,
                    'end': _i,
                    'type': _type,
                })
                _start, _type = -1, None
        if _type is not None:
            _entities.append({
                'text': ' '.join(_tokens[_start:len(_tokens)]),
                'start': _start,
                'end': len(_tokens),
                'type': _type,
            })
            _start, _type = -1, None
        return _entities

    task_name = 'conll03_ner'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/processed/{}'.format(task_name)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    tokens, tags = [], []
    for line in tqdm(read_file('data/raw/conll03/train.txt')):
        line = line.strip()
        if line.startswith('-DOCSTART-'):
            continue
        if len(line) == 0:
            if len(tokens) == 0:
                continue
            entities = _process(tokens, tags)
            outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
            tokens, tags = [], []
        else:
            token, _, _, tag = line.strip().split()
            tokens.append(token)
            tags.append(tag)
    save_json_lines(outputs, os.path.join(output_dir, 'data_train.json'))

    outputs = []
    tokens, tags = [], []
    for line in tqdm(read_file('data/raw/conll03/dev.txt')):
        line = line.strip()
        if line.startswith('-DOCSTART-'):
            continue
        if len(line) == 0:
            if len(tokens) == 0:
                continue
            entities = _process(tokens, tags)
            outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
            tokens, tags = [], []
        else:
            token, _, _, tag = line.strip().split()
            tokens.append(token)
            tags.append(tag)
    save_json_lines(outputs, os.path.join(output_dir, 'data_valid.json'))

    outputs = []
    tokens, tags = [], []
    for line in tqdm(read_file('data/raw/conll03/test.txt')):
        line = line.strip()
        if line.startswith('-DOCSTART-'):
            continue
        if len(line) == 0:
            if len(tokens) == 0:
                continue
            entities = _process(tokens, tags)
            outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
            tokens, tags = [], []
        else:
            token, _, _, tag = line.strip().split()
            tokens.append(token)
            tags.append(tag)
    save_json_lines(outputs, os.path.join(output_dir, 'data_test.json'))


def process_conll04_re():
    task_name = 'conll04_re'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/processed/{}'.format(task_name)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    for entry in tqdm(read_json('data/raw/conll04/conll04_train.json')):
        tokens = entry['tokens']
        entities = []
        for entity in entry['entities']:
            entities.append({
                'text': ' '.join(tokens[entity['start']:entity['end']]),
                'start': entity['start'],
                'end': entity['end'],
                'type': entity['type'],
            })
        relations = []
        for relation in entry['relations']:
            relations.append({
                'head': relation['head'],
                'tail': relation['tail'],
                'type': relation['type'],
            })
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': relations, 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_train.json'))

    outputs = []
    for entry in tqdm(read_json('data/raw/conll04/conll04_dev.json')):
        tokens = entry['tokens']
        entities = []
        for entity in entry['entities']:
            entities.append({
                'text': ' '.join(tokens[entity['start']:entity['end']]),
                'start': entity['start'],
                'end': entity['end'],
                'type': entity['type'],
            })
        relations = []
        for relation in entry['relations']:
            relations.append({
                'head': relation['head'],
                'tail': relation['tail'],
                'type': relation['type'],
            })
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': relations, 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_valid.json'))

    outputs = []
    for entry in tqdm(read_json('data/raw/conll04/conll04_test.json')):
        tokens = entry['tokens']
        entities = []
        for entity in entry['entities']:
            entities.append({
                'text': ' '.join(tokens[entity['start']:entity['end']]),
                'start': entity['start'],
                'end': entity['end'],
                'type': entity['type'],
            })
        relations = []
        for relation in entry['relations']:
            relations.append({
                'head': relation['head'],
                'tail': relation['tail'],
                'type': relation['type'],
            })
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': relations, 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_test.json'))


def process_conll05_srl():
    def _process(_tokens, _tags):
        _entities = []
        _start, _type = -1, None
        for _i, _tag in enumerate(_tags):
            if _tag.startswith('B'):
                _prefix, _suffix = _tag[0], _tag[2:]
                if _type is None:
                    _start, _type = _i, _suffix
                else:
                    _entities.append({
                        'text': ' '.join(_tokens[_start:_i]),
                        'start': _start,
                        'end': _i,
                        'type': _type,
                    })
                    _start, _type = _i, _suffix
            elif _tag.startswith('O') and _type is not None:
                _entities.append({
                    'text': ' '.join(_tokens[_start:_i]),
                    'start': _start,
                    'end': _i,
                    'type': _type,
                })
                _start, _type = -1, None
        if _type is not None:
            _entities.append({
                'text': ' '.join(_tokens[_start:len(_tokens)]),
                'start': _start,
                'end': len(_tokens),
                'type': _type,
            })
            _start, _type = -1, None
        return _entities

    task_name = 'conll05_srl'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/processed/{}'.format(task_name)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    tokens, bag_of_tags = [], []
    for line in tqdm(read_file('data/raw/CoNLL2005-SRL/train-set.gz.parse.sdeps.combined.bio')):
        line = line.strip()
        if len(line) == 0:
            if len(tokens) != 0:
                for i in range(len(bag_of_tags[0])):
                    tags = [bag[i] for bag in bag_of_tags]
                    entities = _process(tokens, tags)
                    outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
            tokens, bag_of_tags = [], []
        else:
            columns = line.strip().split()
            tokens.append(columns[3])
            bag_of_tags.append(columns[14:])
    save_json_lines(outputs, os.path.join(output_dir, 'data_train.json'))

    outputs = []
    tokens, bag_of_tags = [], []
    for line in tqdm(read_file('data/raw/CoNLL2005-SRL/dev-set.gz.parse.sdeps.combined.bio')):
        line = line.strip()
        if len(line) == 0:
            if len(tokens) != 0:
                for i in range(len(bag_of_tags[0])):
                    tags = [bag[i] for bag in bag_of_tags]
                    entities = _process(tokens, tags)
                    outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
            tokens, bag_of_tags = [], []
        else:
            columns = line.strip().split()
            tokens.append(columns[3])
            bag_of_tags.append(columns[14:])
    save_json_lines(outputs, os.path.join(output_dir, 'data_valid.json'))

    outputs = []
    tokens, bag_of_tags = [], []
    for line in tqdm(read_file('data/raw/CoNLL2005-SRL/test.wsj.gz.parse.sdeps.combined.bio')):
        line = line.strip()
        if len(line) == 0:
            if len(tokens) != 0:
                for i in range(len(bag_of_tags[0])):
                    tags = [bag[i] for bag in bag_of_tags]
                    entities = _process(tokens, tags)
                    outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
            tokens, bag_of_tags = [], []
        else:
            columns = line.strip().split()
            tokens.append(columns[3])
            bag_of_tags.append(columns[14:])
    save_json_lines(outputs, os.path.join(output_dir, 'data_test.json'))


def process_conll12_srl():
    def _process(_tokens, _tags):
        _entities = []
        _queue, _queue_head = [], 0
        for _i, _predicate in enumerate(_tags):
            _match = re.search(r'^\((.+)\*', _predicate)
            if _match:
                for _entity_type in _match.group(1).split('('):
                    _queue.append((_entity_type, _i))
            for _ in _predicate[_predicate.index('*') + 1:]:
                _entities.append({
                    'text': ' '.join(_tokens[_queue[_queue_head][1]:_i + 1]),
                    'start': _queue[_queue_head][1],
                    'end': _i + 1,
                    'type': _queue[_queue_head][0],
                })
                _queue_head += 1
        return _entities

    task_name = 'conll12_srl'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/processed/{}'.format(task_name)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    tokens, all_tags, num_triggers = [], [], -1
    for line in tqdm(list(read_file('data/raw/Conll2012-SRL/train.english.v4_gold_conll'))):
        line = line.strip()
        if line.startswith('#begin document') or line.startswith('#end document'): continue
        if len(line) == 0:
            if len(tokens) == 0: continue
            for tags in all_tags:
                entities = _process(tokens, tags)
                outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': 'conll12_srl'})
            tokens, all_tags, num_triggers = [], [], -1
        else:
            columns = line.split()
            if num_triggers == -1:
                num_triggers = len(columns) - 12
                all_tags = [[] for _ in range(num_triggers)]
            tokens.append(columns[3])
            for i, predicate in enumerate(columns[11:-1]):
                all_tags[i].append(predicate)
    save_json_lines(outputs, os.path.join(output_dir, 'data_train.json'))

    outputs = []
    tokens, all_tags, num_triggers = [], [], -1
    for line in tqdm(list(read_file('data/raw/Conll2012-SRL/dev.english.v4_gold_conll'))):
        line = line.strip()
        if line.startswith('#begin document') or line.startswith('#end document'): continue
        if len(line) == 0:
            if len(tokens) == 0: continue
            for tags in all_tags:
                entities = _process(tokens, tags)
                outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': 'conll12_srl'})
            tokens, all_tags, num_triggers = [], [], -1
        else:
            columns = line.split()
            if num_triggers == -1:
                num_triggers = len(columns) - 12
                all_tags = [[] for _ in range(num_triggers)]
            tokens.append(columns[3])
            for i, predicate in enumerate(columns[11:-1]):
                all_tags[i].append(predicate)
    save_json_lines(outputs, os.path.join(output_dir, 'data_valid.json'))

    outputs = []
    tokens, all_tags, num_triggers = [], [], -1
    for line in tqdm(list(read_file('data/raw/Conll2012-SRL/test.english.v4_gold_conll'))):
        line = line.strip()
        if line.startswith('#begin document') or line.startswith('#end document'): continue
        if len(line) == 0:
            if len(tokens) == 0: continue
            for tags in all_tags:
                entities = _process(tokens, tags)
                outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': 'conll12_srl'})
            tokens, all_tags, num_triggers = [], [], -1
        else:
            columns = line.split()
            if num_triggers == -1:
                num_triggers = len(columns) - 12
                all_tags = [[] for _ in range(num_triggers)]
            tokens.append(columns[3])
            for i, predicate in enumerate(columns[11:-1]):
                all_tags[i].append(predicate)
    save_json_lines(outputs, os.path.join(output_dir, 'data_test.json'))


def process_genia_ner():
    task_name = 'genia_ner'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/processed/{}'.format(task_name)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    lines = list(read_file('data/raw/genia/train.data'))
    for i in tqdm(range(len(lines) // 4)):
        tokens = lines[i * 4].strip().split()
        annotations = lines[i * 4 + 2].strip()
        entities = []
        for info in annotations.split('|'):
            if len(info) != 0:
                position, entity_type = info.split()
                start, end = map(int, position.split(','))
                entities.append({
                    'text': tokens[start:end],
                    'start': start,
                    'end': end,
                    'type': entity_type,
                })
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_train.json'))

    outputs = []
    lines = list(read_file('data/raw/genia/dev.data'))
    for i in tqdm(range(len(lines) // 4)):
        tokens = lines[i * 4].strip().split()
        annotations = lines[i * 4 + 2].strip()
        entities = []
        for info in annotations.split('|'):
            if len(info) != 0:
                position, entity_type = info.split()
                start, end = map(int, position.split(','))
                entities.append({
                    'text': tokens[start:end],
                    'start': start,
                    'end': end,
                    'type': entity_type,
                })
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_valid.json'))

    outputs = []
    lines = list(read_file('data/raw/genia/test.data'))
    for i in tqdm(range(len(lines) // 4)):
        tokens = lines[i * 4].strip().split()
        annotations = lines[i * 4 + 2].strip()
        entities = []
        for info in annotations.split('|'):
            if len(info) != 0:
                position, entity_type = info.split()
                start, end = map(int, position.split(','))
                entities.append({
                    'text': tokens[start:end],
                    'start': start,
                    'end': end,
                    'type': entity_type,
                })
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_test.json'))


def process_nyt_re():
    task_name = 'nyt_re'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/processed/{}'.format(task_name)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    for entry in tqdm(read_json('data/raw/nyt/train.json')):
        tokens = entry['tokens']
        entities = []
        relations = []
        for head_start, head_end, head_type, relation_type, tail_start, tail_end, tail_type in entry['spo_details']:
            head = {'text': ' '.join(tokens[head_start:head_end]), 'start': head_start, 'end': head_end, 'type': head_type}
            tail = {'text': ' '.join(tokens[tail_start:tail_end]), 'start': tail_start, 'end': tail_end, 'type': tail_type}

            if head in entities:
                head_index = entities.index(head)
            else:
                head_index = len(entities)
                entities.append(head)
            if tail in entities:
                tail_index = entities.index(tail)
            else:
                tail_index = len(entities)
                entities.append(tail)
            relations.append({'head': head_index, 'tail': tail_index, 'type': relation_type})
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': relations, 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_train.json'))

    outputs = []
    for entry in tqdm(read_json('data/raw/nyt/dev.json')):
        tokens = entry['tokens']
        entities = []
        relations = []
        for head_start, head_end, head_type, relation_type, tail_start, tail_end, tail_type in entry['spo_details']:
            head = {'text': ' '.join(tokens[head_start:head_end]), 'start': head_start, 'end': head_end, 'type': head_type}
            tail = {'text': ' '.join(tokens[tail_start:tail_end]), 'start': tail_start, 'end': tail_end, 'type': tail_type}

            if head in entities:
                head_index = entities.index(head)
            else:
                head_index = len(entities)
                entities.append(head)
            if tail in entities:
                tail_index = entities.index(tail)
            else:
                tail_index = len(entities)
                entities.append(tail)
            relations.append({'head': head_index, 'tail': tail_index, 'type': relation_type})
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': relations, 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_valid.json'))

    outputs = []
    for entry in tqdm(read_json('data/raw/nyt/test.json')):
        tokens = entry['tokens']
        entities = []
        relations = []
        for head_start, head_end, head_type, relation_type, tail_start, tail_end, tail_type in entry['spo_details']:
            head = {'text': ' '.join(tokens[head_start:head_end]), 'start': head_start, 'end': head_end, 'type': head_type}
            tail = {'text': ' '.join(tokens[tail_start:tail_end]), 'start': tail_start, 'end': tail_end, 'type': tail_type}

            if head in entities:
                head_index = entities.index(head)
            else:
                head_index = len(entities)
                entities.append(head)
            if tail in entities:
                tail_index = entities.index(tail)
            else:
                tail_index = len(entities)
                entities.append(tail)
            relations.append({'head': head_index, 'tail': tail_index, 'type': relation_type})
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': relations, 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_test.json'))


def process_ontonotes_ner():
    task_name = 'ontonotes_ner'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/processed/{}'.format(task_name)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    for entry in tqdm(read_json_lines('data/raw/ontonotes/train.json')):
        shift = 0
        for tokens, ner in zip(entry['sentences'], entry['ner']):
            entities = []
            for start, end, entity_type in ner:
                start, end = start - shift, end - shift + 1
                entities.append({'text': ' '.join(tokens[start:end]), 'start': start, 'end': end, 'type': entity_type})
            shift += len(tokens)
            outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_train.json'))

    outputs = []
    for entry in tqdm(read_json_lines('data/raw/ontonotes/dev.json')):
        shift = 0
        for tokens, ner in zip(entry['sentences'], entry['ner']):
            entities = []
            for start, end, entity_type in ner:
                start, end = start - shift, end - shift + 1
                entities.append({'text': ' '.join(tokens[start:end]), 'start': start, 'end': end, 'type': entity_type})
            shift += len(tokens)
            outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_valid.json'))

    outputs = []
    for entry in tqdm(read_json_lines('data/raw/ontonotes/test.json')):
        shift = 0
        for tokens, ner in zip(entry['sentences'], entry['ner']):
            entities = []
            for start, end, entity_type in ner:
                start, end = start - shift, end - shift + 1
                entities.append({'text': ' '.join(tokens[start:end]), 'start': start, 'end': end, 'type': entity_type})
            shift += len(tokens)
            outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_test.json'))


def process_tacred_rc():
    task_name = 'tacred_rc'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/processed/{}'.format(task_name)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    for entry in tqdm(read_json('data/raw/tacred/json/train.json')):
        tokens = entry['token']
        head_start, head_end, head_type = entry['subj_start'], entry['subj_end'] + 1, entry['subj_type']
        tail_start, tail_end, tail_type = entry['obj_start'], entry['obj_end'] + 1, entry['obj_type']
        entities = [
            {'text': ' '.join(tokens[head_start:head_end]), 'start': head_start, 'end': head_end, 'type': head_type},
            {'text': ' '.join(tokens[tail_start:tail_end]), 'start': tail_start, 'end': tail_end, 'type': tail_type},
        ]
        relations = [{'head': 0, 'tail': 1, 'type': entry['relation']}]
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': relations, 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_train.json'))

    outputs = []
    for entry in tqdm(read_json('data/raw/tacred/json/dev.json')):
        tokens = entry['token']
        head_start, head_end, head_type = entry['subj_start'], entry['subj_end'] + 1, entry['subj_type']
        tail_start, tail_end, tail_type = entry['obj_start'], entry['obj_end'] + 1, entry['obj_type']
        entities = [
            {'text': ' '.join(tokens[head_start:head_end]), 'start': head_start, 'end': head_end, 'type': head_type},
            {'text': ' '.join(tokens[tail_start:tail_end]), 'start': tail_start, 'end': tail_end, 'type': tail_type},
        ]
        relations = [{'head': 0, 'tail': 1, 'type': entry['relation']}]
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': relations, 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_valid.json'))

    outputs = []
    for entry in tqdm(read_json('data/raw/tacred/json/test.json')):
        tokens = entry['token']
        head_start, head_end, head_type = entry['subj_start'], entry['subj_end'] + 1, entry['subj_type']
        tail_start, tail_end, tail_type = entry['obj_start'], entry['obj_end'] + 1, entry['obj_type']
        entities = [
            {'text': ' '.join(tokens[head_start:head_end]), 'start': head_start, 'end': head_end, 'type': head_type},
            {'text': ' '.join(tokens[tail_start:tail_end]), 'start': tail_start, 'end': tail_end, 'type': tail_type},
        ]
        relations = [{'head': 0, 'tail': 1, 'type': entry['relation']}]
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': relations, 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_test.json'))


def generate_schema(root_dir):
    for task_name in os.listdir(root_dir):
        task_dir = os.path.join(root_dir, task_name)
        if not os.path.isdir(task_dir): continue
        entity_schema = set()
        relation_schema = set()
        for role in ['train', 'valid', 'test']:
            for entry in read_json_lines(os.path.join(task_dir, 'data_{}.json'.format(role))):
                entity_schema.update(_['type'] for _ in entry['entities'])
                relation_schema.update(_['type'] for _ in entry['relations'])
        save_file(entity_schema, os.path.join(task_dir, 'schema_entity.txt'))
        save_file(relation_schema, os.path.join(task_dir, 'schema_relation.txt'))


def main():
    init_logger(logging.INFO)

    # process_ace2005_trigger()
    # process_ace2005_argument()
    # process_ace2005_ner()
    process_ace2005_re()
    # process_ade_re()
    # process_conll03_ner()
    # process_conll04_re()
    # process_conll05_srl()
    # process_conll12_srl()
    # process_genia_ner()
    # process_nyt_re()
    process_ontonotes_ner()
    # process_tacred_rc()

    generate_schema('data/processed')


if __name__ == '__main__':
    main()

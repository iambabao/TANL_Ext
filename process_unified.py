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
from tqdm import tqdm

from src.utils.my_utils import init_logger, read_file, save_file, read_json, save_json, read_json_lines, save_json_lines

logger = logging.getLogger(__name__)


def process_ace2005_joint_er():
    task_name = 'ace2005_joint_er'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/formatted_unified/{}'.format(task_name)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    for entry in tqdm(read_json_lines('data/raw/ace2005_joint_er/train.json')):
        tokens = []
        for sentence in entry['sentences']:
            tokens.extend(sentence)
        entities = []
        for block in entry['ner']:
            for entity in block:
                entities.append({
                    'text': ' '.join(tokens[entity[0]:entity[1] + 1]),
                    'start': entity[0],
                    'end': entity[1] + 1,
                    'type': entity[-1],
                })
        relations = []
        for block in entry['relations']:
            for relation in block:
                head_start, head_end = relation[0], relation[1] + 1
                tail_start, tail_end = relation[2], relation[3] + 1
                head_entity, tail_entity = -1, -1
                for i, entity in enumerate(entities):
                    if entity['start'] == head_start and entity['end'] == head_end:
                        head_entity = i
                    if entity['start'] == tail_start and entity['end'] == tail_end:
                        tail_entity = i
                relations.append({
                    'head': head_entity,
                    'tail': tail_entity,
                    'type': relation[-1],
                })
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': relations, 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_train.json'))

    outputs = []
    for entry in tqdm(read_json_lines('data/raw/ace2005_joint_er/dev.json')):
        tokens = []
        for sentence in entry['sentences']:
            tokens.extend(sentence)
        entities = []
        for block in entry['ner']:
            for entity in block:
                entities.append({
                    'text': ' '.join(tokens[entity[0]:entity[1] + 1]),
                    'start': entity[0],
                    'end': entity[1] + 1,
                    'type': entity[-1],
                })
        relations = []
        for block in entry['relations']:
            for relation in block:
                head_start, head_end = relation[0], relation[1] + 1
                tail_start, tail_end = relation[2], relation[3] + 1
                head_entity, tail_entity = -1, -1
                for i, entity in enumerate(entities):
                    if entity['start'] == head_start and entity['end'] == head_end:
                        head_entity = i
                    if entity['start'] == tail_start and entity['end'] == tail_end:
                        tail_entity = i
                relations.append({
                    'head': head_entity,
                    'tail': tail_entity,
                    'type': relation[-1],
                })
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': relations, 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_valid.json'))

    outputs = []
    for entry in tqdm(read_json_lines('data/raw/ace2005_joint_er/test.json')):
        tokens = []
        for sentence in entry['sentences']:
            tokens.extend(sentence)
        entities = []
        for block in entry['ner']:
            for entity in block:
                entities.append({
                    'text': ' '.join(tokens[entity[0]:entity[1] + 1]),
                    'start': entity[0],
                    'end': entity[1] + 1,
                    'type': entity[-1],
                })
        relations = []
        for block in entry['relations']:
            for relation in block:
                head_start, head_end = relation[0], relation[1] + 1
                tail_start, tail_end = relation[2], relation[3] + 1
                head_entity, tail_entity = -1, -1
                for i, entity in enumerate(entities):
                    if entity['start'] == head_start and entity['end'] == head_end:
                        head_entity = i
                    if entity['start'] == tail_start and entity['end'] == tail_end:
                        tail_entity = i
                relations.append({
                    'head': head_entity,
                    'tail': tail_entity,
                    'type': relation[-1],
                })
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': relations, 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_test.json'))


def process_ace2005_ner():
    task_name = 'ace2005_ner'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/formatted_unified/{}'.format(task_name)
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


def process_ace2005_event():
    task_name = 'ace2005_event'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/formatted_unified/{}'.format(task_name)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    for entry in tqdm(read_json('data/raw/ace2005event/train.json')):
        tokens = entry['words']
        entities = []
        for ent in entry['golden-entity-mentions']:
            entities.append({
                'text': ent['text'],
                'start': ent['start'],
                'end': ent['end'],
                'type': ent['entity-type'],
            })
        relations = []
        for event in entry['golden-event-mentions']:
            trigger_index = len(entities)
            entities.append({
                'text': event['trigger']['text'],
                'start': event['trigger']['start'],
                'end': event['trigger']['end'],
                'type': 'trigger',
            })
            for arg in event['arguments']:
                arg_index = entities.index({
                    'text': arg['text'],
                    'start': arg['start'],
                    'end': arg['end'],
                    'type': arg['entity-type'],
                })
                relations.append({'head': trigger_index, 'tail': arg_index, 'type': arg['role']})
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': relations, 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_train.json'))

    outputs = []
    for entry in tqdm(read_json('data/raw/ace2005event/dev.json')):
        tokens = entry['words']
        entities = []
        for ent in entry['golden-entity-mentions']:
            entities.append({
                'text': ent['text'],
                'start': ent['start'],
                'end': ent['end'],
                'type': ent['entity-type'],
            })
        relations = []
        for event in entry['golden-event-mentions']:
            trigger_index = len(entities)
            entities.append({
                'text': event['trigger']['text'],
                'start': event['trigger']['start'],
                'end': event['trigger']['end'],
                'type': 'trigger',
            })
            for arg in event['arguments']:
                arg_index = entities.index({
                    'text': arg['text'],
                    'start': arg['start'],
                    'end': arg['end'],
                    'type': arg['entity-type'],
                })
                relations.append({'head': trigger_index, 'tail': arg_index, 'type': arg['role']})
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': relations, 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_valid.json'))

    outputs = []
    for entry in tqdm(read_json('data/raw/ace2005event/test.json')):
        tokens = entry['words']
        entities = []
        for ent in entry['golden-entity-mentions']:
            entities.append({
                'text': ent['text'],
                'start': ent['start'],
                'end': ent['end'],
                'type': ent['entity-type'],
            })
        relations = []
        for event in entry['golden-event-mentions']:
            trigger_index = len(entities)
            entities.append({
                'text': event['trigger']['text'],
                'start': event['trigger']['start'],
                'end': event['trigger']['end'],
                'type': 'trigger',
            })
            for arg in event['arguments']:
                arg_index = entities.index({
                    'text': arg['text'],
                    'start': arg['start'],
                    'end': arg['end'],
                    'type': arg['entity-type'],
                })
                relations.append({'head': trigger_index, 'tail': arg_index, 'type': arg['role']})
        outputs.append({'tokens': tokens, 'entities': entities, 'relations': relations, 'task_name': task_name})
    save_json_lines(outputs, os.path.join(output_dir, 'data_test.json'))


def process_ade():
    task_name = 'ade'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/formatted_unified/{}'.format(task_name)
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


def process_conll03():
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

    task_name = 'conll03'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/formatted_unified/{}'.format(task_name)
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


def process_conll04():
    task_name = 'conll04'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/formatted_unified/{}'.format(task_name)
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
    output_dir = 'data/formatted_unified/{}'.format(task_name)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    tokens, bag_of_tags = [], []
    for line in tqdm(read_file('data/raw/CoNLL2005-SRL/train-set.gz.parse.sdeps.combined.bio')):
        line = line.strip()
        if len(line) == 0:
            if len(tokens) != 0:
                all_entities, all_relations = [], []
                for i in range(len(bag_of_tags[0])):
                    tags = [bag[i] for bag in bag_of_tags]
                    entities = _process(tokens, tags)
                    a2b = {}
                    for a, ent in enumerate(entities):
                        ent = {
                            'text': ent['text'],
                            'start': ent['start'],
                            'end': ent['end'],
                            'type': 'verb' if ent['type'] == 'V' else 'argument',
                        }
                        if ent in all_entities:
                            a2b[a] = all_entities.index(ent)
                        else:
                            a2b[a] = len(all_entities)
                            all_entities.append(ent)

                    head = -1
                    for a, ent in enumerate(entities):
                        if ent['type'] == 'V':
                            head = a2b[a]
                            break
                    for a, ent in enumerate(entities):
                        if ent['type'] != 'V':
                            tail = a2b[a]
                            all_relations.append({
                                'head': head,
                                'tail': tail,
                                'type': ent['type'],
                            })
                outputs.append({'tokens': tokens, 'entities': all_entities, 'relations': all_relations, 'task_name': task_name})
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
                all_entities, all_relations = [], []
                for i in range(len(bag_of_tags[0])):
                    tags = [bag[i] for bag in bag_of_tags]
                    entities = _process(tokens, tags)
                    a2b = {}
                    for a, ent in enumerate(entities):
                        ent = {
                            'text': ent['text'],
                            'start': ent['start'],
                            'end': ent['end'],
                            'type': 'verb' if ent['type'] == 'V' else 'argument',
                        }
                        if ent in all_entities:
                            a2b[a] = all_entities.index(ent)
                        else:
                            a2b[a] = len(all_entities)
                            all_entities.append(ent)

                    head = -1
                    for a, ent in enumerate(entities):
                        if ent['type'] == 'V':
                            head = a2b[a]
                            break
                    for a, ent in enumerate(entities):
                        if ent['type'] != 'V':
                            tail = a2b[a]
                            all_relations.append({
                                'head': head,
                                'tail': tail,
                                'type': ent['type'],
                            })
                outputs.append({'tokens': tokens, 'entities': all_entities, 'relations': all_relations, 'task_name': task_name})
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
                all_entities, all_relations = [], []
                for i in range(len(bag_of_tags[0])):
                    tags = [bag[i] for bag in bag_of_tags]
                    entities = _process(tokens, tags)
                    a2b = {}
                    for a, ent in enumerate(entities):
                        ent = {
                            'text': ent['text'],
                            'start': ent['start'],
                            'end': ent['end'],
                            'type': 'verb' if ent['type'] == 'V' else 'argument',
                        }
                        if ent in all_entities:
                            a2b[a] = all_entities.index(ent)
                        else:
                            a2b[a] = len(all_entities)
                            all_entities.append(ent)

                    head = -1
                    for a, ent in enumerate(entities):
                        if ent['type'] == 'V':
                            head = a2b[a]
                            break
                    for a, ent in enumerate(entities):
                        if ent['type'] != 'V':
                            tail = a2b[a]
                            all_relations.append({
                                'head': head,
                                'tail': tail,
                                'type': ent['type'],
                            })
                outputs.append({'tokens': tokens, 'entities': all_entities, 'relations': all_relations, 'task_name': task_name})
            tokens, bag_of_tags = [], []
        else:
            columns = line.strip().split()
            tokens.append(columns[3])
            bag_of_tags.append(columns[14:])
    save_json_lines(outputs, os.path.join(output_dir, 'data_test.json'))


def process_conll12_srl():
    pass


def process_genia():
    task_name = 'genia'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/formatted_unified/{}'.format(task_name)
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


def process_nyt():
    task_name = 'nyt'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/formatted_unified/{}'.format(task_name)
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


def process_ontonotes():
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

    task_name = 'ontonotes'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/formatted_unified/{}'.format(task_name)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    tokens, tags = [], []
    for line in tqdm(read_file('data/raw/ontonotes/train.txt')):
        line = line.strip()
        if len(line) == 0:
            if len(tokens) == 0:
                continue
            entities = _process(tokens, tags)
            outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
            tokens, tags = [], []
        else:
            token, _, _, tag = line.split()
            tokens.append(token)
            tags.append(tag)
    save_json_lines(outputs, os.path.join(output_dir, 'data_train.json'))

    outputs = []
    tokens, tags = [], []
    for line in tqdm(read_file('data/raw/ontonotes/dev.txt')):
        line = line.strip()
        if len(line) == 0:
            if len(tokens) == 0:
                continue
            entities = _process(tokens, tags)
            outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
            tokens, tags = [], []
        else:
            token, _, _, tag = line.split()
            tokens.append(token)
            tags.append(tag)
    save_json_lines(outputs, os.path.join(output_dir, 'data_valid.json'))

    outputs = []
    tokens, tags = [], []
    for line in tqdm(read_file('data/raw/ontonotes/test.txt')):
        line = line.strip()
        if len(line) == 0:
            if len(tokens) == 0:
                continue
            entities = _process(tokens, tags)
            outputs.append({'tokens': tokens, 'entities': entities, 'relations': [], 'task_name': task_name})
            tokens, tags = [], []
        else:
            token, _, _, tag = line.split()
            tokens.append(token)
            tags.append(tag)
    save_json_lines(outputs, os.path.join(output_dir, 'data_test.json'))


def process_tacred():
    task_name = 'tacred'
    logger.info('Processing: {}'.format(task_name))
    output_dir = 'data/formatted_unified/{}'.format(task_name)
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


def main():
    init_logger(logging.INFO)

    process_ace2005_joint_er()
    process_ace2005_ner()
    process_ace2005_event()
    process_ade()
    process_conll03()
    process_conll04()
    process_conll05_srl()
    process_conll12_srl()
    process_genia()
    process_nyt()
    process_ontonotes()
    process_tacred()


if __name__ == '__main__':
    main()

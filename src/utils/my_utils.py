# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/1/1
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/5/14
"""

import json
import random
import logging
from argparse import Namespace
from nltk.translate.bleu_score import corpus_bleu

logger = logging.getLogger(__name__)


def init_logger(level, filename=None, mode='w', encoding='utf-8'):
    logging_config = {
        'format': '%(asctime)s - %(levelname)s - %(name)s:\t%(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S',
        'level': level,
        'handlers': [logging.StreamHandler()]
    }
    if filename:
        logging_config['handlers'].append(logging.FileHandler(filename, mode, encoding))
    logging.basicConfig(**logging_config)


def log_title(title, sep='='):
    return sep * 50 + '  {}  '.format(title) + sep * 50


def read_file(filename, mode='r', encoding='utf-8', skip=0):
    with open(filename, mode, encoding=encoding) as fin:
        for line in fin:
            if skip > 0:
                skip -= 1
                continue
            yield line


def save_file(data, filename, mode='w', encoding='utf-8', skip=0):
    with open(filename, mode, encoding=encoding) as fout:
        for line in data:
            if skip > 0:
                skip -= 1
                continue
            print(line, file=fout)


def read_json(filename, mode='r', encoding='utf-8'):
    with open(filename, mode, encoding=encoding) as fin:
        return json.load(fin)


def save_json(data, filename, mode='w', encoding='utf-8'):
    with open(filename, mode, encoding=encoding) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4)


def read_json_lines(filename, mode='r', encoding='utf-8', skip=0):
    with open(filename, mode, encoding=encoding) as fin:
        for line in fin:
            if skip > 0:
                skip -= 1
                continue
            yield json.loads(line)


def save_json_lines(data, filename, mode='w', encoding='utf-8', skip=0):
    with open(filename, mode, encoding=encoding) as fout:
        for line in data:
            if skip > 0:
                skip -= 1
                continue
            print(json.dumps(line, ensure_ascii=False), file=fout)


def read_txt_dict(filename, sep=None, mode='r', encoding='utf-8', skip=0):
    key_2_id = dict()
    with open(filename, mode, encoding=encoding) as fin:
        for line in fin:
            if skip > 0:
                skip -= 1
                continue
            key, _id = line.strip().split(sep)
            key_2_id[key] = _id
    id_2_key = dict(zip(key_2_id.values(), key_2_id.keys()))

    return key_2_id, id_2_key


def save_txt_dict(key_2_id, filename, sep=None, mode='w', encoding='utf-8', skip=0):
    with open(filename, mode, encoding=encoding) as fout:
        for key, value in key_2_id.items():
            if skip > 0:
                skip -= 1
                continue
            if sep:
                print('{} {}'.format(key, value), file=fout)
            else:
                print('{}{}{}'.format(key, sep, value), file=fout)


def read_json_dict(filename, mode='r', encoding='utf-8'):
    with open(filename, mode, encoding=encoding) as fin:
        key_2_id = json.load(fin)
        id_2_key = dict(zip(key_2_id.values(), key_2_id.keys()))

    return key_2_id, id_2_key


def save_json_dict(data, filename, mode='w', encoding='utf-8'):
    with open(filename, mode, encoding=encoding) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4)


def pad_list(item_list, pad, max_len):
    item_list = item_list[:max_len]
    return item_list + [pad] * (max_len - len(item_list))


def pad_batch(data_batch, pad, max_len=None):
    if max_len is None:
        max_len = len(max(data_batch, key=len))
    return [pad_list(data, pad, max_len) for data in data_batch]


def convert_item(item, convert_dict, unk):
    return convert_dict[item] if item in convert_dict else unk


def convert_list(item_list, convert_dict, pad, unk, max_len=None):
    item_list = [convert_item(item, convert_dict, unk) for item in item_list]
    if max_len is not None:
        item_list = pad_list(item_list, pad, max_len)

    return item_list


def make_batch_iter(data, batch_size, shuffle):
    data_size = len(data)
    num_batches = (data_size + batch_size - 1) // batch_size

    if shuffle:
        random.shuffle(data)
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min(data_size, (i + 1) * batch_size)
        yield data[start_index:end_index]


# ====================
def parse_data_args(args, dataset_name, dataset_split):
    data_args = Namespace()
    data_args.model_name_or_path = args.model_name_or_path
    data_args.data_dir = args.data_dir
    data_args.dataset_name = dataset_name
    data_args.dataset_split = dataset_split
    data_args.max_seq_length = args.max_src_length
    data_args.max_output_seq_length = args.max_tgt_length
    data_args.max_seq_length_eval = args.max_src_length
    data_args.max_output_seq_length_eval = args.max_tgt_length
    data_args.prefix = args.prefix
    data_args.do_lower_case = args.do_lower_case
    data_args.overwrite_cache = args.overwrite_cache
    return data_args


def format_data(tokens, entities, relations):
    entities = [
        {'text': ' '.join(tokens[start:end]), 'start': start, 'end': end, 'type': d_type}
        for d_type, start, end in entities
    ]
    entities = sorted(entities, key=lambda x: (x['start'], x['end']))

    relations = [
        {
            'head': ' '.join(tokens[h_start:h_end]),
            'tail': ' '.join(tokens[t_start:t_end]),
            'h_start': h_start, 'h_end': h_end, 'h_type': h_type,
            't_start': t_start, 't_end': t_end, 't_type': t_type,
            'type': d_type
        }
        for d_type, (h_type, h_start, h_end), (t_type, t_start, t_end) in relations
    ]
    relations = sorted(relations, key=lambda x: (x['h_start'], x['h_end'], x['t_start'], x['t_end']))

    return entities, relations


# def refine_outputs(args, outputs, data_processor):
#     shift = 0
#     refined_outputs = defaultdict(list)
#     for task in args.task_list:
#         dataset = data_processor.datasets[task]
#         for example, generated_ids in zip(dataset.examples, outputs[shift:]):
#             generated_sentence = data_processor.tokenizer.decode(
#                 generated_ids,
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=False,
#             )
#             generated_output = dataset.output_format.run_inference(
#                 example,
#                 generated_sentence,
#                 entity_types=dataset.entity_types,
#                 relation_types=dataset.relation_types,
#             )
#             generated_entities, generated_relations = generated_output[:2]
#             generated_entities, generated_relations = format_data(
#                 tokens=example.tokens,
#                 entities=generated_entities,
#                 relations=generated_relations,
#             )
#             refined_outputs[task].append({
#                 'content': ' '.join(example.tokens),
#                 'output': generated_output,
#                 'entities': generated_entities,
#                 'relations': generated_relations,
#             })
#         shift += len(dataset.examples)
#     return refined_outputs
#
#
# def compute_metrics(args, outputs, data_processor):
#     results = {}
#     entity_f1 = []
#     relation_f1 = []
#     entity_f1_no_type = []
#     for task in args.task_list:
#         dataset = data_processor.datasets[task]
#         results[task] = dataset.evaluate_generated_outputs(outputs[task])
#         entity_f1.append(results[task]["entity_f1"])
#         relation_f1.append(results[task]["relation_f1"])
#         entity_f1_no_type.append(results[task]["entity_f1_no_type"])
#     results["entity_f1"] = sum(entity_f1) / len(entity_f1)
#     results["relation_f1"] = sum(relation_f1) / len(relation_f1)
#     results["entity_f1_no_type"] = sum(entity_f1_no_type) / len(entity_f1_no_type)
#     results["score"] = (results["entity_f1"] + results["relation_f1"]) / 2
#     return results


def generate_outputs(outputs, tokenizer):
    generated = []
    for line in outputs:
        generated.append(tokenizer.decode(line, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return generated


def refine_outputs(examples, outputs):
    refined_outputs = []
    for example, generated in zip(examples, outputs):
        refined_outputs.append({
            'source': example.source,
            'target': example.target,
            'generated': generated,
            'task_name': example.task_name,
        })
    return refined_outputs


def compute_metrics(outputs):
    references = []
    hypotheses = []
    for entry in outputs:
        references.append([entry['target'].strip().split()])  # ref is a list of words
        hypotheses.append(entry['generated'].strip().split())  # hyp is a list of words

    bleu1 = corpus_bleu(references, hypotheses, (1., 0., 0., 0.))
    bleu2 = corpus_bleu(references, hypotheses, (0.5, 0.5, 0., 0.))
    bleu3 = corpus_bleu(references, hypotheses, (0.33, 0.33, 0.33, 0.))
    bleu4 = corpus_bleu(references, hypotheses, (0.25, 0.25, 0.25, 0.25))
    result = {'Bleu_1': bleu1, 'Bleu_2': bleu2, 'Bleu_3': bleu3, 'Bleu_4': bleu4, }

    return result

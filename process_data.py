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
import random

from src.utils.my_utils import init_logger, read_file, save_file, read_json_lines, save_json_lines

logger = logging.getLogger(__name__)


def generate_stage_one(input_dir, output_dir, keep_ratio=1.00):
    entity_pattern = re.compile(r'\[([^=\[\]|]+)\|([^=\[\]|]+)]')
    relation_pattern = re.compile(r'\[([^=\[\]|]+)\|([^=\[\]|]+)\|([^\[\]]+)]')

    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if not filename.endswith('.target'): continue
        logger.info('processing {}'.format(filename))

        outputs = []
        for line in read_file(os.path.join(input_dir, filename)):
            line, raw_line = line.strip(), line.strip()
            for match in relation_pattern.finditer(line):
                line = line.replace(match.group(), '[ {} | {} ]'.format(match.group(1), match.group(2)))
            for match in entity_pattern.finditer(line):
                if random.random() < keep_ratio: continue
                line = line.replace(match.group(), match.group(1))
            outputs.append({'source': line, 'target': raw_line})
        save_json_lines(outputs, os.path.join(output_dir, '{}.{:1.2f}.json'.format(filename[:-7], keep_ratio)))


def main():
    init_logger(logging.INFO)

    tasks = ['ace2005_joint_er', 'ade', 'conll04', 'nyt']
    for task in tasks:
        generate_stage_one(
            'data/formatted/{}_t5-base_512/episode_00'.format(task),
            'data/stage_two/{}'.format(task),
            keep_ratio=1.00,
        )
        generate_stage_one(
            'data/formatted/{}_t5-base_512/episode_00'.format(task),
            'data/stage_two/{}'.format(task),
            keep_ratio=0.75,
        )
        generate_stage_one(
            'data/formatted/{}_t5-base_512/episode_00'.format(task),
            'data/stage_two/{}'.format(task),
            keep_ratio=0.50,
        )
        generate_stage_one(
            'data/formatted/{}_t5-base_512/episode_00'.format(task),
            'data/stage_two/{}'.format(task),
            keep_ratio=0.25,
        )


if __name__ == '__main__':
    main()

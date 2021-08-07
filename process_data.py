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


def generate_pipeline(task, role):
    relation_pattern = re.compile(r'\[(.+?)\|([^=\[\]|]+?)\|([^\[\]]+?)\]')

    sources = []
    for line in read_file('data/formatted/{}_t5-base_512/episode_00/{}.source'.format(task, role)):
        sources.append(line.strip())

    targets = []
    for line in read_file('data/formatted/{}_t5-base_512/episode_00/{}.target'.format(task, role)):
        targets.append(line.strip())

    middles = []
    for line in targets:
        match = relation_pattern.search(line)
        while match:
            line = line.replace(match.group(), '[ {} | {} ]'.format(match.group(1).strip(), match.group(2).strip()))
            match = relation_pattern.search(line)
        middles.append(line)

    os.makedirs('data/pipeline/{}/stage_one'.format(task), exist_ok=True)
    stage_one = []
    for src, tgt in zip(sources, middles):
        stage_one.append({"source": src, "target": tgt})
    save_json_lines(stage_one, 'data/pipeline/{}/stage_one/{}.json'.format(task, role))

    os.makedirs('data/pipeline/{}/stage_two'.format(task), exist_ok=True)
    stage_two = []
    for src, tgt in zip(middles, targets):
        stage_two.append({"source": src, "target": tgt})
    save_json_lines(stage_two, 'data/pipeline/{}/stage_two/{}.json'.format(task, role))


def generate_stage_two(task, role, keep_ratio=1.00):
    entity_pattern = re.compile(r'\[([^=\[\]|]+?)\|([^=\[\]|]+?)\]')
    relation_pattern = re.compile(r'\[(.+?)\|([^=\[\]|]+?)\|([^\[\]]+?)\]')

    outputs = []
    for line in read_file('data/formatted/{}_t5-base_512/episode_00/{}.target'.format(task, role)):
        line, raw_line = line.strip(), line.strip()

        # remove relation
        match = relation_pattern.search(line)
        while match:
            line = line.replace(match.group(), '[ {} | {} ]'.format(match.group(1).strip(), match.group(2).strip()))
            match = relation_pattern.search(line)

        # remove entity
        for match in entity_pattern.finditer(line):
            if random.random() < keep_ratio: continue
            line = line.replace(match.group(), match.group(1).strip())

        # remove nested entity when keep_ratio == 0.0
        if keep_ratio == 0.0:
            match = entity_pattern.search(line)
            while match:
                line = line.replace(match.group(), match.group(1).strip())
                match = entity_pattern.search(line)

        outputs.append({'source': line, 'target': raw_line})

    output_dir = 'data/stage_two/{}'.format(task)
    os.makedirs(output_dir, exist_ok=True)
    save_json_lines(outputs, os.path.join(output_dir, 'data_{}_{:03d}.json'.format(role, int(keep_ratio * 100))))


def main():
    init_logger(logging.INFO)

    # generate data for pipeline training
    tasks = ['ace2005_joint_er', 'ade', 'conll04', 'nyt']
    for task in tasks:
        logger.info('processing: {}'.format(task))
        generate_pipeline(task, 'train')
        if task != 'ade':
            generate_pipeline(task, 'valid')
        generate_pipeline(task, 'test')

    # generate data for stage two training
    tasks = ['ace2005_joint_er', 'ade', 'conll04', 'nyt']
    ratios = [0.00, 0.25, 0.50, 0.75, 1.00]
    for task in tasks:
        for ratio in ratios:
            logger.info('processing: {} {}'.format(task, ratio))
            generate_stage_two(task, 'train', ratio)
            if task != 'ade':
                generate_stage_two(task, 'valid', ratio)
            generate_stage_two(task, 'test', ratio)


if __name__ == '__main__':
    main()

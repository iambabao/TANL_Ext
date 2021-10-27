# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2021/10/19
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/10/19
"""

import torch
from tqdm import tqdm
from itertools import combinations


def get_score(tasks, mat):
    task_scores = {task: 0.0 for task in tasks}
    for tgt_task in tasks:
        scores = []
        for src_task in tasks:
            if src_task == tgt_task:
                continue
            scores.append(mat[src_task][tgt_task])
        if len(scores) > 0:
            task_scores[tgt_task] = sum(scores) / len(scores)

    return task_scores


def solver(mat, k):
    results = []
    tasks = list(mat.keys())
    for i in range(2, len(tasks) + 1):
        for group in combinations(tasks, i):
            results.append((get_score(group, mat), group))

    best = -1e5
    best_comb = None
    for comb in tqdm(combinations(results, k)):
        current = []
        comb_tasks = tuple()
        for item in comb:
            current.append(item[0])
            comb_tasks += item[1]
        if len(set(comb_tasks)) == len(tasks):
            task_scores = {task: -1e5 for task in tasks}
            for scores in current:
                for k, v in scores.items():
                    task_scores[k] = max(task_scores[k], v)
            if sum(task_scores.values()) > best:
                best = sum(task_scores.values())
                best_comb = comb

    for item in best_comb:
        print(item[0])
        print(item[1])


def main():
    affinities = torch.load('outputs/temp/checkpoints/affinities.bin')

    avg_aff = {}
    for src in affinities.keys():
        avg_aff[src] = {}
        for tgt in affinities[src].keys():
            avg_aff[src][tgt] = (sum(affinities[src][tgt]) / len(affinities[src][tgt])) if src != tgt else 0.0

    for key, value in avg_aff.items():
        print(key)
        for k, v in value.items():
            print(k, v)

    solver(avg_aff, 4)


if __name__ == '__main__':
    main()

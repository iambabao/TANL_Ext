# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2021/3/30
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/3/30
"""

import argparse
import logging
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM

from src.utils.my_utils import init_logger, read_json_lines, save_json_lines

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True, type=str, help="")
    parser.add_argument("--input_file", required=True, type=str, help="")
    parser.add_argument("--output_file", required=True, type=str, help="")
    parser.add_argument("--max_src_length", default=128, type=int, help="")
    parser.add_argument("--max_tgt_length", default=128, type=int, help="")
    parser.add_argument("--batch_size", default=16, type=int, help="")
    parser.add_argument("--prefix", default=None, help="")
    parser.add_argument("--no_cuda", action="store_true", help="")
    args = parser.parse_args()

    init_logger(logging.INFO)

    logger.info('Loading model from {}'.format(args.model_name_or_path))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model.to(device)
    model.eval()

    logger.info('Processing {}'.format(args.input_file))
    entries, results = list(read_json_lines(args.input_file)), []
    for step in tqdm(range((len(entries) + args.batch_size - 1) // args.batch_size), desc="Inference"):
        start_index = step * args.batch_size
        end_index = min(len(entries), (step + 1) * args.batch_size)
        if args.prefix is not None:
            if args.prefix == "":
                batch_text = ["{}: {}".format(_['task_name'], _['source']) for _ in entries[start_index:end_index]]
            else:
                batch_text = ["{}: {}".format(args.prefix, _['source']) for _ in entries[start_index:end_index]]
        else:
            batch_text = [_['source'] for _ in entries[start_index:end_index]]

        encoded = tokenizer.batch_encode_plus(
            batch_text,
            padding="max_length",
            truncation=True,
            max_length=args.max_src_length,
            return_tensors='pt',
        )
        inputs = {
            "input_ids": encoded["input_ids"].to(model.device),
            "attention_mask": encoded["attention_mask"].to(model.device),
        }
        outputs = model.generate(**inputs, max_length=args.max_tgt_length)
        generated = tokenizer.batch_decode(
            outputs.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for entry, gen in zip(entries[start_index:end_index], generated):
            results.append({'source': entry['source'], 'target': entry['target'], 'generated': gen})
    output_dir = os.path.split(args.output_file)[0]
    os.makedirs(output_dir, exist_ok=True)
    save_json_lines(results, args.output_file)

    logger.info("Done!")


if __name__ == '__main__':
    main()

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
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM

from src.utils.my_utils import init_logger, read_json_lines, save_json_lines

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path_one", required=True, type=str, help="")
    parser.add_argument("--model_path_two", required=True, type=str, help="")
    parser.add_argument("--input_file", required=True, type=str, help="")
    parser.add_argument("--output_file", required=True, type=str, help="")
    parser.add_argument("--max_src_length", default=128, type=int, help="")
    parser.add_argument("--max_tgt_length", default=128, type=int, help="")
    parser.add_argument("--batch_size", default=16, type=int, help="")
    parser.add_argument("--cache_dir", default=None, type=str, help="")
    parser.add_argument("--no_cuda", action="store_true", help="")
    args = parser.parse_args()

    init_logger(logging.INFO)

    tokenizer = AutoTokenizer.from_pretrained("t5-base", cache_dir=args.cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    logger.info('Loading stage one model from {}'.format(args.model_path_one))
    config = AutoConfig.from_pretrained(args.model_path_one)
    model_one = AutoModelForSeq2SeqLM.from_pretrained(args.model_path_one, config=config)
    model_one.to(device)
    model_one.eval()

    logger.info('Loading stage two model from {}'.format(args.model_path_two))
    config = AutoConfig.from_pretrained(args.model_path_two)
    model_two = AutoModelForSeq2SeqLM.from_pretrained(args.model_path_two, config=config)
    model_two.to(device)
    model_two.eval()

    logger.info('processing {}'.format(args.input_file))
    entries, results = list(read_json_lines(args.input_file)), []
    for step in tqdm(range((len(entries) + args.batch_size - 1) // args.batch_size), desc="Generating"):
        start_index = step * args.batch_size
        end_index = min(len(entries), (step + 1) * args.batch_size)

        # stage one
        encoded = tokenizer.batch_encode_plus(
            [_['source'] for _ in entries[start_index:end_index]],
            padding="max_length",
            truncation=True,
            max_length=args.max_src_length,
            return_tensors='pt',
        )
        inputs = {
            "input_ids": encoded["input_ids"].to(model_one.device),
            "attention_mask": encoded["attention_mask"].to(model_one.device),
        }
        outputs = model_one.generate(**inputs, max_length=args.max_tgt_length)
        generated = tokenizer.batch_decode(
            outputs.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # stage two
        encoded = tokenizer.batch_encode_plus(
            generated,
            padding="max_length",
            truncation=True,
            max_length=args.max_src_length,
            return_tensors='pt',
        )
        inputs = {
            "input_ids": encoded["input_ids"].to(model_two.device),
            "attention_mask": encoded["attention_mask"].to(model_two.device),
        }
        outputs = model_two.generate(**inputs, max_length=args.max_tgt_length)
        generated = tokenizer.batch_decode(
            outputs.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # final results
        for entry, gen in zip(entries, generated):
            results.append({'source': entry['source'], 'generated': gen})

    save_json_lines(results, args.output_file)

    logger.info("Done!")


if __name__ == '__main__':
    main()

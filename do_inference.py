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

from src.utils.my_utils import init_logger, read_file, save_file

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True, type=str, help="")
    parser.add_argument("--input_dir", required=True, type=str, help="")
    parser.add_argument("--output_dir", required=True, type=str, help="")
    parser.add_argument("--splits", required=True, type=str, help="")
    parser.add_argument("--max_src_length", default=128, type=int, help="")
    parser.add_argument("--max_tgt_length", default=128, type=int, help="")
    parser.add_argument("--batch_size", default=16, type=int, help="")
    parser.add_argument("--no_cuda", action="store_true", help="")
    args = parser.parse_args()

    init_logger(logging.INFO)

    logger.info('Loading model from {}'.format(args.model_name_or_path))
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model.to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    for split in args.splits.strip().split(","):
        sources, targets = [_.strip() for _ in read_file(os.path.join(args.input_dir, "{}.source".format(split)))], []
        for step in tqdm(range((len(sources) + args.batch_size - 1) // args.batch_size), desc="Generating"):
            start_index = step * args.batch_size
            end_index = min(len(sources), (step + 1) * args.batch_size)
            encoded = tokenizer.batch_encode_plus(
                sources[start_index:end_index],
                padding="max_length",
                truncation=True,
                max_length=args.max_src_length,
                return_tensors='pt',
            )
            inputs = {
                "input_ids": encoded["input_ids"].to(model.device),
                "attention_mask": encoded["attention_mask"].to(model.device),
            }
            outputs = model.generate(
                **inputs,
                max_length=args.max_tgt_length,
                num_beams=1,
                num_return_sequences=1,
            )
            generated = tokenizer.batch_decode(outputs.detach().cpu().tolist(), skip_special_tokens=True)
            targets.extend(generated)
        save_file(targets, os.path.join(args.output_dir, "{}.target".format(split)))

    logger.info("Done!")


if __name__ == '__main__':
    main()

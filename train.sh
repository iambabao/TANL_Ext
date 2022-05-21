#!/usr/bin/env bash

set -e

SEED=2333
#DATE="$(date +%m%d)"

MODEL_NAME_OR_PATH="t5-base"
MAX_SRC_LENGTH=256
MAX_TGT_LENGTH=256

TASK="conll03_ner"
DATA_DIR="data/processed"
OUTPUT_DIR="checkpoints/${TASK}"
CACHE_DIR="${HOME}/003_downloads/cache_transformers"
LOG_DIR="log/${TASK}"

NUM_TRAIN_EPOCH=50
PER_DEVICE_TRAIN_BATCH_SIZE=16
PER_DEVICE_EVAL_BATCH_SIZE=16
LEARNING_RATE=5e-5
LOGGING_STEPS=5000

CUDA_VISIBLE_DEVICES=0 python do_train.py \
--task ${TASK} \
--prefix \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--max_src_length ${MAX_SRC_LENGTH} \
--max_tgt_length ${MAX_TGT_LENGTH} \
--data_dir ${DATA_DIR} \
--output_dir ${OUTPUT_DIR} \
--cache_dir ${CACHE_DIR} \
--log_dir ${LOG_DIR} \
--do_train \
--do_eval \
--evaluate_during_training \
--overwrite_output_dir \
--num_train_epochs ${NUM_TRAIN_EPOCH} \
--per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
--per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
--learning_rate ${LEARNING_RATE} \
--logging_steps ${LOGGING_STEPS} \
--seed ${SEED}

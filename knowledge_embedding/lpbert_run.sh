#!/usr/bin/env bash

#BERT_DIR='../embedding/KeBioLM'

#BERT_DIR='../embedding/bert-base-uncased'

# BERT_DIR='../embedding/scibert_scivocab_uncased'
BERT_DIR='../embedding/scibert_scivocab_uncased'

## 全部数据的实体抽取任务示例

python lpbert_train.py \
            --save_model=True \
            --num_epochs=20 \
            --use_wandb=True\
            --learning_rate=1e-5 \
            --use_scheduler=True\
            --gradient_accumulation_steps=1 \
            --max_len=512 \
            --use_gpu=True\
            --use_n_gpu=True\
            --use_fp16=True\
            --batch_size=6 \
            --warmup_proportion=0.05 \
            --eval_batch_size=1024 \
            --gpu_id='0' \
            --bert_name='scibert' \
            --bert_dir=$BERT_DIR  \
            --logfile_name='drug_demo' \
            --model_name='star'  \
            --metric_verbose=True\

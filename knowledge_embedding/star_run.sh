#!/usr/bin/env bash

#BERT_DIR='../embedding/KeBioLM'

#BERT_DIR='../embedding/bert-base-uncased'

# BERT_DIR='../embedding/scibert_scivocab_uncased'

BERT_DIR='../embedding/biobert-base-cased-v1.1'


## 全部数据的实体抽取任务示例

python train.py \
            --save_model=True \
            --num_epochs=20 \
            --learning_rate=1e-5 \
            --use_gpu=True \
            --gradient_accumulation_steps=1 \
            --max_len=50 \
            --batch_size=32 \
            --warmup_proportion=0.05 \
            --eval_batch_size=1024 \
            --gpu_id='0' \
            --bert_name='biobert' \
            --dataset_name='new_myumls'  \
            --bert_dir=$BERT_DIR  \
            --logfile_name='drug_demo' \
            --model_name='star'  \
            --metric_verbose=True\

#python train.py \
#            --use_wandb=True \
#            --save_model=True \
#            --num_epochs=20 \
#            --learning_rate=1e-5 \
#            --use_gpu=True \
#            --gradient_accumulation_steps=1 \
#            --max_len=20 \
#            --batch_size=32 \
#            --warmup_proportion=0.05 \
#            --eval_batch_size=1024 \
#            --gpu_id='0' \
#            --bert_name='biobert' \
#            --dataset_name='umls'  \
#            --bert_dir=$BERT_DIR  \
#            --logfile_name='drug_demo' \
#            --model_name='kge'  \
#            --metric_verbose=True\

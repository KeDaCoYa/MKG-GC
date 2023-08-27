#!/usr/bin/env bash

#BERT_DIR="../embedding/scibert_scivocab_uncased/"

BERT_DIR="../embedding/SapBERT-from-PubMedBERT-fulltext"
#
#python ./my_multi_predicate.py --bert_dir=$BERT_DIR \
#               --model_name='模型预测' \
#               --dataset_name='1009abstracts'\
#               --use_gpu=True \
#               --use_n_gpu=True  \
#               --gpu_id='0' \
#               --save_model=True \
#               --batch_size=64 \


python ./my_multi_predicate.py --bert_dir=$BERT_DIR \
               --model_name='模型预测' \
               --dataset_name='1009abstracts'\
               --use_gpu=True \
               --use_n_gpu=True  \
               --gpu_id='0' \
               --save_model=True \
               --batch_size=128 \



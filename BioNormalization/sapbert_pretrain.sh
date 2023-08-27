#!/usr/bin/env bash

#BERT_DIR="../embedding/scibert_scivocab_uncased/"
#BERT_DIR="/root/code/bioner/embedding/biobert-base-cased-v1.2/"
BERT_DIR="/root/code/bioner/embedding/SapBERT-from-PubMedBERT-fulltext"

python ./train.py --bert_dir=$BERT_DIR \
               --model_name='模型训练' \
               --dataset_name='bc5cdr-disease' \
               --use_gpu=True \
               --use_n_gpu=True  \
               --gpu_id='0' \
               --save_model=True \
               --num_epochs=10 \
               --batch_size=64


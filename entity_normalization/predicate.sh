#!/usr/bin/env bash

#BERT_DIR="../embedding/scibert_scivocab_uncased/"
BERT_DIR="../embedding/SapBERT-from-PubMedBERT-fulltext"
#
python ./predicate.py --bert_dir=$BERT_DIR \
               --model_name='模型预测' \
               --dataset_name='bc5cdr-disease' \
               --use_gpu=True \
               --use_n_gpu=True  \
               --gpu_id='0' \
               --save_model=True \
               --num_epochs=40 \
               --batch_size=64 \



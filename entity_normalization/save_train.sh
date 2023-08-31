#!/usr/bin/env bash

#BERT_DIR="../embedding/scibert_scivocab_uncased/"
#BERT_DIR="/root/code/bioner/embedding/biobert-base-cased-v1.2/"
BERT_DIR="/root/code/bioner/embedding/SapBERT-from-PubMedBERT-fulltext"

python ./save_train.py --bert_dir=$BERT_DIR \
              --use_wandb=True\
               --model_name='模型训练' \
               --freeze_bert=True\
               --use_gpu=True \
               --use_amp=True \
               --gpu_id='0' \
               --num_epochs=5 \
               --batch_size=8 \
               --task_encoder_nums=3\
               --encoder_type='bert'\
               --model_name='biosyn' \

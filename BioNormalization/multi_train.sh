#!/usr/bin/env bash

#BERT_DIR="../embedding/scibert_scivocab_uncased/"
#BERT_DIR="/root/code/bioner/embedding/biobert-base-cased-v1.2/"
BERT_DIR="../embedding/SapBERT-from-PubMedBERT-fulltext"

#python ./multi_train.py --bert_dir=$BERT_DIR \
#              --use_wandb=True\
#               --freeze_bert=True\
#               --use_gpu=True \
#               --use_n_gpu=True \
#               --use_fp16=True \
#               --gpu_id='0' \
#               --num_epochs=8 \
#               --batch_size=12 \
#               --task_encoder_nums=1\
#               --encoder_type='gau'\
#               --model_name='biosyn' \

#
#python ./multi_train.py  --bert_dir=$BERT_DIR \
#              --use_wandb=True\
#              --use_gpu=True \
#              --use_n_gpu=True \
#              --use_fp16=True \
#              --use_scheduler=True\
#              --gpu_id='0' \
#              --num_epochs=8 \
#              --learning_rate=1e-5\
#              --batch_size=32 \
#               --task_encoder_nums=1\
#               --encoder_type='gau'\
#               --model_name='biosyn' \

python ./multi_train.py  --bert_dir=$BERT_DIR \
              --use_wandb=True\
              --use_gpu=True \
              --use_n_gpu=True \
              --use_fp16=True \
              --freeze_bert=True\
              --use_scheduler=True\
              --gpu_id='0' \
              --num_epochs=8 \
              --learning_rate=1e-5\
              --batch_size=32 \
               --task_encoder_nums=1\
               --encoder_type='gau'\
               --model_name='biosyn'\

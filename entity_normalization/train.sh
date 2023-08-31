#!/usr/bin/env bash

#BERT_DIR="../embedding/scibert_scivocab_uncased/"
#BERT_DIR="/root/code/bioner/embedding/biobert-base-cased-v1.2/"
BERT_DIR="../embedding/SapBERT-from-PubMedBERT-fulltext"

#python ./train.py --bert_dir=$BERT_DIR \
#               --model_name='single_model' \
#               --encoder_type='bert' \
#               --freeze_bert=True\
#               --dataset_name='mesh_chemical_drug' \
#               --use_wandb=True\
#               --gpu_id='0' \
#               --use_gpu=True\
#               --use_fp16=True\
#               --num_epochs=8 \
#               --batch_size=32 \
#               --model_name='biosyn' \
#
#python ./train.py --bert_dir=$BERT_DIR \
#               --model_name='single_model' \
#               --encoder_type='bert' \
#               --freeze_bert=True\
#               --dataset_name='gene_protein' \
#               --use_wandb=True\
#               --gpu_id='0' \
#               --use_gpu=True\
#               --use_fp16=True\
#               --num_epochs=8 \
#               --batch_size=16 \
#               --model_name='biosyn' \

python ./train.py --bert_dir=$BERT_DIR \
               --model_name='single_model' \
               --encoder_type='bert' \
               --debug=True\
               --freeze_bert=True\
               --dataset_name='bc5cdr-disease' \
               --gpu_id='0' \
               --use_gpu=True\
               --bert_dir=${BERT_DIR}\
               --use_fp16=True\
               --num_epochs=8 \
               --batch_size=16 \
               --model_name='biosyn'\
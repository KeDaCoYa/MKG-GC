#!/bin/bash


BERT_DIR='../embedding/biobert-base-cased-v1.1'

# 注意多个

python ./multi_re_main.py  \
              --use_wandb=True\
              --use_scheduler=True\
              --num_epochs=15 \
              --gpu_id='0' \
              --bert_lr=1e-5 \
              --use_gpu=True \
              --use_n_gpu=True \
              --freeze_bert=True\
              --use_fp16=True \
              --fixed_batch_length=True \
              --max_len=512 \
              --batch_size=64 \
              --bert_name='biobert' \
              --model_name='multi_entity_marker' \
              --scheme=-12 \
              --data_format='single' \
              --dataset_type='original_dataset' \
              --dataset_name='AllDataset' \
              --class_type='multi' \
              --run_type='normal' \
              --print_step=1 \
              --save_model=True\
              --bert_dir=${BERT_DIR}\
              --train_verbose=True

python ./multi_re_main.py  \
              --use_wandb=True\
              --use_scheduler=True\
              --num_epochs=15 \
              --gpu_id='0' \
              --bert_lr=1e-5 \
              --use_gpu=True \
              --use_n_gpu=True \
              --use_fp16=True \
              --fixed_batch_length=True \
              --max_len=512 \
              --batch_size=64 \
              --bert_name='biobert' \
              --model_name='multi_entity_marker' \
              --scheme=-12 \
              --data_format='single' \
              --dataset_type='original_dataset' \
              --dataset_name='AllDataset' \
              --class_type='multi' \
              --run_type='normal' \
              --print_step=1 \
              --save_model=True\
              --bert_dir=${BERT_DIR}\
              --train_verbose=True

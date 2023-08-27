#!/bin/bash


BERT_DIR='../embedding/biobert-base-cased-v1.1'




python ./multi_re_predicate.py  \
              --gpu_id='0' \
              --use_gpu=True\
              --max_len=512 \
              --batch_size=128 \
              --bert_name='biobert' \
              --scheme=-12 \
              --data_format='single' \
              --dataset_type='my_dataset' \
              --dataset_name='1009abstracts' \
              --class_type='single' \
              --run_type='normal' \
              --bert_dir=${BERT_DIR}
#
# # 预测的时候不要用fix batch
#python ./multi_re_predicate.py  \
#              --num_epochs=1 \
#              --gpu_id='0' \
#              --batch_size=128 \
#              --use_scheduler=True\
#              --use_gpu=True\
#              --freeze_bert=True\
#              --max_len=512 \
#              --bert_name='biobert' \
#              --scheme=-12 \
#              --data_format='single' \
#              --dataset_type='my_dataset' \
#              --dataset_name='3400abstracts' \
#              --class_type='single' \
#              --run_type='normal' \
#              --print_step=1 \
#              --bert_dir=${BERT_DIR}

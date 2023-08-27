#!/bin/bash


BERT_DIR='../embedding/biobert-base-cased-v1.1'




python ./re_predicate.py  \
            --fixed_batch_length=True \
            --max_len=512 \
            --batch_size=128\
            --use_gpu=True \
            --bert_name='biobert' \
            --model_name='single_entity_marker' \
            --scheme=-12  \
            --data_format='single' \
            --dataset_type='my_dataset' \
            --dataset_name='1009abstracts' \
            --class_type='single' \
            --print_step=1 \
            --num_labels=6  \
            --bert_dir=${BERT_DIR}

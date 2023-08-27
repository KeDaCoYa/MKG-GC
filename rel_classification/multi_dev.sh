#!/bin/bash


BERT_DIR='../embedding/biobert-base-cased-v1.1'




python ./multi_re_dev.py  \
              --num_epochs=1 \
              --gpu_id='0' \
              --bert_lr=2e-5 \
              --use_scheduler=True\
              --use_gpu=True\
              --freeze_bert=True\
              --fixed_batch_length=True \
              --max_len=512 \
              --batch_size=16 \
              --bert_name='biobert' \
              --model_name='sing_entity_marker' \
              --scheme=-12 \
              --data_format='single' \
              --dataset_type='original_dataset' \
              --dataset_name='AllDataset' \
              --class_type='multi' \
              --run_type='normal' \
              --print_step=1 \
              --bert_dir=${BERT_DIR}

#!/bin/bash


BERT_DIR='../embedding/biobert-base-cased-v1.1'

python ./multi_ner_main.py \
            --use_wandb=True \
            --run_type='normal' \
            --num_epochs=15 \
            --freeze_bert=True\
            --learning_rate=1e-5 \
            --fixed_batch_length=True \
            --use_scheduler=True \
            --gradient_accumulation_steps=1 \
            --max_len=512 \
            --batch_size=64 \
            --gpu_id='0' \
            --use_gpu=True\
            --use_n_gpu=True\
            --use_fp16=True\
            --bert_name='biobert' \
            --decoder_layer='span' \
            --ner_dataset_name='multi_all_dataset_v1_lite'  \
            --entity_type='multiple'  \
            --bert_dir=$BERT_DIR  \
            --logfile_name='inter_binary_bert_linear_mid_span_epochs15' \
            --model_name='inter_binary_bert_bilstm_span'  \
            --verbose=True\

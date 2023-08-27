#!/bin/bash


BERT_DIR='../embedding/biobert-base-cased-v1.1'



# 全部数据的实体抽取任务示例
python ./ner_predicate.py \
            --which_model='bert' \
            --run_type='normal' \
            --save_model=True \
            --num_epochs=1 \
            --use_gpu=True \
            --fixed_batch_length=True \
            --gradient_accumulation_steps=1 \
            --max_len=512 \
            --batch_size=1 \
            --gpu_id='0' \
            --bert_name='biobert' \
            --decoder_layer='span' \
            --ner_dataset_name='AllDataset'  \
            --entity_type='multiple'  \
            --bert_dir=$BERT_DIR  \
            --logfile_name='normalize_dataset_model' \
            --model_name='bert_span'
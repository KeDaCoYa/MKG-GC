#!/bin/bash

BERT_DIR='../embedding/KeBioLM'
python ./ner_dev.py \
            --which_model='bert' \
            --run_type='normal' \
            --save_model=True \
            --num_epochs=4 \
            --use_gpu=True \
            --use_n_gpu=True \
            --use_fp16=True \
            --fixed_batch_length=True \
            --gradient_accumulation_steps=1 \
            --max_len=256 \
            --batch_size=8 \
            --gpu_id='0' \
            --bert_name='biobert' \
            --decoder_layer='mlp' \
            --ner_dataset_name='AllDataset'  \
            --entity_type='multiple'  \
            --bert_dir=$BERT_DIR  \
            --logfile_name='test_log' \
            --model_name='bert_mlp'  \
# ---------------Normal Model--------------------

#python ./ner_dev.py --which_model='normal' --use_pretrained_embedding=True --num_epochs=8  --use_gpu=True --fixed_batch_length=True  --gradient_accumulation_steps=1 --max_len=256  --batch_size=32 --gpu_id='0' --logfile_name='无' --model_name='bilstm_globalpointer' --decoder_layer='globalpointer' --ner_dataset_name='jnlpba' --entity_type='multiple'

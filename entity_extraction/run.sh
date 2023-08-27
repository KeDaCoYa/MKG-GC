#!/bin/bash



#python ./ner_main.py --which_model='bert' --run_type='normal' --save_best_model=True --num_epochs=8  --use_gpu=True --fixed_batch_length=True  --gradient_accumulation_steps=1 --max_len=256  --batch_size=16 --gpu_id='0' --bert_name='kebiolm' --logfile_name='kebiolm_crf_cv5' --decoder_layer='crf'  --ner_dataset_name='jnlpba' --entity_type='multiple' --bert_dir=$BERT_DIR --model_name='bert_crf'

#BERT_DIR='../embedding/bert-base-uncased'

#BERT_DIR='../embedding/flash_quad-base-uncased'


BERT_DIR='../embedding/biobert-base-cased-v1.1'

#

python ./ner_main.py \
            --use_wandb=True \
            --run_type='normal' \
            --num_epochs=15 \
            --learning_rate=1e-5 \
            --freeze_bert=True\
            --fixed_batch_length=True \
            --use_scheduler=True \
            --gradient_accumulation_steps=1 \
            --max_len=512 \
            --batch_size=16 \
            --gpu_id='0' \
            --use_gpu=True\
            --use_n_gpu=True\
            --use_fp16=True\
            --bert_name='biobert' \
            --decoder_layer='span' \
            --ner_dataset_name='BC7DrugProt'  \
            --entity_type='multiple'  \
            --bert_dir=$BERT_DIR  \
            --logfile_name='single_task_bert_span' \
            --model_name='bert_span'  \



#!/usr/bin/env bash


BERT_DIR="../embedding/SapBERT-from-PubMedBERT-fulltext"

python ./my_multi_train.py --bert_dir=$BERT_DIR \
              --use_wandb=True\
              --use_n_gpu=True\
               --model_name='multi_task_five_model' \
               --task_name='freeze4层'\
               --use_gpu=True \
               --use_fp16=True \
               --gpu_id='0' \
               --num_epochs=8 \
               --freeze_bert=True\
               --learning_rate=2e-5\
               --batch_size=24 \
               --task_encoder_nums=2\
               --encoder_type='gau'\
               --save_model=True\
               --model_name='biosyn' \

python ./my_multi_train.py --bert_dir=$BERT_DIR \
              --use_wandb=True\
              --use_n_gpu=True\
               --model_name='multi_task_five_model' \
               --task_name='不再freeze'\
               --use_gpu=True \
               --use_fp16=True \
               --gpu_id='0' \
               --num_epochs=8 \
               --learning_rate=2e-5\
               --batch_size=24 \
               --task_encoder_nums=2\
               --encoder_type='gau'\
               --save_model=True\
               --model_name='biosyn' \


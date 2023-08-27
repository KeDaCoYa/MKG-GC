# -*- encoding: utf-8 -*-
"""
@File    :   DTI_train.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/5/1 21:36   
@Description :   None 

"""
from src.utils.dti_dataset_utils import DTIProcessor


def train(config):
    processor = DTIProcessor()

    vocab2id,id2vocab = processor.get_vocab2id(config.data_dir)
    train_examples = processor.get_train_examples(config.data_dir)
    dev_examples = processor.get_dev_examples(config.data_dir)


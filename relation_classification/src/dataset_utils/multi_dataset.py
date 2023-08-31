# -*- encoding: utf-8 -*-
"""
@File    :   multi_dataset.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/5/8 9:29   
@Description :   None 

"""
from src.dataset_utils.data_process_utils import process_normal_data, process_mtb_data


def read_multi_data(config,type_='train'):
    """
    这个是用于最终的总的数据集的读取,读取所有的数据集进行
    :param config:
    :param type:
    :return:
    """


    if config.data_format == 'single': #格式为<CLS>sentence a<sep>sentence b <sep>
        if type_ == 'train':
            examples = process_normal_data(config.train_normal_path, config.dataset_name)
        elif type_ == 'dev':
            examples = process_normal_data(config.dev_normal_path, config.dataset_name)
        elif type_ == 'test':
            examples = process_normal_data(config.test_normal_path, config.dataset_name)
    elif config.data_format == 'cross':
        if type_ == 'train':
            examples = process_mtb_data(config.train_mtb_path, config.dataset_name)
        elif type_ == 'dev':
            examples = process_mtb_data(config.dev_mtb_path, config.dataset_name)
        elif type_ == 'test':
            examples = process_mtb_data(config.test_mtb_path, config.dataset_name)
    else:
       raise ValueError("data_format错误")
    return examples
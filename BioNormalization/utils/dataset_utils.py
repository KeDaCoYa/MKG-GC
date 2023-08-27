# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2022/01/19
   Copyright:      (c) kedaxia 2022
-------------------------------------------------
   Change Activity:
                   2022/01/19: 
-------------------------------------------------
"""
import glob
import os
import pickle
import logging

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertForPreTraining

logger = logging.getLogger('main.dataset_utils')

def choose_model_tokenizer(config):


    tokenizer = AutoTokenizer.from_pretrained('../embedding/scibert_scivocab_uncased/')
    config.vocab_size = tokenizer.vocab_size
    model = BertForPreTraining(config)
    model.bert.resize_token_embeddings(len(tokenizer))

    device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')
    return model,tokenizer,device


class QueryDataset(Dataset):

    def __init__(self, data_dir,filter_composite=False,filter_duplicate=False,filter_cuiless=False):
        """

        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            是否过滤掉重复的
        draft : bool
            这个表示只使用一部分的数据，有利于之后的debug
        """

        self.data = self.load_data(
            data_dir=data_dir,
            filter_composite=filter_composite,
            filter_duplicate=filter_duplicate,
            filter_cuiless=filter_cuiless
        )

    def load_data(self, data_dir, filter_composite, filter_duplicate, filter_cuiless):
        """
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            是否过滤复合mention，就是一个mention中由其他特殊符号，这里限制为'+',’|‘
        filter_duplicate : bool
            filter duplicate queries
        filter_cuiless : bool
            remove samples with cuiless
        Returns
        -------
        data : np.array
            mention, cui pairs
        """
        data = []
        # 读取所有的concept文件
        concept_files = glob.glob(os.path.join(data_dir, "*.concept"))
        for concept_file in tqdm(concept_files):
            # 这里读取训练集的每个文件
            # 文件的内容非常固定，一个是entity mention，另一个是对应的ID
            with open(concept_file, "r", encoding='utf-8') as f:
                concepts = f.readlines()

            for concept in concepts:
                concept = concept.split("||")
                mention = concept[3].strip()
                cui = concept[4].strip()
                is_composite = (cui.replace("+", "|").count("|") > 0)

                # filter composite cui
                if filter_composite and is_composite:
                    continue
                # filter cuiless
                if filter_cuiless and cui == '-1':
                    continue

                data.append((mention, cui))

        if filter_duplicate:
            data = list(dict.fromkeys(data))

        # return np.array data
        data = np.array(data)

        return data

class MyQueryDataset(Dataset):

    def __init__(self, data_path):
        """
        这里不同于上面，这里给定具体的文件路径
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            是否过滤掉重复的
        draft : bool
            这个表示只使用一部分的数据，有利于之后的debug
        """

        self.data = self.load_data(
            data_path=data_path,
        )

    def load_data(self, data_path):
        """
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            是否过滤复合mention，就是一个mention中由其他特殊符号，这里限制为'+',’|‘
        filter_duplicate : bool
            filter duplicate queries
        filter_cuiless : bool
            remove samples with cuiless
        Returns
        -------
        data : np.array
            mention, cui pairs
        """
        data = []
        # 读取所有的concept文件
        with open(data_path,'r',encoding='utf-8') as f:
            all_data = f.readlines()
        for line in all_data:
            id_,mention = line.split('||')
            data.append((mention.strip(), id_))
        data = np.array(data)

        return data
def load_my_data(data_path):
    """
    Parameters
    ----------
    data_dir : str
        a path of data
    filter_composite : bool
        是否过滤复合mention，就是一个mention中由其他特殊符号，这里限制为'+',’|‘
    filter_duplicate : bool
        filter duplicate queries
    filter_cuiless : bool
        remove samples with cuiless
    Returns
    -------
    data : np.array
        mention, cui pairs
    """
    data = []
    # 读取所有的concept文件
    with open(data_path,'r',encoding='utf-8') as f:
        all_data = f.readlines()
    for line in all_data:
        id_,mention = line.split('||')
        data.append((mention.strip(), id_))
    data = np.array(data)

    return data
class DictionaryDataset:

    def __init__(self, dictionary_path):
        '''

        :param dictionary_path:
        '''

        self.data = self.load_data(dictionary_path)

    def load_data(self, dictionary_path):
        """
        加载字典的内容
        :param dictionary_path:
        :return:
        """
        data = []
        with open(dictionary_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip()
                if line == "":
                    continue
                cui, name = line.split("||")
                data.append((name, cui))

        data = np.array(data)
        return data


def load_dictionary(dictionary_path):
    '''
    读取字典文件中的内容

    这个字典每一行由两部分组成，一个是ID，另一个是mention(或者是别名)
    :param dictionary_path:
    :return:
    '''
    dictionary = DictionaryDataset(
        dictionary_path=dictionary_path
    )

    return dictionary.data


def load_queries(data_dir, filter_composite, filter_duplicate, filter_cuiless):
    """
    这是加载标准的数据集，数据格式和dictionary的一样，一行由CUI和mention组成

    Parameters
    ----------
    is_train : bool
        train or dev
    filter_composite : bool
        filter composite mentions
    filter_duplicate : bool
        filter duplicate queries
    filter_cuiless : bool
        filter samples with cuiless
    """
    if 'rocessed' in data_dir:
        dataset = QueryDataset(
            data_dir=data_dir,
            filter_composite=filter_composite,
            filter_duplicate=filter_duplicate,
            filter_cuiless=filter_cuiless
        )
    else:
        dataset = MyQueryDataset(
            data_path=data_dir,
        )

    return dataset.data
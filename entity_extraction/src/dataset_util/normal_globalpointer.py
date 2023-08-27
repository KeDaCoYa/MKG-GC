# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2021/11/25
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/11/25: 
-------------------------------------------------
"""

from ipdb import set_trace

import torch
import numpy as np
from torch.utils.data import Dataset
import logging

from src.dataset_util.base_dataset import globalpointer_collate, sequence_padding

logger = logging.getLogger('main.globalpointer_dataset')

class NormalGlobalPointerDataset(Dataset):
    def __init__(self,data,word2id,config):
        '''
        这是dynamic
        :param data: 这个数据就是load_data的结果，以列表的形式存储
            [(raw_text,start_off,end_offset,entity_type_id),(),....]
        :param tokenizer:分词器
        :num_tags:表示需要识别的实体类别
        '''


        super(NormalGlobalPointerDataset,self).__init__()
        self.data = data
        self.config = config
        self.PAD = config.PAD
        self.UNK = config.UNK
        self.num_tags = config.num_gp_class
        self.config = config

        self.word2id = word2id
        self.nums = len(data)


    def __len__(self):
        return self.nums

    def collate_predicate(self,examples):
        '''
        这里开始处理数据，在DataLoader读取了一个batch的数据之后进行处理
        :param examples:
        :return:
        '''
        raw_text_list = []
        batch_token_ids = []
        attention_masks = []
        max_seq_len = max(len(x.text) for x in examples)
        batch_max_len = 0
        for item in examples:
            raw_text = item.text
            actual_len = len(raw_text)
            if batch_max_len<actual_len:
                batch_max_len = actual_len

            # 这里是规整一下实体边界
            mask  = np.zeros(max_seq_len)
            mask[:actual_len] = 1
            attention_masks.append(mask)
            token_ids = []
            for word in raw_text:
                word_id = self.word2id.get(word, self.word2id.get(word.lower(), self.UNK))

                token_ids.append(word_id)

            batch_token_ids.append(token_ids)

            raw_text_list.append(raw_text)


        if self.config.fixed_batch_length:
            pad_len = self.config.max_len
        else:
            pad_len = min(batch_max_len, self.config.max_len)

        # 这里batch_input_ids,batch_segment_ids,batch_attention_mask都没有进行pad，这里需要进行pad，将长度进行统一补齐...
        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids,length=pad_len)).long()
        # batch_labels.shape = (batch_size,entity_type,seq_len,seq_len)


        attention_masks = torch.tensor(attention_masks).long()

        return raw_text_list, batch_token_ids,attention_masks


    def collate(self,examples):
        '''
        这里开始处理数据，在DataLoader读取了一个batch的数据之后进行处理
        :param examples:
        :return:
        '''
        raw_text_list,batch_labels = [], []
        batch_token_ids = []
        attention_masks = []
        max_seq_len = max(len(x.text) for x in examples)
        batch_max_len = 0
        for item in examples:
            raw_text = item.text
            actual_len = len(raw_text)
            if batch_max_len<actual_len:
                batch_max_len = actual_len
            true_labels = item.labels
            labels = np.zeros((self.num_tags, max_seq_len, max_seq_len)) # 普通条件下是(1,seq_len,seq_len)
            # 这里是规整一下实体边界
            mask  = np.zeros(max_seq_len)
            mask[:actual_len] = 1
            attention_masks.append(mask)
            token_ids = []
            for word in raw_text:
                word_id = self.word2id.get(word, self.word2id.get(word.lower(), self.UNK))
                #word_id = self.word2id.get(word,0)
                #word_id = self.word2id.get(word, self.word2id.get('unk'))
                token_ids.append(word_id)

            batch_token_ids.append(token_ids)
            # 需要转变为start_offset,end_offset类别

            globalpointer_label2id = self.config.globalpointer_label2id

            globalpointer_collate(true_labels, labels, globalpointer_label2id)


            raw_text_list.append(raw_text)

            # 这里按照实际举例进行一个slice
            batch_labels.append(labels[:, :actual_len, :actual_len])
        if self.config.fixed_batch_length:
            pad_len = self.config.max_len
        else:
            pad_len = min(batch_max_len, self.config.max_len)

        # 这里batch_input_ids,batch_segment_ids,batch_attention_mask都没有进行pad，这里需要进行pad，将长度进行统一补齐...
        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids,length=pad_len)).long()
        # batch_labels.shape = (batch_size,entity_type,seq_len,seq_len)

        batch_labels = torch.tensor(sequence_padding(batch_labels, seq_dims=3)).long()
        attention_masks = torch.tensor(attention_masks).long()

        return raw_text_list, batch_token_ids,  batch_labels,attention_masks
    def __getitem__(self, item):
        return self.data[item]


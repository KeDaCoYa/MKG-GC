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



logger = logging.getLogger('main.normal_crf_dataset')



class NormlCRFDataset(Dataset):
    def __init__(self, features,word2id,config):
        '''
            这是CRF动态batch的数据处理
            在这里将所有数据转变为tensor，可以直接使用
        :param data:
        '''
        super(NormlCRFDataset, self).__init__()
        self.nums = len(features)
        self.data = features
        self.word2id = word2id
        self.config = config
        self.UNK = config.UNK
        self.PAD = config.PAD
        # 因为这里一次只取一个，所以一个tensor就行了...


    def __len__(self):
        return self.nums

    def __getitem__(self, index):

        return self.data[index]
    def collate_predicate(self,examples):
        '''
        这里开始处理数据，在DataLoader读取了一个batch的数据之后进行处理
        :param examples:
        :return:
        '''
        raw_text_list = []
        batch_token_ids = []
        attention_masks = []
        batch_max_len = max([len(example.text) for example in examples])
        if self.config.fixed_batch_length:
            batch_max_len = self.config.max_len
        else:
            batch_max_len = min(batch_max_len,self.config.max_len)
        for item in examples:
            raw_text = item.text
            raw_text_list.append(raw_text)
            actual_len = len(raw_text)

            mask = np.zeros(batch_max_len)
            mask[:actual_len] = 1

            # 开始对token进行向量化
            token_ids = []
            for word in raw_text:
                word_id = self.word2id.get(word, self.word2id.get(word.lower(), self.UNK))
                token_ids.append(word_id)



            batch_token_ids.append(token_ids)
            # 需要转变为start_offset,end_offset类别
            attention_masks.append(mask)

        batch_token_ids = torch.tensor(batch_token_ids).long()


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
        batch_max_len = max([len(example.text) for example in examples])
        if self.config.fixed_batch_length:
            batch_max_len = self.config.max_len
        else:
            batch_max_len = min(batch_max_len,self.config.max_len)
        for item in examples:
            raw_text = item.text
            raw_text_list.append(raw_text)
            actual_len = len(raw_text)
            labels = item.labels

            mask = np.zeros(batch_max_len)
            mask[:actual_len] = 1

            # 开始对token进行向量化
            token_ids = []
            for word in raw_text:
                word_id = self.word2id.get(word, self.word2id.get(word.lower(), self.UNK))
                token_ids.append(word_id)
            label_ids = [self.config.crf_label2id.get(c) for c in labels]

            pad_len = batch_max_len - len(label_ids)
            # 全部补齐为 O
            label_ids = label_ids + [0] * pad_len
            token_ids = token_ids + [self.PAD] * pad_len


            batch_token_ids.append(token_ids)
            # 需要转变为start_offset,end_offset类别
            attention_masks.append(mask)
            # 这里按照实际举例进行一个slice
            batch_labels.append(label_ids)
        # 这里batch_input_ids,batch_segment_ids,batch_attention_mask都没有进行pad，这里需要进行pad，将长度进行统一补齐...
        batch_token_ids = torch.tensor(batch_token_ids).long()

        # batch_labels.shape = (batch_size,entity_type,seq_len,seq_len)
        batch_labels = torch.tensor(batch_labels).long()

        attention_masks = torch.tensor(attention_masks).long()

        return raw_text_list, batch_token_ids,  batch_labels,attention_masks


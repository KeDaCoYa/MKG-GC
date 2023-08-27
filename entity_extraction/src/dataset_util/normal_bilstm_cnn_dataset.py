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

import string

from ipdb import set_trace

import torch
from torch.utils.data import Dataset
import numpy as np



from config import NormalConfig
from src.dataset_util.base_dataset import sequence_padding


class NormlBiLSTMCNNCRFDataset(Dataset):
    def __init__(self,data,word2id,config:NormalConfig):
        '''
        这是dynamic
        :param data: 这个数据就是load_data的结果，以列表的形式存储
            [(raw_text,start_off,end_offset,entity_type_id),(),....]
        :param tokenizer:分词器
        :num_tags:表示需要识别的实体类别
        :param is_train:这个并没有体现出其作用
        '''

        super(NormlBiLSTMCNNCRFDataset,self).__init__()

        self.data = data
        self.num_tags = config.num_crf_class
        self.UNK = config.UNK
        self.PAD = config.PAD
        self.config = config
        self.max_len = config.max_len
        self.word2id = word2id
        self.nums = len(data)


    def __len__(self):
        return self.nums

    def collate_predicate(self,examples):
        '''
        这里开始处理数据，在DataLoader读取了一个batch的数据之后进行处理
        这里还要处理char数据
        对数据进行pad
        :param examples:InputExample
        :return:
        '''
        word_len = self.config.word_len #character需要填充的最大值...
        char2id = self.config.char2id
        batch_seq_max_len = max([len(x.text) for x in examples])

        raw_text_list = []
        batch_token_ids = []
        batch_char_token_ids = [] #(batch_size,seq_len,word_len)
        attention_masks = []

        for item in examples:
            raw_text = item.text

            mask = np.zeros((batch_seq_max_len))

            mask[:len(raw_text)] = 1
            attention_masks.append(mask)
            token_ids = []
            for word in raw_text:
                word_id = self.word2id.get(word, self.word2id.get(word.lower(),self.UNK))
                token_ids.append(word_id)

            char_ids = np.zeros((batch_seq_max_len,word_len))

            # 这里限制长度为20
            for i,word in enumerate(raw_text):
                for j,c in enumerate(word):
                    if j>=20:
                        break
                    char_ids[i][j] = self.config.char2id.get(c,1)

            raw_text_list.append(raw_text)
            batch_char_token_ids.append(char_ids)
            batch_token_ids.append(token_ids)

        # 这里batch_input_ids,batch_segment_ids,batch_attention_mask都没有进行pad，这里需要进行pad，将长度进行统一补齐...
        if self.config.fixed_batch_length:
            pad_len = self.config.max_len
        else:
            pad_len = min(batch_seq_max_len,self.config.max_len)
        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids,length=pad_len)).long()

        batch_char_token_ids = torch.tensor(batch_char_token_ids).long() #shape=(batch_size,seq_len,word_len)
        attention_masks = torch.tensor(attention_masks).long() #shape=(batch_size,seq_len,word_len)



        return raw_text_list, batch_token_ids,batch_char_token_ids,attention_masks

    def collate(self,examples):
        '''
        这里开始处理数据，在DataLoader读取了一个batch的数据之后进行处理
        这里还要处理char数据
        对数据进行pad
        :param examples:InputExample
        :return:
        '''
        word_len = self.config.word_len #character需要填充的最大值...
        char2id = self.config.char2id
        batch_seq_max_len = max([len(x.text) for x in examples])

        raw_text_list,batch_labels = [], []
        batch_token_ids = []
        batch_char_token_ids = [] #(batch_size,seq_len,word_len)
        attention_masks = []

        for item in examples:
            raw_text = item.text
            labels = item.labels
            mask = np.zeros((batch_seq_max_len))

            mask[:len(raw_text)] = 1
            attention_masks.append(mask)
            token_ids = []
            for word in raw_text:
                word_id = self.word2id.get(word, self.word2id.get(word.lower(),self.UNK))
                token_ids.append(word_id)
            label_ids = [self.config.crf_label2id.get(c) for c in labels]
            char_ids = np.zeros((batch_seq_max_len,word_len))

            # 这里限制长度为20
            for i,word in enumerate(raw_text):
                for j,c in enumerate(word):
                    if j>=20:
                        break
                    char_ids[i][j] = self.config.char2id.get(c,1)

            raw_text_list.append(raw_text)
            batch_char_token_ids.append(char_ids)
            batch_token_ids.append(token_ids)
            batch_labels.append(label_ids)
        # 这里batch_input_ids,batch_segment_ids,batch_attention_mask都没有进行pad，这里需要进行pad，将长度进行统一补齐...
        if self.config.fixed_batch_length:
            pad_len = self.config.max_len
        else:
            pad_len = min(batch_seq_max_len,self.config.max_len)
        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids,length=pad_len)).long()
        batch_labels = torch.tensor(sequence_padding(batch_labels,length=pad_len)).long()
        batch_char_token_ids = torch.tensor(batch_char_token_ids).long() #shape=(batch_size,seq_len,word_len)
        attention_masks = torch.tensor(attention_masks).long() #shape=(batch_size,seq_len,word_len)



        return raw_text_list, batch_token_ids,batch_char_token_ids, batch_labels,attention_masks
    def __getitem__(self, item):
        return self.data[item]

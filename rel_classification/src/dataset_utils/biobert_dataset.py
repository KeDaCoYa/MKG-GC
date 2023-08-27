# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2021/12/22
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/22: 
-------------------------------------------------
"""
import re

from ipdb import set_trace
import numpy as np

import torch
from torch.utils.data import Dataset

from transformers import BertTokenizer





from config import BertConfig
from src.utils.function_utils import get_pos_feature
from src.dataset_utils.data_process_utils import sequence_padding


class BioBERT_Dataset(Dataset):
    def __init__(self, config: BertConfig, sents, labels, tokenizer, label2id):
        super(BioBERT_Dataset, self).__init__()
        self.config = config
        self.sents = sents
        self.labels = labels
        self.tokenizer = tokenizer
        # 将特殊实体加入到分词器，防止给切分

        self.label2id = label2id

        self.max_len = config.max_len

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, item):

        sent = self.sents[item]
        label = self.labels[item]
        return sent, label

    def collate_fn(self, features):
        '''
        这里也是保证每一句话中只有一对实体
        :param features:
        :return:
        '''
        raw_text_li = []
        batch_input_ids = []
        batch_token_type_ids = []
        batch_labels = []
        batch_attention_masks = []
        batch_entities_positions = []


        e1_pattern = None
        e2_pattern = None
        pattern = None
        if self.config.ent1_end_token == '$':
            pattern = re.compile(r'@.*?\$')

        else:
            e1_pattern = re.compile(r'{}.*?{}'.format(self.config.ent1_start_token,self.config.ent1_end_token))
            e2_pattern = re.compile(r'{}.*?{}'.format(self.config.ent2_start_token,self.config.ent2_end_token))

        batch_max_len = 0
        for sent, label in features:
            # sent 是word-level list: ['Feadeal','ABVDF','the',...]
            raw_text_li.append(sent)

            subword_tokens = self.tokenizer.tokenize(sent)

            # 进行分词
            if self.config.ent1_end_token == '$':

                e1,e2 = pattern.findall(sent)
            else:
                e1 = e1_pattern.findall(sent)[0]
                e2 = e2_pattern.findall(sent)[0]
            pos1,pos2 = None,None
            # 由于[CLS]
            for pos,word in enumerate(subword_tokens):
                if pos1 is None:
                    if e1.lower() == word:
                        pos1 = pos+1
                if pos2 is None:
                    if e2.lower() == word:
                        pos2 = pos+1
                        break
            if batch_max_len<len(subword_tokens):
                batch_max_len = len(subword_tokens)

            encoder_res = self.tokenizer.encode_plus(subword_tokens, truncation=True, max_length=self.max_len,add_special_tokens=True)

            input_ids = encoder_res['input_ids']
            token_type_ids = encoder_res['token_type_ids']
            attention_mask = encoder_res['attention_mask']

            batch_input_ids.append(input_ids)
            batch_token_type_ids.append(token_type_ids)
            batch_attention_masks.append(attention_mask)
            batch_entities_positions.append([pos1, pos2])
            batch_labels.append(int(label))


        if self.config.fixed_batch_length:
            pad_length = self.config.max_len
        else:
            pad_length = min(batch_max_len,self.config.max_len)

        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids,length=pad_length)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids,length=pad_length)).long()
        batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks,length=pad_length)).long()
        batch_entities_positions = torch.tensor(batch_entities_positions).long()
        batch_labels = torch.tensor(batch_labels).long()

        if batch_max_len > self.config.max_len:
            raise ValueError

        return batch_input_ids, batch_token_type_ids, batch_attention_masks,batch_labels,batch_entities_positions




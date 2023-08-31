# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2021/12/03
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/03: 
-------------------------------------------------
"""
import torch
from ipdb import set_trace
from torch.utils.data import Dataset
import numpy as np
from transformers import BertTokenizer

from config import BertConfig
from src.utils.function_utils import get_pos_feature
from src.dataset_utils.data_process_utils import sequence_padding



class RBERT_Dataset(Dataset):
    def __init__(self,config:BertConfig,sents,labels,tokenizer:BertTokenizer,label2id,max_len):
        '''
        这个是专门处理类似semeval 2010 数据集的情况
        :param config:
        :param sents:
        :param labels:
        :param tokenizer:
        :param label2id:
        :param max_len:
        '''
        super(RBERT_Dataset, self).__init__()
        self.config = config
        self.sents = sents
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len


    def __len__(self):
        return len(self.sents)
    def __getitem__(self, item):

        ent1,ent2,sent = self.sents[item]
        label = self.labels[item]
        return ent1,ent2,sent,label

    def collate_fn(self,features):
        '''
        在这里将数据转换为模型需要的数据类别
        这里针对的数据类别为semeval 2010 数据集
            一行三部分组成，实体1，实体2，句子
            configuration	elements	The system as described above has its greatest application in an arrayed configuration of antenna elements
            这里需要保证实体只出现在句子一次
        :param features:一个batch_size的输入数据
        :return:
        '''
        raw_text_li = []
        batch_input_ids = []
        batch_token_type_ids = []
        batch_labels = []
        batch_attention_masks = []
        batch_e1_mask = []
        batch_e2_mask = []

        batch_max_len = 0
        for ent1,ent2,sent,label in features:

            # sent 是word-level list: ['Feadeal','ABVDF','the',...]
            raw_text_li.append(sent)

            # 进行分词
            label = self.label2id.get(label)
            batch_labels.append(label)

            # 实体的第一个word
            e1_start = ent1.split(' ')[0] if ' ' in ent1 else ent1
            e2_start = ent2.split(' ')[0] if ' ' in ent2 else ent2

            # 获得两个实体在分词之后的index
            e1_idx = sent.index(e1_start)  # 实体1在此句子的位置
            e1_len = len(ent1.split(' '))
            e2_idx = sent.index(e2_start)  # 实体2在这个句子上的位置
            e2_len = len(ent2.split(' '))

            # 通过列表来修改数据集
            sent_li = list(sent)

            # 修正句子，添加$$和##

            sent = sent_li[:e1_idx] + ['$'] + sent_li[e1_idx:e1_idx +e1_len] + ['$'] + sent_li[e1_idx +e1_len:e2_idx]+['#']+sent_li[e2_idx:e2_idx+e2_len]+['#']+sent_li[e2_idx+e2_len:]
            tokenize_sent = self.tokenizer.tokenize(" ".join(sent))

            e1_start_idx = tokenize_sent.index('$')
            e1_end_idx = e1_start_idx+len(self.tokenizer.tokenize(ent1))+1
            e2_start_idx = tokenize_sent.index('#')
            e2_end_idx = e2_start_idx+len(self.tokenizer.tokenize(ent2))+1


            encode_res = self.tokenizer.encode_plus(tokenize_sent)

            input_ids = encode_res['input_ids']
            attention_mask = encode_res['attention_mask']
            token_type_ids = encode_res['token_type_ids']

            if len(input_ids) > batch_max_len:
                batch_max_len = len(input_ids)
            # 构建其对应的mask
            # 由于增加[CLS]这个token
            e1_start_idx += 1
            e1_end_idx += 1
            e2_start_idx += 1
            e2_end_idx += 1

            e1_mask = np.zeros(len(input_ids))
            e2_mask = np.zeros(len(input_ids))
            e1_mask[e1_start_idx:e1_end_idx + 1] = 1
            e2_mask[e2_start_idx:e2_end_idx + 1] = 1


            batch_input_ids.append(input_ids)
            batch_token_type_ids.append(token_type_ids)
            batch_attention_masks.append(attention_mask)
            batch_e1_mask.append(e1_mask)
            batch_e2_mask.append(e2_mask)



        #开始pad

        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()
        batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks)).long()
        batch_e1_mask = torch.tensor(sequence_padding(batch_e1_mask)).long()
        batch_e2_mask = torch.tensor(sequence_padding(batch_e2_mask)).long()
        batch_labels = torch.tensor(batch_labels).long()

        if batch_max_len>self.config.max_len:
            raise ValueError

        return batch_input_ids,batch_token_type_ids,batch_attention_masks,batch_e1_mask,batch_e2_mask,batch_labels


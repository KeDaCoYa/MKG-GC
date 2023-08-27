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

from src.dataset_util.base_dataset import globalpointer_collate, sequence_padding, tokenize_text, \
    tokenize_text_predicate

logger = logging.getLogger('main.glopointer_dataset')


class BertGlobalPointerDataset(Dataset):
    def __init__(self,data,tokenizer,config):
        '''

        :param data: 这个数据就是load_data的结果，以列表的形式存储
            [(raw_text,start_off,end_offset,entity_type_id),(),....]
        :param tokenizer:分词器
        :num_tags:表示需要识别的实体类别
        :param is_train:这个并没有体现出其作用
        '''

        super(BertGlobalPointerDataset,self).__init__()
        self.data = data
        self.config = config
        self.num_tags = config.num_gp_class
        self.max_len = config.max_len
        self.nums = len(data)
        self.tokenizer = tokenizer



    def __len__(self):
        return self.nums


    def encoder(self,item):
        '''
        这是对一个数据的encode
        :param item:
        :return:
        '''

        raw_text = item.text

        encoder_res = self.tokenizer.encode_plus(raw_text,truncation=True,max_length=self.max_len)
        input_ids = encoder_res['input_ids']
        token_type_ids = encoder_res['token_type_ids']
        attention_mask = encoder_res['attention_mask']
        return raw_text,input_ids,token_type_ids,attention_mask

    def collate(self,examples):
        '''
        这里开始处理数据，在DataLoader读取了一个batch的数据之后进行处理
        :param examples:
        :return:
        '''
        raw_text_list, batch_input_ids, batch_attention_mask, batch_labels, batch_token_type_ids = [], [], [], [], []


        globalpointer_label2id = self.config.globalpointer_label2id
        for item in examples:
            # encoder是主要的转换方法
            raw_text,input_ids, token_type_ids, attention_mask = self.encoder(item)

            labels = np.zeros((self.num_tags, self.max_len, self.max_len)) # 普通条件下是(1,seq_len,seq_len)
            # 这里是规整一下实体边界
            true_labels = item.labels

            globalpointer_collate(true_labels,labels,globalpointer_label2id)


            raw_text_list.append(raw_text)
            batch_input_ids.append(input_ids)
            batch_token_type_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)

            batch_labels.append(labels[:, :len(input_ids), :len(input_ids)])
        # 这里batch_input_ids,batch_segment_ids,batch_attention_mask都没有进行pad，这里需要进行pad，将长度进行统一补齐...
        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()
        batch_attention_mask = torch.tensor(sequence_padding(batch_attention_mask)).float()
        # batch_labels.shape = (batch_size,entity_type,seq_len,seq_len)
        batch_labels = torch.tensor(sequence_padding(batch_labels, seq_dims=3)).long()

        return raw_text_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels

    def tokenize_encoder_predicate(self,item):
        '''
        这是对一个数据(InputExample)的encoder
        :param item:
            raw_text: 这个就是未分词之前的word list
            true_labels:
        :return:
        '''

        raw_text = item.text

        subword_tokens,origin_to_subword_index = tokenize_text_predicate(self.tokenizer,item,self.max_len)

        #


        encoder_res = self.tokenizer.encode_plus(subword_tokens,truncation=True,max_length=self.max_len)
        subword_input_ids = encoder_res['input_ids']
        subword_token_type_ids = encoder_res['token_type_ids']
        subword_attention_mask = encoder_res['attention_mask']
        # 这里+1是因为bert会增加[CLS],[SEP]
        origin_to_subword_index = [x+1 for x in origin_to_subword_index]

        return raw_text,subword_input_ids,subword_token_type_ids,subword_attention_mask,origin_to_subword_index

    def tokenize_encoder(self,item):
        '''
        这是对一个数据(InputExample)的encoder
        :param item:
            raw_text: 这个就是未分词之前的word list
            true_labels:
        :return:
        '''

        raw_text = item.text

        subword_tokens,origin_to_subword_index = tokenize_text(self.tokenizer,item,self.max_len)

        #


        encoder_res = self.tokenizer.encode_plus(subword_tokens,truncation=True,max_length=self.max_len)
        subword_input_ids = encoder_res['input_ids']
        subword_token_type_ids = encoder_res['token_type_ids']
        subword_attention_mask = encoder_res['attention_mask']
        # 这里+1是因为bert会增加[CLS],[SEP]
        origin_to_subword_index = [x+1 for x in origin_to_subword_index]

        return raw_text,subword_input_ids,subword_token_type_ids,subword_attention_mask,origin_to_subword_index

    def collate_tokenize(self, examples):
        '''
        这里开始处理数据，在DataLoader读取了一个batch的数据之后进行处理
        :param examples:
        :return:
        '''
        raw_text_list = []
        batch_subword_input_ids = []  # 这是针对token-level，这里数据一般
        batch_subword_attention_masks = []

        batch_subword_token_type_ids = []

        origin_to_subword_indexs = []
        batch_globalpointer_labels = []

        # subword之后的值
        batch_subword_max_len = 0

        globalpointer_label2id = self.config.globalpointer_label2id
        for item in examples:
            true_labels = item.labels
            # encoder是主要的转换方法
            raw_text, subword_input_ids, subword_token_type_ids, subword_attention_mask, origin_to_subword_index = self.tokenize_encoder(item)
            if sum(subword_attention_mask) - 2 > batch_subword_max_len:
                batch_subword_max_len = sum(subword_attention_mask) - 2


            # 截断长度
            raw_text = raw_text[:len(origin_to_subword_index)]
            true_labels = true_labels[:len(origin_to_subword_index)]

            # 普通条件下是(1,seq_len,seq_len)
            # 这是globalpointer需要的label

            gp_labels = np.zeros((self.num_tags, self.max_len, self.max_len))
            globalpointer_collate(true_labels, gp_labels, globalpointer_label2id)

            raw_text_list.append(raw_text)
            origin_to_subword_indexs.append(origin_to_subword_index)
            batch_subword_input_ids.append(subword_input_ids)
            batch_subword_token_type_ids.append(subword_token_type_ids)

            batch_subword_attention_masks.append(subword_attention_mask)
            # 这里按照真实长度+2截断就可以了，
            batch_globalpointer_labels.append(gp_labels[:, :len(raw_text), :len(raw_text)])

        if self.config.fixed_batch_length:
            pad_length = self.max_len
        else:
            pad_length = min(batch_subword_max_len, self.max_len)
        # 也需要对origin_to_subword_index进行检查
        new_origin_to_subword_indexs = []
        for i, subword_index in enumerate(origin_to_subword_indexs):
            new_index = []
            for ele in subword_index:
                if ele >= pad_length:
                    break
                else:
                    new_index.append(ele)
            new_origin_to_subword_indexs.append(new_index)
        input_true_length = [len(x) for x in new_origin_to_subword_indexs]
        input_true_length = torch.tensor(input_true_length).long()

        # 这里batch_input_ids,batch_segment_ids,batch_attention_mask都没有进行pad，这里需要进行pad，将长度进行统一补齐...

        batch_subword_input_ids = torch.tensor(sequence_padding(batch_subword_input_ids,length=pad_length)).long()
        batch_origin_to_subword_indexs = torch.tensor(sequence_padding(new_origin_to_subword_indexs,length=pad_length)).long()

        batch_subword_token_type_ids = torch.tensor(sequence_padding(batch_subword_token_type_ids,length=pad_length)).long()
        batch_subword_attention_masks = torch.tensor(sequence_padding(batch_subword_attention_masks,length=pad_length)).float()

        batch_globalpointer_labels = torch.tensor(sequence_padding(batch_globalpointer_labels, seq_dims=3,length=pad_length)).long()



        return raw_text_list, batch_globalpointer_labels,batch_subword_input_ids,batch_subword_token_type_ids,batch_subword_attention_masks,batch_origin_to_subword_indexs,None,input_true_length
    def collate_predicate(self, examples):
        '''
        这里开始处理数据，在DataLoader读取了一个batch的数据之后进行处理
        :param examples:
        :return:
        '''
        raw_text_list = []
        batch_subword_input_ids = []  # 这是针对token-level，这里数据一般
        batch_subword_attention_masks = []

        batch_subword_token_type_ids = []

        origin_to_subword_indexs = []



        batch_subword_max_len = 0


        for item in examples:


            raw_text, subword_input_ids, subword_token_type_ids, subword_attention_mask, origin_to_subword_index = self.tokenize_encoder_predicate(item)
            if sum(subword_attention_mask) - 2 > batch_subword_max_len:
                batch_subword_max_len = sum(subword_attention_mask) - 2


            # 截断长度
            raw_text = raw_text[:len(origin_to_subword_index)]


            # 普通条件下是(1,seq_len,seq_len)
            # 这是globalpointer需要的label




            raw_text_list.append(raw_text)
            origin_to_subword_indexs.append(origin_to_subword_index)
            batch_subword_input_ids.append(subword_input_ids)
            batch_subword_token_type_ids.append(subword_token_type_ids)

            batch_subword_attention_masks.append(subword_attention_mask)
            # 这里按照真实长度+2截断就可以了，

        # 这里batch_input_ids,batch_segment_ids,batch_attention_mask都没有进行pad，这里需要进行pad，将长度进行统一补齐...

        batch_subword_input_ids = torch.tensor(sequence_padding(batch_subword_input_ids)).long()
        batch_subword_token_type_ids = torch.tensor(sequence_padding(batch_subword_token_type_ids)).long()
        batch_subword_attention_masks = torch.tensor(sequence_padding(batch_subword_attention_masks)).float()



        max_len = self.max_len
        if batch_subword_max_len > self.max_len:
            batch_subword_input_ids = torch.tensor(batch_subword_input_ids).long()[:, max_len]
            batch_subword_token_type_ids = torch.tensor(batch_subword_token_type_ids).long()[:, max_len]
            batch_subword_attention_masks = torch.tensor(batch_subword_attention_masks).long()[:, max_len]



        return raw_text_list,batch_subword_input_ids,batch_subword_token_type_ids,batch_subword_attention_masks,origin_to_subword_indexs

    def __getitem__(self, item):
        return self.data[item]







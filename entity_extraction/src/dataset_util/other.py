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
import torch
from torch.utils.data import Dataset

from src.dataset_util.base_dataset import sequence_padding


class NormlMyDataset(Dataset):
    def __init__(self, features):
        '''
            这是针对固定长度
            在这里将所有数据转变为tensor，可以直接使用
        :param data:
        '''
        super(NormlMyDataset, self).__init__()
        self.nums = len(features)
        # 因为这里一次只取一个，所以一个tensor就行了...
        self.tokens_ids = [torch.tensor(example.token_ids).long() for example in features]

        self.labels = [torch.tensor(example.labels).long() for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {
            'token_ids': self.tokens_ids[index],

            'labels': self.labels[index],

        }
        return data


class NormlMyDataset_dynamic(Dataset):
    def __init__(self, features):
        '''
            在这里将所有数据转变为tensor，可以直接使用
        :param data:
        '''
        super(NormlMyDataset_dynamic, self).__init__()
        self.nums = len(features)
        self.data = features

    def __len__(self):
        return self.nums

    def collate_fn(self, features):
        '''
        这个函数用于DataLoader，一次处理一个batch的数据

        :param features: 这个就是一个InputExample
        :return:
        '''

        raw_text_list, batch_input_ids, batch_attention_masks, batch_start_ids, batch_end_ids, batch_token_type_ids = [], [], [], [], [], []

        for item in features:
            # encoder是主要的转换方法
            # 这里已经经过了encode_plus

            raw_text, input_ids, token_type_ids, attention_mask = self.encoder(item)

            start_ids = [0] * len(raw_text)
            end_ids = [0] * len(raw_text)

            labels = item.labels

            # 开始根据BIO的标注模式转变为ＳＰＡＮ
            start_index = 0

            while start_index < len(raw_text):

                if labels[start_index] == 'B':
                    start_ids[start_index] = 1
                    start_index += 1
                    while start_index < len(raw_text) and labels[start_index] == 'I':
                        start_index += 1
                    if start_index == len(raw_text) or labels[start_index] == 'O':
                        end_ids[start_index - 1] = 1
                else:
                    start_index += 1

            start_ids = [0] + start_ids + [0]
            end_ids = [0] + end_ids + [0]

            raw_text_list.append(raw_text)
            batch_input_ids.append(input_ids)
            batch_token_type_ids.append(token_type_ids)
            batch_attention_masks.append(attention_mask)
            # 这里按照实际举例进行一个slice
            batch_start_ids.append(start_ids)
            batch_end_ids.append(end_ids)

        # 这里batch_input_ids,batch_segment_ids,batch_attention_mask都没有进行pad，这里需要进行pad，将长度进行统一补齐...

        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()
        batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks)).float()
        # batch_labels.shape = (batch_size,entity_type,seq_len,seq_len)
        batch_start_ids = torch.tensor(sequence_padding(batch_start_ids)).long()
        batch_end_ids = torch.tensor(sequence_padding(batch_start_ids)).long()

        return raw_text_list, batch_input_ids, batch_attention_masks, batch_token_type_ids, batch_start_ids, batch_end_ids

    def __getitem__(self, index):

        return self.data[index]

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
                   2022年3月28日：这里面代码大量都重复使用....
-------------------------------------------------
"""

from ipdb import set_trace
import logging

import torch
from torch.utils.data import Dataset

from config import BertConfig
from src.dataset_util.base_dataset import sequence_padding, tokenize_text, tokenize_text_predicate

logger = logging.getLogger('main.span_dataset')

class BertSpanDataset(Dataset):
    def __init__(self, features):
        '''
            在这里将所有数据转变为tensor，可以直接使用
        :param data:
        '''
        super(BertSpanDataset, self).__init__()
        self.nums = len(features)
        # 因为这里一次只取一个，所以一个tensor就行了...
        self.tokens_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks).long() for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]
        self.start_ids = [torch.tensor(example.start_ids).long() for example in features]
        self.end_ids = [torch.tensor(example.end_ids).long() for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {
            'token_ids': self.tokens_ids[index],
            'attention_masks': self.attention_masks[index],
            'token_type_ids': self.token_type_ids[index],
            'start_ids': self.start_ids[index],
            'end_ids': self.end_ids[index],

        }
        return data


class BertSpanDataset_dynamic(Dataset):
    def __init__(self, config: BertConfig, data, tokenizer, is_train=True):

        super(BertSpanDataset_dynamic, self).__init__()
        self.nums = len(data)

        # 这里的data就是InputExamples，格式为
        # ipdb> train_examples[0].text
        # ['Identification', 'of', 'APC2', ',', 'a', 'homologue', 'of', 'the', 'adenomatous', 'polyposis', 'coli', 'tumour', 'suppressor', '.']
        # train_examples[0].labels
        # ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'O', 'O']
        self.data = data
        self.config = config
        self.num_tags = config.num_span_class
        self.max_len = config.max_len
        self.nums = len(data)
        self.tokenizer = tokenizer
        self.label2id = config.span_label2id
        self.is_train = is_train

    def __len__(self):
        return self.nums

    def encoder(self, item):
        '''
        这是对一个数据的encode
        :param item:
        :return:
        '''
        if self.is_train:
            raw_text = item.text

            encoder_res = self.tokenizer.encode_plus(raw_text, truncation=True, max_length=self.max_len)
            input_ids = encoder_res['input_ids']
            token_type_ids = encoder_res['token_type_ids']
            attention_mask = encoder_res['attention_mask']
            return raw_text, input_ids, token_type_ids, attention_mask
        else:
            pass

    def label_decode(self, labels, span_label2id, mode=1):
        '''
        根据BIO标注，生成span需要的数据集
        :param actual_lens:句子的实际长度
        :param labels:这就是句子的实际label
        :param mode: 0表示只有一种实体类别，1表示有多种实体类别
        :return:
        '''
        actual_lens = len(labels)
        start_ids = [0] * actual_lens
        end_ids = [0] * actual_lens
        if mode == 0:
            start_index = 0
            while start_index < actual_lens:
                if labels[start_index] == 'O':
                    start_index += 1
                    continue

                BIO_format = labels[start_index]
                entity_type_id = 1

                if BIO_format == 'B':  # 找到一个实体
                    if start_index < actual_lens:  # 给start_ids标注这个实体
                        start_ids[start_index] = entity_type_id
                        start_index += 1

                    while start_index < actual_lens and labels[start_index] == 'I':
                        start_index += 1

                    if start_index == actual_lens or labels[start_index] == 'O':
                        end_ids[start_index - 1] = entity_type_id
                else:
                    start_index += 1
        else:
            # 开始根据BIO的标注模式转变为ＳＰＡＮ
            start_index = 0
            while start_index < actual_lens:
                if labels[start_index] == 'O':
                    start_index += 1
                    continue

                BIO_format, entity_type = labels[start_index].split('-')
                entity_type_id = span_label2id[entity_type]

                if BIO_format == 'B':  # 找到一个实体
                    if start_index < actual_lens:  # 给start_ids标注这个实体
                        start_ids[start_index] = entity_type_id
                        start_index += 1

                    while start_index < actual_lens and labels[start_index].split('-')[0] == 'I':
                        start_index += 1

                    if start_index == actual_lens or labels[start_index].split('-')[0] == 'O':
                        end_ids[start_index - 1] = entity_type_id
                else:
                    start_index += 1

        return start_ids, end_ids

    def tokenize_encoder(self, item):
        '''
        这是对一个数据(InputExample)的encoder
        :param item:
            raw_text: 这个就是未分词之前的word list
            true_labels:
        :return:
        '''
        if self.is_train:
            raw_text = item.text

            subword_tokens, origin_to_subword_index = tokenize_text(self.tokenizer, item, self.max_len)

            encoder_res = self.tokenizer.encode_plus(subword_tokens, truncation=True, max_length=self.max_len)
            subword_input_ids = encoder_res['input_ids']
            subword_token_type_ids = encoder_res['token_type_ids']
            subword_attention_mask = encoder_res['attention_mask']
            origin_to_subword_index = [x + 1 for x in origin_to_subword_index]

            return raw_text, subword_input_ids, subword_token_type_ids, subword_attention_mask, origin_to_subword_index
        else:
            pass

    def tokenize_encoder_predicate(self, item):
        """
        这是对一个数据(InputExample)的encoder
        :param item:
            raw_text: 这个就是未分词之前的word list
            true_labels:
        :return:
        """
        if self.is_train:
            raw_text = item.text

            subword_tokens, origin_to_subword_index = tokenize_text_predicate(self.tokenizer, item, self.max_len)

            encoder_res = self.tokenizer.encode_plus(subword_tokens, truncation=True, max_length=self.max_len)
            subword_input_ids = encoder_res['input_ids']
            subword_token_type_ids = encoder_res['token_type_ids']
            subword_attention_mask = encoder_res['attention_mask']
            origin_to_subword_index = [x + 1 for x in origin_to_subword_index]

            return raw_text, subword_input_ids, subword_token_type_ids, subword_attention_mask, origin_to_subword_index
        else:
            pass

    def collate_fn_predicate(self, features):
        '''
            专用于span的predicate
         '''

        raw_text_list, batch_input_ids, batch_attention_masks, batch_token_type_ids = [], [], [], []
        span_label2id = self.config.span_label2id

        origin_to_subword_indexs = []

        batch_subword_max_len = 0
        for item in features:
            # encoder是主要的转换方法
            # 这里已经经过了encode_plus

            raw_text, subword_input_ids, subword_token_type_ids, subword_attention_mask, origin_to_subword_index = self.tokenize_encoder_predicate(
                item)

            ## 这里需要截断，因为因为subword可能太长，超过了max_len,需要截断，因此true label和raw_text都要截断
            # 这里硬阶段，可能导致bug，某些实体被截断
            raw_text = raw_text[:len(origin_to_subword_index)]

            subword_len = sum(subword_attention_mask)
            if subword_len - 2 > batch_subword_max_len:
                batch_subword_max_len = subword_len - 2

            raw_text_list.append(raw_text)
            batch_input_ids.append(subword_input_ids)
            batch_token_type_ids.append(subword_token_type_ids)
            batch_attention_masks.append(subword_attention_mask)
            origin_to_subword_indexs.append(origin_to_subword_index)
            # 这里按照实际举例进行一个slice
        # 也需要对origin_to_subword_index进行检查
        new_origin_to_subword_indexs = []
        for i, subword_index in enumerate(origin_to_subword_indexs):
            new_index = []
            for ele in subword_index:
                if ele >= self.max_len:
                    break
                else:
                    new_index.append(ele)
            new_origin_to_subword_indexs.append(new_index)
        # 这里batch_input_ids,batch_segment_ids,batch_attention_mask都没有进行pad，这里需要进行pad，将长度进行统一补齐...
        input_true_length = [len(x) for x in new_origin_to_subword_indexs]
        input_true_length = torch.tensor(input_true_length).long()
        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()
        batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks)).float()

        max_len = self.max_len
        if batch_subword_max_len > self.max_len:
            batch_input_ids = torch.tensor(batch_input_ids).long()[:, max_len]
            batch_token_type_ids = torch.tensor(batch_token_type_ids).long()[:, max_len]
            batch_attention_masks = torch.tensor(batch_attention_masks).long()[:, max_len]

        return raw_text_list, batch_input_ids, batch_attention_masks, batch_token_type_ids, new_origin_to_subword_indexs, input_true_length

    def collate_fn_tokenize(self, features):
        '''
        这个函数用于DataLoader，一次处理一个batch的数据

        :param features: 这个就是一个InputExample
        :return: start_ids,end_ids是对应原数据，不进行任何修改
        '''

        raw_text_list, batch_input_ids, batch_attention_masks, batch_token_type_ids, batch_start_ids, batch_end_ids = [], [], [], [], [], []
        span_label2id = self.config.span_label2id
        mode = 0
        if len(span_label2id) > 2:
            mode = 1
        if self.config.ner_dataset_name in ['jnlpba-dna', 'jnlpba-rna', 'jnlpba-celltype', 'jnlpba-cellline']:
            mode = 1

        origin_to_subword_indexs = []

        batch_subword_max_len = 0
        for item in features:
            # encoder是主要的转换方法
            # 这里已经经过了encode_plus

            raw_text, subword_input_ids, subword_token_type_ids, subword_attention_mask, origin_to_subword_index = self.tokenize_encoder(
                item)
            labels = item.labels



            start_ids, end_ids = self.label_decode(labels, span_label2id=span_label2id, mode=mode)


            ## 这里需要截断，因为因为subword可能太长，超过了max_len,需要截断，因此true label和raw_text都要截断
            # todo:这里采用硬截斷，可能导致bug，某些实体被截断
            raw_text = raw_text[:len(origin_to_subword_index)]
            start_ids = start_ids[:len(origin_to_subword_index)]
            end_ids = end_ids[:len(origin_to_subword_index)]
            subword_len = sum(subword_attention_mask)
            if subword_len - 2 > batch_subword_max_len:
                batch_subword_max_len = subword_len - 2

            raw_text_list.append(raw_text)
            batch_input_ids.append(subword_input_ids)
            batch_token_type_ids.append(subword_token_type_ids)
            batch_attention_masks.append(subword_attention_mask)
            origin_to_subword_indexs.append(origin_to_subword_index)
            # 这里按照实际举例进行一个slice

            # 之前忘记补充[CLS]对应的label

            # start_ids = [0]+start_ids
            # end_ids = [0]+end_ids
            batch_start_ids.append(start_ids)
            batch_end_ids.append(end_ids)


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

        # 这里batch_input_ids,batch_segment_ids,batch_attention_mask都没有进行pad，这里需要进行pad，将长度进行统一补齐...

        # 这个是记录每一句话的真实单词个数，tokenize之后会产生许多token
        input_true_length = [len(x) for x in new_origin_to_subword_indexs]
        input_true_length = torch.tensor(input_true_length).long()
        batch_origin_to_subword_indexs = torch.tensor(
            sequence_padding(new_origin_to_subword_indexs, length=pad_length)).long()
        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids, length=pad_length)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=pad_length)).long()
        batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks, length=pad_length)).float()
        batch_start_ids = torch.tensor(sequence_padding(batch_start_ids, length=pad_length)).long()
        batch_end_ids = torch.tensor(sequence_padding(batch_end_ids, length=pad_length)).long()

        return raw_text_list, batch_input_ids, batch_attention_masks, batch_token_type_ids, batch_start_ids, batch_end_ids, batch_origin_to_subword_indexs, input_true_length


    def collate_fn(self, features):
        '''
        这个函数用于DataLoader，一次处理一个batch的党史数据

        :param features: 这个就是一个InputExample
        :return:
        '''

        raw_text_list, batch_input_ids, batch_attention_masks, batch_start_ids, batch_end_ids, batch_token_type_ids = [], [], [], [], [], []
        span_label2id = self.config.span_label2id
        mode = 0

        for item in features:
            # encoder是主要的转换方法
            # 这里已经经过了encode_plus

            raw_text, input_ids, token_type_ids, attention_mask = self.encoder(item)
            labels = item.labels
            if len(span_label2id) > 2:
                mode = 1
            start_ids, end_ids = self.label_decode(labels, span_label2id=span_label2id, mode=mode)

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

        # 这里的length要+2，因此

        batch_start_ids = torch.tensor(sequence_padding(batch_start_ids)).long()
        batch_end_ids = torch.tensor(sequence_padding(batch_end_ids)).long()

        return raw_text_list, batch_input_ids, batch_attention_masks, batch_token_type_ids, batch_start_ids, batch_end_ids
    def multi_collate_fn_tokenize(self, features):

        """
        这是对待mutli task
        这个函数用于DataLoader，一次处理一个batch的数据

        :param features: 这个就是一个InputExample
        :return: start_ids,end_ids是对应原数据，不进行任何修改
        """

        raw_text_list, batch_input_ids, batch_attention_masks, batch_token_type_ids, batch_start_ids, batch_end_ids = [], [], [], [], [], []
        entity_type_ids = []
        span_label2id = self.config.span_label2id
        mode = 0
        origin_to_subword_indexs = []

        batch_subword_max_len = 0
        for item in features:
            # encoder是主要的转换方法
            # 这里已经经过了encode_plus

            raw_text, subword_input_ids, subword_token_type_ids, subword_attention_mask, origin_to_subword_index = self.tokenize_encoder(
                item)
            labels = item.labels
            entity_type_id = item.entity_type_id

            if len(span_label2id) > 2:
                mode = 1
            start_ids, end_ids = self.label_decode(labels, span_label2id=span_label2id, mode=mode)

            ## 这里需要截断，因为因为subword可能太长，超过了max_len,需要截断，因此true label和raw_text都要截断
            # todo:这里采用硬截斷，可能导致bug，某些实体被截断
            raw_text = raw_text[:len(origin_to_subword_index)]
            start_ids = start_ids[:len(origin_to_subword_index)]
            end_ids = end_ids[:len(origin_to_subword_index)]
            subword_len = sum(subword_attention_mask)
            if subword_len - 2 > batch_subword_max_len:
                batch_subword_max_len = subword_len - 2

            raw_text_list.append(raw_text)
            batch_input_ids.append(subword_input_ids)
            batch_token_type_ids.append(subword_token_type_ids)
            batch_attention_masks.append(subword_attention_mask)
            origin_to_subword_indexs.append(origin_to_subword_index)
            # 这里按照实际举例进行一个slice
            batch_start_ids.append(start_ids)

            batch_end_ids.append(end_ids)
            # todo: 一般只会使用fixed_batch_length,所以这里存在bug
            entity_type_ids.append([entity_type_id]*self.max_len)

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

        # 这里batch_input_ids,batch_segment_ids,batch_attention_mask都没有进行pad，这里需要进行pad，将长度进行统一补齐...

        # 这个是记录每一句话的真实单词个数，tokenize之后会产生许多token
        input_true_length = [len(x) for x in new_origin_to_subword_indexs]
        input_true_length = torch.tensor(input_true_length).long()
        batch_origin_to_subword_indexs = torch.tensor(
            sequence_padding(new_origin_to_subword_indexs, length=pad_length)).long()
        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids, length=pad_length)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=pad_length)).long()
        batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks, length=pad_length)).float()
        batch_start_ids = torch.tensor(sequence_padding(batch_start_ids, length=pad_length)).long()
        batch_end_ids = torch.tensor(sequence_padding(batch_end_ids, length=pad_length)).long()
        batch_entity_type_ids = torch.tensor(entity_type_ids).long()

        return raw_text_list, batch_input_ids, batch_attention_masks, batch_token_type_ids, batch_start_ids, batch_end_ids, batch_origin_to_subword_indexs, input_true_length,batch_entity_type_ids

    def multi_collate_fn_predicate(self, features):
        """
            专用于abstract text的span的predicate
         """

        raw_text_list, batch_input_ids, batch_attention_masks, batch_token_type_ids = [], [], [], []
        span_label2id = self.config.span_label2id

        origin_to_subword_indexs = []

        batch_subword_max_len = 0
        for item in features:
            # encoder是主要的转换方法
            # 这里已经经过了encode_plus

            raw_text, subword_input_ids, subword_token_type_ids, subword_attention_mask, origin_to_subword_index = self.tokenize_encoder_predicate(
                item)

            ## 这里需要截断，因为因为subword可能太长，超过了max_len,需要截断，因此true label和raw_text都要截断
            # 这里硬阶段，可能导致bug，某些实体被截断
            raw_text = raw_text[:len(origin_to_subword_index)]

            subword_len = sum(subword_attention_mask)
            if subword_len - 2 > batch_subword_max_len:
                batch_subword_max_len = subword_len - 2

            raw_text_list.append(raw_text)
            batch_input_ids.append(subword_input_ids)
            batch_token_type_ids.append(subword_token_type_ids)
            batch_attention_masks.append(subword_attention_mask)
            origin_to_subword_indexs.append(origin_to_subword_index)
            # 这里按照实际举例进行一个slice
        # 也需要对origin_to_subword_index进行检查
        new_origin_to_subword_indexs = []
        for i, subword_index in enumerate(origin_to_subword_indexs):
            new_index = []
            for ele in subword_index:
                if ele >= self.max_len:
                    break
                else:
                    new_index.append(ele)
            new_origin_to_subword_indexs.append(new_index)
        # 这里batch_input_ids,batch_segment_ids,batch_attention_mask都没有进行pad，这里需要进行pad，将长度进行统一补齐...
        input_true_length = [len(x) for x in new_origin_to_subword_indexs]
        input_true_length = torch.tensor(input_true_length).long()
        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()
        batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks)).float()
        batch_entity_type_ids = torch.zeros_like(batch_input_ids)
        max_len = self.max_len
        if batch_subword_max_len > self.max_len:
            batch_input_ids = torch.tensor(batch_input_ids).long()[:, max_len]
            batch_token_type_ids = torch.tensor(batch_token_type_ids).long()[:, max_len]
            batch_attention_masks = torch.tensor(batch_attention_masks).long()[:, max_len]

        return raw_text_list, batch_input_ids, batch_attention_masks, batch_token_type_ids, new_origin_to_subword_indexs, input_true_length,batch_entity_type_ids
    def __getitem__(self, index):

        return self.data[index]

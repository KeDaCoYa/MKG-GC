# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :   这里对候选实体对进行特殊符号包住，相当于额外增加特殊符号....
                    例如<s1></e1>,<s2></e2>等特殊符号来包含
   Author :        kedaxia
   date：          2021/12/22
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/22: 
-------------------------------------------------
"""

import logging
from ipdb import set_trace

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import BertTokenizer


from config import BertConfig
from src.utils.function_utils import get_pos_feature
from src.dataset_utils.data_process_utils import sequence_padding

logger = logging.getLogger('main.entity_marker')


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()


class MTBDataset(Dataset):
    def __init__(self,examples,config:BertConfig,tokenizer:BertTokenizer,label2id,device):
        '''
        使用这个读取数据的时候，所有需要的数据都会放在同一个文件之中，
        :param config:
        :param sents:
        :param tokenizer:
        :param label2id:
        :param max_len:
        '''
        super(MTBDataset, self).__init__()
        self.config = config
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = config.max_len
        self.device = device


    def __len__(self):
        return len(self.examples)
    def __getitem__(self, item):

        return self.examples[item]

    def collate_fn_predicate(self, examples):
        '''
        专用于model的predicate
        input: <cls>sent1<sep>sent2<sep>
        - sent1和sent2为实体1和2所在的位置

        :param examples:一个batch_size的输入数据,
        :return:
        '''
        raw_text_li_a = []
        raw_text_li_b = []
        batch_input_ids = []
        batch_token_type_ids = []
        batch_attention_masks = []
        batch_e1_mask = []
        batch_e2_mask = []

        batch_max_len = 0

        for exam in examples:
            text_a = exam.text_a
            text_b = exam.text_b
            raw_text_li_a.append(text_a)
            raw_text_li_b.append(text_b)

            tokenize_text_a = self.tokenizer.tokenize(text_a)
            tokenize_text_b = self.tokenizer.tokenize(text_b)
            # 如果长度过长，那么开始裁剪长度
            if len(tokenize_text_a) + len(tokenize_text_b) > self.config.max_len - self.config.total_special_toks:
                logger.info('长度为{},开始裁剪长度'.format(len(tokenize_text_a) + len(tokenize_text_b)))

                res = self._process_seq_len(text_a, text_b)
                if res:
                    text_a, text_b = res
                else:
                    logger.warning('发生了裁剪死循环...')
                    continue
                tokenize_text_a = self.tokenizer.tokenize(text_a)
                tokenize_text_b = self.tokenizer.tokenize(text_b)
                logger.info('裁剪之后的长度为{}'.format(len(tokenize_text_a) + len(tokenize_text_b)))
                if len(tokenize_text_a) + len(tokenize_text_b) > self.config.max_len - self.config.total_special_toks:
                    logger.warning("放弃这条数据,数据太长了....")
                    continue

            encode_res = self.tokenizer.encode_plus(text_a, text_b)

            input_ids = encode_res['input_ids']
            attention_mask = encode_res['attention_mask']
            token_type_ids = encode_res['token_type_ids']

            if len(input_ids) > batch_max_len:
                batch_max_len = len(input_ids)
            # 构建其对应的mask
            # 由于增加[CLS]这个token

            e1_start_idx = input_ids.index(self.config.ent1_start_tag_id)
            e1_end_idx = input_ids.index(self.config.ent1_end_tag_id)
            e2_start_idx = input_ids.index(self.config.ent2_start_tag_id)
            e2_end_idx = input_ids.index(self.config.ent2_end_tag_id)

            e1_mask = np.zeros(len(input_ids))
            e2_mask = np.zeros(len(input_ids))
            e1_mask[e1_start_idx:e1_end_idx + 1] = 1
            e2_mask[e2_start_idx:e2_end_idx + 1] = 1

            batch_input_ids.append(input_ids)
            batch_token_type_ids.append(token_type_ids)
            batch_attention_masks.append(attention_mask)
            batch_e1_mask.append(e1_mask)
            batch_e2_mask.append(e2_mask)
        # 开始pad
        if self.config.fixed_batch_length:
            pad_len = self.config.max_len
        else:
            pad_len = min(self.config.max_len, batch_max_len)
        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids, length=pad_len), device=self.device).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=pad_len),
                                            device=self.device).long()
        batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks, length=pad_len),
                                             device=self.device).long()
        batch_e1_mask = torch.tensor(sequence_padding(batch_e1_mask, length=pad_len), device=self.device).long()
        batch_e2_mask = torch.tensor(sequence_padding(batch_e2_mask, length=pad_len), device=self.device).long()

        return batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_e1_mask, batch_e2_mask
    def collate_fn(self,examples):
        '''
        在这里将数据转换为模型需要的数据格式
        input: <cls>sent1<sep>sent2<sep>
        - sent1和sent2为实体1和2所在的位置

        :param examples:一个batch_size的输入数据,
        :return:
        '''
        raw_text_li_a = []
        raw_text_li_b = []
        batch_input_ids = []
        batch_token_type_ids = []
        batch_labels = []
        batch_attention_masks = []
        batch_e1_mask = []
        batch_e2_mask = []

        batch_max_len = 0

        for exam in examples:
            text_a = exam.text_a
            text_b = exam.text_b

            label = exam.label

            raw_text_li_a.append(text_a)
            raw_text_li_b.append(text_b)

            label = self.label2id.get(label)

            tokenize_text_a = self.tokenizer.tokenize(text_a)
            tokenize_text_b = self.tokenizer.tokenize(text_b)
            # 如果长度过长，那么开始裁剪长度
            if len(tokenize_text_a) + len(tokenize_text_b) > self.config.max_len - self.config.total_special_toks:
                logger.info('长度为{},开始裁剪长度'.format(len(tokenize_text_a) + len(tokenize_text_b) ))

                res = self._process_seq_len(text_a,text_b)
                if res:
                    text_a, text_b = res
                else:
                    logger.warning('发生了裁剪死循环...')
                    continue
                tokenize_text_a = self.tokenizer.tokenize(text_a)
                tokenize_text_b = self.tokenizer.tokenize(text_b)
                logger.info('裁剪之后的长度为{}'.format(len(tokenize_text_a)+len(tokenize_text_b)))
                if len(tokenize_text_a) + len(tokenize_text_b) > self.config.max_len - self.config.total_special_toks:
                    logger.warning("放弃这条数据,数据太长了....")
                    continue

            batch_labels.append(label)
            encode_res = self.tokenizer.encode_plus(text_a,text_b)

            input_ids = encode_res['input_ids']
            attention_mask = encode_res['attention_mask']
            token_type_ids = encode_res['token_type_ids']

            if len(input_ids) > batch_max_len:
                batch_max_len = len(input_ids)
            # 构建其对应的mask
            # 由于增加[CLS]这个token

            e1_start_idx  = input_ids.index(self.config.ent1_start_tag_id)
            e1_end_idx  = input_ids.index(self.config.ent1_end_tag_id)
            e2_start_idx  = input_ids.index(self.config.ent2_start_tag_id)
            e2_end_idx  = input_ids.index(self.config.ent2_end_tag_id)


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
        if self.config.fixed_batch_length:
            pad_len = self.config.max_len
        else:
            pad_len = min(self.config.max_len,batch_max_len)
        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids,length=pad_len),device=self.device).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids,length=pad_len),device=self.device).long()
        batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks,length=pad_len),device=self.device).long()
        batch_e1_mask = torch.tensor(sequence_padding(batch_e1_mask,length=pad_len),device=self.device).long()
        batch_e2_mask = torch.tensor(sequence_padding(batch_e2_mask,length=pad_len),device=self.device).long()
        batch_labels = torch.tensor(batch_labels,device=self.device).long()

        return batch_input_ids,batch_token_type_ids,batch_attention_masks,batch_e1_mask,batch_e2_mask,batch_labels

    def _process_seq_len(self, text_a, text_b, total_special_toks=3):
        """
            裁切句子的方法，直接使用clinicalTransformer提供的方法
            This function is used to truncate sequences with len > max_seq_len
            Truncate strategy:
            1. find all the index for special tags
            3. count distances between leading word to first tag and second tag to last.
            first -1- tag1 entity tag2 -2- last
            4. pick the longest distance from (1, 2), if 1 remove first token, if 2 remove last token
            5. repeat until len is equal to max_seq_len
        """
        flag = True
        # 防止死循环的
        no_loop = 0

        while len(self.tokenizer.tokenize(text_a) + self.tokenizer.tokenize(text_b)) > (self.config.max_len - total_special_toks):

            if flag:
                text_a = self._truncate_helper(text_a)
            else:
                text_b = self._truncate_helper(text_b)

            flag = not flag
            no_loop += 1
            if no_loop>50:
                return

        return text_a, text_b


    def _truncate_helper(self,text):
        '''
        这是一个句子一个句子的找
        这里对原始的的text进行去除，并不是tokenize之后的....
        :param text:
        :return:
        '''
        tokens = text.split(" ")
        # 这是得到 word-level的index
        tags_li =  [self.config.ent1_start_tag,self.config.ent1_end_tag,self.config.ent2_start_tag ,  self.config.ent2_end_tag ]
        spec_tag_idx1, spec_tag_idx2 = [idx for (idx, tk) in enumerate(tokens) if tk.lower() in tags_li]
        start_idx, end_idx = 0, len(tokens) - 1
        truncate_space_head = spec_tag_idx1 - start_idx
        truncate_space_tail = end_idx - spec_tag_idx2

        if truncate_space_head == truncate_space_tail == 0: #这是表示如果实体1和实体2 都已经在句子的首尾，那么就不要继续删除了....
            return text

        if truncate_space_head > truncate_space_tail: # 如果离头更远，那么先抛弃头部的word...
            tokens.pop(0)
        else:
            tokens.pop(-1)

        return " ".join(tokens)



class InterMTBDataset(Dataset):
    def __init__(self,examples,config:BertConfig,tokenizer:BertTokenizer,label2id,device):
        '''
        使用这个读取数据的时候，所有需要的数据都会放在同一个文件之中，
        :param config:
        :param sents:
        :param tokenizer:
        :param label2id:
        :param max_len:
        '''
        super(InterMTBDataset, self).__init__()
        self.config = config
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = config.max_len
        self.device = device


    def __len__(self):
        return len(self.examples)
    def __getitem__(self, item):

        return self.examples[item]

    def collate_fn_predicate(self, examples):
        '''
        专用于model的predicate
        input: <cls>sent1<sep>sent2<sep>
        - sent1和sent2为实体1和2所在的位置

        :param examples:一个batch_size的输入数据,
        :return:
        '''
        raw_text_li_a = []
        raw_text_li_b = []

        batch_input_ids1 = []
        batch_token_type_ids1 = []
        batch_attention_masks1 = []

        batch_e1_mask = []
        batch_e2_mask = []

        batch_input_ids2 = []
        batch_token_type_ids2 = []
        batch_attention_masks2 = []
        batch_max_len = 0

        for exam in examples:
            text_a = exam.text_a
            text_b = exam.text_b
            raw_text_li_a.append(text_a)
            raw_text_li_b.append(text_b)

            tokenize_text_a = self.tokenizer.tokenize(text_a)
            tokenize_text_b = self.tokenizer.tokenize(text_b)
            # 如果长度过长，那么开始裁剪长度
            if len(tokenize_text_a) + len(tokenize_text_b) > self.config.max_len - self.config.total_special_toks:
                logger.info('长度为{},开始裁剪长度'.format(len(tokenize_text_a) + len(tokenize_text_b)))

                res = self._process_seq_len(text_a, text_b)
                if res:
                    text_a, text_b = res
                else:
                    logger.warning('发生了裁剪死循环...')
                    continue
                tokenize_text_a = self.tokenizer.tokenize(text_a)
                tokenize_text_b = self.tokenizer.tokenize(text_b)
                logger.info('裁剪之后的长度为{}'.format(len(tokenize_text_a) + len(tokenize_text_b)))
                if len(tokenize_text_a) + len(tokenize_text_b) > self.config.max_len - self.config.total_special_toks:
                    logger.warning("放弃这条数据,数据太长了....")
                    continue

            encode_res1 = self.tokenizer.encode_plus(text_a)
            encode_res2 = self.tokenizer.encode_plus(text_b)

            input_ids1 = encode_res1['input_ids']
            attention_mask1 = encode_res1['attention_mask']
            token_type_ids1 = encode_res1['token_type_ids']

            input_ids2 = encode_res2['input_ids']
            attention_mask2 = encode_res2['attention_mask']
            token_type_ids2 = encode_res2['token_type_ids']

            if len(input_ids1) > batch_max_len:
                batch_max_len1 = len(input_ids1)
            if len(input_ids2) > batch_max_len:
                batch_max_len2 = len(input_ids2)
            # 构建其对应的mask
            # 由于增加[CLS]这个token

            e1_start_idx = input_ids1.index(self.config.ent1_start_tag_id)
            e1_end_idx = input_ids1.index(self.config.ent1_end_tag_id)

            e2_start_idx = input_ids2.index(self.config.ent2_start_tag_id)
            e2_end_idx = input_ids2.index(self.config.ent2_end_tag_id)

            e1_mask = np.zeros(len(input_ids1))
            e2_mask = np.zeros(len(input_ids2))
            e1_mask[e1_start_idx:e1_end_idx + 1] = 1
            e2_mask[e2_start_idx:e2_end_idx + 1] = 1

            batch_input_ids1.append(input_ids1)
            batch_token_type_ids1.append(attention_mask1)
            batch_attention_masks1.append(token_type_ids1)

            batch_input_ids2.append(input_ids2)
            batch_token_type_ids2.append(attention_mask2)
            batch_attention_masks2.append(token_type_ids2)

            batch_e1_mask.append(e1_mask)
            batch_e2_mask.append(e2_mask)
        # 开始pad
        if self.config.fixed_batch_length:
            pad_len = self.config.max_len
        else:
            pad_len = min(self.config.max_len, batch_max_len)
        batch_input_ids1 = torch.tensor(sequence_padding(batch_input_ids1, length=pad_len), device=self.device).long()
        batch_token_type_ids1 = torch.tensor(sequence_padding(batch_token_type_ids1, length=pad_len),
                                            device=self.device).long()
        batch_attention_masks1 = torch.tensor(sequence_padding(batch_attention_masks1, length=pad_len),
                                             device=self.device).long()
        batch_input_ids2 = torch.tensor(sequence_padding(batch_input_ids2, length=pad_len), device=self.device).long()
        batch_token_type_ids2 = torch.tensor(sequence_padding(batch_token_type_ids2, length=pad_len),
                                             device=self.device).long()
        batch_attention_masks2 = torch.tensor(sequence_padding(batch_attention_masks2, length=pad_len),
                                              device=self.device).long()

        batch_e1_mask = torch.tensor(sequence_padding(batch_e1_mask, length=pad_len), device=self.device).long()
        batch_e2_mask = torch.tensor(sequence_padding(batch_e2_mask, length=pad_len), device=self.device).long()

        return batch_input_ids1, batch_token_type_ids1, batch_attention_masks1,batch_input_ids2, batch_token_type_ids2, batch_attention_masks2, batch_e1_mask, batch_e2_mask
    def collate_fn(self,examples):
        '''
        在这里将数据转换为模型需要的数据格式
        input: <cls>sent1<sep>sent2<sep>
        - sent1和sent2为实体1和2所在的位置

        :param examples:一个batch_size的输入数据,
        :return:
        '''
        raw_text_li_a = []
        raw_text_li_b = []
        batch_input_ids = []

        batch_input_ids1 = []
        batch_token_type_ids1 = []
        batch_attention_masks1 = []

        batch_e1_mask = []
        batch_e2_mask = []
        batch_labels = []
        batch_input_ids2 = []
        batch_token_type_ids2 = []
        batch_attention_masks2 = []

        batch_max_len = 0

        for exam in examples:
            text_a = exam.text_a
            text_b = exam.text_b

            label = exam.label

            raw_text_li_a.append(text_a)
            raw_text_li_b.append(text_b)

            label = self.label2id.get(label)

            tokenize_text_a = self.tokenizer.tokenize(text_a)
            tokenize_text_b = self.tokenizer.tokenize(text_b)
            # 如果长度过长，那么开始裁剪长度
            if len(tokenize_text_a) + len(tokenize_text_b) > self.config.max_len - self.config.total_special_toks:
                logger.info('长度为{},开始裁剪长度'.format(len(tokenize_text_a) + len(tokenize_text_b) ))

                res = self._process_seq_len(text_a,text_b)
                if res:
                    text_a, text_b = res
                else:
                    logger.warning('发生了裁剪死循环...')
                    continue
                tokenize_text_a = self.tokenizer.tokenize(text_a)
                tokenize_text_b = self.tokenizer.tokenize(text_b)
                logger.info('裁剪之后的长度为{}'.format(len(tokenize_text_a)+len(tokenize_text_b)))
                if len(tokenize_text_a) + len(tokenize_text_b) > self.config.max_len - self.config.total_special_toks:
                    logger.warning("放弃这条数据,数据太长了....")
                    continue


            encode_res1 = self.tokenizer.encode_plus(text_a)
            encode_res2 = self.tokenizer.encode_plus(text_b)

            input_ids1 = encode_res1['input_ids']
            attention_mask1 = encode_res1['attention_mask']
            token_type_ids1 = encode_res1['token_type_ids']

            input_ids2 = encode_res2['input_ids']
            attention_mask2 = encode_res2['attention_mask']
            token_type_ids2 = encode_res2['token_type_ids']

            if len(input_ids1) > batch_max_len:
                batch_max_len1 = len(input_ids1)
            if len(input_ids2) > batch_max_len:
                batch_max_len2 = len(input_ids2)
            # 构建其对应的mask
            # 由于增加[CLS]这个token

            e1_start_idx = input_ids1.index(self.config.ent1_start_tag_id)
            e1_end_idx = input_ids1.index(self.config.ent1_end_tag_id)

            e2_start_idx = input_ids2.index(self.config.ent2_start_tag_id)
            e2_end_idx = input_ids2.index(self.config.ent2_end_tag_id)

            e1_mask = np.zeros(len(input_ids1))
            e2_mask = np.zeros(len(input_ids2))
            e1_mask[e1_start_idx:e1_end_idx + 1] = 1
            e2_mask[e2_start_idx:e2_end_idx + 1] = 1

            batch_input_ids1.append(input_ids1)
            batch_token_type_ids1.append(attention_mask1)
            batch_attention_masks1.append(token_type_ids1)

            batch_input_ids2.append(input_ids2)
            batch_token_type_ids2.append(attention_mask2)
            batch_attention_masks2.append(token_type_ids2)

            batch_e1_mask.append(e1_mask)
            batch_e2_mask.append(e2_mask)

            batch_labels.append(label)
        #开始pad
        if self.config.fixed_batch_length:
            pad_len = self.config.max_len
        else:
            pad_len = min(self.config.max_len,batch_max_len)
        batch_input_ids1 = torch.tensor(sequence_padding(batch_input_ids1, length=pad_len), device=self.device).long()
        batch_token_type_ids1 = torch.tensor(sequence_padding(batch_token_type_ids1, length=pad_len),
                                             device=self.device).long()
        batch_attention_masks1 = torch.tensor(sequence_padding(batch_attention_masks1, length=pad_len),
                                              device=self.device).long()
        batch_input_ids2 = torch.tensor(sequence_padding(batch_input_ids2, length=pad_len), device=self.device).long()
        batch_token_type_ids2 = torch.tensor(sequence_padding(batch_token_type_ids2, length=pad_len),
                                             device=self.device).long()
        batch_attention_masks2 = torch.tensor(sequence_padding(batch_attention_masks2, length=pad_len),
                                              device=self.device).long()

        batch_e1_mask = torch.tensor(sequence_padding(batch_e1_mask, length=pad_len), device=self.device).long()
        batch_e2_mask = torch.tensor(sequence_padding(batch_e2_mask, length=pad_len), device=self.device).long()

        batch_labels = torch.tensor(batch_labels,device=self.device).long()

        return batch_input_ids1, batch_token_type_ids1, batch_attention_masks1,batch_input_ids2, batch_token_type_ids2, batch_attention_masks2, batch_e1_mask, batch_e2_mask,batch_labels

    def _process_seq_len(self, text_a, text_b, total_special_toks=3):
        """
            裁切句子的方法，直接使用clinicalTransformer提供的方法
            This function is used to truncate sequences with len > max_seq_len
            Truncate strategy:
            1. find all the index for special tags
            3. count distances between leading word to first tag and second tag to last.
            first -1- tag1 entity tag2 -2- last
            4. pick the longest distance from (1, 2), if 1 remove first token, if 2 remove last token
            5. repeat until len is equal to max_seq_len
        """
        flag = True
        # 防止死循环的
        no_loop = 0

        while len(self.tokenizer.tokenize(text_a) + self.tokenizer.tokenize(text_b)) > (self.config.max_len - total_special_toks):

            if flag:
                text_a = self._truncate_helper(text_a)
            else:
                text_b = self._truncate_helper(text_b)

            flag = not flag
            no_loop += 1
            if no_loop>50:
                return

        return text_a, text_b


    def _truncate_helper(self,text):
        '''
        这是一个句子一个句子的找
        这里对原始的的text进行去除，并不是tokenize之后的....
        :param text:
        :return:
        '''
        tokens = text.split(" ")
        # 这是得到 word-level的index
        tags_li =  [self.config.ent1_start_tag,self.config.ent1_end_tag,self.config.ent2_start_tag ,  self.config.ent2_end_tag ]
        spec_tag_idx1, spec_tag_idx2 = [idx for (idx, tk) in enumerate(tokens) if tk.lower() in tags_li]
        start_idx, end_idx = 0, len(tokens) - 1
        truncate_space_head = spec_tag_idx1 - start_idx
        truncate_space_tail = end_idx - spec_tag_idx2

        if truncate_space_head == truncate_space_tail == 0: #这是表示如果实体1和实体2 都已经在句子的首尾，那么就不要继续删除了....
            return text

        if truncate_space_head > truncate_space_tail: # 如果离头更远，那么先抛弃头部的word...
            tokens.pop(0)
        else:
            tokens.pop(-1)

        return " ".join(tokens)




class MultiMTBDataset(Dataset):
    def __init__(self,examples,config:BertConfig,tokenizer:BertTokenizer,label2id,device):
        '''
        使用这个读取数据的时候，所有需要的数据都会放在同一个文件之中，
        :param config:
        :param sents:
        :param tokenizer:
        :param label2id:
        :param max_len:
        '''
        super(MultiMTBDataset, self).__init__()
        self.config = config
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = config.max_len
        self.device = device


    def __len__(self):
        return len(self.examples)
    def __getitem__(self, item):

        return self.examples[item]

    def collate_fn_predicate(self, examples):
        '''
        专用于model的predicate
        input: <cls>sent1<sep>sent2<sep>
        - sent1和sent2为实体1和2所在的位置

        :param examples:一个batch_size的输入数据,
        :return:
        '''
        raw_text_li_a = []
        raw_text_li_b = []
        batch_input_ids = []
        batch_token_type_ids = []
        batch_attention_masks = []
        batch_e1_mask = []
        batch_e2_mask = []
        batch_rel_type = []

        batch_max_len = 0

        for exam in examples:
            text_a = exam.text_a
            text_b = exam.text_b
            raw_text_li_a.append(text_a)
            raw_text_li_b.append(text_b)


            tokenize_text_a = self.tokenizer.tokenize(text_a)
            tokenize_text_b = self.tokenizer.tokenize(text_b)
            # 如果长度过长，那么开始裁剪长度
            if len(tokenize_text_a) + len(tokenize_text_b) > self.config.max_len - self.config.total_special_toks:
                logger.info('长度为{},开始裁剪长度'.format(len(tokenize_text_a) + len(tokenize_text_b)))

                res = self._process_seq_len(text_a, text_b)
                if res:
                    text_a, text_b = res
                else:
                    logger.warning('发生了裁剪死循环...')
                    continue
                tokenize_text_a = self.tokenizer.tokenize(text_a)
                tokenize_text_b = self.tokenizer.tokenize(text_b)
                logger.info('裁剪之后的长度为{}'.format(len(tokenize_text_a) + len(tokenize_text_b)))
                if len(tokenize_text_a) + len(tokenize_text_b) > self.config.max_len - self.config.total_special_toks:
                    logger.warning("放弃这条数据,数据太长了....")
                    continue

            encode_res = self.tokenizer.encode_plus(text_a, text_b)

            input_ids = encode_res['input_ids']
            attention_mask = encode_res['attention_mask']
            token_type_ids = encode_res['token_type_ids']

            if len(input_ids) > batch_max_len:
                batch_max_len = len(input_ids)
            # 构建其对应的mask
            # 由于增加[CLS]这个token

            e1_start_idx = input_ids.index(self.config.ent1_start_tag_id)
            e1_end_idx = input_ids.index(self.config.ent1_end_tag_id)
            e2_start_idx = input_ids.index(self.config.ent2_start_tag_id)
            e2_end_idx = input_ids.index(self.config.ent2_end_tag_id)

            e1_mask = np.zeros(len(input_ids))
            e2_mask = np.zeros(len(input_ids))
            e1_mask[e1_start_idx:e1_end_idx + 1] = 1
            e2_mask[e2_start_idx:e2_end_idx + 1] = 1

            batch_input_ids.append(input_ids)
            batch_token_type_ids.append(token_type_ids)
            batch_attention_masks.append(attention_mask)
            batch_e1_mask.append(e1_mask)
            batch_e2_mask.append(e2_mask)
            batch_rel_type.append(exam.rel_type)
        # 开始pad
        if self.config.fixed_batch_length:
            pad_len = self.config.max_len
        else:
            pad_len = min(self.config.max_len, batch_max_len)
        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids, length=pad_len), device=self.device).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=pad_len),
                                            device=self.device).long()
        batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks, length=pad_len),
                                             device=self.device).long()
        batch_e1_mask = torch.tensor(sequence_padding(batch_e1_mask, length=pad_len), device=self.device).long()
        batch_e2_mask = torch.tensor(sequence_padding(batch_e2_mask, length=pad_len), device=self.device).long()
        batch_rel_type = torch.tensor(batch_rel_type, device=self.device).long()

        return batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_e1_mask, batch_e2_mask,batch_rel_type
    def collate_fn(self,examples):
        '''
        在这里将数据转换为模型需要的数据格式
        input: <cls>sent1<sep>sent2<sep>
        - sent1和sent2为实体1和2所在的位置

        :param examples:一个batch_size的输入数据,
        :return:
        '''
        raw_text_li_a = []
        raw_text_li_b = []
        batch_input_ids = []
        batch_token_type_ids = []
        batch_labels = []
        batch_attention_masks = []
        batch_e1_mask = []
        batch_e2_mask = []
        batch_rel_type = []

        batch_max_len = 0

        for exam in examples:
            text_a = exam.text_a
            text_b = exam.text_b

            label = exam.label

            raw_text_li_a.append(text_a)
            raw_text_li_b.append(text_b)

            label = self.label2id.get(label)

            tokenize_text_a = self.tokenizer.tokenize(text_a)
            tokenize_text_b = self.tokenizer.tokenize(text_b)
            # 如果长度过长，那么开始裁剪长度
            if len(tokenize_text_a) + len(tokenize_text_b) > self.config.max_len - self.config.total_special_toks:
                logger.info('长度为{},开始裁剪长度'.format(len(tokenize_text_a) + len(tokenize_text_b) ))

                res = self._process_seq_len(text_a,text_b)
                if res:
                    text_a, text_b = res
                else:
                    logger.warning('发生了裁剪死循环...')
                    continue
                tokenize_text_a = self.tokenizer.tokenize(text_a)
                tokenize_text_b = self.tokenizer.tokenize(text_b)
                logger.info('裁剪之后的长度为{}'.format(len(tokenize_text_a)+len(tokenize_text_b)))
                if len(tokenize_text_a) + len(tokenize_text_b) > self.config.max_len - self.config.total_special_toks:
                    logger.warning("放弃这条数据,数据太长了....")
                    continue

            batch_labels.append(label)
            encode_res = self.tokenizer.encode_plus(text_a,text_b)

            input_ids = encode_res['input_ids']
            attention_mask = encode_res['attention_mask']
            token_type_ids = encode_res['token_type_ids']

            if len(input_ids) > batch_max_len:
                batch_max_len = len(input_ids)
            # 构建其对应的mask
            # 由于增加[CLS]这个token

            e1_start_idx  = input_ids.index(self.config.ent1_start_tag_id)
            e1_end_idx  = input_ids.index(self.config.ent1_end_tag_id)
            e2_start_idx  = input_ids.index(self.config.ent2_start_tag_id)
            e2_end_idx  = input_ids.index(self.config.ent2_end_tag_id)


            e1_mask = np.zeros(len(input_ids))
            e2_mask = np.zeros(len(input_ids))
            e1_mask[e1_start_idx:e1_end_idx + 1] = 1
            e2_mask[e2_start_idx:e2_end_idx + 1] = 1


            batch_input_ids.append(input_ids)
            batch_token_type_ids.append(token_type_ids)
            batch_attention_masks.append(attention_mask)
            batch_e1_mask.append(e1_mask)
            batch_e2_mask.append(e2_mask)
            batch_rel_type.append(exam.rel_type)
        #开始pad
        if self.config.fixed_batch_length:
            pad_len = self.config.max_len
        else:
            pad_len = min(self.config.max_len,batch_max_len)
        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids,length=pad_len),device=self.device).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids,length=pad_len),device=self.device).long()
        batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks,length=pad_len),device=self.device).long()
        batch_e1_mask = torch.tensor(sequence_padding(batch_e1_mask,length=pad_len),device=self.device).long()
        batch_e2_mask = torch.tensor(sequence_padding(batch_e2_mask,length=pad_len),device=self.device).long()
        batch_labels = torch.tensor(batch_labels,device=self.device).long()
        batch_rel_type = torch.tensor(batch_rel_type,device=self.device).long()

        return batch_input_ids,batch_token_type_ids,batch_attention_masks,batch_e1_mask,batch_e2_mask,batch_labels,batch_rel_type

    def _process_seq_len(self, text_a, text_b, total_special_toks=3):
        """
            裁切句子的方法，直接使用clinicalTransformer提供的方法
            This function is used to truncate sequences with len > max_seq_len
            Truncate strategy:
            1. find all the index for special tags
            3. count distances between leading word to first tag and second tag to last.
            first -1- tag1 entity tag2 -2- last
            4. pick the longest distance from (1, 2), if 1 remove first token, if 2 remove last token
            5. repeat until len is equal to max_seq_len
        """
        flag = True
        # 防止死循环的
        no_loop = 0

        while len(self.tokenizer.tokenize(text_a) + self.tokenizer.tokenize(text_b)) > (self.config.max_len - total_special_toks):

            if flag:
                text_a = self._truncate_helper(text_a)
            else:
                text_b = self._truncate_helper(text_b)

            flag = not flag
            no_loop += 1
            if no_loop>50:
                return

        return text_a, text_b


    def _truncate_helper(self,text):
        '''
        这是一个句子一个句子的找
        这里对原始的的text进行去除，并不是tokenize之后的....
        :param text:
        :return:
        '''
        tokens = text.split(" ")
        # 这是得到 word-level的index
        tags_li =  [self.config.ent1_start_tag,self.config.ent1_end_tag,self.config.ent2_start_tag ,  self.config.ent2_end_tag ]
        spec_tag_idx1, spec_tag_idx2 = [idx for (idx, tk) in enumerate(tokens) if tk.lower() in tags_li]
        start_idx, end_idx = 0, len(tokens) - 1
        truncate_space_head = spec_tag_idx1 - start_idx
        truncate_space_tail = end_idx - spec_tag_idx2

        if truncate_space_head == truncate_space_tail == 0: #这是表示如果实体1和实体2 都已经在句子的首尾，那么就不要继续删除了....
            return text

        if truncate_space_head > truncate_space_tail: # 如果离头更远，那么先抛弃头部的word...
            tokens.pop(0)
        else:
            tokens.pop(-1)

        return " ".join(tokens)
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  这里的数据集针对的是entity type类别的数据
        所以数据集的sentence.txt一般就一句话，实体对已经用entity type给替代了...
   Author :        kedaxia
   date：          2021/12/22
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/22: 
-------------------------------------------------
"""
import re
import logging
from ipdb import set_trace

import numpy as np
import torch
from torch.utils.data import Dataset

from config import BertConfig
from src.utils.function_utils import get_pos_feature
from src.dataset_utils.data_process_utils import sequence_padding, InputExamples

logger = logging.getLogger('main.entity_type_marker')


class NormalDataset(Dataset):
    def __init__(self, examples, config: BertConfig, tokenizer, label2id, device):
        super(NormalDataset, self).__init__()
        self.config = config
        self.examples = examples

        self.tokenizer = tokenizer
        # 将特殊实体加入到分词器，防止给切分

        self.label2id = label2id

        self.max_len = config.max_len
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):

        return self.examples[item]

    def collate_fn_predicate(self, features):
        '''
        这个专用于模型的predicate的collate_fn
        和collate_fn的不同是没有label的处理
        :param features:
        :return:
        '''
        raw_text_li = []
        batch_input_ids = []
        batch_token_type_ids = []
        batch_attention_masks = []
        batch_e1_mask = []
        batch_e2_mask = []

        batch_max_len = 0
        for example in features:
            sent = example.text
            # sent 是word-level list: ['Feadeal','ABVDF','the',...]
            raw_text_li.append(sent)
            subword_tokens = self.tokenizer.tokenize(sent)
            # 如果长度过长，那么开始裁剪长度
            if len(subword_tokens) > self.config.max_len - self.config.total_special_toks:

                logger.info('长度为{},开始裁剪长度'.format(len(subword_tokens)))
                sent = self._process_seq_len(sent)
                if not sent:
                    logger.warning('此数据难以裁剪，进行抛弃......')
                    continue
                subword_tokens = self.tokenizer.tokenize(sent)
                logger.info('裁剪之后的长度为{}'.format(len(subword_tokens)))
                if len(subword_tokens) > self.config.max_len - self.config.total_special_toks:
                    continue

            if batch_max_len < len(subword_tokens):
                batch_max_len = len(subword_tokens)

            encoder_res = self.tokenizer.encode_plus(subword_tokens, truncation=True, max_length=self.max_len,
                                                     add_special_tokens=True)

            input_ids = encoder_res['input_ids']
            token_type_ids = encoder_res['token_type_ids']
            attention_mask = encoder_res['attention_mask']

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

        if self.config.fixed_batch_length:
            pad_length = self.config.max_len
        else:
            pad_length = min(batch_max_len, self.config.max_len)

        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids, length=pad_length), device=self.device).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=pad_length),
                                            device=self.device).long()
        batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks, length=pad_length),
                                             device=self.device).long()
        batch_e1_mask = torch.tensor(sequence_padding(batch_e1_mask, length=pad_length), device=self.device).long()
        batch_e2_mask = torch.tensor(sequence_padding(batch_e2_mask, length=pad_length), device=self.device).long()

        return batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_e1_mask, batch_e2_mask

    def collate_fn(self, features):
        '''

        :param features:
        :return:
        '''
        raw_text_li = []
        batch_input_ids = []
        batch_token_type_ids = []
        batch_labels = []
        batch_attention_masks = []
        batch_entities_positions = []
        batch_e1_mask = []
        batch_e2_mask = []

        batch_max_len = 0
        for example in features:

            sent = example.text
            label = self.label2id[example.label]

            # sent 是word-level list: ['Feadeal','ABVDF','the',...]
            raw_text_li.append(sent)
            subword_tokens = self.tokenizer.tokenize(sent)
            # 如果长度过长，那么开始裁剪长度
            if len(subword_tokens) > self.config.max_len - self.config.total_special_toks:

                logger.info('长度为{},开始裁剪长度'.format(len(subword_tokens)))
                sent = self._process_seq_len(sent)
                if not sent:
                    logger.warning('此数据难以裁剪，进行抛弃......')
                    continue
                subword_tokens = self.tokenizer.tokenize(sent)
                logger.info('裁剪之后的长度为{}'.format(len(subword_tokens)))
                if len(subword_tokens) > self.config.max_len - self.config.total_special_toks:
                    continue

            if batch_max_len < len(subword_tokens):
                batch_max_len = len(subword_tokens)
            batch_labels.append(label)
            encoder_res = self.tokenizer.encode_plus(subword_tokens, truncation=True, max_length=self.max_len,
                                                     add_special_tokens=True)

            input_ids = encoder_res['input_ids']
            token_type_ids = encoder_res['token_type_ids']
            attention_mask = encoder_res['attention_mask']

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

        if self.config.fixed_batch_length:
            pad_length = self.config.max_len
        else:
            pad_length = min(batch_max_len, self.config.max_len)

        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids, length=pad_length), device=self.device).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=pad_length),
                                            device=self.device).long()
        batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks, length=pad_length),
                                             device=self.device).long()
        batch_e1_mask = torch.tensor(sequence_padding(batch_e1_mask, length=pad_length), device=self.device).long()
        batch_e2_mask = torch.tensor(sequence_padding(batch_e2_mask, length=pad_length), device=self.device).long()

        batch_labels = torch.tensor(batch_labels, device=self.device).long()

        return batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_e1_mask, batch_e2_mask, batch_labels

    def _process_seq_len(self, text, total_special_toks=3):
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
        loop_count = 0
        while len(self.tokenizer.tokenize(text)) > (self.config.max_len - total_special_toks):
            text = self._truncate_helper(text)
            loop_count += 1
            if loop_count > 50:
                return
        return text

    def _truncate_helper(self, text):
        '''
        这是一个句子一个句子的找
        这里对原始的的text进行去除，并不是tokenize之后的....
        :param text:
        :return:
        '''
        tokens = text.split(" ")
        # 这是得到 word-level的index
        spec_tag_idx1, spec_tag_idx2 = [idx for (idx, tk) in enumerate(tokens) if
                                        tk.lower() in [self.config.ent1_start_tag, self.config.ent2_end_tag]]
        start_idx, end_idx = 0, len(tokens) - 1
        truncate_space_head = spec_tag_idx1 - start_idx
        truncate_space_tail = end_idx - spec_tag_idx2

        if truncate_space_head == truncate_space_tail == 0:  # 这是表示如果实体1和实体2 都已经在句子的首尾，那么就不要继续删除了....
            return text

        if truncate_space_head > truncate_space_tail:  # 如果离头更远，那么先抛弃头部的word...
            tokens.pop(0)
        else:
            tokens.pop(-1)

        return " ".join(tokens)

class MultiNormalDataset(Dataset):
    def __init__(self, examples, config: BertConfig, tokenizer, label2id, device):
        super(MultiNormalDataset, self).__init__()
        self.config = config
        self.examples = examples

        self.tokenizer = tokenizer
        # 将特殊实体加入到分词器，防止给切分

        self.label2id = label2id

        self.max_len = config.max_len
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):

        return self.examples[item]

    def collate_fn_predicate(self, features):
        '''
        这个专用于模型的predicate的collate_fn
        和collate_fn的不同是没有label的处理
        :param features:
        :return:
        '''
        raw_text_li = []
        batch_input_ids = []
        batch_token_type_ids = []
        batch_attention_masks = []
        batch_e1_mask = []
        batch_e2_mask = []
        batch_rel_type = []

        batch_max_len = 0
        for example in features:
            sent = example.text

            # sent 是word-level list: ['Feadeal','ABVDF','the',...]
            raw_text_li.append(sent)
            subword_tokens = self.tokenizer.tokenize(sent)
            # 如果长度过长，那么开始裁剪长度
            if len(subword_tokens) > self.config.max_len - self.config.total_special_toks:

                logger.info('长度为{},开始裁剪长度'.format(len(subword_tokens)))
                sent = self._process_seq_len(sent)
                if not sent:
                    logger.warning('此数据难以裁剪，进行抛弃......')
                    continue
                subword_tokens = self.tokenizer.tokenize(sent)
                logger.info('裁剪之后的长度为{}'.format(len(subword_tokens)))
                if len(subword_tokens) > self.config.max_len - self.config.total_special_toks:
                    continue

            if batch_max_len < len(subword_tokens):
                batch_max_len = len(subword_tokens)

            encoder_res = self.tokenizer.encode_plus(subword_tokens, truncation=True, max_length=self.max_len,
                                                     add_special_tokens=True)

            input_ids = encoder_res['input_ids']
            token_type_ids = encoder_res['token_type_ids']
            attention_mask = encoder_res['attention_mask']

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
            batch_rel_type.append(example.rel_type)

        if self.config.fixed_batch_length:
            pad_length = self.config.max_len
        else:
            pad_length = min(batch_max_len, self.config.max_len)

        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids, length=pad_length), device=self.device).long()

        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=pad_length),
                                            device=self.device).long()
        batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks, length=pad_length),
                                             device=self.device).long()
        batch_e1_mask = torch.tensor(sequence_padding(batch_e1_mask, length=pad_length), device=self.device).long()
        batch_e2_mask = torch.tensor(sequence_padding(batch_e2_mask, length=pad_length), device=self.device).long()
        batch_rel_type = torch.tensor(batch_rel_type, device=self.device).long()

        return batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_e1_mask, batch_e2_mask,batch_rel_type

    def collate_fn(self, features):
        """

        :param features:
        :return:
        """
        raw_text_li = []
        batch_input_ids = []
        batch_token_type_ids = []
        batch_labels = []
        batch_attention_masks = []
        batch_entities_positions = []
        batch_e1_mask = []
        batch_e2_mask = []
        batch_rel_type = []

        batch_max_len = 0
        for example in features:

            sent = example.text
            label = self.label2id[example.label]

            # sent 是word-level list: ['Feadeal','ABVDF','the',...]
            raw_text_li.append(sent)
            subword_tokens = self.tokenizer.tokenize(sent)
            # 如果长度过长，那么开始裁剪长度
            if len(subword_tokens) > self.config.max_len - self.config.total_special_toks:

                logger.info('长度为{},开始裁剪长度'.format(len(subword_tokens)))
                sent = self._process_seq_len(sent)
                if not sent:
                    logger.warning('此数据难以裁剪，进行抛弃......')
                    continue
                subword_tokens = self.tokenizer.tokenize(sent)
                logger.info('裁剪之后的长度为{}'.format(len(subword_tokens)))
                if len(subword_tokens) > self.config.max_len - self.config.total_special_toks:
                    continue

            if batch_max_len < len(subword_tokens):
                batch_max_len = len(subword_tokens)
            batch_labels.append(label)
            encoder_res = self.tokenizer.encode_plus(subword_tokens, truncation=True, max_length=self.max_len,
                                                     add_special_tokens=True)

            input_ids = encoder_res['input_ids']
            token_type_ids = encoder_res['token_type_ids']
            attention_mask = encoder_res['attention_mask']

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
            batch_rel_type.append(example.rel_type)

        if self.config.fixed_batch_length:
            pad_length = self.config.max_len
        else:
            pad_length = min(batch_max_len, self.config.max_len)

        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids, length=pad_length), device=self.device).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=pad_length),
                                            device=self.device).long()
        batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks, length=pad_length),
                                             device=self.device).long()
        batch_e1_mask = torch.tensor(sequence_padding(batch_e1_mask, length=pad_length), device=self.device).long()
        batch_e2_mask = torch.tensor(sequence_padding(batch_e2_mask, length=pad_length), device=self.device).long()

        batch_labels = torch.tensor(batch_labels, device=self.device).long()
        batch_rel_type = torch.tensor(batch_rel_type, device=self.device).long()

        return batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_e1_mask, batch_e2_mask, batch_labels,batch_rel_type

    def _process_seq_len(self, text, total_special_toks=3):
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
        loop_count = 0
        while len(self.tokenizer.tokenize(text)) > (self.config.max_len - total_special_toks):
            text = self._truncate_helper(text)
            loop_count += 1
            if loop_count > 50:
                return
        return text

    def _truncate_helper(self, text):
        '''
        这是一个句子一个句子的找
        这里对原始的的text进行去除，并不是tokenize之后的....
        :param text:
        :return:
        '''
        tokens = text.split(" ")
        # 这是得到 word-level的index
        spec_tag_idx1, spec_tag_idx2 = [idx for (idx, tk) in enumerate(tokens) if
                                        tk.lower() in [self.config.ent1_start_tag, self.config.ent2_end_tag]]
        start_idx, end_idx = 0, len(tokens) - 1
        truncate_space_head = spec_tag_idx1 - start_idx
        truncate_space_tail = end_idx - spec_tag_idx2

        if truncate_space_head == truncate_space_tail == 0:  # 这是表示如果实体1和实体2 都已经在句子的首尾，那么就不要继续删除了....
            return text

        if truncate_space_head > truncate_space_tail:  # 如果离头更远，那么先抛弃头部的word...
            tokens.pop(0)
        else:
            tokens.pop(-1)

        return " ".join(tokens)

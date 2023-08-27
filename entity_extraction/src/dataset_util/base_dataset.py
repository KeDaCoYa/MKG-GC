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

import numpy as np
from ipdb import set_trace


class InputExample:
    def __init__(self, text, labels,entity_type_id=None):
        self.text = text
        self.labels = labels
        self.entity_type_id = entity_type_id


class BaseFeature:
    def __init__(self, token_ids, attention_masks, token_type_ids, raw_text=None):
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.raw_text = raw_text


class CRFFeature(BaseFeature):
    def __init__(self, token_ids, attention_masks, token_type_ids, labels, raw_text=None):
        super().__init__(token_ids, attention_masks, token_type_ids, raw_text)

        self.labels = labels


class SpanFeature(BaseFeature):
    def __init__(self, token_ids, attention_masks, token_type_ids, start_ids, end_ids, raw_text=None):
        super().__init__(token_ids, attention_masks, token_type_ids, raw_text)

        self.start_ids = start_ids
        self.end_ids = end_ids


class NormalCRFFeature:
    def __init__(self, token_ids, labels, raw_text=None):
        self.token_ids = token_ids
        # self.attention_masks=attention_masks

        self.labels = labels

        self.raw_text = raw_text


class NERProcessor(object):
    def __init__(self,entity_type_id=None):
        """
        这个参数用于multiner，标志实体的类别
        """
        self.entity_type_id = entity_type_id

    def get_examples(self, raw_text, labels):
        if labels is None:  # predicate的时候使用
            examples = []
            for text in raw_text:
                example = InputExample(text, None)
                examples.append(example)
            return examples
        examples = []
        for text, label in zip(raw_text, labels):
            if self.entity_type_id:
                example = InputExample(text, label,self.entity_type_id)
            else:
                example = InputExample(text, label)
            examples.append(example)
        return examples


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """
    这个方法是对inputs按照length进行对齐
        length长，则补充value
        length短，则删除过长的

    Numpy函数，将序列padding到同一长度
    按照一个batch的最大长度进行padding
    :param inputs:(batch_size,None),每个序列的长度不一样
    :param seq_dim: 表示对哪些维度进行pad，默认为1，只有当对label进行pad的时候，seq_dim=3,因为labels.shape=(batch_size,entity_type,seq_len,seq_len)
        因为一般都是对(batch_size,seq_len)进行pad，，，
    :param length: 这个是设置补充之后的长度，一般为None，根据batch的实际长度进行pad
    :param value:
    :param mode:
    :return:
    """
    pad_length = length
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)  # length=np.array([max_batch_length])
    elif not hasattr(length, '__getitem__'):  # 如果这个length的类别不是列表....,就进行转变

        length = [length]
    # logger.info('这个batch下面的最长长度为{}'.format(length[0]))
    if seq_dims == 3: # 这个只针对globalpointer的情况

        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        length[1] = pad_length
        length[2] = pad_length
        slices = [np.s_[:length[i]] for i in
                  range(seq_dims)]  # 获得针对针对不同维度的slice，对于seq_dims=0,slice=[None:max_len:None],max_len是seq_dims的最大值
        slices = tuple(slices) if len(slices) > 1 else slices[0]
    else:
        slices = [np.s_[:length[i]] for i in range(seq_dims)]  # 获得针对针对不同维度的slice，对于seq_dims=0,slice=[None:max_len:None],max_len是seq_dims的最大值
        slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]  # 有多少个维数，就需要多少个(0,0),一般是一个

    outputs = []
    for x in inputs:
        # X为一个列表
        # 这里就是截取长度
        x = x[slices]
        for i in range(seq_dims):  # 对不同的维度逐步进行扩充
            if mode == 'post':
                # np.shape(x)[i]是获得当前的实际长度
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)


def globalpointer_collate(true_labels, labels, globalpointer_label2id):
    '''
    这是根据true label转变为globalpointer所需要的格式
    :param true_labels:
    :param labels:
    :param globalpointer_label2id:
    :return:
    '''
    start_index = 0
    # 获取正确的set
    actual_len = len(true_labels)

    while start_index < actual_len:
        cur_label = true_labels[start_index].split('-')
        if len(cur_label) == 2:
            # 这个entity type和entity type id是全局的数据
            BIO_format, entity_type = cur_label
            entity_type_id = globalpointer_label2id[entity_type]
        else:
            BIO_format = cur_label[0]
            entity_type_id = 0

        if start_index + 1 < actual_len:
            next_label = true_labels[start_index + 1].split('-')
            if len(next_label) == 2:
                BIO_, _ = next_label
            elif len(next_label) == 1:
                BIO_ = next_label[0]

        if BIO_format == 'B' and start_index + 1 < actual_len and BIO_ == 'O':  # 实体是一个单词
            labels[entity_type_id, start_index, start_index] = 1
            start_index += 1
        elif BIO_format == 'B' and start_index + 1 >= actual_len:  # 最后只有一个实体，并且只有一个单词，到达了最后
            labels[entity_type_id, start_index, start_index] = 1
            break
        elif BIO_format == 'B':
            j = start_index + 1
            while j < actual_len:
                j_label = true_labels[j].split('-')
                if len(j_label) == 2:
                    BIO_, _ = j_label
                elif len(j_label) == 1:
                    BIO_ = j_label[0]

                if BIO_ == 'I':
                    j += 1
                else:

                    labels[entity_type_id, start_index, j - 1] = 1

                    break
            if j >= actual_len:
                j_label = true_labels[j - 1].split('-')
                if len(j_label) == 2:
                    BIO_, _ = j_label
                elif len(j_label) == 1:
                    BIO_ = j_label[0]

                if BIO_ == 'I':
                    labels[entity_type_id, start_index, j - 1] = 1

            start_index = j
        else:
            start_index += 1


def get_i_label(beginning_label, label_map):
    """To properly label segments of words broken by BertTokenizer=.
    """
    if "B-" in beginning_label:
        i_label = "I-" + beginning_label.split("B-")[-1]
        return i_label
    elif "I-" in beginning_label:
        i_label = "I-" + beginning_label.split("I-")[-1]
        return i_label
    else:
        return "O"


def tokenize_text(bert_tokenizer, example, max_len):
    """
    一次处理一个InputExample
    首先经过这个分词之后才会进行encode_plus
    :param bert_tokenizer:
    :param example: 这个就是InputExample
    :param max_len: 一个句子的最大长度
    :return:
    """
    raw_text_list = example.text
    true_labels = example.labels
    subword_tokens = []

    # 这是表示原先的token与分词之后的一个对应关系
    origin_to_subword_index = []
    # 这里的labels类似于[B-DNA,I-DNA,....]
    for word_idx, (word, label) in enumerate(zip(raw_text_list, true_labels)):
        tmp_index = len(subword_tokens)
        origin_to_subword_index.append(tmp_index)
        subword = bert_tokenizer.tokenize(word)

        subword_tokens.extend(subword)
        if len(subword_tokens) >= max_len - 2:  # 超过最大长度，则break
            break
    return subword_tokens, origin_to_subword_index


def tokenize_text_predicate(bert_tokenizer, example, max_len):
    """
    一次处理一个InputExample
    首先经过这个分词之后才会进行encode_plus
    :param bert_tokenizer:
    :param example: 这个就是InputExample

    :return:
    """
    raw_text_list = example.text

    subword_tokens = []

    # 这是表示原先的token与分词之后的一个对应关系
    origin_to_subword_index = []
    # 这里的labels类似于[B-DNA,I-DNA,....]
    for word_idx, word in enumerate(raw_text_list):
        tmp_index = len(subword_tokens)
        origin_to_subword_index.append(tmp_index)
        subword = bert_tokenizer.tokenize(word)

        subword_tokens.extend(subword)
        if len(subword_tokens) >= max_len - 2:
            raise ValueError("句子太长了为:{}".format(len(subword_tokens)))
    return subword_tokens, origin_to_subword_index

# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  一句话一句话的预测,也就是用于之后的abstract预测
   Author :        kedaxia
   date：          2021/11/20
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/11/20: 
-------------------------------------------------
"""


import json
import time
import logging
from collections import defaultdict

from ipdb import set_trace
from tqdm import  tqdm
import numpy as np

import torch
from transformers import BertModel, BertTokenizer

from config import BertConfig
from utils.data_process_utils import read_data, build_vocab


logger = logging.getLogger()


def span_predicate(start_logits,end_logits,raw_text_li,span_id2label):
    '''
    一般只有bert采用这个span，使用bilstm，这个span没啥意思
    :param config:
    :param best_model_ckpt_path:
    :param kwargs:
    :return:
    '''
    batch_size = len(raw_text_li)


    entities = []
    for id,length in enumerate(range(batch_size)):
        # 这里是一个集合，共由四部分组成(entity,entity_type,start_idx,end_idx)
        tmp_start_logits = start_logits[id]
        tmp_end_logits = end_logits[id]


        for i,s_type in enumerate(tmp_start_logits):
            if s_type == 0: # 忽略Other 的标签
                continue
            for j,e_type in enumerate(tmp_end_logits[i:]):
                if s_type == e_type:

                    entities.append({
                    'entity_type': span_id2label[s_type],
                    'start_idx': str(i),
                    'end_idx': str(i+j),
                    'entity_name': " ".join(raw_text_li[id][i:i+j+1])
                    })

                    break

    return entities



def crf_predicate(predicate_tokens, id2label, raw_texts):
    '''
    针对多类别的BIO进行decode...
    '''

    entities = []
    batch_size = len(predicate_tokens)
    for i in range(batch_size):

        predicate_token = predicate_tokens[i]



        start_index = 0
        # 获取正确的set
        raw_text = raw_texts[i]
        actual_len = len(raw_text)


        while start_index < actual_len:


            if id2label[predicate_token[start_index]][0] == 'B' and start_index + 1 < actual_len and  id2label[predicate_token[start_index+1]] == 'O':
                entity_type = id2label[predicate_token[start_index]].split('-')[-1]
                entities.append({
                    'entity_type':entity_type,
                    'start_idx': str(start_index),
                    'end_idx': str(start_index),
                    'entity_name': " ".join(raw_text[start_index:start_index + 1])
                })
                start_index += 1
            elif id2label[predicate_token[start_index]][0] == 'B' and start_index + 1 >= actual_len:
                entity_type = id2label[predicate_token[start_index]].split('-')[-1]
                entities.append({
                    'entity_type': entity_type,
                    'start_idx': str(start_index),
                    'end_idx': str(start_index),
                    'entity_name': " ".join(raw_text[start_index:start_index + 1])
                })

                break
            elif id2label[predicate_token[start_index]][0] == 'B':
                j = start_index + 1
                while j < actual_len:
                    if id2label[predicate_token[j]][0] == 'I':
                        j += 1
                    else:
                        entity_type = id2label[predicate_token[start_index]].split('-')[-1]
                        entities.append({
                            'entity_type': entity_type,
                            'start_idx': str(start_index),
                            'end_idx': str(j - 1),
                            'entity_name': " ".join(raw_text[start_index:j])
                        })

                        break
                if j >= actual_len:
                    if id2label[predicate_token[j-1]][0] == 'I':
                        entity_type = id2label[predicate_token[start_index]].split('-')[-1]
                        entities.append({
                            'entity_type': entity_type,
                            'start_idx': str(start_index),
                            'end_idx': str(j - 1),
                            'entity_name': " ".join(raw_text[start_index:j])
                        })


                start_index = j
            else:
                start_index += 1



    return entities

def crf_predicate_BIO(predicate_token,raw_text):
    '''
    这是针对BIO label的解码，label只有BIO三种类别
    这个得保证B=2,I=1,O=0,这个id2label和label2id

    '''


    entities = []
    start_ = 0
    start_index = 0
    # 获取正确的set
    raw_text = raw_text[0]
    actual_len = len(raw_text)
    while start_index < actual_len:

        if predicate_token[start_index] == 2 and start_index + 1 < actual_len and predicate_token[start_index + 1] == 0:  # 实体是一个单词
            entities.append({

                'start_idx': str(start_index),
                'end_idx': str(start_index),
                'entity_name': " ".join(raw_text[start_index:start_index + 1])
            })
            start_index += 1
        elif predicate_token[start_index] == 2 and start_index + 1 >= actual_len:

            entities.append({

                'start_idx': str(start_index),
                'end_idx': str(start_index),
                'entity_name': raw_text[start_index]
            })
            break
        elif predicate_token[start_index] == 2:
            j = start_index + 1
            while j < actual_len:
                if predicate_token[j] == 1:
                    j += 1
                else:
                    entities.append({

                        'start_idx': str(start_index),
                        'end_idx': str(j-1),
                        'entity_name': raw_text[start_index:j]
                    })

                    break
            if j >= actual_len:
                if predicate_token[j - 1] == 1:
                    entities.append({

                        'start_idx': str(start_index),
                        'end_idx': str(j - 1),
                        'entity_name': raw_text[start_index:j]
                    })

            start_index = j
        else:
            start_index += 1

    print(entities)
    return entities


def bert_globalpointer_predicate(scores_li,globalpointer_id2label,raw_text_li):
    '''

    这个也是一句话，一句话的进行decode....
    这个只能处理一句话的，

    '''
    entities = []
    batch_size = len(scores_li)


    for i in range(batch_size):
        scores = scores_li[i]
        raw_text = raw_text_li[i]
        # 这里在decode的时候将开头和末尾给去掉，因为这是[CLS],[SEP]
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        # 这里相当于直接将预测结果进行解码，

        for l, start, end in zip(*np.where(scores > 0)):
            entities.append({
                'start_idx': str(start),
                'end_idx': str(end),
                'entity_type': globalpointer_id2label[l],
                'entity_name': " ".join(raw_text[start:end + 1])
            })

    return entities

def normal_globalpointer_predicate(scores,globalpointer_id2label,raw_text):

    '''
    实体是一句话一句话的抽取，
    scores:shape=(1,type,seq_len,seq_len)
    '''

    raw_text = raw_text[0]
    entities = []

    for b, l, start, end  in zip(*np.where(scores > 0)):

        entities.append({

            'start_idx': str(start),
            'end_idx': str(end),
            'entity_type': globalpointer_id2label[l],
            'entity_name': " ".join(raw_text[start:end + 1])
        })


    return entities


def vote(entities_list,threshold=0.9):
    '''
    entities_list:每个模型对同一个文本的预测，len(entities_list)为model的个数
    entities_list[0]:为字典{'type','start','end','ent_name'}
    '''
    threshold_nums = int(len(entities_list) * threshold)  # 这个表示只有当某个实体被大于这个个数的模型预测到，才算是预测完成....
    entities_dict = defaultdict(int)
    entities = []

    for _entities in entities_list:
        for ent_ in _entities:
            ent_type = ent_['entity_type']
            start_idx = ent_['start_idx']
            end_idx = ent_['end_idx']
            ent_name = ent_['entity_name']
            entities_dict[(ent_type, start_idx, end_idx,ent_name)] += 1

    for key in entities_dict:
        if entities_dict[key] >= threshold_nums:
            entities.append({
                'start_idx': key[1],
                'end_idx': key[2],
                'entity_type': key[0],
                'entity_name': key[3],
            })


    return entities



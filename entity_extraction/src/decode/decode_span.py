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
import logging
import numpy as np
from collections import defaultdict
from ipdb import set_trace

from utils.function_utils import  gentl_print

logger = logging.getLogger('main.span_evaluate')
import json

def decode_raw_text_span(start_logits,end_logits,raw_text,save_path,span_id2label):
    '''
    这里是将span进行decode，格式如下
        {
        raw_text:
        entity_type:
        start_idx:
        end_idx:
        entity:

        }
    :param start_logits:
    :param end_logits:
    :param start_ids:
    :param end_ids:
    :param raw_text:
    :param span_label2id:
    :return:
    '''
    lens_li = [len(i) for i in raw_text]  # 使用raw_text作为具体长度
    predicate_dict = defaultdict(list)
    predicate_results = []
    for id, length in enumerate(lens_li):
        # 这里是一个集合，共由四部分组成(entity,entity_type,start_idx,end_idx)

        tmp_start_logits = start_logits[id][1:length + 1]
        tmp_end_logits = end_logits[id][1:length + 1]


        for i, s_type in enumerate(tmp_start_logits):
            if s_type == 0:
                continue
            for j, e_type in enumerate(tmp_end_logits[i:]):
                if s_type == e_type:
                    predicate_dict[s_type].append((s_type, i, j))
                    predicate_results.append({
                        'raw_text':raw_text[id],
                        'start_idx':i,
                        'end_ids':j,
                        'entity_type':span_id2label[s_type],
                        'entity':raw_text[i:j]
                    })
                    break

    # 保存保存结果
    json.dump(
        predicate_results,
        open(save_path, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )





def evaluate_span_micro(start_logits,end_logits,start_ids,end_ids,raw_text,span_label2id,type_weight,verbose=False):
    '''
    Bert解码的时候注意<CLS>,<SEP>
    这是macro的计算方式
    :param start_logits:
    :param end_logits:
    :param start_ids:
    :param end_ids:
    :param mask:
    :param raw_text:
    :param type_weight:各个entity type对应的权重
    :return:
    '''
    span_id2label = {value:key for key,value in span_label2id.items()}
    lens_li = [len(i) for i in raw_text] #使用raw_text作为具体长度
    #这两个是用于评估
    predicate_dict = defaultdict(list)
    true_dict = defaultdict(list)

    # 这两个是用于decode
    decode_predicate = defaultdict(list)
    decode_true = defaultdict(list)

    for id,length in enumerate(lens_li):
        # 这里是一个集合，共由四部分组成(entity,entity_type,start_idx,end_idx)


        tmp_start_logits = start_logits[id][1:length+1]
        tmp_end_logits = end_logits[id][1:length+1]
        tmp_start_id = start_ids[id][1:length+1]
        tmp_end_id = end_ids[id][1:length+1]


        for i,s_type in enumerate(tmp_start_logits):
            if s_type == 0:
                continue
            for j,e_type in enumerate(tmp_end_logits[i:]):
                if s_type == e_type:
                    predicate_dict[s_type].append((s_type, id,i, i+j+1))
                    decode_predicate[s_type].append((raw_text[id][i:i+j+1], i, i+j+1))
                    break

        for i,s_type in enumerate(tmp_start_id):
            if s_type == 0:
                continue
            for j,e_type in enumerate(tmp_end_id[i:]):
                if s_type == e_type:
                    decode_true[s_type].append((raw_text[id][i:i + j + 1], i, i + j + 1))
                    true_dict[s_type].append((s_type, id,i, j+i+1))
                    break
    f1 = 0.
    p = 0.
    r = 0.

    if type_weight is not None:
        gentl_li = []
        for key_ in type_weight.keys():
            key = span_label2id[key_]

            R = set(predicate_dict[key])
            T = set(true_dict[key])
            X = len(R & T)
            Y = len(R)
            Z = len(T)
            tmp_f1 = 2 * X / (Y + Z) if Y + Z != 0 else 0.
            tmp_p = X / Y if Y != 0 else 0.
            tmp_r = X / Z if Z != 0 else 0.
            f1 += type_weight[key_] * tmp_f1
            p += type_weight[key_] * tmp_p
            r += type_weight[key_] * tmp_r
            gentl_li.append([key_,tmp_f1,tmp_p,tmp_r])


        if verbose:
            gentl_print(gentl_li)

    else:
        # 这是单类别实体的计算
        R = set(predicate_dict[1])
        T = set(true_dict[1])
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        f1 = 2 * X / (Y + Z) if Y + Z != 0 else 0.
        p = X / Y if Y != 0 else 0.
        r = X / Z if Z != 0 else 0.
    return f1, p, r



def evaluate_span_macro(start_logits,end_logits,start_ids,end_ids,raw_text):
    '''
    Bert解码的时候注意<CLS>,<SEP>
    这是macro的计算方式
    :param start_logits:
    :param end_logits:
    :param start_ids:
    :param end_ids:
    :param mask:
    :param raw_text:
    :return:
    '''
    lens_li = [len(i) for i in raw_text] #使用raw_text作为具体长度


    sum_ = 0
    A, B, C = 1e-10, 1e-10, 1e-10

    for id,length in enumerate(lens_li):
        # 这里是一个集合，共由四部分组成(entity,entity_type,start_idx,end_idx)
        predicate_set = set()



        tmp_start_logits = start_logits[id][1:length+1]
        tmp_end_logits = end_logits[id][1:length+1]
        tmp_start_id = start_ids[id][1:length+1]
        tmp_end_id = end_ids[id][1:length+1]



        for i,s_type in enumerate(tmp_start_logits):
            if s_type == 0:
                continue
            for j,e_type in enumerate(tmp_end_logits[i:]):
                if s_type == e_type:
                    predicate_set.add((s_type, i, j+i+1))
                    break

        true_set = set()
        for i,s_type in enumerate(tmp_start_id):
            if s_type == 0:
                continue
            for j,e_type in enumerate(tmp_end_id[i:]):
                if s_type == e_type:
                    true_set.add((s_type, i, j+i+1))
                    break

        T = set(true_set)
        A += len(predicate_set & T)
        B += len(predicate_set)
        C += len(T)

        sum_ += len(true_set)
    f1 = 2 * A / (B + C) if (B + C) > 1e-10 else 0
    P = A / B if B > 1e-10 and A > 1e-10 else 0
    R = A / C if C > 1e-10 else 0
    return P,R,f1


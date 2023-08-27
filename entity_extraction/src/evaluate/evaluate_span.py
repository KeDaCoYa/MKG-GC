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

from utils.function_utils import gentl_print, gentl_print_confusion_matrix

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


def transfer_span_to_BIO(start_logits,end_logits,raw_text):
    """
    将bert_span预测的结果给转变为BIO label
    """
    lens_li = [len(i) for i in raw_text]  # 使用raw_text作为具体长度
    # 这两个是用于评估
    predicate_dict = defaultdict(list)
    true_dict = defaultdict(list)

    # 这两个是用于decode
    decode_predicate = defaultdict(list)

    all_BIO = []
    for id, length in enumerate(lens_li):
        # 这里是一个集合，共由四部分组成(entity,entity_type,start_idx,end_idx)
        BIO = ['O']*length
        tmp_start_logits = start_logits[id][:length]
        tmp_end_logits = end_logits[id][:length]

        for i, s_type in enumerate(tmp_start_logits):
            if s_type == 0:  # 忽略Other 的标签
                continue
            for j, e_type in enumerate(tmp_end_logits[i:]):
                if s_type == e_type:
                    for tmp_idx in range(i,i+j+1):
                        if tmp_idx == i:
                            BIO[tmp_idx] = 'B'
                        else:
                            BIO[tmp_idx] = 'I'

                    predicate_dict[s_type].append((s_type, id, i, i + j + 1))
                    decode_predicate[s_type].append((raw_text[id][i:i + j + 1], i, i + j + 1))
                    break

        all_BIO.append(BIO)
    return all_BIO

def error_analysis(start_logits,end_logits,start_ids,end_ids,raw_text,span_label2id):



    lens_li = [len(i) for i in raw_text] #使用raw_text作为具体长度
    #这两个是用于评估

    predicate_dict = defaultdict(list)
    true_dict = defaultdict(list)
    # 这两个是用于decode
    decode_predicate = defaultdict(list)
    decode_true = defaultdict(list)

    for id,length in enumerate(lens_li):
        # 这里是一个集合，共由四部分组成(entity,entity_type,start_idx,end_idx)
        tmp_start_logits = start_logits[id][:length]
        tmp_end_logits = end_logits[id][:length]
        tmp_start_id = start_ids[id][:length]
        tmp_end_id = end_ids[id][:length]

        for i,s_type in enumerate(tmp_start_logits):
            if s_type == 0: # 忽略Other 的标签
                continue
            for j,e_type in enumerate(tmp_end_logits[i:]):
                if s_type == e_type:
                    predicate_dict[s_type].append((s_type, id,i, i+j+1))
                    decode_predicate[s_type].append((' '.join(raw_text[id][i:i+j+1]),id, i, i+j+1))
                    break

        for i,s_type in enumerate(tmp_start_id):
            if s_type == 0:
                continue
            for j,e_type in enumerate(tmp_end_id[i:]):
                if s_type == e_type:
                    decode_true[s_type].append((' '.join(raw_text[id][i:i + j + 1]), id,i, i + j + 1))
                    true_dict[s_type].append((s_type, id,i, j+i+1))
                    break


    f1 = 0.
    p = 0.
    r = 0.
    # 这个列表是记录每一种entity type对应的f1,p,r
    gentl_li = []
    # 这个混淆矩阵则是记录TP,TN,FP,FN的值
    confusion_matrix = []



    ALL_X = 0
    All_Y = 0
    All_Z = 0

    for key_ in span_label2id.keys():
        if key_ == 'Other':
            continue
        key = span_label2id[key_]
        R = set(decode_predicate[key])
        T = set(decode_true[key])

        union_set = R&T
        predicate_surplus = R-union_set
        no_predicate = T-union_set




    f1 = 2 * ALL_X / (All_Y + All_Z) if (All_Y + All_Z) > 1e-10 else 0
    p = ALL_X / All_Y if All_Y > 1e-10 and ALL_X > 1e-10 else 0
    r= ALL_X / All_Z if All_Z > 1e-10 else 0



    return f1, p, r
def evaluate_span_fpr(start_logits,end_logits,start_ids,end_ids,raw_text,span_label2id,type_weight,average=None,verbose=False,wandb_dict=None,wandb=None):
    """
    这是对span结果的评估，包括解码和评估
    共三种评估方式:micro,macro和weight-cro

    如果使用Bert解码的时候注意<CLS>,<SEP>占用的位置
    :param start_logits:
    :param end_logits:
    :param start_ids:
    :param end_ids:
    :param mask:
    :param raw_text:
    :param type_weight:各个entity type对应的权重
    :return:
    """


    lens_li = [len(i) for i in raw_text] #使用raw_text作为具体长度
    #这两个是用于评估
    predicate_dict = defaultdict(list)
    true_dict = defaultdict(list)

    # 这两个是用于decode
    decode_predicate = defaultdict(list)
    decode_true = defaultdict(list)

    for id,length in enumerate(lens_li):
        # 这里是一个集合，共由四部分组成(entity,entity_type,start_idx,end_idx)
        tmp_start_logits = start_logits[id][:length]
        tmp_end_logits = end_logits[id][:length]
        tmp_start_id = start_ids[id][:length]
        tmp_end_id = end_ids[id][:length]

        for i,s_type in enumerate(tmp_start_logits):
            if s_type == 0: # 忽略Other 的标签
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
    # 这个列表是记录每一种entity type对应的f1,p,r
    gentl_li = []
    # 这个混淆矩阵则是记录TP,TN,FP,FN的值
    confusion_matrix = []
    if average == 'weight':

        for key_ in type_weight.keys():
            key = span_label2id[key_]

            R = set(predicate_dict[key])
            T = set(true_dict[key])
            X = len(R & T) # TP
            Y = len(R)     # TP+FP
            Z = len(T)     # TP+FN
            tmp_f1 = 2 * X / (Y + Z) if Y + Z != 0 else 0.
            tmp_p = X / Y if Y != 0 else 0.
            tmp_r = X / Z if Z != 0 else 0.
            f1 += type_weight[key_] * tmp_f1
            p += type_weight[key_] * tmp_p
            r += type_weight[key_] * tmp_r
            gentl_li.append([key_,tmp_f1,tmp_p,tmp_r,len(true_dict[key])])
            # 只计算TP,FP,FN
            # TN在ner中没意义
            confusion_matrix.append([key_,X,Y-X,Z-X,len(true_dict[key])])
    elif average == 'micro':

        ALL_X = 0
        All_Y = 0
        All_Z = 0

        for key_ in span_label2id.keys():
            if key_ == 'Other':
                continue
            key = span_label2id[key_]
            R = set(predicate_dict[key])
            T = set(true_dict[key])
            X = len(R & T) # TP
            Y = len(R)     # TP+FP
            Z = len(T)     # TP+FN
            ALL_X += X
            All_Y += Y
            All_Z += Z
            tmp_f1 = 2 * X / (Y + Z) if Y + Z != 0 else 0.
            tmp_p = X / Y if Y != 0 else 0.
            tmp_r = X / Z if Z != 0 else 0.

            gentl_li.append([key_, tmp_f1, tmp_p, tmp_r,len(true_dict[key])])
            confusion_matrix.append([key_,X,Y-X,Z-X,len(true_dict[key])])

        f1 = 2 * ALL_X / (All_Y + All_Z) if (All_Y + All_Z) > 1e-10 else 0
        p = ALL_X / All_Y if All_Y > 1e-10 and ALL_X > 1e-10 else 0
        r= ALL_X / All_Z if All_Z > 1e-10 else 0


    elif average == 'macro':

        weight = 1/(len(span_label2id)-1)
        for key_ in span_label2id.keys():
            if key_ == 'Other':
                continue
            key = span_label2id[key_]
            R = set(predicate_dict[key])
            T = set(true_dict[key])
            X = len(R & T)
            Y = len(R)
            Z = len(T)
            tmp_f1 = 2 * X / (Y + Z) if Y + Z != 0 else 0.
            tmp_p = X / Y if Y != 0 else 0.
            tmp_r = X / Z if Z != 0 else 0.
            f1 += weight * tmp_f1
            p += weight* tmp_p
            r += weight * tmp_r
            gentl_li.append([key_, tmp_f1, tmp_p, tmp_r,len(true_dict[key])])
            confusion_matrix.append([key_,X,Y-X,Z-X,len(true_dict[key])])
    else:
        raise ValueError("选择正确的evaluate mode:micro,macro,weight")

    average_f1 = 0.
    average_p = 0.
    average_r = 0.
    for line in gentl_li:
        entity_type,f1,p,r,_ = line
        average_f1 += f1
        average_p += p
        average_r += r
    
    average_f1 = average_f1/len(confusion_matrix)
    average_p = average_p/len(confusion_matrix)
    average_r = average_r/len(confusion_matrix)
    if wandb_dict:

        global_step = wandb_dict['global_step']
        epoch = wandb_dict['epoch']
        type_ = wandb_dict['type_']

        f1_key = '{}_{}_f1'.format(type_, 'macro')
        p_key = '{}_{}_p'.format(type_, 'macro')
        r_key = '{}_{}_r'.format(type_, 'macro')

        wandb.log(
            {"{}-epoch".format(type_): epoch, f1_key: average_f1, p_key: average_p, r_key: average_r},
            step=global_step)

    logger.info('average maro f1:{:.5f},macro-p:{:.5f},macro-r:{:.5f}'.format(average_f1,average_p,average_r))

    if verbose:
        gentl_print(gentl_li,average=average)
        gentl_print_confusion_matrix(confusion_matrix)

    return f1, p, r



def evaluate_span_micro(start_logits,end_logits,start_ids,end_ids,raw_text):
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


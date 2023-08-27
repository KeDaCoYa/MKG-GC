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
import json

import torch
import logging
import numpy as np
from collections import defaultdict
from ipdb import set_trace

from utils.function_utils import  gentl_print

logger = logging.getLogger('main.globalpointer_evaluate')



class GlobalPointerMetrics(object):

    def __init__(self):
        '''
            这个是global_pointer的专用评估器，包括分数计算和最终解码
            下面的y_pred和y_true的shape都是(batch_size,entity_type,seq_len,seq_len)
        '''
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        '''

        预测结果的f1值
        y_pred为预测的结果，下三角为很小的值，只关注上三角
        '''

        # 这一步直接获得所有可能为实体
        y_pred = torch.gt(y_pred, 0).float()
        # y_true * y_pred直接求得预测正确的个数
        # 然后得到f1
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        if y_pred.sum() == 0:
            return 0.
        else:
            return torch.sum(y_pred[y_true == 1]) / (y_pred.sum())

    def get_evaluate_fpr_micro(self, y_pred, y_true, type_weight, globalpointer_label2id, verbose=False):
        '''

        :param y_pred:
        :param y_true:
        :return:f1,p,r是已经经过加权之后的值
        '''

        predicate_dict = defaultdict(list)
        true_dict = defaultdict(list)

        y_pred = y_pred.data.cpu().numpy()
        y_true = y_true.data.cpu().numpy()


        for b, l, start, end in zip(*np.where(y_pred > 0)):
            predicate_dict[l].append((b, start, end))

        for b, l, start, end in zip(*np.where(y_true > 0)):
            true_dict[l].append((b, start, end))



        if type_weight is not None:
            gentl_li = []
            entity_type_score = {}
            for key_ in type_weight.keys():
                key = globalpointer_label2id[key_]
                R = set(predicate_dict[key])
                T = set(true_dict[key])
                X = len(R & T)
                Y = len(R)
                Z = len(T)
                tmp_f1 = 2 * X / (Y + Z) if Y + Z != 0 else 0.
                tmp_p = X / Y if Y != 0 else 0.
                tmp_r = X / Z if Z != 0 else 0.

                gentl_li.append([key_, tmp_f1, tmp_p, tmp_r])
                entity_type_score[key_] = {'p':tmp_p,'r':tmp_r,'f1':tmp_f1}

            if verbose:
                gentl_print(gentl_li)

            return entity_type_score
        else:

            R = set(predicate_dict[0])
            T = set(true_dict[0])
            X = len(R & T)
            Y = len(R)
            Z = len(T)
            tmp_f1 = 2 * X / (Y + Z) if Y + Z != 0 else 0.
            tmp_p = X / Y if Y != 0 else 0.
            tmp_r = X / Z if Z != 0 else 0.
            entity_type_score = {'p': tmp_p, 'r': tmp_r, 'f1': tmp_f1}
            return entity_type_score

    def get_evaluate_fpr_macro(self, y_pred, y_true):
        '''
            这是macro的评估方式
        :param y_pred:
        :param y_true:
        :return:
        '''
        y_pred = y_pred.data.cpu().numpy()
        y_true = y_true.data.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)

        f1, precision, recall = 2 * X / (Y + Z) if Y + Z != 0 else 0., X / Y if Y != 0 else 0., X / Z if Z != 0 else 0.
        return f1, precision, recall


    def decoder(self,raw_text,y_pred,y_true=None):
        '''
        这里一次只对一个数据进行decode
        :param y_pred:
        :param y_true:
        :return:
        '''
        predicate_dict = defaultdict(list)
        true_dict = defaultdict(list)

        y_pred = y_pred.data.cpu().numpy()

        all_ = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            predicate_dict[l].append((raw_text[start:end+1], start, end))


        if y_true:
            y_true = y_true.data.cpu().numpy()
            for b, l, start, end in zip(*np.where(y_true > 0)):
                true_dict[l].append((raw_text[start:end+1], start, end))

            return predicate_dict,true_dict
        return predicate_dict


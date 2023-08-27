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
from prettytable import PrettyTable


logger = logging.getLogger('main.globalpointer_evaluate')



def gentl_print(gentl_li,average='micro'):

    tb = PrettyTable()
    tb.field_names = ['实体类别','{}-f1'.format(average), '{}-precision'.format(average), '{}-recall'.format(average),'support']
    for a,f1,p,r,support in gentl_li:
        tb.add_row([a,round(f1,5),round(p,5),round(r,5),support])

    logger.info(tb)
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
    def transfer_to_BIO(self,y_pred):
        """
        :todo 转变为BIO格式
        """
    def get_evaluate_fpr(self, y_pred, y_true, globalpointer_label2id, type_weight=None,average='micro',verbose=False):
        """

        :param y_pred:
        :param y_true:
        :return:f1,p,r是已经经过加权之后的值
        """

        predicate_dict = defaultdict(list)
        true_dict = defaultdict(list)

        y_pred = y_pred.data.cpu().numpy()
        y_true = y_true.data.cpu().numpy()

        for b, l, start, end in zip(*np.where(y_pred > 0)):
            predicate_dict[l].append((b, start, end))

        for b, l, start, end in zip(*np.where(y_true > 0)):
            true_dict[l].append((b, start, end))

        f1 = 0.
        p = 0.
        r = 0.
        gentl_li = []
        confusion_matrix = []
        if average == 'weight':

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
                f1 += type_weight[key_] * tmp_f1
                p += type_weight[key_] * tmp_p
                r += type_weight[key_] * tmp_r
                gentl_li.append([key_, tmp_f1, tmp_p, tmp_r,len(true_dict[key])])
                confusion_matrix.append([key_, X, Y - X, Z - X, len(true_dict[key])])
        elif average == 'micro':

            X,Y,Z = 0,0,0
            for key_ in globalpointer_label2id.keys():

                key = globalpointer_label2id[key_]
                R = set(predicate_dict[key])
                T = set(true_dict[key])

                tmp_X = len(R & T)
                tmp_Y = len(R)
                tmp_Z = len(T)
                tmp_f1 = 2 * tmp_X / (tmp_Y + tmp_Z) if tmp_Y + tmp_Z != 0 else 0.
                tmp_p = tmp_X / tmp_Y if tmp_Y != 0 else 0.
                tmp_r = tmp_X / tmp_Z if tmp_Z != 0 else 0.
                gentl_li.append([key_, tmp_f1, tmp_p, tmp_r,len(true_dict[key])])
                confusion_matrix.append([key_, X, Y - X, Z - X, len(true_dict[key])])
                X += tmp_X
                Y += tmp_Y
                Z += tmp_Z
            f1 = 2 * X / (Y + Z) if Y + Z != 0 else 0.
            p = X / Y if Y != 0 else 0.
            r = X / Z if Z != 0 else 0.

        elif average == 'macro':

            weight = 1/(len(globalpointer_label2id)+1)
            for key_ in globalpointer_label2id.keys():
                key = globalpointer_label2id[key_]
                R = set(predicate_dict[key])
                T = set(true_dict[key])
                X = len(R & T)
                Y = len(R)
                Z = len(T)
                tmp_f1 = 2 * X / (Y + Z) if Y + Z != 0 else 0.
                tmp_p = X / Y if Y != 0 else 0.
                tmp_r = X / Z if Z != 0 else 0.
                f1 += weight * tmp_f1
                p += weight * tmp_p
                r += weight * tmp_r
                gentl_li.append([key_, tmp_f1, tmp_p, tmp_r,len(true_dict[key])])
                confusion_matrix.append([key_, X, Y - X, Z - X, len(true_dict[key])])
        else:
            raise ValueError("只能选择micro,macro,weight")
        if verbose:
            gentl_print(gentl_li, average=average)
            gentl_print_confusion_matrix(confusion_matrix)
        return f1,p,r


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


def gentl_print_confusion_matrix(gentl_li):
    """
    这个只是针对TP,TN,FP,FN的
    """
    tb = PrettyTable()
    tb.field_names = ['实体类别', 'TP', 'FP', 'FN','support']
    for a,TP, FP, FN, support in gentl_li:
        tb.add_row([a,TP,FP,FN, support])

    logger.info(tb)

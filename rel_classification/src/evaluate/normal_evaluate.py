# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2021/12/02
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/02: 
-------------------------------------------------
"""
from ipdb import set_trace


def normal_evalute(pred,true,label2id,type_weight,mode='micro'):
    '''
    暂时不精确到每个关系类别的具体performance
    :param pred:
    :param true:
    :param type_weight:
    :param mode:
    :return:
    '''

    acc = sum(pred==true)/len(pred)

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in enumerate(pred):
        if pred[i] == true[i]:
            tp += 1
        else:
            f


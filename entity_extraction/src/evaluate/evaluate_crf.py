# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  这是通用型的评估，主要是entity-level的评估
   Author :        kedaxia
   date：          2022/01/11
   Copyright:      (c) kedaxia 2022
-------------------------------------------------
   Change Activity:
                   2022/01/11: 
-------------------------------------------------
"""
from collections import defaultdict

from utils.function_utils import gentl_print, gentl_print_confusion_matrix


def entities_evaluate_fpr(predicate_entities_li,true_entities_li,label2id,type_weight,average='micro',verbose=False):
    '''

    '''
    # 首先将entities转变为dict

    predicate_dict = defaultdict(list)
    true_dict = defaultdict(list)
    for ent in predicate_entities_li:
        ent_type = ent['entity_type']
        predicate_dict[ent_type].append((ent_type,ent['start_idx'],ent['end_idx'],ent['entity_name']))
    for ent in true_entities_li:
        ent_type = ent['entity_type']
        true_dict[ent_type].append((ent_type,ent['start_idx'],ent['end_idx'],ent['entity_name']))

    f1 = 0.
    p = 0.
    r = 0.
    gentl_li = []
    confusion_matrix = []
    if average == 'weight':

        for key in type_weight.keys():


            R = set(predicate_dict[key])
            T = set(true_dict[key])
            X = len(R & T)
            Y = len(R)
            Z = len(T)
            tmp_f1 = 2 * X / (Y + Z) if Y + Z != 0 else 0.
            tmp_p = X / Y if Y != 0 else 0.
            tmp_r = X / Z if Z != 0 else 0.
            f1 += type_weight[key] * tmp_f1
            p += type_weight[key] * tmp_p
            r += type_weight[key] * tmp_r
            gentl_li.append([key, tmp_f1, tmp_p, tmp_r, len(true_dict[key])])
            confusion_matrix.append([key, X, Y - X, Z - X, len(true_dict[key])])

    elif average == 'micro':

        ALL_X = 0
        All_Y = 0
        All_Z = 0
        for key in label2id.keys():
            if key == 'O':
                continue

            R = set(predicate_dict[key])
            T = set(true_dict[key])
            X = len(R & T)
            Y = len(R)
            Z = len(T)
            ALL_X += X
            All_Y += Y
            All_Z += Z
            tmp_f1 = 2 * X / (Y + Z) if Y + Z != 0 else 0.
            tmp_p = X / Y if Y != 0 else 0.
            tmp_r = X / Z if Z != 0 else 0.

            gentl_li.append([key, tmp_f1, tmp_p, tmp_r, len(true_dict[key])])
            confusion_matrix.append([key, X, Y - X, Z - X, len(true_dict[key])])

        f1 = 2 * ALL_X / (All_Y + All_Z) if (All_Y + All_Z) > 1e-10 else 0
        p = ALL_X / All_Y if All_Y > 1e-10 and ALL_X > 1e-10 else 0
        r = ALL_X / All_Z if All_Z > 1e-10 else 0


    elif average == 'macro':

        weight = 1 / (len(label2id) - 1)
        for key in label2id.keys():
            if key == 'Other':
                continue

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
            gentl_li.append([key, tmp_f1, tmp_p, tmp_r, len(true_dict[key])])
            confusion_matrix.append([key, X, Y - X, Z - X, len(true_dict[key])])
    else:
        raise ValueError("选择正确的evaluate mode:micro,macro,weight")
    if verbose:
        gentl_print(gentl_li, average=average)
        gentl_print_confusion_matrix(confusion_matrix)
    return f1, p, r
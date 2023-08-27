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
from ipdb import set_trace


def align_predicted_labels_with_original_sentence_tokens(predicted_labels,orig_to_token_index,mask_length):
    """
        如果分词之后，那么就需要进行对齐，
        :param predicted_labels:这是预测之后的结果，shape=(batch_size,seq_len)#seq_len分词之后的长度
            但是这里面含有对[CLS],[SEP]等所有标签的预测
        我们只需要真正的labels...
        :param orig_to_token_index:这个长度是word-level的长度...
        :param mask_length:这是encode_plus之后的效果
    """
    mask_length = [len(x) for x in orig_to_token_index]

    aligned_predicted_labels = []
    for idx, (o_t_t_i, p_l_s) in enumerate(zip(orig_to_token_index, predicted_labels[1:-1])):
        # print(idx)
        temp = []
        actual_len = mask_length[idx]
        for i in range(len(o_t_t_i)):

            # 这里＋1是因为p_l_s里有[CLS]的预测，跳过这一个
            token_idx = o_t_t_i[i]
            if token_idx<len(p_l_s):
                temp.append(p_l_s[token_idx])
            else:
                temp.append(0) #对应O标签


        aligned_predicted_labels.append(temp)

    return aligned_predicted_labels



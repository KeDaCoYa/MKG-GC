# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2021/12/04
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/04: 
-------------------------------------------------
"""
import numpy as np

def sequence_padding(inputs,length=None,value=0,seq_dims=1,mode='post'):
    '''
    这里对数据进行pad，不同的batch里面使用不同的长度
    这个方法从多个方面考虑pad，写的很高级
    这个方法一般写不出来，阿西吧


    Numpy函数，将序列padding到同一长度
    按照一个batch的最大长度进行padding
    :param inputs:(batch_size,None),每个序列的长度不一样
    :param seq_dim: 表示对哪些维度进行pad，默认为1，只有当对label进行pad的时候，seq_dim=3,因为labels.shape=(batch_size,entity_type,seq_len,seq_len)
        因为一般都是对(batch_size,seq_len)进行pad，，，
    :param length: 这个是设置补充之后的长度，一般为None，根据batch的实际长度进行pad
    :param value:
    :param mode:
    :return:
    '''
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs],axis=0)  # length=np.array([max_batch_length])
    elif not hasattr(length,'__getitem__'): # 如果这个length的类别不是列表....,就进行转变
        length = [length]
    #logger.info('这个batch下面的最长长度为{}'.format(length[0]))

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
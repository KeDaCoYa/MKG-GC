# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :   这里实现各种Embedding，可能主要是position embedding
   Author :        kedaxia
   date：          2021/11/10
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/11/10: 
-------------------------------------------------
"""
import math
import torch
import torch.nn as nn

class TrainablePositionEmbedding(nn.Module):
    def __init__(self,model_dim,max_len=512,merge_mode='add'):
        '''
        这是苏剑林的，但是还不知道实际效果咋样
        这是可训练位置编码,这个相当于获得一个绝对位置，然后绝对位置经过Embedding得到位置编码
        :param model_dim: 相当于word embedding
        :param max_len:
        :param merge_mode:
        '''
        super(TrainablePositionEmbedding,self).__init__()
        self.max_len = max_len
        self.v_dim = model_dim
        self.merge_mode = merge_mode
        #这个embedding首先设定为0
        self.embeddings = nn.Embedding(self.max_len,model_dim) #这个位置编码，最长为max_len
        # 首次将其初始化为0
        nn.init.zeros_(self.embeddings.weight)
        nn.init.zeros_(self.embeddings.bias)

    def forward(self,x,r = 0):
        '''

        :param x:
        :param r: 表示起始位置，这样可以计算相对距离
        :return:
        '''
        batch_size,seq_len = x.shape()[0],x.shape()[1]
        pid = torch.arange(seq_len)
        pid = pid.unsqueeze(0) # pid变为(1,seq_len)
        pid = pid.repeat(batch_size,1) #shape=(batch_size,seq_len)
        pid = torch.abs(pid-torch.IntTensor(r))
        res = self.embeddings(pid)
        if self.merge_mode == 'add':
            return res+x
        else:
            return torch.cat(x,res)





class SinCosPositionEmbedding(nn.Module):
    def __init__(self,model_dim,max_len=512):
        '''
        BERT里的位置编码向量
        :param model_dim: 这是embedding需要变成的dim
        :param max_len:
        '''
        super(SinCosPositionEmbedding,self).__init__()
        # 因为对于一个序列，按照构建对应的embedding，即为(seq_len,model_dim)
        pe = torch.zeros(max_len,model_dim).float()
        pe.requires_grad = False  #注意这个不能训练


        position = torch.arange(0,max_len).float().unsqueeze(1) #shape为(seq_len,1)
        div_term = ((torch.arange(0,max_len,2).float()*-(math.log(10000.0)/model_dim))).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]







# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :   这里实现各种注意力机制
   Author :        kedaxia
   date：          2021/11/10
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/11/10: 
-------------------------------------------------
"""
import torch
import torch.nn as nn
from math import sqrt
from ipdb import set_trace
class MultiHeadAttention(nn.Module):
    def __init__(self,input_dim,num_heads,head_dims):
        '''
        多头注意力机制
        :param input_dim: 一般为word_embedding_dim,或者hidden_dim
        :param num_heads:8
        :param head_dims: 有时候也可以细分为dimk和dimv,类似selfAttention下面的样例，16
        '''
        super(MultiHeadAttention,self).__init__()
        self.num_heads = num_heads
        self.head_dims = head_dims
        self.output_dims = num_heads*head_dims
        self.q = nn.Linear(input_dim,self.output_dims)
        self.k = nn.Linear(input_dim,self.output_dims)
        self.v = nn.Linear(input_dim,self.output_dims)
        self._norm_fact = 1 / sqrt(self.output_dims)

    def mask(self, x, mask, mode='mul'):
        '''
        对计算完成的注意力进行mask
        :param mask:
        :param mode:
        :return:
        '''
        if mask is None:
            return
        else:
            # 这里进行补充维数，让其和x.shape一致
            for _ in range(len(x.shape) - len(mask.shape)):
                mask = mask.unsqueeze(len(mask.shape))  # 在最后一维加上,变为[batch_size,seq_len,1,1]
            if mode == 'mul':  # mul相当于直接进行掩码，相当于对非mask的地方进行保留，其他地方去掉
                return torch.mul(x, mask)
            else:  # 'add'  这相当于将非mask的地方给变得非常小，这个区别并不知道...
                return x - (1 - mask) * 1e10

    def forward(self, x, mask=None):
        '''
        这里可能也直接输入[]
        :param x:
        :param mask:(batch_size,seq_len)
        :return:output:shape = (batch_size,seq_len,dim_v)
        '''

        batch_size,seq_len = x.shape[0],x.shape[1]
        Q = self.q(x)  # Q,K.shape = (batch_size,seq_len,dim_k)
        K = self.k(x)
        V = self.v(x)  # V.shape = (batch_size,seq_len,dim_v)
        # 转换为多头形式
        Q = Q.reshape(-1,batch_size,seq_len,self.head_dims)  #shape = (num_heads,batch_size,seq_len,head_dims)
        K = K.reshape(-1,batch_size,seq_len,self.head_dims)  #shape = (num_heads,batch_size,seq_len,head_dims)
        V = V.reshape(-1,batch_size,seq_len,self.head_dims)  #shape = (num_heads,batch_size,seq_len,head_dims)


        # 首先计算Q*K的值
        K = K.permute(0,1, 3, 2)  # K.shape = (num_heads,batch_size,head_dims,seq_len)
        res = torch.matmul(Q, K)  # shape = (num_heads,batch_size,seq_len,seq_len)

        # 开始mask
        res = res.permute(1, 2, 3, 0)
        res = self.mask(res, mask)
        res = res.permute(3, 0, 1, 2) # shape=(num_heads,batch_size,seq_len,seq_len)

        # 这里有一个修正因子，但是Su的有时候没有添加
        attn = nn.Softmax(dim=-1)(res) * self._norm_fact  #attn.shape = num_head,batch_size ,seq_len ,seq_len

        # 计算最后的输出
        output = torch.matmul(attn, V)  #num_head,batch_size,seq_len,head_dims
        #转变
        output = output.view(-1,seq_len,self.output_dims) #转变为(batch_size,seq_len,output_dims)

        return output


class SelfAttention(nn.Module):
    def __init__(self,input_dim,dim_k,dim_v):
        '''

        :param input_dim: 这个其实就是word_dim
        :param dim_k:  Q,K使用的是相同的dim
        :param dim_v:  V使用的是不同的dim
        '''
        super(SelfAttention,self).__init__()
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.v = nn.Linear(input_dim,dim_v)
        self._norm_fact = 1 / sqrt(dim_k)
    def mask(self,x,mask,mode='mul'):
        '''
        对计算完成的注意力进行mask
        :param mask:
        :param mode:共两种方式'mul','add', 相当于一个是0进行处理
        :return:
        '''
        if mask is None:
            return x
        else:
            #这里进行补充维数，让其和x.shape一致
            for _ in range(len(x.shape) - len(mask.shape)):
                mask = mask.unsqueeze(len(mask.shape))  # 在最后一维加上,变为[batch_size,seq_len,1,1]
            if mode == 'mul':  # mul相当于直接进行掩码，相当于对非mask的地方进行保留，其他地方去掉
                return torch.mul(x, mask)
            else:  # 'add'  这相当于将非mask的地方给变得非常小，
                return x - (1 - mask) * 1e10
    def forward(self,x,mask=None):
        '''
        这里可能也直接输入[]
        :param x:
        :param mask
        :return:output:shape = (batch_size,seq_len,dim_v)
        '''
        Q = self.q(x) #Q,K.shape = (batch_size,seq_len,dim_k)
        K = self.k(x)
        V = self.v(x) # V.shape = (batch_size,seq_len,dim_v)
        #首先计算Q*K的值
        K = K.permute(0,2,1)  #K.shape = (batch_size,dim_k,seq_len)

        res = torch.bmm(Q,K) #shape = (batch_size,seq_len,seq_len)
        res = self.mask(res,mask)
        #开始mask

        attn = nn.Softmax(dim=-1)(res)*self._norm_fact

        #计算最后的输出
        output = torch.matmul(attn,V) #shape=(batch_size,seq_len,dim_v)
        return output





class NormalAttention(nn.Module):
    def __init__(self):
        super(NormalAttention, self).__init__()


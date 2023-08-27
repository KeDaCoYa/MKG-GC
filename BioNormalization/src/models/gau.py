# -*- encoding: utf-8 -*-
"""
@File    :   gau.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/5/5 22:41   
@Description :   None 

"""
import torch
import torch.nn as nn
from einops import rearrange
from ipdb import set_trace
from torch import einsum
import torch.nn.functional as F

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder


class OffsetScale(nn.Module):
    def __init__(self, dim, heads=1):
        """

        :param dim: 这个是qk_dim,128
        :param heads: 在GAU=2,在FLASH=4
        """
        super().__init__()
        # 这个相当于W,b
        # 只是初始化为1和0
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        """

        :param x: shape=(batch_size,seq_len,qk_dim)
        :return:
        """
        # out.shape = (batch_size,seq_len,head,qk_dim)
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        # 返回两个x.shape
        return out.unbind(dim=-2)


class GAU(nn.Module):
    def __init__(
            self,
            config,
            dim=768,
            query_key_dim=128,
            expansion_factor=2.,
            add_residual=True,
            causal=False,
            dropout=0.,
            norm_klass=nn.LayerNorm
    ):
        """
        这个是用于构建FLASH_QUAD模型
        :param dim: hidden_dim=512
        :param query_key_dim: 这是注意力的dim ,128
        :param expansion_factor: 表示扩展一下维度
        :param add_residual: 是否残差连接
        :param causal:
        :param dropout:
        :param norm_klass: 选择标准化类别
        """
        super().__init__()

        hidden_dim = int(expansion_factor * dim)

        self.norm = norm_klass(dim)
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )

        self.offsetscale = OffsetScale(query_key_dim, heads=2)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.add_residual = add_residual

    def forward(self, hidden_states, attention_mask=None, rel_pos_bias=None):

        seq_len, device = hidden_states.shape[-2], hidden_states.device

        normed_x = self.norm(hidden_states)
        # to_hidden是将normed_x转变为(batch_size,seq_len,hidden_dim*2)
        # 这里其实就是将x给分为注意力：v,以及用于gate的input
        # v.shape=gate.shape=(batch_size,seq_len,hidden_dim)
        # v用于之后的qk乘机，得到注意力
        v, gate = self.to_hidden(normed_x).chunk(2, dim=-1)
        # qk.shape=(batch_size,seq_len,qk_dim)
        qk = self.to_qk(normed_x)
        # 这个其实将qk进行分开，得到q,k
        q, k = self.offsetscale(qk)
        # 这就是注意力的前面结果，shape=(batch_size,seq_len,seq_len)
        sim = einsum('b i d, b j d -> b i j', q, k) / seq_len

        # 添加相对位置编码
        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        attn = F.relu(sim) ** 2
        attn = self.dropout(attn)

        if exists(attention_mask):
            mask = rearrange(attention_mask, 'b j -> b 1 j')
            attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)
        # out.shape = (batch_size,seq_len,hidden_dim)
        # attn.shape = (batch_size,seq_len,seq_len)
        out = einsum('b i j, b j d -> b i d', attn, v)
        # 直接点乘

        out = out * gate

        out = self.to_out(out)  # out.shape=(batch_size,seq_len,hidden_dim)

        if self.add_residual:
            out = out + hidden_states
        # output.shape=(batch_size,max_seq,dim*3)
        return (out,)

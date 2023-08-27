import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from ipdb import set_trace



def silu(x):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """
    return x * torch.sigmoid(x)


def rope(x, dim):
    """
        RoPE position embedding.
        RoPE位置编码
    """
    shape = x.shape
    if isinstance(dim, int):
        dim = [dim]
    spatial_shape = [shape[i] for i in dim]
    total_len = 1
    for i in spatial_shape:
        total_len *= i
    position = torch.reshape(
        torch.arange(total_len, dtype=x.dtype,
                     device=x.device), spatial_shape
    )
    for i in range(dim[-1] + 1, len(shape) - 1, 1):
        position = position.unsqueeze(-1)
    half_size = shape[-1] // 2
    freq_seq = -torch.arange(half_size, dtype=x.dtype, device=x.device) / float(
        half_size
    )
    inv_freq = 10000 ** freq_seq
    sinusoid = torch.einsum("...,d->...d", position, inv_freq)
    sin = sinusoid.sin()
    cos = sinusoid.cos()
    x1, x2 = torch.chunk(x, 2, dim=-1)

    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class ScaleNorm(nn.Module):
    """
        这个就是在原论文中经常用到的scale_offset
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scala = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """
        一个学习参数scala
        然后将x进行一个简单的标准化
        """
        # 这是均方根值
        mean_square = (x ** 2).mean(dim=-1, keepdim=True)
        # 这相当于为scala*x/sqrt(mean_square+eps),
        x = x * torch.rsqrt(mean_square + self.eps) * self.scala
        return x


class GAU(nn.Module):
    """
    GAU block.
    Input shape: (batch_size,seq_len,hidden_size)
    """

    def __init__(
        self,
        hidden_size=768,
        expansion_factor=2,
        s=128,
        norm_type="layer_norm",
        eps=1e-5,
        hidden_act="silu",
        max_position_embeddings=512,
    ):
        """
        expansion_factor:其实就是一个维数扩充因子，没啥太大含义
            在标准transformer的中一般为4*hidden，即expansion_factor=4
            这里为2
        max_position_embeddings:序列的最大长度
        """
        super().__init__()
        self.s = s  # 这是head size
        self.e = int(hidden_size * expansion_factor)

        self.uv = nn.Linear(hidden_size, 2 * self.e + self.s)
        # self.weight.shape=(2,128)
        # 下面两个参数就是为了从z生成q,k:也就是注意力
        self.weight = nn.Parameter(torch.randn(2, self.s))
        self.bias = nn.Parameter(torch.zeros(2, self.s))

        # 这是GAU的最后输出层
        self.o = nn.Linear(self.e, hidden_size)
        self.LayerNorm = (
            nn.LayerNorm(hidden_size, eps=eps)
            if norm_type == "layer_norm"
            else ScaleNorm(eps=eps)
        )
        self.w = nn.Parameter(torch.randn(2 * max_position_embeddings - 1))
        self.a = nn.Parameter(torch.randn(1, self.s))
        self.b = nn.Parameter(torch.randn(1, self.s))

        self.act_fn = silu # 这是门控机制用到的激活函数，为silu

        self.max_position_embeddings = max_position_embeddings

        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.w, std=0.02)
        nn.init.normal_(self.a, std=0.02)
        nn.init.normal_(self.b, std=0.02)

    def rel_pos_bias(self, seq_len):
        """
            生成相对位置偏置
            Relative position bias.
        """
        if seq_len <= 512:
            # Construct Toeplitz matrix directly when the sequence length is less than 512
            t = F.pad(self.w[: 2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
            t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
            r = (2 * seq_len - 1) // 2
            t = t[..., r:-r]
        else:
            # Construct Toeplitz matrix using RoPE when the sequence length is over 512.
            a = rope(self.a.repeat(seq_len, 1), dim=0)
            b = rope(self.b.repeat(seq_len, 1), dim=0)
            t = torch.einsum("mk,nk ->mn", a, b)

        return t

    def forward(self, x, attention_mask=None, output_attentions=False, causal=False):
        """

        """

        seq_len = x.shape[1]
        shortcut, x = x, self.LayerNorm(x)
        # uv.shape=(bs,seq_len,self.e*2+self.s)
        uv = self.uv(x)
        # 对uv结果进行切分，u.shape=(batch_size,seq_len,self.e=2*hidden_size)
        # v.shape=(batch_size,seq_len,self.e=2*hidden_size)
        # base.shape=(batch_size,seq_len,self.s),base就是论文中的z
        u, v, base = torch.split(self.act_fn(uv), [self.e, self.e, self.s], dim=-1)
        # Generate Query (q) and Key (k) from base.
        base = torch.einsum("...r,hr->...hr", base, self.weight) + self.bias #base.shape=(batch_size,seq_len,2,s=128)
        # 这是得到rope 位置编码
        base = rope(base, dim=1)
        #q.shape=k.shape = (batch_size,seq_len,s)
        q, k = torch.unbind(base, dim=-2)
        # Calculate the quadratic attention.
        # qk.shape=(batch_size,seq_len,seq_len)
        qk = torch.einsum("bnd,bmd->bnm", q, k)
        # 这个是相对位置偏置，我也不知道有啥用...
        bias = self.rel_pos_bias(self.max_position_embeddings)[:, :seq_len, :seq_len]

        # 计算得到A，也就是attention
        kernel = torch.square(torch.relu(qk / self.max_position_embeddings + bias))
        # attention_mask
        if attention_mask is not None:
            assert attention_mask.ndim == 2
            # attention_mask.shape=(batch_size,seq_len)
            # attention_mask[:, None, :].shape=(batch_size,1,seq_len),attention_mask[:, :, None].shape=(batch_size,seq_len,1)
            # attn_mask.shape=(batch_size,)
            attn_mask = (attention_mask[:, None, :] * attention_mask[:, :, None]).type_as(x)
            kernel *= attn_mask
        # 如果是auto regressive language model
        if causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0)
            kernel *= causal_mask

        x = u * torch.einsum("bnm,bme->bne", kernel, v)
        x = self.o(x)
        if output_attentions:
            return x + shortcut, kernel
        return (x + shortcut,)

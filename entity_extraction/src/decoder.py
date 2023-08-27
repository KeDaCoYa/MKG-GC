# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2021/11/21
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/11/21: 
-------------------------------------------------
"""
import torch
import torch.nn as nn
from ipdb import set_trace


class GlobalPointer(nn.Module):
    def __init__(self,config,hidden_size,inner_dim=64,use_RoPE=True):
        '''

        :param config:
        :param num_tags: 实体类别个数，CNeEE为9种类别
        :param inner_dim: 超参数，模型的一个dim
        :param hidden_size:接在globalpointer前面的dim
        :param use_RoPE:使用RoPE位置编码
        '''
        super(GlobalPointer,self).__init__()
        self.num_tags = config.num_gp_class
        self.inner_dim = inner_dim
        self.hidden_size = hidden_size

        self.use_RoPE = use_RoPE


        # 最后的一个全连接层
        self.dense = nn.Linear(self.hidden_size, self.num_tags * self.inner_dim * 2)#Linear(in_features=1024, out_features=1152, bias=True)


    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        '''
        param
        '''
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)  # shape=(seq_len,1)
        # 这里是针对out_dim进行，对同一个位置的不同dim进行，因为原始为2i，2i+1，所以只能截断一半
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        sin_embeddings = torch.sin(embeddings)  # shape = (seq_len,output_dim/2)
        cos_embeddings = torch.cos(embeddings)

        # 这里dim=-1，从而可以让其进行交叉
        embeddings = torch.stack([sin_embeddings, cos_embeddings], dim=-1)  # embeddings.shape=(seq_len,output_dim/2,2)
        # 这里表示只对batchsize进行重复，其他位置不改变
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))

        return embeddings
    def forward(self,encoder_output,token_ids=None, attention_mask=None):
        '''

        :param encoder_output:  这是encoder的结果
        :param token_ids: 这个好像就是为了制造attention
        :param attention_mask: 有bert_tokenizer得到的mask，shape=(batch_size,seq_len),没有的则是根据token_ids构建
        :param device:
        :return: logits
        '''
        self.device = token_ids.device

        # last_hidden_state:(batch_size, seq_len, hidden_size) = torch.Size([16, 128, 1024])
        last_hidden_state = encoder_output

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]  # seq_len=128

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state)  # torch.Size([16, 128, 1152])
        # 这里给切分为多个元组了，对outputs进行切分，对dim=-1维进行切分，其中每一块的大小维self.inner_dim*2,最终切了9块
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)  # (batch_size,seq_len=128,dim=128)

        outputs = torch.stack(outputs,dim=-2)  # outputs.shape = outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)=torch.Size([16, 128, 9, 128])
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim) = torch.Size([16, 128, 9, 64])
        # 这是打分函数的两个组成
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        # 这个位置编码好像是不用训练学习的，直接计算学习得到
        if self.use_RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim).to(device=self.device)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)

            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # logits:(batch_size, ent_type_size, seq_len, seq_len)=(16,9,128,128)  #这个就是最终的形式
        # 这里的einsum非常高级，优点看不懂，其实它相等于
        # 通俗理解写法
        qw = qw.permute(0, 2, 1, 3)
        kw = kw.permute(0, 2, 3, 1)
        logits = torch.matmul(qw, kw)
        # -----高级写法------------
        # logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        if attention_mask is None:
            attention_mask = torch.gt(token_ids, 0).type(torch.int32)

        # attention_mask.shape=(batch_size,1,1,seq_len)
        # attention_mask必须按照上述的shape进行unsqueeze,然后进行expand
        # 若是attention_mask.shape=(batch_size,1,seq_len，1)然后进行mask，那么就会出现错误，无法以sentence为单位进行mask掉..
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.num_tags, seq_len, seq_len)

        logits = logits * pad_mask - (1 - pad_mask) * 1e12  # 这是mask的一种方式，不再是让其他区域为0，而是变为很小的值

        # 排除下三角，只保留上三角的分数
        # 这里mask就是下三角为1(不包含对角线)
        mask = torch.tril(torch.ones_like(logits), -1)
        # 将下三角区域的值弄得非常小
        logits = logits - mask * 1e12
        logits = logits / self.inner_dim ** 0.5


        return logits


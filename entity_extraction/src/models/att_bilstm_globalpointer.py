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
import torch
import torch.nn as nn


import logging
from config import NormalConfig
from src.attentions import SelfAttention,MultiHeadAttention
from src.decoder import GlobalPointer
from utils.loss_utils import multilabel_categorical_crossentropy

logger = logging.getLogger('main.att_bilstm_globalpointer')




class Att_BiLSTM_GlobalPointer(nn.Module):
    def __init__(self,config:NormalConfig):
        super(Att_BiLSTM_GlobalPointer,self).__init__()
        self.word_embedding_dim = config.word_embedding_dim
        self.vocab_size = config.vocab_size
        self.learning_rate = config.learning_rate
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_bilstm_layers  # 表示bilstm的层数
        self.dropout_prob = config.dropout_prob

        self.num_class = config.num_gp_class

        if config.use_pretrained_embedding:
            logger.info('将预训练的词嵌入加载到nn.Embedding')
            self.word_embedding = nn.Embedding(self.vocab_size, self.word_embedding_dim)
            self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embeddings))
        else:
            self.word_embedding = nn.Embedding(self.vocab_size, self.word_embedding_dim)
        # batch_first,是将batch_size作为第一个，即接受输入(batch_size,input_dim,hidden_dim)的数据
        self.lstm = nn.LSTM(self.word_embedding_dim, self.hidden_size, batch_first=True, bidirectional=True,
                            num_layers=self.num_layers, dropout=self.dropout_prob)
        if config.attention_mechanism == 'sa':
            self.attention = SelfAttention(self.hidden_size*2,128,128)
        elif config.attention_mechanism == 'mha':
            self.attention = MultiHeadAttention(self.hidden_size*2,8,16)
        # 因为是双层的lstm，将前向和后向的lstm输出进行叠加，所以hidden_size*2

        # 这是CRF的转移矩阵

        self.globalpointer = GlobalPointer(config, hidden_size=self.hidden_size, use_RoPE=True)

    def forward(self, token_ids,labels=None):
        '''

        :param x:
        :param lengths:这是一个batch中，每个batch的真实长度，因为每个文本序列长度都不同，需要pad为统一长度
        :return:
        '''
        out = self.word_embedding(token_ids)
        # out.shape=(batch_size,seq_len,hidden_size*2)
        # 不对填充的pad进行计算
        out, _ = self.lstm(out)

        # out.shape = [batch.seq_len,num_class)

        attention_masks = torch.gt(torch.unsqueeze(token_ids, 2), 0).type(torch.int32)
        attention_masks = attention_masks.squeeze(-1)

        attn_output = self.attention(out,attention_masks) #attn_output.shape = (batch_size,seq_len,hidden_dim)


        logits = self.globalpointer(encoder_output=attn_output, token_ids=token_ids)
        if labels is None:
            return logits
        else:
            loss = multilabel_categorical_crossentropy(logits, labels)
            return loss,logits




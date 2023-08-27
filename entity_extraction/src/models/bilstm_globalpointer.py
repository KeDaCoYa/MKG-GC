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

from src.decoder import GlobalPointer
from utils.loss_utils import multilabel_categorical_crossentropy

logger = logging.getLogger('main.bilstm_globalpointer')




class BiLSTM_GlobalPointer(nn.Module):

    def __init__(self, config, inner_dim=64):
        super(BiLSTM_GlobalPointer, self).__init__()

        self.word_embedding_dim = config.word_embedding_dim
        self.vocab_size = config.vocab_size
        self.learning_rate = config.learning_rate
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_bilstm_layers  # 表示bilstm的层数
        self.num_gp_class = config.num_gp_class

        self.dropout = config.dropout_prob

        self.inner_dim = inner_dim

        if config.use_pretrained_embedding:
            logger.info('将预训练的词嵌入加载到nn.Embedding')
            self.word_embedding = nn.Embedding(self.vocab_size, self.word_embedding_dim,padding_idx=0)
            self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embeddings))
        else:
            self.word_embedding = nn.Embedding(self.vocab_size, self.word_embedding_dim)


        # batch_first,是将batch_size作为第一个，即接受输入(batch_size,input_dim,hidden_dim)的数据
        self.lstm = nn.LSTM(self.word_embedding_dim, self.hidden_size, batch_first=True, bidirectional=True,
                            num_layers=self.num_layers, dropout=self.dropout)

        # 因为是双层的lstm，将前向和后向的lstm输出进行叠加，所以hidden_size*2

        self.mid_linear = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_prob)
        )

        # 需要加入一个中间层

        self.globalpointer = GlobalPointer(config, hidden_size=self.hidden_size*2,use_RoPE=True)
        self.criterion = multilabel_categorical_crossentropy


    def forward(self, token_ids,labels=None):
        '''

        :param token_ids:shape = (batch_size,seq_len)
        :return:
        '''

        out = self.word_embedding(token_ids)
        # out = self.dropout1(out)
        # out.shape=(batch_size,seq_len,hidden_size*2)
        # 不对填充的pad进行计算
        out, _ = self.lstm(out)

        #out = self.mid_linear(out)

        logits = self.globalpointer(encoder_output=out,token_ids=token_ids)
        if labels is None:
            return logits
        else:

            loss = multilabel_categorical_crossentropy(logits, labels)

            return loss,logits

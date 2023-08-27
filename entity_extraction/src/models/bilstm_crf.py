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

from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import logging
from config import NormalConfig


logger = logging.getLogger('main.model_bilstm_crf')


class BiLSTM_CRF(nn.Module):

    def __init__(self, config: NormalConfig):
        super(BiLSTM_CRF, self).__init__()

        self.word_embedding_dim = config.word_embedding_dim
        self.vocab_size = config.vocab_size
        self.learning_rate = config.learning_rate
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_bilstm_layers  # 表示bilstm的层数
        self.dropout = 0.5

        self.num_class = config.num_crf_class

        # 创建一个根据训练集训练的词嵌入

        if config.use_pretrained_embedding:
            logger.info('首次将预训练的词嵌入加载到nn.Embedding')
            self.embedding = nn.Embedding(self.vocab_size, self.word_embedding_dim)
            self.embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embeddings))
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.word_embedding_dim)

        # batch_first,是将batch_size作为第一个，即接受输入(batch_size,input_dim,hidden_dim)的数据
        self.lstm = nn.LSTM(self.word_embedding_dim, self.hidden_size, batch_first=True, bidirectional=True,
                            num_layers=self.num_layers, dropout=self.dropout)

        # 因为是双层的lstm，将前向和后向的lstm输出进行叠加，所以hidden_size*2
        self.classification = nn.Linear(self.hidden_size * 2, self.num_class)
        # 这是CRF的转移矩阵

        self.crf_model = CRF(num_tags=self.num_class, batch_first=True)

    def forward(self, token_ids, labels,attention_masks):
        '''

        :param x:
        :param lengths:这是一个batch中，每个batch的真实长度，因为每个文本序列长度都不同，需要pad为统一长度
        :return:
        '''

        out = self.embedding(token_ids)
        # out.shape=(batch_size,seq_len,hidden_size*2)
        # 不对填充的pad进行计算
        out, _ = self.lstm(out)

        # out.shape = [batch.seq_len,num_class)
        emissions = self.classification(out)


        if labels is not None:

            loss = -1. * self.crf_model(emissions=emissions, tags=labels.long(), mask=attention_masks.byte(),
                                        reduction='mean')
            tokens_out = self.crf_model.decode(emissions=emissions, mask=attention_masks.byte())
            return loss, tokens_out
        else:
            tokens_out = self.crf_model.decode(emissions=emissions, mask=attention_masks.byte())
            return tokens_out

    def forward1(self, x, lengths, labels):
        '''
        这里采用pad的方式...
        :param x:
        :param lengths:这是一个batch中，每个batch的真实长度，因为每个文本序列长度都不同，需要pad为统一长度
        :return:
        '''
        out = self.embedding(x)
        # out.shape=(batch_size,seq_len,hidden_size*2)
        # 不对填充的pad进行计算

        packed = pack_padded_sequence(out, lengths, batch_first=True)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        # out.shape = [batch.seq_len,num_class)
        emissions = self.classification(out)
        attention_masks = torch.gt(torch.unsqueeze(x, 2), 0).type(torch.int32)
        attention_masks = attention_masks.squeeze(-1)

        if labels is not None:
            loss = -1. * self.crf_model(emissions=emissions, tags=y.long(), mask=attention_masks.byte(),reduction='mean')
            tokens_out = self.crf_model.decode(emissions=emissions, mask=attention_masks.byte())
            return loss,tokens_out
        else:
            tokens_out = self.crf_model.decode(emissions=emissions, mask=attention_masks.byte())
            return tokens_out
        # batch_size,max_len,out_size = out.size()
        # #将crf扩展为(batch_size,max_len,num_class,num_class),这个加号相当于将transition加到crf_score中的每个batch上
        # crf_score = out.unsqueeze(2).expand(-1,-1,out_size,-1)+self.transition.unsqueeze(0)
        # #crf_score = [batch_size,seq_len,out_size,out_size]
        # return crf_score


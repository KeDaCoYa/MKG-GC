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
import torch
from torch.nn.utils.rnn import pad_sequence

from config import MyBertConfig
from src.models.bert_model import BaseBert

from ipdb import set_trace

import torch.nn as nn

from torchcrf import CRF

import logging



class Bert_BiLSTM_CRF(BaseBert):

    def __init__(self, config:MyBertConfig):
        super(Bert_BiLSTM_CRF, self).__init__(config=config)


        # bert的输出层dim
        out_dims = self.bert_config.hidden_size
        self.config = config
        self.num_layers = 1
        self.hidden_size = 128
        self.lstm = nn.LSTM(out_dims, self.hidden_size, batch_first=True, bidirectional=True,
                            num_layers=self.num_layers, dropout=0.5)

        # 因为是双层的lstm，将前向和后向的lstm输出进行叠加，所以hidden_size*2
        self.classifier = nn.Linear(self.hidden_size * 2,config.num_crf_class)
        # 这是CRF的转移矩阵
        self.dropout = nn.Dropout(config.dropout_prob)
        self.crf_model = CRF(num_tags=config.num_crf_class, batch_first=True)

        # todo:是否初始化的影响
        init_blocks = [self.classifier]  #不知道bilstm是否可以初始化
        #self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)
        #冻结这些参数，只留下最后一点点进行微调


    def forward(self, input_ids, token_type_ids=None, attention_masks=None, labels=None,input_token_starts=None):
        '''
        这是tokenizer的前向传播
        :param x:
        :param lengths:这是一个batch中，每个batch的真实长度，因为每个文本序列长度都不同，需要pad为统一长度
        :return:
        '''
        if self.config.bert_name == 'biobert':

            bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_masks,
                                           token_type_ids=token_type_ids.long())
            sequence_output = bert_outputs[0]  # shape=(batch_size,seq_len,hidden_dim)=[32, 55, 768]
        elif self.config.bert_name == 'kebiolm':
            bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_masks,
                                           token_type_ids=token_type_ids, return_dict=False)
            sequence_output = bert_outputs[2]  # shape=(batch_size,seq_len,hidden_dim)=[32, 55, 768]
        else:
            raise ValueError


        # 将subwords的第一个subword作为之前的token representation
        origin_sequence_output = []

        for layer, starts in zip(sequence_output, input_token_starts):
            res = layer[starts]
            origin_sequence_output.append(res)

        # 这里的max_len和上面的seq_len已经不一样了，因为这里是按照token-level,而不是subword-level
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)

        # seq_out.shape=(batch_size,seq_len,768)
        # 不对填充的pad进行计算
        out, _ = self.lstm(padded_sequence_output) #out.shape=(batch_size,seq_len,hidden_size*2 = 256)
        out = self.dropout(out)
        emissions = self.classifier(out)

        #todo： 这里测试两个lossmask是否一样，应该是一样的...
        loss_mask = labels.gt(-1)

        # loss_mask = torch.zeros((out.shape[0], out.shape[1])).to(input_ids.device)
        # for i, tmp in enumerate(input_token_starts):
        #     lens = len(tmp)
        #     loss_mask[i][:lens] = 1


        if labels is not None:
            loss = -1. * self.crf_model(emissions=emissions, tags=labels.long(), mask=loss_mask.byte(),
                                        reduction='mean')
            tokens_out = self.crf_model.decode(emissions=emissions, mask=loss_mask.byte())
            return loss,tokens_out
        else:
            tokens_out = self.crf_model.decode(emissions=emissions, mask=loss_mask.byte())
            return tokens_out

    def origin_forward(self, token_ids, attention_masks, token_type_ids, labels):
        '''
        这是token_level的前向传播
        :param x:
        :param lengths:这是一个batch中，每个batch的真实长度，因为每个文本序列长度都不同，需要pad为统一长度
        :return:
        '''
        bert_outputs = self.bert_model(input_ids=token_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
        seq_out = bert_outputs[0]
        # seq_out.shape=(batch_size,seq_len,768)
        # 不对填充的pad进行计算
        out, _ = self.lstm(seq_out) #out.shape=(batch_size,seq_len,hidden_size*2 = 256)


        emissions = self.classifier(out)
        attention_masks = torch.gt(torch.unsqueeze(token_ids, 2), 0).type(torch.int32)
        attention_masks = attention_masks.squeeze(-1)

        if labels is not None:
            loss = -1. * self.crf_model(emissions=emissions, tags=labels.long(), mask=attention_masks.byte(),
                                        reduction='mean')
            tokens_out = self.crf_model.decode(emissions=emissions, mask=attention_masks.byte())
            return loss,tokens_out
        else:
            tokens_out = self.crf_model.decode(emissions=emissions, mask=attention_masks.byte())
            return tokens_out

# -*- encoding: utf-8 -*-
"""
@File    :   base_layers.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/5/8 8:36   
@Description :   None 

"""
import torch.nn as nn
from transformers import BertLayer


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)



class EncoderLayer(nn.Module):
    def __init__(self, config, dropout_rate=0.2):
        super(EncoderLayer, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)

        config.hidden_act='swish'
        self.bert_layer = BertLayer(config)


    def forward(self, x):

        x = self.dropout(x)
        x = self.bert_layer(x)[0]
        x = self.dropout(x)

        return x

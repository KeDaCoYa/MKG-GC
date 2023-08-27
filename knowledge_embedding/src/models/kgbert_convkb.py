# -*- encoding: utf-8 -*-
"""
@File    :   kg_bert.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/17 15:24   
@Description :   这是我自创的模型，但是没啥用，因为kgbert本身就不太行...
"""
import logging

import torch
from ipdb import set_trace

from src.models.bert_model import BaseBert

logger = logging.getLogger("main.kgbert_convkb")

import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel


class KGBertConvKBModel(BertPreTrainedModel):
    """
    这个就是一个基于BERT的简单二分类模型，使用[CLS]的输出结果
    """

    def __init__(self, config, num_labels):
        super(KGBertConvKBModel, self).__init__(config)
        self.config = config
        self.num_labels = num_labels

        if config.bert_name in ['biobert', 'scibert','bert']:
            logger.info("加载{}模型".format(config.bert_dir))
            self.bert = BertModel.from_pretrained(config.bert_dir, output_hidden_states=True,
                                                  hidden_dropout_prob=config.dropout_prob)
        if config.freeze_bert:
            self.freeze_parameter(config.freeze_layers)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)


        # convkb部分

        self.config.out_channels = 64
        self.config.kernel_size = 1
        self.config.convkb_drop_prob = 0.5

        self.conv1_bn = nn.BatchNorm2d(1)
        # Conv2d(1, 64, kernel_size=(1, 3), stride=(1, 1))
        self.conv_layer = nn.Conv2d(1, self.config.out_channels, (self.config.kernel_size, 3))  # kernel size x 3
        # BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_bn = nn.BatchNorm2d(self.config.out_channels)
        # dropout=0.5
        self.dropout = nn.Dropout(self.config.convkb_drop_prob)
        self.non_linearity = nn.ReLU()  # you should also tune with torch.tanh() or torch.nn.Tanh()
        # Linear(in_features=3200, out_features=1, bias=False)

        self.fc_layer = nn.Linear((self.config.hidden_size - self.config.kernel_size + 1) * self.config.out_channels, 1,
                                  bias=False)

        self.classifier = nn.Linear((self.config.hidden_size - self.config.kernel_size + 1) * self.config.out_channels,
                                    num_labels)



    def freeze_parameter(self, freeze_layers):
        """
        对指定的layers进行冻结参数
        :param freeze_layers: 格式为['layer.10','layer.11','bert.pooler','out.']
        :return:
        """

        for name, param in self.bert.named_parameters():

            for ele in freeze_layers:
                if ele in name:
                    param.requires_grad = False
        # 验证一下实际情况
        # for name,param in self.bert_model.named_parameters():
        #     if param.requires_grad:
        #         print(name,param.size())

    @staticmethod
    def _init_weights(blocks, **kwargs):
        '''
        对指定的blocks进行参数初始化,只对指定layer进行初始化
        主要是对BERT之后的一些layer进行初始化
        :param blocks:
        :param kwargs:
        :return:
        '''
        for block in blocks:
            for module in block.modules():  # 就是获取Sequential的里面的每一个layer
                if isinstance(module, nn.Linear):  # 只对全连接层进行初始化
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                    elif isinstance(module, nn.Embedding):
                        nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                    elif isinstance(module, nn.LayerNorm):  # 这个没看懂为什么这样子进行初始化,全初始化为1和0
                        nn.init.ones_(module.weight)
                        nn.init.zeros_(module.bias)

    def _calc(self, h, r, t):
        """

        :param h:shape = (batch_size,1,dim)
        :param r:
        :param t:
        :return:
        """

        # h = h.unsqueeze(1)  # bs x 1 x dim
        # r = r.unsqueeze(1)
        # t = t.unsqueeze(1)

        conv_input = torch.cat([h, r, t], 1)  # bs x 3 x dim
        # conv_input.shape = (batch_size,hidden_size,3)
        conv_input = conv_input.transpose(1, 2)
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)
        conv_input = self.conv1_bn(conv_input)
        out_conv = self.conv_layer(conv_input)
        out_conv = self.conv2_bn(out_conv)
        out_conv = self.non_linearity(out_conv)
        out_conv = out_conv.view(-1, (self.config.hidden_size - self.config.kernel_size + 1) * self.config.out_channels)


        return out_conv

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None,head_mask=None,rel_mask=None,tail_mask=None):
        """

        :param input_ids:
        :param token_type_ids:
        :param attention_mask:
        :param labels:
        :param head_mask:  下面三个为ehad,tail,rel的mask,shape=(batch_size,seq_len)
        :param rel_mask:
        :param tail_mask:
        :return:
        """
        encoder_output, pooled_output, _ = self.bert(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids )

        # pooled_output = self.dropout(pooled_output)

        head_mask = head_mask.unsqueeze(1)
        tail_mask = tail_mask.unsqueeze(1)
        rel_mask = rel_mask.unsqueeze(1)

        head_representation = self.dropout(torch.matmul(head_mask,encoder_output))
        tail_representation = self.dropout(torch.matmul(rel_mask,encoder_output))
        rel_representation = self.dropout(torch.matmul(tail_mask,encoder_output))
        # todo: 由于head_representation等未姜伟，其dim会很大，之后考虑来神经网络进行降维
        convkb_out = self._calc(head_representation,tail_representation,rel_representation)

        logits = self.classifier(convkb_out)

        #logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()

            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return logits, loss
        else:
            return logits

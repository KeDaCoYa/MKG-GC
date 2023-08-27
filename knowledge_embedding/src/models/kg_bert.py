# -*- encoding: utf-8 -*-
"""
@File    :   kg_bert.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/17 15:24   
@Description :   None 
"""
import logging

from ipdb import set_trace

from src.models.bert_model import BaseBert

logger = logging.getLogger("main.kgbert")

import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel


class KGBertModel(BertPreTrainedModel):
    """
    这个就是一个基于BERT的简单二分类模型，使用[CLS]的输出结果
    """

    def __init__(self, config, num_labels):
        super(KGBertModel, self).__init__(config)
        self.config = config
        self.num_labels = num_labels

        if config.bert_name in ['biobert', 'scibert','bert']:
            logger.info("加载{}模型".format(config.bert_dir))
            self.bert = BertModel.from_pretrained(config.bert_dir, output_hidden_states=True,
                                                  hidden_dropout_prob=config.dropout_prob)
        if config.freeze_bert:
            self.freeze_parameter(config.freeze_layers)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

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

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):

        _, pooled_output, _ = self.bert(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()

            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return logits, loss
        else:
            return logits

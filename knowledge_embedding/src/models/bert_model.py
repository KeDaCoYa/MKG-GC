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

import os
from ipdb import set_trace
import logging

import torch
import torch.nn as nn
from transformers import BertModel



from config import MyBertConfig

logger = logging.getLogger('main.bert_model')

class BaseBert(nn.Module):
    def __init__(self,config:MyBertConfig):
        '''
        这是最基础的BERT模型加载，加载预训练的模型
        :param config:
        :param bert_dir:
        :param dropout_prob:
        '''
        super(BaseBert, self).__init__()
        if config.bert_name in ['biobert','scibert']:
            self.bert_model = BertModel.from_pretrained(config.bert_dir,output_hidden_states=True,hidden_dropout_prob=config.dropout_prob)


        # elif config.bert_name == 'flash_quad':
        #
        #     self.set_flash_quad_config(config)
        #
        #     #model = FLASHQuadForMaskedLM(config)
        #     checkpoint = torch.load(os.path.join(config.bert_dir, 'model.pt'))
        #     model.load_state_dict(checkpoint)
        #     self.bert_model = model.flash_quad
        #
        # elif config.bert_name == 'wwm_bert':
        #
        #     self.set_wwm_bert_config(config)
        #     #model = WWWMBertForPreTraining(config)
        #
        #     checkpoint = torch.load(os.path.join(config.bert_dir,'model.pt'))
        #     model.load_state_dict(checkpoint)
        #     self.bert_model = model.bert
        #
        # self.bert_config = self.bert_model.config

        if config.freeze_bert:
            self.freeze_parameter(config.freeze_layers)


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
            for module in block.modules(): #就是获取Sequential的里面的每一个layer
                if isinstance(module, nn.Linear): # 只对全连接层进行初始化
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                    elif isinstance(module, nn.Embedding):
                        nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                    elif isinstance(module, nn.LayerNorm):  # 这个没看懂为什么这样子进行初始化,全初始化为1和0
                        nn.init.ones_(module.weight)
                        nn.init.zeros_(module.bias)

    def freeze_parameter(self,freeze_layers):
        '''
        对指定的layers进行冻结参数
        :param layers: 格式为['layer.10','layer.11','bert.pooler','out.']
        :return:
        '''

        for name,param in self.bert_model.named_parameters():

            for ele in freeze_layers:
                if ele in name:
                    param.requires_grad = False
        #验证一下实际情况
        # for name,param in self.bert_model.named_parameters():
        #     if param.requires_grad:
        #         print(name,param.size())

    def set_wwm_bert_config(self,config):
        """
            这个是专门为wwm_bert的成功使用而增加的参数
        """
        config.attention_probs_dropout_prob = 0.1
        config.hidden_act = "gelu"
        config.hidden_dropout_prob = 0.1
        config.hidden_size = 768
        config.initializer_range = 0.02
        config.intermediate_size = 3072
        config.max_position_embeddings = 512
        config.num_attention_heads = 12
        config.num_hidden_layers = 12
        config.type_vocab_size = 2
        config.vocab_size = 31090
        config.layer_norm_eps=1e-5
        config.train_type='scratch'

    def set_flash_quad_config(self,config):
        """
            这个是专门为wwm_bert的成功使用而增加的参数
        """

        config.vocab_size = 31090
        config.hidden_size = 768
        config.num_hidden_layers = 24
        config.max_position_embeddings = 512
        config.type_vocab_size = 2
        config.initializer_range = 0.02
        config.layer_norm_eps = 1e-5
        config.pad_token_id = 0
        config.expansion_factor = 2
        config.s = 128
        config.norm_type = "scale_norm"
        config.gradient_checkpointing = False
        config.dropout = 0.0
        config.hidden_act = "swish"
        config.classifier_dropout = 0.1
        config.train_type = 'scratch'
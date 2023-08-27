# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2022/01/20
   Copyright:      (c) kedaxia 2022
-------------------------------------------------
   Change Activity:
                   2022/01/20: 
-------------------------------------------------
"""


import logging
from ipdb import set_trace

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

from transformers.activations import ACT2FN

from config import MyBertConfig

logger = logging.getLogger('main.bert_model')


class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config: MyBertConfig):
        super().__init__(config)

        # BERT模型

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.config = config
        if config.train_type == 'scratch':
            self.init_weights()
        else:
            logger.info('加载预训练模型权重:{}'.format(config.bert_dir))
            self.bert = BertModel.from_pretrained(config.bert_dir)
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, mlm_labels=None, nsp_labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_attentions=True,
                            output_hidden_states=True,
                            )

        sequence_output, pooled_output = outputs[:2]
        # 得到的是模型的输出，
        mlm_scores, nsp_scores = self.cls(sequence_output, pooled_output)
        if mlm_labels is not None and nsp_labels is not None:
            mlm_loss = self.loss(mlm_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            nsp_loss = self.loss(nsp_scores.view(-1, 2), nsp_labels.view(-1))
            return mlm_scores, nsp_scores, mlm_loss, nsp_loss
        return mlm_scores, nsp_scores


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        """
        定义在预训练过程中的所有无监督任务
        1. MLM
        2. NST
        :param config:
        """
        super().__init__()
        # MLM任务
        self.predictions = BertLMPredictionHead(config)
        # NSP任务，就是一个二分类
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        '''
        传入的是bert的outputs
        :param sequence_output: shape=(batch_size,seq_len,hidden_size)
        :param pooled_output: shape=(batch_size,hidden_size) 这是[CLS]的输出...
        :return:
        '''
        # 这是MLM task
        prediction_scores = self.predictions(sequence_output)
        # 这是NSP task
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        """

        :param config:
        """
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        # 选择一个激活函数，一般是gelu
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

logger = logging.getLogger('main.wwm_bert')
class WWWMBertForPreTraining(BertPreTrainedModel):
    def __init__(self, config:MyBertConfig):
        super().__init__(config)

        # BERT模型
        self.bert = BertModel(config)
        self.cls = WWMBertPreTrainingHeads(config)
        self.config=config

        if config.train_type == 'scratch':
            self.init_weights()
        else:
            logger.info('加载预训练模型权重:{}'.format(config.bert_dir))
            self.bert = BertModel.from_pretrained(config.bert_dir)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(self,input_ids=None,token_type_ids=None,attention_mask=None,mlm_labels=None):

        outputs = self.bert(input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
            output_hidden_states=True,
        )

        sequence_output = outputs[0]
        # 得到的是模型的输出，
        mlm_scores = self.cls(sequence_output)
        mlm_loss = self.loss_fn(mlm_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
        return mlm_scores,mlm_loss


class WWMBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        '''
        定义在预训练过程中的所有无监督任务
        只有Whole word mask 一个无监督任务
        :param config:
        '''
        super().__init__()
        # MLM任务
        self.predictions = BertLMPredictionHead(config)


    def forward(self, sequence_output):
        '''
        传入的是bert的outputs
        :param sequence_output: shape=(batch_size,seq_len,hidden_size)

        :return:
        '''
        # 这是MLM task
        prediction_scores = self.predictions(sequence_output)

        return prediction_scores

# -*- encoding: utf-8 -*-
"""
@File    :   biolpbert.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/5/6 21:30   
@Description :   None 

"""
from ipdb import set_trace
from torch import nn
from transformers import PreTrainedModel, BertPreTrainedModel, BertModel
from transformers.modeling_bert import BertLMPredictionHead


class LpBertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # BERT模型
        self.bert = BertModel(config)
        self.bert.from_pretrained(config.bert_dir)
        # 这个是mrm任务的预测
        self.mrm = WMM(config)

        self.head_mem = WMM(config)
        self.tail_mem = WMM(config)
        self.config=config

        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(self,head_mem,tail_mem,mrm,head_lem_label_ids=None,tail_mem_label_ids=None,mrm_label_ids=None):
        """
            三个参数都是字典形式存储，共四个key
                input_ids
                attention_mask
                token_type_ids
                label_ids
        """


        head_mem_outputs = self.bert(**head_mem)[0]
        tail_mem_outputs = self.bert(**tail_mem)[0]
        mrm_outputs = self.bert(**mrm)[0]

        # 得到的是模型的输出，
        mrm_scores = self.mrm(mrm_outputs)
        head_mem_scores = self.head_mem(head_mem_outputs)
        tail_mem_scores = self.tail_mem(tail_mem_outputs)

        mrm_loss = self.loss_fn(mrm_scores.view(-1, self.config.vocab_size), mrm_label_ids.view(-1))
        head_mem_loss = self.loss_fn(head_mem_scores.view(-1, self.config.vocab_size),head_lem_label_ids.view(-1))
        tail_mem_loss = self.loss_fn(tail_mem_scores.view(-1, self.config.vocab_size),tail_mem_label_ids.view(-1))



        return mrm_scores,head_mem_scores,tail_mem_scores,head_mem_loss,tail_mem_loss,mrm_loss


class WMM(nn.Module):
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


# -*- encoding: utf-8 -*-
"""
@File    :   multi_rbert_large.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/5/13 14:48   
@Description :   None 

"""
import copy

import torch
from ipdb import set_trace
from torch import nn

from config import MyBertConfig
from src.models.base_layers import FCLayer, EncoderLayer
from src.models.bert_model import EntityMarkerBaseModel


class MultiRBERTForAlldataLarge(EntityMarkerBaseModel):

    def __init__(self, config: MyBertConfig, scheme=1):
        super(MultiRBERTForAlldataLarge, self).__init__(config)

        self.num_labels = 2
        self.config = config
        self.scheme = scheme
        # 下面这两个dim可以进行修改
        self.cls_dim = self.bert_config.hidden_size
        self.entity_dim = self.bert_config.hidden_size

        self.cls_fc_layer = FCLayer(self.bert_config.hidden_size, self.cls_dim, self.config.dropout_prob)
        self.entity_fc_layer = FCLayer(self.bert_config.hidden_size, self.entity_dim, self.config.dropout_prob)

        # classificer1 PPI
        # classificer2 DDI
        # classificer3 CPI
        # classificer4 GDI
        # classificer5 CDI

        self.encoder1 = EncoderLayer(config)
        self.encoder2 = EncoderLayer(config)
        self.encoder3 = EncoderLayer(config)
        self.encoder4 = EncoderLayer(config)
        self.encoder5 = EncoderLayer(config)

        self.classifier1 = nn.Linear(self.classifier_dim, self.num_labels)
        self.classifier2 = nn.Linear(self.classifier_dim, self.num_labels)
        self.classifier3 = nn.Linear(self.classifier_dim, self.num_labels)
        self.classifier4 = nn.Linear(self.classifier_dim, self.num_labels)
        self.classifier5 = nn.Linear(self.classifier_dim, self.num_labels)

        if self.config.freeze_bert:
            self.freeze_parameter(config.freeze_layers)

            # 模型层数的初始化初始化
        nn.init.xavier_normal_(self.cls_fc_layer.linear.weight)
        nn.init.constant_(self.cls_fc_layer.linear.bias, 0.)

        nn.init.xavier_normal_(self.entity_fc_layer.linear.weight)
        nn.init.constant_(self.entity_fc_layer.linear.bias, 0.)

        nn.init.xavier_normal_(self.classifier1.weight)
        nn.init.constant_(self.classifier1.bias, 0.)

        nn.init.xavier_normal_(self.classifier2.weight)
        nn.init.constant_(self.classifier2.bias, 0.)

        nn.init.xavier_normal_(self.classifier3.weight)
        nn.init.constant_(self.classifier3.bias, 0.)

        nn.init.xavier_normal_(self.classifier4.weight)
        nn.init.constant_(self.classifier4.bias, 0.)

        nn.init.xavier_normal_(self.classifier5.weight)
        nn.init.constant_(self.classifier5.bias, 0.)



    def forward(self, input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask, rel_type=None):
        """
        但是这里将实体对放在一个[CLS]sent<SEP>中，而不是两个sent之中


        :param input_ids:
        :param token_type_ids:
        :param attention_masks:
        :param e1_mask:  这里e1_mask和e2_mask覆盖了special tag([s1][e1],[s2][e2])，所以这里需要需要切片以下
        :param e2_mask:
        :param labels:
        :param rel_type: 这个是表明每个输入数据的是哪种类别，DDI,CPI,PPI,GDI,CDI五种类别
        :return:
        """

        outputs = self.bert_model(
            input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]  # shape=(batch_size,seq_len,hidden_size)
        pooled_output = outputs[1]  # [CLS],shape = (batch_size,hidden_size)=(16,768)

        # shape=(batch_size,seq_len,hidden_size)
        concat_h = self.get_pool_output(sequence_output, pooled_output, input_ids, e1_mask, e2_mask)

        # 在这里开始多任务的分支
        if rel_type is None or rel_type[0].item() == 0:
            rel1_logits = self.classifier1(concat_h)
            rel2_logits = self.classifier2(concat_h)
            rel3_logits = self.classifier3(concat_h)
            rel4_logits = self.classifier4(concat_h)
            rel5_logits = self.classifier5(concat_h)

            return rel1_logits + rel2_logits + rel3_logits + rel4_logits + rel5_logits
        else:

            rel1_idx = copy.deepcopy(rel_type)
            rel2_idx = copy.deepcopy(rel_type)
            rel3_idx = copy.deepcopy(rel_type)
            rel4_idx = copy.deepcopy(rel_type)
            rel5_idx = copy.deepcopy(rel_type)

            rel1_idx[rel1_idx != 1] = 0
            rel2_idx[rel2_idx != 2] = 0
            rel3_idx[rel3_idx != 3] = 0
            rel4_idx[rel4_idx != 4] = 0
            rel5_idx[rel5_idx != 5] = 0

            rel1_seq_out =rel1_idx.unsqueeze(-1).unsqueeze(-1)*sequence_output
            rel1_seq_out = self.encoder1(rel1_seq_out)
            concat_h = self.get_pool_output(rel1_seq_out, rel1_seq_out[:, 0, :], input_ids, e1_mask, e2_mask)
            rel1_output = rel1_idx.unsqueeze(-1) * concat_h
            rel1_logits = self.classifier1(rel1_output)

            rel2_seq_out = rel2_idx.unsqueeze(-1).unsqueeze(-1) * sequence_output
            rel2_seq_out = self.encoder2(rel2_seq_out)
            concat_h = self.get_pool_output(rel2_seq_out, rel2_seq_out[:, 0, :], input_ids, e1_mask, e2_mask)
            rel2_output = rel2_idx.unsqueeze(-1) * concat_h
            rel2_logits = self.classifier2(rel2_output)

            rel3_seq_out = rel3_idx.unsqueeze(-1).unsqueeze(-1) * sequence_output
            rel3_seq_out = self.encoder3(rel3_seq_out)
            concat_h = self.get_pool_output(rel3_seq_out, rel3_seq_out[:, 0, :], input_ids, e1_mask, e2_mask)
            rel3_output = rel3_idx.unsqueeze(-1) * concat_h
            rel3_logits = self.classifier3(rel3_output)

            rel4_seq_out = rel4_idx.unsqueeze(-1).unsqueeze(-1) * sequence_output
            rel4_seq_out = self.encoder4(rel4_seq_out)
            concat_h = self.get_pool_output(rel4_seq_out, rel4_seq_out[:, 0, :], input_ids, e1_mask, e2_mask)
            rel4_output = rel4_idx.unsqueeze(-1) * concat_h
            rel4_logits = self.classifier4(rel4_output)

            rel5_seq_out = rel5_idx.unsqueeze(-1).unsqueeze(-1) * sequence_output
            rel5_seq_out = self.encoder5(rel5_seq_out)
            concat_h = self.get_pool_output(rel5_seq_out, rel5_seq_out[:, 0, :], input_ids, e1_mask, e2_mask)
            rel5_output = rel5_idx.unsqueeze(-1) * concat_h
            rel5_logits = self.classifier5(rel5_output)

            logits = rel1_logits + rel2_logits + rel3_logits + rel4_logits + rel5_logits
            # Softmax
        if labels is not None:
            labels = (labels > 0).long()
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()

                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            return loss, logits

        return logits  # (loss), logits, (hidden_states), (attentions)

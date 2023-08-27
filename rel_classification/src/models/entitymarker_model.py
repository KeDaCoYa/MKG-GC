# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :   这里的模型是single entity marker
   Author :        kedaxia
   date：          2021/12/03
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/03: 
-------------------------------------------------
"""

from ipdb import set_trace

from config import BertConfig
from src.models.base_layers import FCLayer
from src.models.bert_model import EntityMarkerBaseModel
import torch.nn as nn




class SingleEntityMarkersREModel(EntityMarkerBaseModel):
    def __init__(self, config: BertConfig, scheme=1):
        super(SingleEntityMarkersREModel, self).__init__(config)

        self.num_labels = config.num_labels
        self.config = config
        self.scheme = scheme
        # 下面这两个dim可以进行修改
        self.cls_dim = self.bert_config.hidden_size
        self.entity_dim = self.bert_config.hidden_size

        self.cls_fc_layer = FCLayer(self.bert_config.hidden_size, self.cls_dim, self.config.dropout_prob)
        if self.scheme == -113:
            self.entity_fc_layer = FCLayer(self.bert_config.hidden_size*3, self.entity_dim, self.config.dropout_prob)
        else:
            self.entity_fc_layer = FCLayer(self.bert_config.hidden_size, self.entity_dim, self.config.dropout_prob)


        self.classifier = FCLayer(
            self.classifier_dim,
            self.config.num_labels,
            self.config.dropout_prob,
            use_activation=False,
        )
        if self.config.freeze_bert:
            self.freeze_parameter(config.freeze_layers)

        self.init_layer()

        if self.num_labels == 1:
            self.loss_fct = nn.MSELoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()



    def init_layer(self):
        # 模型层数的初始化初始化
        nn.init.xavier_normal_(self.cls_fc_layer.linear.weight)
        nn.init.constant_(self.cls_fc_layer.linear.bias, 0.)

        nn.init.xavier_normal_(self.entity_fc_layer.linear.weight)
        nn.init.constant_(self.entity_fc_layer.linear.bias, 0.)

        nn.init.xavier_normal_(self.classifier.linear.weight)
        nn.init.constant_(self.classifier.linear.bias, 0.)

    def forward(self, input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask):
        """
        但是这里将实体对放在一个[CLS]sent<SEP>中，而不是两个sent之中

        :param input_ids:
        :param token_type_ids:
        :param attention_masks:
        :param e1_mask:  这里e1_mask和e2_mask覆盖了special tag([s1][e1],[s2][e2])，所以这里需要需要切片以下
        :param e2_mask:
        :param labels:
        :return:
        """

        outputs = self.bert_model(
            input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]  # shape=(batch_size,seq_len,hidden_size)
        pooled_output = outputs[1]  # [CLS],shape = (batch_size,hidden_size)

        concat_h = self.get_pool_output(sequence_output, pooled_output, input_ids, e1_mask, e2_mask,attention_mask=attention_masks)
        logits = self.classifier(concat_h)


        if labels is not None:
            if self.num_labels == 1:
                loss = self.loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            return loss, logits

        return logits


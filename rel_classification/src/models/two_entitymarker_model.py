# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2021/12/22
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/22: 
-------------------------------------------------
"""

from ipdb import set_trace
import numpy as np

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss,BCELoss

from config import MyBertConfig
from src.models.bert_model import EntityMarkerBaseModel
from src.models.entitymarker_model import FCLayer


class CrossEntityMarkerReModel(EntityMarkerBaseModel):
    def __init__(self, config:MyBertConfig):
        super(CrossEntityMarkerReModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.dropout = nn.Dropout(config.dropout_prob)
        self.loss_fn = CrossEntropyLoss(reduction='none')

        self.scheme = config.scheme


        self.cls_dim = self.bert_config.hidden_size
        self.entity_dim = self.bert_config.hidden_size
        self.cls_fc_layer = FCLayer(self.bert_config.hidden_size, self.cls_dim, self.config.dropout_prob)
        self.entity_fc_layer = FCLayer(self.bert_config.hidden_size, self.entity_dim, self.config.dropout_prob)

        self.classifier = nn.Linear(self.classifier_dim, self.num_labels)

        nn.init.xavier_normal_(self.cls_fc_layer.linear.weight)
        nn.init.constant_(self.cls_fc_layer.linear.bias, 0.)

        nn.init.xavier_normal_(self.entity_fc_layer.linear.weight)
        nn.init.constant_(self.entity_fc_layer.linear.bias, 0.)

        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.)


    def get_entity_representation(self, sequence_output, sequence_pool_output, input_ids, e1_mask=None, e2_mask=None):

        if self.scheme == -1:
            # 这是rbert的方式,[[CLS]],[s1]ent1[e1],[s2]ent2[e2]]
            # 这个还对pool，ent1,ent2额外使用MLP进行转变...
            e1_h = self.entity_average(sequence_output, e1_mask)
            e2_h = self.entity_average(sequence_output, e2_mask)

            # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
            # pooled_output.shape=(batch_size,768)
            pooled_output = self.cls_fc_layer(sequence_pool_output)

            e1_h = self.entity_fc_layer(e1_h)
            e2_h = self.entity_fc_layer(e2_h)

            # Concat -> fc_layer
            concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)  # torch.Size([16, 2304])
        elif self.scheme == 1:
            # 这个和1相反，不适用额外的linear层
            # 这是rbert的方式,[[CLS]],[s1]ent1[e1],[s2]ent2[e2]]

            e1_h = self.entity_average(sequence_output, e1_mask)
            e2_h = self.entity_average(sequence_output, e2_mask)

            concat_h = torch.cat([sequence_pool_output, e1_h, e2_h], dim=-1)  # torch.Size([16, 2304])
        elif self.scheme == 2:
            # [pool_output,[s1],[e1],[s2],[e2]]
            seq_tags = []
            for each_tag in [self.config.ent1_start_tag_id, self.config.ent1_end_tag_id, self.config.ent2_start_tag_id,
                             self.config.ent2_end_tag_id]:
                seq_tags.append(self.special_tag_representation(sequence_output, input_ids, each_tag))
            concat_h = torch.cat((sequence_pool_output, *seq_tags), dim=1)
        elif self.scheme == -2:
            # [pool_output,[s1],[e1],[s2],[e2]]


            ent1_start = self.special_tag_representation(sequence_output, input_ids, self.config.ent1_start_tag_id)
            ent1_end = self.special_tag_representation(sequence_output, input_ids, self.config.ent1_end_tag_id)

            ent2_start = self.special_tag_representation(sequence_output, input_ids, self.config.ent2_start_tag_id)
            ent2_end = self.special_tag_representation(sequence_output, input_ids, self.config.ent2_end_tag_id)

            ent1_start = self.entity_fc_layer(ent1_start)
            ent1_end = self.entity_fc_layer(ent1_end)
            ent2_start = self.entity_fc_layer(ent2_start)
            ent2_end = self.entity_fc_layer(ent2_end)

            sequence_pool_output = self.cls_fc_layer(sequence_pool_output)

            concat_h = torch.cat([sequence_pool_output, ent1_start,ent1_end,ent2_start,ent2_end], dim=-1)  # torch.Size([16, 2304])

        elif self.scheme == 3:
            # [[CLS],[s1],[s2]]
            seq_tags = []
            for each_tag in [self.config.ent1_start_tag_id, self.config.ent2_start_tag_id]:
                seq_tags.append(self.special_tag_representation(sequence_output, input_ids, each_tag))
            concat_h = torch.cat((sequence_pool_output, *seq_tags), dim=1)
        elif self.scheme == -3:
            # [[CLS],[s1],[s2]]
            ent1_start = self.special_tag_representation(sequence_output, input_ids, self.config.ent1_start_tag_id)
            ent2_start = self.special_tag_representation(sequence_output, input_ids, self.config.ent2_start_tag_id)

            ent1_start = self.entity_fc_layer(ent1_start)
            ent2_start = self.entity_fc_layer(ent2_start)
            sequence_pool_output = self.cls_fc_layer(sequence_pool_output)
            concat_h = torch.cat([sequence_pool_output, ent1_start, ent2_start],
                                 dim=-1)  # torch.Size([16, 2304])
        elif self.scheme == 4:
            # [[s1],[s2]]
            seq_tags = []
            for each_tag in [self.config.ent1_start_tag_id, self.config.ent2_start_tag_id]:
                seq_tags.append(self.special_tag_representation(sequence_output, input_ids, each_tag))
            concat_h = torch.cat(seq_tags, dim=1)
        elif self.scheme == -4:
            ent1_start = self.special_tag_representation(sequence_output, input_ids, self.config.ent1_start_tag_id)
            ent2_start = self.special_tag_representation(sequence_output, input_ids, self.config.ent2_start_tag_id)

            ent1_start = self.entity_fc_layer(ent1_start)
            ent2_start = self.entity_fc_layer(ent2_start)

            concat_h = torch.cat([ent1_start, ent2_start],dim=-1)  # torch.Size([16, 2304])
        elif self.scheme == 5:
            # [[CLS]]
            concat_h = sequence_pool_output  # shape=(batch_size,hidden_size*2)
        elif self.scheme == 6:
            # [[s1]ent1[e1],[s2]ent2[ent2]]

            e1_mask = e1_mask.unsqueeze(1)
            e2_mask = e2_mask.unsqueeze(1)
            ent1_rep = torch.bmm(e1_mask.float(), sequence_output)
            ent2_rep = torch.bmm(e2_mask.float(), sequence_output)
            ent1_rep = ent1_rep.squeeze(1)
            ent2_rep = ent2_rep.squeeze(1)
            concat_h = torch.cat([ent1_rep, ent2_rep], dim=1)
        elif self.scheme == -6:
            e1_h = self.entity_average(sequence_output, e1_mask)
            e2_h = self.entity_average(sequence_output, e2_mask)

            e1_h = self.entity_fc_layer(e1_h)
            e2_h = self.entity_fc_layer(e2_h)

            concat_h = torch.cat([e1_h, e2_h], dim=-1)  # torch.Size([16, 2304])
        elif self.scheme == 7:
            # [ent1,ent2]
            # 取消e1_mask,e2_mask在[s1][e1],[s2][e2]的label，也就是直接设为0
            # e1_start_idx, e1_end_idx = self.get_ent_position(e1_mask)
            # e2_start_idx, e2_end_idx = self.get_ent_position(e2_mask)
            # e1_mask[e1_start_idx] = 0
            # e1_mask[e1_end_idx] = 0
            # e2_mask[e2_start_idx] = 0
            # e2_mask[e2_end_idx] = 0
            bs, seq_len = e1_mask.shape
            tmp_e1_mask = e1_mask.cpu().numpy().tolist()
            tmp_e2_mask = e2_mask.cpu().numpy().tolist()
            for i in range(bs):
                tmp_e1 = tmp_e1_mask[i]
                tmp_e2 = tmp_e2_mask[i]
                start_idx_e1 = tmp_e1.index(0)
                end_idx_e1 = start_idx_e1 + sum(tmp_e1) - 1
                start_idx_e2 = tmp_e2.index(0)
                end_idx_e2 = start_idx_e2 + sum(tmp_e2) - 1
                e1_mask[start_idx_e1][end_idx_e1] = 0
                e2_mask[start_idx_e2][end_idx_e2] = 0

            e1_mask = e1_mask.unsqueeze(1)
            e2_mask = e2_mask.unsqueeze(1)
            ent1_rep = torch.bmm(e1_mask.float(), sequence_output)
            ent2_rep = torch.bmm(e2_mask.float(), sequence_output)
            ent1_rep = ent1_rep.squeeze(1)
            ent2_rep = ent2_rep.squeeze(1)
            concat_h = torch.cat([ent1_rep, ent2_rep], dim=1)
        elif self.scheme == -7:
            bs, seq_len = e1_mask.shape
            tmp_e1_mask = e1_mask.cpu().numpy().tolist()
            tmp_e2_mask = e2_mask.cpu().numpy().tolist()
            for i in range(bs):
                tmp_e1 = tmp_e1_mask[i]
                tmp_e2 = tmp_e2_mask[i]
                start_idx_e1 = tmp_e1.index(0)
                end_idx_e1 = start_idx_e1 + sum(tmp_e1) - 1
                start_idx_e2 = tmp_e2.index(0)
                end_idx_e2 = start_idx_e2 + sum(tmp_e2) - 1
                e1_mask[start_idx_e1][end_idx_e1] = 0
                e2_mask[start_idx_e2][end_idx_e2] = 0

            e1_mask = e1_mask.unsqueeze(1)
            e2_mask = e2_mask.unsqueeze(1)
            ent1_rep = torch.bmm(e1_mask.float(), sequence_output)
            ent2_rep = torch.bmm(e2_mask.float(), sequence_output)
            ent1_rep = ent1_rep.squeeze(1)
            ent2_rep = ent2_rep.squeeze(1)

            ent1_rep = self.entity_fc_layer(ent1_rep)
            ent2_rep = self.entity_fc_layer(ent2_rep)

            concat_h = torch.cat([ent1_rep, ent2_rep], dim=1)
        elif self.scheme == 8:
            # [[CLS],ent1,ent2]
            bs, seq_len = e1_mask.shape
            tmp_e1_mask = e1_mask.cpu().numpy().tolist()
            tmp_e2_mask = e2_mask.cpu().numpy().tolist()
            for i in range(bs):
                tmp_e1 = tmp_e1_mask[i]
                tmp_e2 = tmp_e2_mask[i]
                start_idx_e1 = tmp_e1.index(0)
                end_idx_e1 = start_idx_e1 + sum(tmp_e1) - 1
                start_idx_e2 = tmp_e2.index(0)
                end_idx_e2 = start_idx_e2 + sum(tmp_e2) - 1
                e1_mask[start_idx_e1][end_idx_e1] = 0
                e2_mask[start_idx_e2][end_idx_e2] = 0

            e1_mask = e1_mask.unsqueeze(1)
            e2_mask = e2_mask.unsqueeze(1)
            ent1_rep = torch.bmm(e1_mask.float(), sequence_output)
            ent2_rep = torch.bmm(e2_mask.float(), sequence_output)
            ent1_rep = ent1_rep.squeeze(1)
            ent2_rep = ent2_rep.squeeze(1)
            concat_h = torch.cat([sequence_pool_output, ent1_rep, ent2_rep], dim=1)
        elif self.scheme == -8:
            # [[CLS],ent1,ent2]
            bs, seq_len = e1_mask.shape
            tmp_e1_mask = e1_mask.cpu().numpy().tolist()
            tmp_e2_mask = e2_mask.cpu().numpy().tolist()
            for i in range(bs):
                tmp_e1 = tmp_e1_mask[i]
                tmp_e2 = tmp_e2_mask[i]
                start_idx_e1 = tmp_e1.index(0)
                end_idx_e1 = start_idx_e1 + sum(tmp_e1) - 1
                start_idx_e2 = tmp_e2.index(0)
                end_idx_e2 = start_idx_e2 + sum(tmp_e2) - 1
                e1_mask[start_idx_e1][end_idx_e1] = 0
                e2_mask[start_idx_e2][end_idx_e2] = 0

            e1_mask = e1_mask.unsqueeze(1)
            e2_mask = e2_mask.unsqueeze(1)
            ent1_rep = torch.bmm(e1_mask.float(), sequence_output)
            ent2_rep = torch.bmm(e2_mask.float(), sequence_output)
            ent1_rep = ent1_rep.squeeze(1)
            ent2_rep = ent2_rep.squeeze(1)

            ent1_rep = self.entity_fc_layer(ent1_rep)
            ent2_rep = self.entity_fc_layer(ent2_rep)
            sequence_pool_output = self.cls_fc_layer(sequence_pool_output)

            concat_h = torch.cat([sequence_pool_output, ent1_rep, ent2_rep], dim=1)


        else:
            raise ValueError

        return concat_h


    def forward(self, input_ids, token_type_ids,attention_masks,labels,e1_mask,e2_mask):
        '''
        这个应该支持多卡训练
        :param input_ids: (batch_size,seq_len,hidden_size)
        :param token_type_ids:
        :param attention_masks:
        :param labels:
        :param entity_positions:
        :return:
        '''

        bert_outputs = self.bert_model(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)  #返回 sequence_output, pooled_output, (hidden_states), (attentions)

        bert_output = bert_outputs[0] # shape=(batch_size,seq_len,hidden_size)
        pooled_outputs = bert_outputs[1] # shape=(batch_size,seq_len,hidden_size)



        #pooled_output.shape = (batch_size,hiddensize*2)

        # 这里提取目标representation...

        pooled_output = self.get_entity_representation(bert_output,pooled_outputs,input_ids,e1_mask=e1_mask,e2_mask=e2_mask)


        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)


        if labels is not None:
            loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))
            return loss,logits

        return logits



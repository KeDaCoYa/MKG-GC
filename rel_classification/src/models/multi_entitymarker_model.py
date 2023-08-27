# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2021/12/03
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/03: 
-------------------------------------------------
"""
import copy

import torch
from ipdb import set_trace

from config import BertConfig
from src.models.bert_model import BaseBert, EntityMarkerBaseModel
import torch.nn as nn

from src.models.entitymarker_model import FCLayer

#
# class MultiRBERT(BaseBert):
#     def __init__(self, config: BertConfig, scheme=1):
#         super(MultiRBERT, self).__init__(config)
#
#         self.num_labels = config.num_labels
#         self.config = config
#         self.scheme = scheme
#         # 下面这两个dim可以进行修改
#         self.cls_dim = self.bert_config.hidden_size
#         self.entity_dim = self.bert_config.hidden_size
#
#         self.cls_fc_layer = FCLayer(self.bert_config.hidden_size, self.cls_dim, self.config.dropout_prob)
#         self.entity_fc_layer = FCLayer(self.bert_config.hidden_size, self.entity_dim, self.config.dropout_prob)
#
#         if self.scheme == 1 or self.scheme == -1:
#             # [pooled_output,e1_mask,e2_mask]
#             self.classifier_dim = self.bert_config.hidden_size * 3
#         elif self.scheme == 2 or self.scheme == -2:
#             self.classifier_dim = self.bert_config.hidden_size * 5
#         elif self.scheme == 3 or self.scheme == -3:
#             self.classifier_dim = self.bert_config.hidden_size * 3
#         elif self.scheme == 4 or self.scheme == -4:
#             self.classifier_dim = self.bert_config.hidden_size * 2
#         elif self.scheme == 5 or self.scheme == -5:
#             self.classifier_dim = self.bert_config.hidden_size
#         elif self.scheme == 6 or self.scheme == -6:
#             self.classifier_dim = self.bert_config.hidden_size * 2
#         elif self.scheme == 7 or self.scheme == -7:
#             self.classifier_dim = self.bert_config.hidden_size * 2
#         elif self.scheme == 8 or self.scheme == -8:
#             self.classifier_dim = self.bert_config.hidden_size * 3
#         else:
#             raise ValueError('scheme没有此:{}'.format(self.scheme))
#
#         self.ggi_classifier = FCLayer(
#             self.classifier_dim,
#             self.config.num_labels,
#             self.config.dropout_prob,
#             use_activation=False,
#         )
#
#         self.ddi_classifier = FCLayer(
#             self.classifier_dim,
#             self.config.num_labels,
#             self.config.dropout_prob,
#             use_activation=False,
#         )
#         self.cpi_classifier = FCLayer(
#             self.classifier_dim,
#             self.config.num_labels,
#             self.config.dropout_prob,
#             use_activation=False,
#         )
#         self.gdi_classifier = FCLayer(
#             self.classifier_dim,
#             self.config.num_labels,
#             self.config.dropout_prob,
#             use_activation=False,
#         )
#         self.cdi_classifier = FCLayer(
#             self.classifier_dim,
#             self.config.num_labels,
#             self.config.dropout_prob,
#             use_activation=False,
#         )
#
#         if self.config.freeze_bert:
#             self.freeze_parameter(config.freeze_layers)
#
#             # 模型层数的初始化初始化
#         nn.init.xavier_normal_(self.cls_fc_layer.linear.weight)
#         nn.init.constant_(self.cls_fc_layer.linear.bias, 0.)
#
#         nn.init.xavier_normal_(self.entity_fc_layer.linear.weight)
#         nn.init.constant_(self.entity_fc_layer.linear.bias, 0.)
#
#         nn.init.xavier_normal_(self.ddi_classifier.linear.weight)
#         nn.init.constant_(self.ddi_classifier.linear.bias, 0.)
#
#         nn.init.xavier_normal_(self.ggi_classifier.linear.weight)
#         nn.init.constant_(self.ggi_classifier.linear.bias, 0.)
#
#         nn.init.xavier_normal_(self.cpi_classifier.linear.weight)
#         nn.init.constant_(self.cpi_classifier.linear.bias, 0.)
#
#         nn.init.xavier_normal_(self.gdi_classifier.linear.weight)
#         nn.init.constant_(self.gdi_classifier.linear.bias, 0.)
#
#         nn.init.xavier_normal_(self.cdi_classifier.linear.weight)
#         nn.init.constant_(self.cdi_classifier.linear.bias, 0.)
#
#     @staticmethod
#     def entity_average(hidden_output, entity_mask):
#         """
#         根据mask来获得对应的输出
#         :param hidden_output:hidden_output是bert的输出，shape=(batch_size,seq_len,hidden_size)=(16,128,756)
#          :param entity_mask: [batch_size, max_seq_len]
#                 e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
#         :return: [batch_size, hidden_dim]
#         """
#         e_mask_unsqueeze = entity_mask.unsqueeze(1)  # shape=(batch_size,1,seq_len)
#         # 这相当于获得实体的实际长度
#         length_tensor = (entity_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
#         # [batch_size, 1, seq_len] * [batch_size, seq_len, hidden_dim] = [batch_size, 1, hidden_dim] -> [batch_size, hidden_dim]
#         sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
#         avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
#
#         return avg_vector
#
#     def get_pool_output(self, sequence_output, sequence_pool_output, input_ids, e1_mask, e2_mask):
#
#         if self.scheme == -1:
#             # 这是rbert的方式,[[CLS]],[s1]ent1[e1],[s2]ent2[e2]]
#             # 这个还对pool，ent1,ent2额外使用MLP进行转变...
#             e1_h = self.entity_average(sequence_output, e1_mask)
#             e2_h = self.entity_average(sequence_output, e2_mask)
#
#             # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
#             # pooled_output.shape=(batch_size,768)
#             pooled_output = self.cls_fc_layer(sequence_pool_output)
#
#             e1_h = self.entity_fc_layer(e1_h)
#             e2_h = self.entity_fc_layer(e2_h)
#
#             # Concat -> fc_layer
#             concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)  # torch.Size([16, 2304])
#         elif self.scheme == 1:
#             # 这个和1相反，不适用额外的linear层
#             # 这是rbert的方式,[[CLS]],[s1]ent1[e1],[s2]ent2[e2]]
#
#             e1_h = self.entity_average(sequence_output, e1_mask)
#             e2_h = self.entity_average(sequence_output, e2_mask)
#
#             concat_h = torch.cat([sequence_pool_output, e1_h, e2_h], dim=-1)  # torch.Size([16, 2304])
#         elif self.scheme == 2:
#             # [pool_output,[s1],[e1],[s2],[e2]]
#             seq_tags = []
#             for each_tag in [self.config.ent1_start_tag_id, self.config.ent1_end_tag_id, self.config.ent2_start_tag_id,
#                              self.config.ent2_end_tag_id]:
#                 seq_tags.append(self.special_tag_representation(sequence_output, input_ids, each_tag))
#             concat_h = torch.cat((sequence_pool_output, *seq_tags), dim=1)
#         elif self.scheme == -2:
#             # [pool_output,[s1],[e1],[s2],[e2]]
#
#             ent1_start = self.special_tag_representation(sequence_output, input_ids, self.config.ent1_start_tag_id)
#             ent1_end = self.special_tag_representation(sequence_output, input_ids, self.config.ent1_end_tag_id)
#
#             ent2_start = self.special_tag_representation(sequence_output, input_ids, self.config.ent2_start_tag_id)
#             ent2_end = self.special_tag_representation(sequence_output, input_ids, self.config.ent2_end_tag_id)
#
#             ent1_start = self.entity_fc_layer(ent1_start)
#             ent1_end = self.entity_fc_layer(ent1_end)
#             ent2_start = self.entity_fc_layer(ent2_start)
#             ent2_end = self.entity_fc_layer(ent2_end)
#
#             sequence_pool_output = self.cls_fc_layer(sequence_pool_output)
#
#             concat_h = torch.cat([sequence_pool_output, ent1_start, ent1_end, ent2_start, ent2_end],
#                                  dim=-1)  # torch.Size([16, 2304])
#
#         elif self.scheme == 3:
#             # [[CLS],[s1],[s2]]
#             seq_tags = []
#             for each_tag in [self.config.ent1_start_tag_id, self.config.ent2_start_tag_id]:
#                 seq_tags.append(self.special_tag_representation(sequence_output, input_ids, each_tag))
#             concat_h = torch.cat((sequence_pool_output, *seq_tags), dim=1)
#         elif self.scheme == -3:
#             # [[CLS],[s1],[s2]]
#             ent1_start = self.special_tag_representation(sequence_output, input_ids, self.config.ent1_start_tag_id)
#             ent2_start = self.special_tag_representation(sequence_output, input_ids, self.config.ent2_start_tag_id)
#
#             ent1_start = self.entity_fc_layer(ent1_start)
#             ent2_start = self.entity_fc_layer(ent2_start)
#             sequence_pool_output = self.cls_fc_layer(sequence_pool_output)
#             concat_h = torch.cat([sequence_pool_output, ent1_start, ent2_start],
#                                  dim=-1)  # torch.Size([16, 2304])
#         elif self.scheme == 4:
#             # [[s1],[s2]]
#             seq_tags = []
#             for each_tag in [self.config.ent1_start_tag_id, self.config.ent2_start_tag_id]:
#                 seq_tags.append(self.special_tag_representation(sequence_output, input_ids, each_tag))
#             concat_h = torch.cat(seq_tags, dim=1)
#         elif self.scheme == -4:
#             ent1_start = self.special_tag_representation(sequence_output, input_ids, self.config.ent1_start_tag_id)
#             ent2_start = self.special_tag_representation(sequence_output, input_ids, self.config.ent2_start_tag_id)
#
#             ent1_start = self.entity_fc_layer(ent1_start)
#             ent2_start = self.entity_fc_layer(ent2_start)
#
#             concat_h = torch.cat([ent1_start, ent2_start], dim=-1)  # torch.Size([16, 2304])
#         elif self.scheme == 5:
#             # [[CLS]]
#             concat_h = sequence_pool_output  # shape=(batch_size,hidden_size*2)
#         elif self.scheme == 6:
#             # [[s1]ent1[e1],[s2]ent2[ent2]]
#
#             e1_mask = e1_mask.unsqueeze(1)
#             e2_mask = e2_mask.unsqueeze(1)
#             ent1_rep = torch.bmm(e1_mask.float(), sequence_output)
#             ent2_rep = torch.bmm(e2_mask.float(), sequence_output)
#             ent1_rep = ent1_rep.squeeze(1)
#             ent2_rep = ent2_rep.squeeze(1)
#             concat_h = torch.cat([ent1_rep, ent2_rep], dim=1)
#         elif self.scheme == -6:
#             e1_h = self.entity_average(sequence_output, e1_mask)
#             e2_h = self.entity_average(sequence_output, e2_mask)
#
#             e1_h = self.entity_fc_layer(e1_h)
#             e2_h = self.entity_fc_layer(e2_h)
#
#             concat_h = torch.cat([e1_h, e2_h], dim=-1)  # torch.Size([16, 2304])
#         elif self.scheme == 7:
#             # [ent1,ent2]
#             # 取消e1_mask,e2_mask在[s1][e1],[s2][e2]的label，也就是直接设为0
#             # e1_start_idx, e1_end_idx = self.get_ent_position(e1_mask)
#             # e2_start_idx, e2_end_idx = self.get_ent_position(e2_mask)
#             # e1_mask[e1_start_idx] = 0
#             # e1_mask[e1_end_idx] = 0
#             # e2_mask[e2_start_idx] = 0
#             # e2_mask[e2_end_idx] = 0
#             bs, seq_len = e1_mask.shape
#             tmp_e1_mask = e1_mask.cpu().numpy().tolist()
#             tmp_e2_mask = e2_mask.cpu().numpy().tolist()
#             for i in range(bs):
#                 tmp_e1 = tmp_e1_mask[i]
#                 tmp_e2 = tmp_e2_mask[i]
#                 start_idx_e1 = tmp_e1.index(0)
#                 end_idx_e1 = start_idx_e1 + sum(tmp_e1) - 1
#                 start_idx_e2 = tmp_e2.index(0)
#                 end_idx_e2 = start_idx_e2 + sum(tmp_e2) - 1
#                 e1_mask[start_idx_e1][end_idx_e1] = 0
#                 e2_mask[start_idx_e2][end_idx_e2] = 0
#
#             e1_mask = e1_mask.unsqueeze(1)
#             e2_mask = e2_mask.unsqueeze(1)
#             ent1_rep = torch.bmm(e1_mask.float(), sequence_output)
#             ent2_rep = torch.bmm(e2_mask.float(), sequence_output)
#             ent1_rep = ent1_rep.squeeze(1)
#             ent2_rep = ent2_rep.squeeze(1)
#             concat_h = torch.cat([ent1_rep, ent2_rep], dim=1)
#         elif self.scheme == -7:
#             bs, seq_len = e1_mask.shape
#             tmp_e1_mask = e1_mask.cpu().numpy().tolist()
#             tmp_e2_mask = e2_mask.cpu().numpy().tolist()
#             for i in range(bs):
#                 tmp_e1 = tmp_e1_mask[i]
#                 tmp_e2 = tmp_e2_mask[i]
#                 start_idx_e1 = tmp_e1.index(0)
#                 end_idx_e1 = start_idx_e1 + sum(tmp_e1) - 1
#                 start_idx_e2 = tmp_e2.index(0)
#                 end_idx_e2 = start_idx_e2 + sum(tmp_e2) - 1
#                 e1_mask[start_idx_e1][end_idx_e1] = 0
#                 e2_mask[start_idx_e2][end_idx_e2] = 0
#
#             e1_mask = e1_mask.unsqueeze(1)
#             e2_mask = e2_mask.unsqueeze(1)
#             ent1_rep = torch.bmm(e1_mask.float(), sequence_output)
#             ent2_rep = torch.bmm(e2_mask.float(), sequence_output)
#             ent1_rep = ent1_rep.squeeze(1)
#             ent2_rep = ent2_rep.squeeze(1)
#
#             ent1_rep = self.entity_fc_layer(ent1_rep)
#             ent2_rep = self.entity_fc_layer(ent2_rep)
#
#             concat_h = torch.cat([ent1_rep, ent2_rep], dim=1)
#         elif self.scheme == 8:
#             # [[CLS],ent1,ent2]
#             bs, seq_len = e1_mask.shape
#             tmp_e1_mask = e1_mask.cpu().numpy().tolist()
#             tmp_e2_mask = e2_mask.cpu().numpy().tolist()
#             for i in range(bs):
#                 tmp_e1 = tmp_e1_mask[i]
#                 tmp_e2 = tmp_e2_mask[i]
#                 start_idx_e1 = tmp_e1.index(0)
#                 end_idx_e1 = start_idx_e1 + sum(tmp_e1) - 1
#                 start_idx_e2 = tmp_e2.index(0)
#                 end_idx_e2 = start_idx_e2 + sum(tmp_e2) - 1
#                 e1_mask[start_idx_e1][end_idx_e1] = 0
#                 e2_mask[start_idx_e2][end_idx_e2] = 0
#
#             e1_mask = e1_mask.unsqueeze(1)
#             e2_mask = e2_mask.unsqueeze(1)
#             ent1_rep = torch.bmm(e1_mask.float(), sequence_output)
#             ent2_rep = torch.bmm(e2_mask.float(), sequence_output)
#             ent1_rep = ent1_rep.squeeze(1)
#             ent2_rep = ent2_rep.squeeze(1)
#             concat_h = torch.cat([sequence_pool_output, ent1_rep, ent2_rep], dim=1)
#         elif self.scheme == -8:
#             # [[CLS],ent1,ent2]
#             bs, seq_len = e1_mask.shape
#             tmp_e1_mask = e1_mask.cpu().numpy().tolist()
#             tmp_e2_mask = e2_mask.cpu().numpy().tolist()
#             for i in range(bs):
#                 tmp_e1 = tmp_e1_mask[i]
#                 tmp_e2 = tmp_e2_mask[i]
#                 start_idx_e1 = tmp_e1.index(0)
#                 end_idx_e1 = start_idx_e1 + sum(tmp_e1) - 1
#                 start_idx_e2 = tmp_e2.index(0)
#                 end_idx_e2 = start_idx_e2 + sum(tmp_e2) - 1
#                 e1_mask[start_idx_e1][end_idx_e1] = 0
#                 e2_mask[start_idx_e2][end_idx_e2] = 0
#
#             e1_mask = e1_mask.unsqueeze(1)
#             e2_mask = e2_mask.unsqueeze(1)
#             ent1_rep = torch.bmm(e1_mask.float(), sequence_output)
#             ent2_rep = torch.bmm(e2_mask.float(), sequence_output)
#             ent1_rep = ent1_rep.squeeze(1)
#             ent2_rep = ent2_rep.squeeze(1)
#
#             ent1_rep = self.entity_fc_layer(ent1_rep)
#             ent2_rep = self.entity_fc_layer(ent2_rep)
#             sequence_pool_output = self.cls_fc_layer(sequence_pool_output)
#
#             concat_h = torch.cat([sequence_pool_output, ent1_rep, ent2_rep], dim=1)
#
#
#         else:
#             raise ValueError
#
#         return concat_h
#
#     def forward(self, input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask, rel_type=None):
#         """
#         但是这里将实体对放在一个[CLS]sent<SEP>中，而不是两个sent之中
#
#
#         :param input_ids:
#         :param token_type_ids:
#         :param attention_masks:
#         :param e1_mask:  这里e1_mask和e2_mask覆盖了special tag([s1][e1],[s2][e2])，所以这里需要需要切片以下
#         :param e2_mask:
#         :param labels:
#         :param rel_type: 这个是表明每个输入数据的是哪种类别，DDI,CPI,PPI,GDI,CDI五种类别
#         :return:
#         """
#
#         outputs = self.bert_model(
#             input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
#         )  # sequence_output, pooled_output, (hidden_states), (attentions)
#         sequence_output = outputs[0]  # shape=(batch_size,seq_len,hidden_size)
#         pooled_output = outputs[1]  # [CLS],shape = (batch_size,hidden_size)=(16,768)
#
#         # shape=(batch_size,seq_len,hidden_size)
#         concat_h = self.get_pool_output(sequence_output, pooled_output, input_ids, e1_mask, e2_mask)
#
#         # 在这里开始多任务的分支
#
#         ggi_idx = copy.deepcopy(rel_type)
#         ddi_idx = copy.deepcopy(rel_type)
#         cpi_idx = copy.deepcopy(rel_type)
#         gdi_idx = copy.deepcopy(rel_type)
#         cdi_idx = copy.deepcopy(rel_type)
#
#         ggi_idx[ggi_idx != 1] = 0
#         ddi_idx[ddi_idx != 2] = 0
#         cpi_idx[cpi_idx != 3] = 0
#         gdi_idx[gdi_idx != 4] = 0
#         cdi_idx[cdi_idx != 5] = 0
#
#         ggi_output = ggi_idx.unsqueeze(-1) * concat_h
#         ddi_output = ddi_idx.unsqueeze(-1) * concat_h
#         cpi_output = cpi_idx.unsqueeze(-1) * concat_h
#         gdi_output = gdi_idx.unsqueeze(-1) * concat_h
#         cdi_output = cdi_idx.unsqueeze(-1) * concat_h
#
#         ggi_logits = self.ggi_classifier(ggi_output)
#         ddi_logits = self.ddi_classifier(ddi_output)
#         cpi_logits = self.cpi_classifier(cpi_output)
#         gdi_logits = self.gdi_classifier(gdi_output)
#         cdi_logits = self.cdi_classifier(cdi_output)
#         logits = ggi_logits + ddi_logits + cpi_logits + gdi_logits + cdi_logits
#         # Softmax
#         if labels is not None:
#
#             if self.num_labels == 1:
#                 loss_fct = nn.MSELoss()
#                 loss = loss_fct(logits.view(-1), labels.view(-1))
#             else:
#                 loss_fct = nn.CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#
#             return loss, logits
#
#         return logits  # (loss), logits, (hidden_states), (attentions)
#

class MultiSingleEntityMarkerForBC6(BaseBert):
    def __init__(self, config: BertConfig, scheme=1):
        super(MultiSingleEntityMarkerForBC6, self).__init__(config)

        self.num_labels = config.num_labels
        self.config = config
        self.scheme = scheme
        # 下面这两个dim可以进行修改
        self.cls_dim = self.bert_config.hidden_size
        self.entity_dim = self.bert_config.hidden_size

        self.cls_fc_layer = FCLayer(self.bert_config.hidden_size, self.cls_dim, self.config.dropout_prob)
        self.entity_fc_layer = FCLayer(self.bert_config.hidden_size, self.entity_dim, self.config.dropout_prob)

        if self.scheme == 1 or self.scheme == -1:
            # [pooled_output,e1_mask,e2_mask]
            self.classifier_dim = self.bert_config.hidden_size * 3
        elif self.scheme == 2 or self.scheme == -2:
            self.classifier_dim = self.bert_config.hidden_size * 5
        elif self.scheme == 3 or self.scheme == -3:
            self.classifier_dim = self.bert_config.hidden_size * 3
        elif self.scheme == 4 or self.scheme == -4:
            self.classifier_dim = self.bert_config.hidden_size * 2
        elif self.scheme == 5 or self.scheme == -5:
            self.classifier_dim = self.bert_config.hidden_size
        elif self.scheme == 6 or self.scheme == -6:
            self.classifier_dim = self.bert_config.hidden_size * 2
        elif self.scheme == 7 or self.scheme == -7:
            self.classifier_dim = self.bert_config.hidden_size * 2
        elif self.scheme == 8 or self.scheme == -8:
            self.classifier_dim = self.bert_config.hidden_size * 3
        else:
            raise ValueError('scheme没有此:{}'.format(self.scheme))

        # self.classifier1 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier2 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier3 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier4 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier5 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier6 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier7 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier8 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier9 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier10 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )

        self.classifier1 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier2 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier3 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier4 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier5 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier6 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier7 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier8 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier9 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier10 = nn.Linear(self.classifier_dim, self.config.num_labels)

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

        nn.init.xavier_normal_(self.classifier6.weight)
        nn.init.constant_(self.classifier6.bias, 0.)

        nn.init.xavier_normal_(self.classifier7.weight)
        nn.init.constant_(self.classifier7.bias, 0.)

        nn.init.xavier_normal_(self.classifier8.weight)
        nn.init.constant_(self.classifier8.bias, 0.)

        nn.init.xavier_normal_(self.classifier9.weight)
        nn.init.constant_(self.classifier9.bias, 0.)

        nn.init.xavier_normal_(self.classifier10.weight)
        nn.init.constant_(self.classifier10.bias, 0.)

        # nn.init.xavier_normal_(self.classifier1.linear.weight)
        # nn.init.constant_(self.classifier1.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier2.linear.weight)
        # nn.init.constant_(self.classifier2.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier3.linear.weight)
        # nn.init.constant_(self.classifier3.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier4.linear.weight)
        # nn.init.constant_(self.classifier4.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier5.linear.weight)
        # nn.init.constant_(self.classifier5.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier6.linear.weight)
        # nn.init.constant_(self.classifier6.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier7.linear.weight)
        # nn.init.constant_(self.classifier7.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier8.linear.weight)
        # nn.init.constant_(self.classifier8.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier9.linear.weight)
        # nn.init.constant_(self.classifier9.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier10.linear.weight)
        # nn.init.constant_(self.classifier10.linear.bias, 0.)

    @staticmethod
    def entity_average(hidden_output, entity_mask):
        """
        根据mask来获得对应的输出
        :param hidden_output:hidden_output是bert的输出，shape=(batch_size,seq_len,hidden_size)=(16,128,756)
         :param entity_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, hidden_dim]
        """
        e_mask_unsqueeze = entity_mask.unsqueeze(1)  # shape=(batch_size,1,seq_len)
        # 这相当于获得实体的实际长度
        length_tensor = (entity_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        # [batch_size, 1, seq_len] * [batch_size, seq_len, hidden_dim] = [batch_size, 1, hidden_dim] -> [batch_size, hidden_dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting

        return avg_vector

    def get_pool_output(self, sequence_output, sequence_pool_output, input_ids, e1_mask, e2_mask):

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

            concat_h = torch.cat([sequence_pool_output, ent1_start, ent1_end, ent2_start, ent2_end],
                                 dim=-1)  # torch.Size([16, 2304])

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

            concat_h = torch.cat([ent1_start, ent2_start], dim=-1)  # torch.Size([16, 2304])
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

        rel1_idx = copy.deepcopy(rel_type)
        rel2_idx = copy.deepcopy(rel_type)
        rel3_idx = copy.deepcopy(rel_type)
        rel4_idx = copy.deepcopy(rel_type)
        rel5_idx = copy.deepcopy(rel_type)
        rel6_idx = copy.deepcopy(rel_type)
        rel7_idx = copy.deepcopy(rel_type)
        rel8_idx = copy.deepcopy(rel_type)
        rel9_idx = copy.deepcopy(rel_type)
        rel10_idx = copy.deepcopy(rel_type)

        rel1_idx[rel1_idx != 1] = 0
        rel2_idx[rel2_idx != 2] = 0
        rel3_idx[rel3_idx != 3] = 0
        rel4_idx[rel4_idx != 4] = 0
        rel5_idx[rel5_idx != 5] = 0
        rel6_idx[rel6_idx != 6] = 0
        rel7_idx[rel7_idx != 7] = 0
        rel8_idx[rel8_idx != 8] = 0
        rel9_idx[rel9_idx != 9] = 0
        rel10_idx[rel10_idx != 10] = 0

        rel1_output = rel1_idx.unsqueeze(-1) * concat_h
        rel2_output = rel2_idx.unsqueeze(-1) * concat_h
        rel3_output = rel3_idx.unsqueeze(-1) * concat_h
        rel4_output = rel4_idx.unsqueeze(-1) * concat_h
        rel5_output = rel5_idx.unsqueeze(-1) * concat_h
        rel6_output = rel6_idx.unsqueeze(-1) * concat_h
        rel7_output = rel7_idx.unsqueeze(-1) * concat_h
        rel8_output = rel8_idx.unsqueeze(-1) * concat_h
        rel9_output = rel9_idx.unsqueeze(-1) * concat_h
        rel10_output = rel10_idx.unsqueeze(-1) * concat_h

        rel1_logits = self.classifier1(rel1_output)
        rel2_logits = self.classifier1(rel2_output)
        rel3_logits = self.classifier1(rel3_output)
        rel4_logits = self.classifier1(rel4_output)
        rel5_logits = self.classifier1(rel5_output)
        rel6_logits = self.classifier1(rel6_output)
        rel7_logits = self.classifier1(rel7_output)
        rel8_logits = self.classifier1(rel8_output)
        rel9_logits = self.classifier1(rel9_output)
        rel10_logits = self.classifier1(rel10_output)

        logits = rel1_logits + rel2_logits + rel3_logits + rel4_logits + rel5_logits + rel6_logits + rel7_logits + rel8_logits + rel9_logits + rel10_logits
        # Softmax
        if labels is not None:

            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            return loss, logits

        return logits  # (loss), logits, (hidden_states), (attentions)


class MultiSingleEntityMarkerForBC7(BaseBert):
    def __init__(self, config: BertConfig, scheme=1):
        super(MultiSingleEntityMarkerForBC7, self).__init__(config)

        self.num_labels = config.num_labels
        self.config = config
        self.scheme = scheme
        # 下面这两个dim可以进行修改
        self.cls_dim = self.bert_config.hidden_size
        self.entity_dim = self.bert_config.hidden_size

        self.cls_fc_layer = FCLayer(self.bert_config.hidden_size, self.cls_dim, self.config.dropout_prob)
        self.entity_fc_layer = FCLayer(self.bert_config.hidden_size, self.entity_dim, self.config.dropout_prob)

        if self.scheme == 1 or self.scheme == -1:
            # [pooled_output,e1_mask,e2_mask]
            self.classifier_dim = self.bert_config.hidden_size * 3
        elif self.scheme == 2 or self.scheme == -2:
            self.classifier_dim = self.bert_config.hidden_size * 5
        elif self.scheme == 3 or self.scheme == -3:
            self.classifier_dim = self.bert_config.hidden_size * 3
        elif self.scheme == 4 or self.scheme == -4:
            self.classifier_dim = self.bert_config.hidden_size * 2
        elif self.scheme == 5 or self.scheme == -5:
            self.classifier_dim = self.bert_config.hidden_size
        elif self.scheme == 6 or self.scheme == -6:
            self.classifier_dim = self.bert_config.hidden_size * 2
        elif self.scheme == 7 or self.scheme == -7:
            self.classifier_dim = self.bert_config.hidden_size * 2
        elif self.scheme == 8 or self.scheme == -8:
            self.classifier_dim = self.bert_config.hidden_size * 3
        else:
            raise ValueError('scheme没有此:{}'.format(self.scheme))

        # self.classifier1 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier2 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier3 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier4 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier5 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier6 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier7 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier8 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier9 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier10 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier11 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier12 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier13 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )

        self.classifier1 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier2 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier3 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier4 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier5 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier6 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier7 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier8 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier9 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier10 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier11 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier12 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier13 = nn.Linear(self.classifier_dim, self.config.num_labels)

        if self.config.freeze_bert:
            self.freeze_parameter(config.freeze_layers)

            # 模型层数的初始化初始化
        nn.init.xavier_normal_(self.cls_fc_layer.linear.weight)
        nn.init.constant_(self.cls_fc_layer.linear.bias, 0.)

        nn.init.xavier_normal_(self.entity_fc_layer.linear.weight)
        nn.init.constant_(self.entity_fc_layer.linear.bias, 0.)

        # nn.init.xavier_normal_(self.classifier1.linear.weight)
        # nn.init.constant_(self.classifier1.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier2.linear.weight)
        # nn.init.constant_(self.classifier2.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier3.linear.weight)
        # nn.init.constant_(self.classifier3.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier4.linear.weight)
        # nn.init.constant_(self.classifier4.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier5.linear.weight)
        # nn.init.constant_(self.classifier5.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier6.linear.weight)
        # nn.init.constant_(self.classifier6.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier7.linear.weight)
        # nn.init.constant_(self.classifier7.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier8.linear.weight)
        # nn.init.constant_(self.classifier8.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier9.linear.weight)
        # nn.init.constant_(self.classifier9.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier10.linear.weight)
        # nn.init.constant_(self.classifier10.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier11.linear.weight)
        # nn.init.constant_(self.classifier11.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier12.linear.weight)
        # nn.init.constant_(self.classifier12.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier13.linear.weight)
        # nn.init.constant_(self.classifier13.linear.bias, 0.)

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

        nn.init.xavier_normal_(self.classifier6.weight)
        nn.init.constant_(self.classifier6.bias, 0.)

        nn.init.xavier_normal_(self.classifier7.weight)
        nn.init.constant_(self.classifier7.bias, 0.)

        nn.init.xavier_normal_(self.classifier8.weight)
        nn.init.constant_(self.classifier8.bias, 0.)

        nn.init.xavier_normal_(self.classifier9.weight)
        nn.init.constant_(self.classifier9.bias, 0.)

        nn.init.xavier_normal_(self.classifier10.weight)
        nn.init.constant_(self.classifier10.bias, 0.)

        nn.init.xavier_normal_(self.classifier11.weight)
        nn.init.constant_(self.classifier11.bias, 0.)

        nn.init.xavier_normal_(self.classifier12.weight)
        nn.init.constant_(self.classifier12.bias, 0.)

        nn.init.xavier_normal_(self.classifier13.weight)
        nn.init.constant_(self.classifier13.bias, 0.)

    @staticmethod
    def entity_average(hidden_output, entity_mask):
        """
        根据mask来获得对应的输出
        :param hidden_output:hidden_output是bert的输出，shape=(batch_size,seq_len,hidden_size)=(16,128,756)
         :param entity_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, hidden_dim]
        """
        e_mask_unsqueeze = entity_mask.unsqueeze(1)  # shape=(batch_size,1,seq_len)
        # 这相当于获得实体的实际长度
        length_tensor = (entity_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        # [batch_size, 1, seq_len] * [batch_size, seq_len, hidden_dim] = [batch_size, 1, hidden_dim] -> [batch_size, hidden_dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting

        return avg_vector

    def get_pool_output(self, sequence_output, sequence_pool_output, input_ids, e1_mask, e2_mask):

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

            concat_h = torch.cat([sequence_pool_output, ent1_start, ent1_end, ent2_start, ent2_end],
                                 dim=-1)  # torch.Size([16, 2304])

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

            concat_h = torch.cat([ent1_start, ent2_start], dim=-1)  # torch.Size([16, 2304])
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

        rel1_idx = copy.deepcopy(rel_type)
        rel2_idx = copy.deepcopy(rel_type)
        rel3_idx = copy.deepcopy(rel_type)
        rel4_idx = copy.deepcopy(rel_type)
        rel5_idx = copy.deepcopy(rel_type)
        rel6_idx = copy.deepcopy(rel_type)
        rel7_idx = copy.deepcopy(rel_type)
        rel8_idx = copy.deepcopy(rel_type)
        rel9_idx = copy.deepcopy(rel_type)
        rel10_idx = copy.deepcopy(rel_type)
        rel11_idx = copy.deepcopy(rel_type)
        rel12_idx = copy.deepcopy(rel_type)
        rel13_idx = copy.deepcopy(rel_type)

        rel1_idx[rel1_idx != 1] = 0
        rel2_idx[rel2_idx != 2] = 0
        rel3_idx[rel3_idx != 3] = 0
        rel4_idx[rel4_idx != 4] = 0
        rel5_idx[rel5_idx != 5] = 0
        rel6_idx[rel6_idx != 6] = 0
        rel7_idx[rel7_idx != 7] = 0
        rel8_idx[rel8_idx != 8] = 0
        rel9_idx[rel9_idx != 9] = 0
        rel10_idx[rel10_idx != 10] = 0
        rel11_idx[rel11_idx != 11] = 0
        rel12_idx[rel12_idx != 12] = 0
        rel13_idx[rel13_idx != 13] = 0

        rel1_output = rel1_idx.unsqueeze(-1) * concat_h
        rel2_output = rel2_idx.unsqueeze(-1) * concat_h
        rel3_output = rel3_idx.unsqueeze(-1) * concat_h
        rel4_output = rel4_idx.unsqueeze(-1) * concat_h
        rel5_output = rel5_idx.unsqueeze(-1) * concat_h
        rel6_output = rel6_idx.unsqueeze(-1) * concat_h
        rel7_output = rel7_idx.unsqueeze(-1) * concat_h
        rel8_output = rel8_idx.unsqueeze(-1) * concat_h
        rel9_output = rel9_idx.unsqueeze(-1) * concat_h
        rel10_output = rel10_idx.unsqueeze(-1) * concat_h
        rel11_output = rel11_idx.unsqueeze(-1) * concat_h
        rel12_output = rel12_idx.unsqueeze(-1) * concat_h
        rel13_output = rel13_idx.unsqueeze(-1) * concat_h

        rel1_logits = self.classifier1(rel1_output)
        rel2_logits = self.classifier1(rel2_output)
        rel3_logits = self.classifier1(rel3_output)
        rel4_logits = self.classifier1(rel4_output)
        rel5_logits = self.classifier1(rel5_output)
        rel6_logits = self.classifier1(rel6_output)
        rel7_logits = self.classifier1(rel7_output)
        rel8_logits = self.classifier1(rel8_output)
        rel9_logits = self.classifier1(rel9_output)
        rel10_logits = self.classifier1(rel10_output)
        rel11_logits = self.classifier1(rel11_output)
        rel12_logits = self.classifier1(rel12_output)
        rel13_logits = self.classifier1(rel13_output)

        logits = rel1_logits + rel2_logits + rel3_logits + rel4_logits + rel5_logits + rel6_logits + rel7_logits + rel8_logits + rel9_logits + rel10_logits + rel11_logits + rel12_logits + rel13_logits
        # Softmax
        if labels is not None:

            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            return loss, logits

        return logits  # (loss), logits, (hidden_states), (attentions)


class MultiSingleEntityMarkerForAlldata(EntityMarkerBaseModel):
    def __init__(self, config: BertConfig):
        super(MultiSingleEntityMarkerForAlldata, self).__init__(config)

        self.num_labels = 2
        self.config = config
        self.scheme = config.scheme
        # 下面这两个dim可以进行修改
        self.cls_dim = self.bert_config.hidden_size
        self.entity_dim = self.bert_config.hidden_size


        self.cls_fc_layer = FCLayer(self.bert_config.hidden_size, self.cls_dim, self.config.dropout_prob)
        if self.scheme == -113:
            self.entity_fc_layer = FCLayer(self.bert_config.hidden_size*3, self.entity_dim, self.config.dropout_prob)
        else:
            self.entity_fc_layer = FCLayer(self.bert_config.hidden_size, self.entity_dim, self.config.dropout_prob)



        self.classifier1 = nn.Linear(self.classifier_dim, self.num_labels)
        self.classifier2 = nn.Linear(self.classifier_dim, self.num_labels)
        self.classifier3 = nn.Linear(self.classifier_dim, self.num_labels)
        self.classifier4 = nn.Linear(self.classifier_dim, self.num_labels)
        self.classifier5 = nn.Linear(self.classifier_dim, self.num_labels)

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

        # nn.init.xavier_normal_(self.classifier1.linear.weight)
        # nn.init.constant_(self.classifier1.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier2.linear.weight)
        # nn.init.constant_(self.classifier2.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier3.linear.weight)
        # nn.init.constant_(self.classifier3.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier4.linear.weight)
        # nn.init.constant_(self.classifier4.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier5.linear.weight)
        # nn.init.constant_(self.classifier5.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier6.linear.weight)
        # nn.init.constant_(self.classifier6.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier7.linear.weight)
        # nn.init.constant_(self.classifier7.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier8.linear.weight)
        # nn.init.constant_(self.classifier8.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier9.linear.weight)
        # nn.init.constant_(self.classifier9.linear.bias, 0.)
        #
        # nn.init.xavier_normal_(self.classifier10.linear.weight)
        # nn.init.constant_(self.classifier10.linear.bias, 0.)

    @staticmethod
    def entity_average(hidden_output, entity_mask):
        """
        根据mask来获得对应的输出
        :param hidden_output:hidden_output是bert的输出，shape=(batch_size,seq_len,hidden_size)=(16,128,756)
         :param entity_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, hidden_dim]
        """
        e_mask_unsqueeze = entity_mask.unsqueeze(1)  # shape=(batch_size,1,seq_len)
        # 这相当于获得实体的实际长度
        length_tensor = (entity_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        # [batch_size, 1, seq_len] * [batch_size, seq_len, hidden_dim] = [batch_size, 1, hidden_dim] -> [batch_size, hidden_dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting

        return avg_vector



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

        if self.config.bert_name in ['biobert','wwm_bert','bert','scibert']:
            bert_outputs = self.bert_model(
            input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
            )
            sequence_output = bert_outputs[0]  # shape=(batch_size,seq_len,hidden_size)
            pooled_output = bert_outputs[1]  # [CLS],shape = (batch_size,hidden_size)=(16,768)
        elif self.config.bert_name in ['flash','flash_quad']:
            bert_outputs = self.bert_model(
                input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
            )
            sequence_output = bert_outputs[0]  # shape=(batch_size,seq_len,hidden_size)
            pooled_output = bert_outputs[0][:,0,:]  # [CLS],shape = (batch_size,hidden_size)=(16,768)
        else:
            raise ValueError


        # outputs = self.bert_model(
        #     input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
        # )  # sequence_output, pooled_output, (hidden_states), (attentions)
        # set_trace()
        # sequence_output = outputs[0]  # shape=(batch_size,seq_len,hidden_size)
        # pooled_output = outputs[1]  # [CLS],shape = (batch_size,hidden_size)=(16,768)

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

            rel1_output = rel1_idx.unsqueeze(-1) * concat_h
            rel2_output = rel2_idx.unsqueeze(-1) * concat_h
            rel3_output = rel3_idx.unsqueeze(-1) * concat_h
            rel4_output = rel4_idx.unsqueeze(-1) * concat_h
            rel5_output = rel5_idx.unsqueeze(-1) * concat_h

            rel1_logits = self.classifier1(rel1_output)
            rel2_logits = self.classifier2(rel2_output)
            rel3_logits = self.classifier3(rel3_output)
            rel4_logits = self.classifier4(rel4_output)
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


class MultiSingleEntityMarkerForBinary(BaseBert):
    def __init__(self, config: BertConfig, scheme=1):
        super(MultiSingleEntityMarkerForBinary, self).__init__(config)

        self.num_labels = config.num_labels
        self.config = config
        self.scheme = scheme
        # 下面这两个dim可以进行修改
        self.cls_dim = self.bert_config.hidden_size
        self.entity_dim = self.bert_config.hidden_size

        self.cls_fc_layer = FCLayer(self.bert_config.hidden_size, self.cls_dim, self.config.dropout_prob)
        self.entity_fc_layer = FCLayer(self.bert_config.hidden_size, self.entity_dim, self.config.dropout_prob)

        if self.scheme == 1 or self.scheme == -1:
            # [pooled_output,e1_mask,e2_mask]
            self.classifier_dim = self.bert_config.hidden_size * 3
        elif self.scheme == 2 or self.scheme == -2:
            self.classifier_dim = self.bert_config.hidden_size * 5
        elif self.scheme == 3 or self.scheme == -3:
            self.classifier_dim = self.bert_config.hidden_size * 3
        elif self.scheme == 4 or self.scheme == -4:
            self.classifier_dim = self.bert_config.hidden_size * 2
        elif self.scheme == 5 or self.scheme == -5:
            self.classifier_dim = self.bert_config.hidden_size
        elif self.scheme == 6 or self.scheme == -6:
            self.classifier_dim = self.bert_config.hidden_size * 2
        elif self.scheme == 7 or self.scheme == -7:
            self.classifier_dim = self.bert_config.hidden_size * 2
        elif self.scheme == 8 or self.scheme == -8:
            self.classifier_dim = self.bert_config.hidden_size * 3
        else:
            raise ValueError('scheme没有此:{}'.format(self.scheme))

        # self.classifier1 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier2 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier3 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier4 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier5 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier6 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier7 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier8 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier9 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier10 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier11 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier12 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )
        # self.classifier13 = FCLayer(
        #     self.classifier_dim,
        #     self.config.num_labels,
        #     self.config.dropout_prob,
        #     use_activation=False,
        # )

        self.classifier1 = nn.Linear(self.classifier_dim, self.config.num_labels)
        self.classifier2 = nn.Linear(self.classifier_dim, self.config.num_labels)

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

    @staticmethod
    def entity_average(hidden_output, entity_mask):
        """
        根据mask来获得对应的输出
        :param hidden_output:hidden_output是bert的输出，shape=(batch_size,seq_len,hidden_size)=(16,128,756)
         :param entity_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, hidden_dim]
        """
        e_mask_unsqueeze = entity_mask.unsqueeze(1)  # shape=(batch_size,1,seq_len)
        # 这相当于获得实体的实际长度
        length_tensor = (entity_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        # [batch_size, 1, seq_len] * [batch_size, seq_len, hidden_dim] = [batch_size, 1, hidden_dim] -> [batch_size, hidden_dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting

        return avg_vector

    def get_pool_output(self, sequence_output, sequence_pool_output, input_ids, e1_mask, e2_mask):

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

            concat_h = torch.cat([sequence_pool_output, ent1_start, ent1_end, ent2_start, ent2_end],
                                 dim=-1)  # torch.Size([16, 2304])

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

            concat_h = torch.cat([ent1_start, ent2_start], dim=-1)  # torch.Size([16, 2304])
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

        rel1_idx = copy.deepcopy(rel_type)
        rel2_idx = copy.deepcopy(rel_type)

        rel1_idx[rel1_idx != 1] = 0
        rel2_idx[rel2_idx != 2] = 0

        rel1_output = rel1_idx.unsqueeze(-1) * concat_h
        rel2_output = rel2_idx.unsqueeze(-1) * concat_h

        rel1_logits = self.classifier1(rel1_output)
        rel2_logits = self.classifier1(rel2_output)

        logits = rel1_logits + rel2_logits
        # Softmax
        if labels is not None:

            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            return loss, logits

        return logits  # (loss), logits, (hidden_states), (attentions)

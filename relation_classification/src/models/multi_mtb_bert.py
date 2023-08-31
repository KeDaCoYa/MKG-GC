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
import copy

from ipdb import set_trace
import numpy as np

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss,BCELoss

from config import BertConfig
from src.models.bert_model import BaseBert
from src.models.entitymarker_model import FCLayer


class MTBRelationClassification(BaseBert):
    def __init__(self, config:BertConfig):
        super(MTBRelationClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.dropout = nn.Dropout(config.dropout_prob)


        self.criterion = CrossEntropyLoss(reduction='none')
        # 初始化部分网络参数
        # nn.init.xavier_normal_(self.test1_entity.weight)
        # nn.init.constant_(self.test1_entity.bias, 0.)
        # nn.init.xavier_normal_(self.test2_entity.weight)
        # nn.init.constant_(self.test2_entity.bias, 0.)
        self.scheme = config.scheme

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

    # def get_entity_representation(self, bert_outputs, pooled_output, input_ids, e1_mask=None, e2_mask=None):
    #     '''
    #     这里使用两个的bert outputs输出...
    #     :param bert_outputs:
    #     :param pool_output:
    #     :param schema:
    #         这里主要是MTB的方法，这里的四种方式和ClinicalTransformer保持一直，之后可能会增加
    #     :return: 直接返回最终的new_pooled_output
    #     '''
    #     if self.scheme == 1:
    #         # [CLS]+[s1]+[e1]+[s2]
    #         seq_tags = []  # 论文中的(2),使用[CLS]和实体的start 标记(也就是<e1>,<e2>或者说<s1><s2>)
    #         for each_tag in [self.config.ent1_start_tag_id, self.config.ent2_start_tag_id]:
    #             seq_tags.append(self.special_tag_representation(bert_outputs, input_ids, each_tag))
    #         new_pooled_output = torch.cat((pooled_output, *seq_tags), dim=1)
    #
    #     elif self.scheme == 2:  # 论文中的(3),使用实体1和实体2的<s1><e1>,<s2><e2>，这个效果在clinicalTransformer论文中效果最好....
    #         seq_tags = []
    #         for each_tag in [self.config.ent1_start_tag_id, self.config.ent1_end_tag_id, self.config.ent2_start_tag_id,
    #                          self.config.ent2_end_tag_id]:
    #             seq_tags.append(self.special_tag_representation(bert_outputs, input_ids, each_tag))
    #         new_pooled_output = torch.cat((pooled_output, *seq_tags), dim=1)
    #     elif self.scheme == 3:  # 论文中的(4),只使用两个实体的开始标志：<s1><s2>...
    #         seq_tags = []
    #         for each_tag in [self.config.ent1_start_tag_id, self.config.ent2_start_tag_id]:
    #             seq_tags.append(self.special_tag_representation(bert_outputs, input_ids, each_tag))
    #         new_pooled_output = torch.cat(seq_tags, dim=1)
    #     elif self.scheme == 4:  # 这是论文中介绍的(1),只使用[CLS]的output
    #         new_pooled_output = pooled_output  # shape=(batch_size,hidden_size*2)
    #     elif self.scheme == 5:  # 这个是最基本的情况，直接将e1_mask和e2_mask对应的全部拿来
    #
    #         e1_mask = e1_mask.unsqueeze(1)
    #         e2_mask = e2_mask.unsqueeze(1)
    #         ent1_rep = torch.bmm(e1_mask.float(), bert_outputs)
    #         ent2_rep = torch.bmm(e2_mask.float(), bert_outputs)
    #         ent1_rep = ent1_rep.squeeze(1)
    #         ent2_rep = ent2_rep.squeeze(1)
    #         new_pooled_output = torch.cat([ent1_rep, ent2_rep], dim=1)
    #     elif self.scheme == 6:  # 只获得真正实体对应的部分，取消掉[s1][e1],[s2][e2]
    #
    #         # 取消e1_mask,e2_mask在[s1][e1],[s2][e2]的label，也就是直接设为0
    #         # e1_start_idx, e1_end_idx = self.get_ent_position(e1_mask)
    #         # e2_start_idx, e2_end_idx = self.get_ent_position(e2_mask)
    #         # e1_mask[e1_start_idx] = 0
    #         # e1_mask[e1_end_idx] = 0
    #         # e2_mask[e2_start_idx] = 0
    #         # e2_mask[e2_end_idx] = 0
    #         bs,seq_len = e1_mask.shape
    #         tmp_e1_mask = e1_mask.cpu().numpy().tolist()
    #         tmp_e2_mask = e2_mask.cpu().numpy().tolist()
    #         for i in range(bs):
    #             tmp_e1 = tmp_e1_mask[i]
    #             tmp_e2 = tmp_e2_mask[i]
    #             start_idx_e1 =tmp_e1.index(0)
    #             end_idx_e1 = start_idx_e1+sum(tmp_e1)-1
    #             start_idx_e2 =tmp_e2.index(0)
    #             end_idx_e2 = start_idx_e2 + sum(tmp_e2) - 1
    #             e1_mask[start_idx_e1][end_idx_e1] = 0
    #             e2_mask[start_idx_e2][end_idx_e2] = 0
    #
    #         e1_mask = e1_mask.unsqueeze(1)
    #         e2_mask = e2_mask.unsqueeze(1)
    #         ent1_rep = torch.bmm(e1_mask.float(), bert_outputs)
    #         ent2_rep = torch.bmm(e2_mask.float(), bert_outputs)
    #         ent1_rep = ent1_rep.squeeze(1)
    #         ent2_rep = ent2_rep.squeeze(1)
    #         new_pooled_output = torch.cat([ent1_rep, ent2_rep], dim=1)
    #     else:
    #         raise ValueError
    #     return new_pooled_output

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


class MultiMtbBertForBC6(BaseBert):
    def __init__(self, config: BertConfig):
        super(MultiMtbBertForBC6, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.dropout = nn.Dropout(config.dropout_prob)

        self.criterion = CrossEntropyLoss(reduction='none')
        # 初始化部分网络参数
        # nn.init.xavier_normal_(self.test1_entity.weight)
        # nn.init.constant_(self.test1_entity.bias, 0.)
        # nn.init.xavier_normal_(self.test2_entity.weight)
        # nn.init.constant_(self.test2_entity.bias, 0.)
        self.scheme = config.scheme

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

        self.cls_dim = self.bert_config.hidden_size
        self.entity_dim = self.bert_config.hidden_size
        self.cls_fc_layer = FCLayer(self.bert_config.hidden_size, self.cls_dim, self.config.dropout_prob)
        self.entity_fc_layer = FCLayer(self.bert_config.hidden_size, self.entity_dim, self.config.dropout_prob)

        #self.classifier = nn.Linear(self.classifier_dim, self.num_labels)
        self.classifier1 = FCLayer(
            self.classifier_dim,
            self.config.num_labels,
            self.config.dropout_prob,
            use_activation=False,
        )
        self.classifier2 = FCLayer(
            self.classifier_dim,
            self.config.num_labels,
            self.config.dropout_prob,
            use_activation=False,
        )
        self.classifier3 = FCLayer(
            self.classifier_dim,
            self.config.num_labels,
            self.config.dropout_prob,
            use_activation=False,
        )
        self.classifier4 = FCLayer(
            self.classifier_dim,
            self.config.num_labels,
            self.config.dropout_prob,
            use_activation=False,
        )
        self.classifier5 = FCLayer(
            self.classifier_dim,
            self.config.num_labels,
            self.config.dropout_prob,
            use_activation=False,
        )
        self.classifier6 = FCLayer(
            self.classifier_dim,
            self.config.num_labels,
            self.config.dropout_prob,
            use_activation=False,
        )
        self.classifier7 = FCLayer(
            self.classifier_dim,
            self.config.num_labels,
            self.config.dropout_prob,
            use_activation=False,
        )
        self.classifier8 = FCLayer(
            self.classifier_dim,
            self.config.num_labels,
            self.config.dropout_prob,
            use_activation=False,
        )
        self.classifier9 = FCLayer(
            self.classifier_dim,
            self.config.num_labels,
            self.config.dropout_prob,
            use_activation=False,
        )
        self.classifier10 = FCLayer(
            self.classifier_dim,
            self.config.num_labels,
            self.config.dropout_prob,
            use_activation=False,
        )

        nn.init.xavier_normal_(self.cls_fc_layer.linear.weight)
        nn.init.constant_(self.cls_fc_layer.linear.bias, 0.)

        nn.init.xavier_normal_(self.entity_fc_layer.linear.weight)
        nn.init.constant_(self.entity_fc_layer.linear.bias, 0.)

        nn.init.xavier_normal_(self.classifier1.linear.weight)
        nn.init.constant_(self.classifier1.linear.bias, 0.)

        nn.init.xavier_normal_(self.classifier2.linear.weight)
        nn.init.constant_(self.classifier2.linear.bias, 0.)

        nn.init.xavier_normal_(self.classifier3.linear.weight)
        nn.init.constant_(self.classifier3.linear.bias, 0.)

        nn.init.xavier_normal_(self.classifier4.linear.weight)
        nn.init.constant_(self.classifier4.linear.bias, 0.)

        nn.init.xavier_normal_(self.classifier5.linear.weight)
        nn.init.constant_(self.classifier5.linear.bias, 0.)

        nn.init.xavier_normal_(self.classifier6.linear.weight)
        nn.init.constant_(self.classifier6.linear.bias, 0.)

        nn.init.xavier_normal_(self.classifier7.linear.weight)
        nn.init.constant_(self.classifier7.linear.bias, 0.)

        nn.init.xavier_normal_(self.classifier8.linear.weight)
        nn.init.constant_(self.classifier8.linear.bias, 0.)

        nn.init.xavier_normal_(self.classifier9.linear.weight)
        nn.init.constant_(self.classifier9.linear.bias, 0.)

        nn.init.xavier_normal_(self.classifier10.linear.weight)
        nn.init.constant_(self.classifier10.linear.bias, 0.)

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


    def forward(self, input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask,rel_type=None):
        '''
        这个应该支持多卡训练
        :param input_ids: (batch_size,seq_len,hidden_size)
        :param token_type_ids:
        :param attention_masks:
        :param labels:
        :param entity_positions:
        :return:
        '''

        bert_outputs = self.bert_model(input_ids, attention_mask=attention_masks,
                                       token_type_ids=token_type_ids,)  # 返回 sequence_output, pooled_output, (hidden_states), (attentions)

        bert_output = bert_outputs[0]  # shape=(batch_size,seq_len,hidden_size)
        pooled_outputs = bert_outputs[1]  # shape=(batch_size,seq_len,hidden_size)

        # pooled_output.shape = (batch_size,hiddensize*2)

        # 这里提取目标representation...

        pooled_output = self.get_entity_representation(bert_output, pooled_outputs, input_ids, e1_mask=e1_mask,
                                                       e2_mask=e2_mask)

        concat_h = self.dropout(pooled_output)

        # 开始多任务的区分
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


        if labels is not None:
            loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits

        return logits



class MultiMtbBertForBC7(BaseBert):
    def __init__(self, config: BertConfig):
        super(MultiMtbBertForBC7, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.dropout = nn.Dropout(config.dropout_prob)

        self.criterion = CrossEntropyLoss(reduction='none')
        # 初始化部分网络参数
        # nn.init.xavier_normal_(self.test1_entity.weight)
        # nn.init.constant_(self.test1_entity.bias, 0.)
        # nn.init.xavier_normal_(self.test2_entity.weight)
        # nn.init.constant_(self.test2_entity.bias, 0.)
        self.scheme = config.scheme

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

        self.cls_dim = self.bert_config.hidden_size
        self.entity_dim = self.bert_config.hidden_size
        self.cls_fc_layer = FCLayer(self.bert_config.hidden_size, self.cls_dim, self.config.dropout_prob)
        self.entity_fc_layer = FCLayer(self.bert_config.hidden_size, self.entity_dim, self.config.dropout_prob)

        #self.classifier = nn.Linear(self.classifier_dim, self.num_labels)
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


    def forward(self, input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask,rel_type=None):
        '''
        这个应该支持多卡训练
        :param input_ids: (batch_size,seq_len,hidden_size)
        :param token_type_ids:
        :param attention_masks:
        :param labels:
        :param entity_positions:
        :return:
        '''

        bert_outputs = self.bert_model(input_ids, attention_mask=attention_masks,
                                       token_type_ids=token_type_ids,)  # 返回 sequence_output, pooled_output, (hidden_states), (attentions)

        bert_output = bert_outputs[0]  # shape=(batch_size,seq_len,hidden_size)
        pooled_outputs = bert_outputs[1]  # shape=(batch_size,seq_len,hidden_size)

        # pooled_output.shape = (batch_size,hiddensize*2)

        # 这里提取目标representation...

        pooled_output = self.get_entity_representation(bert_output, pooled_outputs, input_ids, e1_mask=e1_mask,
                                                       e2_mask=e2_mask)

        concat_h = self.dropout(pooled_output)

        # 开始多任务的区分
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
        rel11_idx[rel11_idx != 10] = 0
        rel12_idx[rel12_idx != 10] = 0
        rel13_idx[rel13_idx != 10] = 0

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

        logits = rel1_logits + rel2_logits + rel3_logits + rel4_logits + rel5_logits + rel6_logits + rel7_logits + rel8_logits + rel9_logits + rel10_logits+rel11_logits+rel12_logits+rel13_logits


        if labels is not None:
            loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits

        return logits



class MultiMtbBertForAlldata(BaseBert):
    def __init__(self, config: BertConfig):
        super(MultiMtbBertForBC7, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.dropout = nn.Dropout(config.dropout_prob)

        self.criterion = CrossEntropyLoss(reduction='none')
        # 初始化部分网络参数
        # nn.init.xavier_normal_(self.test1_entity.weight)
        # nn.init.constant_(self.test1_entity.bias, 0.)
        # nn.init.xavier_normal_(self.test2_entity.weight)
        # nn.init.constant_(self.test2_entity.bias, 0.)
        self.scheme = config.scheme

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

        self.cls_dim = self.bert_config.hidden_size
        self.entity_dim = self.bert_config.hidden_size
        self.cls_fc_layer = FCLayer(self.bert_config.hidden_size, self.cls_dim, self.config.dropout_prob)
        self.entity_fc_layer = FCLayer(self.bert_config.hidden_size, self.entity_dim, self.config.dropout_prob)

        #self.classifier = nn.Linear(self.classifier_dim, self.num_labels)
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


    def forward(self, input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask,rel_type=None):
        '''
        这个应该支持多卡训练
        :param input_ids: (batch_size,seq_len,hidden_size)
        :param token_type_ids:
        :param attention_masks:
        :param labels:
        :param entity_positions:
        :return:
        '''

        bert_outputs = self.bert_model(input_ids, attention_mask=attention_masks,
                                       token_type_ids=token_type_ids,)  # 返回 sequence_output, pooled_output, (hidden_states), (attentions)

        bert_output = bert_outputs[0]  # shape=(batch_size,seq_len,hidden_size)
        pooled_outputs = bert_outputs[1]  # shape=(batch_size,seq_len,hidden_size)

        # pooled_output.shape = (batch_size,hiddensize*2)

        # 这里提取目标representation...

        pooled_output = self.get_entity_representation(bert_output, pooled_outputs, input_ids, e1_mask=e1_mask,
                                                       e2_mask=e2_mask)

        concat_h = self.dropout(pooled_output)

        # 开始多任务的区分
        rel1_idx = copy.deepcopy(rel_type)
        rel2_idx = copy.deepcopy(rel_type)
        rel3_idx = copy.deepcopy(rel_type)
        rel4_idx = copy.deepcopy(rel_type)
        rel5_idx = copy.deepcopy(rel_type)
        rel6_idx = copy.deepcopy(rel_type)


        rel1_idx[rel1_idx != 1] = 0
        rel2_idx[rel2_idx != 2] = 0
        rel3_idx[rel3_idx != 3] = 0
        rel4_idx[rel4_idx != 4] = 0
        rel5_idx[rel5_idx != 5] = 0
        rel6_idx[rel6_idx != 6] = 0

        rel1_output = rel1_idx.unsqueeze(-1) * concat_h
        rel2_output = rel2_idx.unsqueeze(-1) * concat_h
        rel3_output = rel3_idx.unsqueeze(-1) * concat_h
        rel4_output = rel4_idx.unsqueeze(-1) * concat_h
        rel5_output = rel5_idx.unsqueeze(-1) * concat_h
        rel6_output = rel6_idx.unsqueeze(-1) * concat_h



        rel1_logits = self.classifier1(rel1_output)
        rel2_logits = self.classifier1(rel2_output)
        rel3_logits = self.classifier1(rel3_output)
        rel4_logits = self.classifier1(rel4_output)
        rel5_logits = self.classifier1(rel5_output)
        rel6_logits = self.classifier1(rel6_output)


        logits = rel1_logits + rel2_logits + rel3_logits + rel4_logits + rel5_logits + rel6_logits


        if labels is not None:
            loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits

        return logits



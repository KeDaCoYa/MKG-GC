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
import os

from ipdb import set_trace

import numpy as np

import torch
import torch.nn as nn

from transformers import BertModel
import logging

from config import MyBertConfig
from src.models.flash_quad import FLASHQuadForMaskedLM
from src.models.kebiolm_model import KebioModel, KebioForRelationExtraction
from src.models.wwm_flash import FLASHForMaskedLM

logger = logging.getLogger('main.bert_model')


class BaseBert(nn.Module):
    def __init__(self, config):
        """
        这是最基础的BERT模型加载，加载预训练的模型
        :param config:
        :param bert_dir:
        :param dropout_prob:
        """
        super(BaseBert, self).__init__()

        self.config = config
        if config.bert_name == 'kebiolm':

            self.bert_model = KebioForRelationExtraction(config)
            self.bert_model.bert = KebioModel.from_pretrained(config.bert_dir, config=config)
            #self.bert_model = KebioModel.from_pretrained(config.bert_dir, config=config)
        elif config.bert_name == 'flash_quad':



            model = FLASHQuadForMaskedLM(config)
            checkpoint = torch.load(os.path.join(config.bert_dir, 'model.pt'))
            model.load_state_dict(checkpoint)
            self.bert_model = model.flash_quad
        elif config.bert_name == 'flash':
            model = FLASHForMaskedLM(config)
            checkpoint = torch.load(os.path.join(config.bert_dir, 'model.pt'))
            model.load_state_dict(checkpoint)
            self.bert_model = model.flash
        else:
            self.bert_model = BertModel.from_pretrained(config.bert_dir, output_hidden_states=True,
                                                        hidden_dropout_prob=config.dropout_prob)

        self.bert_config = self.bert_model.config
        if config.freeze_bert:
            self.freeze_parameter(config.freeze_layers)

    @staticmethod
    def _init_weights(blocks, **kwargs):
        """
        对指定的blocks进行参数初始化,只对指定layer进行初始化
        主要是对BERT之后的一些layer进行初始化
        :param blocks:
        :param kwargs:
        :return:
        """
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

    def freeze_parameter(self, freeze_layers):
        """
        对指定的layers进行冻结参数
        :param freeze_layers: 格式为['layer.10','layer.11','bert.pooler','out.']
        :return:
        """
        for name, param in self.bert_model.named_parameters():

            for ele in freeze_layers:
                if ele in name:
                    param.requires_grad = False
        # 验证一下实际情况
        # for name,param in self.bert_model.named_parameters():
        #     if param.requires_grad:
        #         print(name,param.size())

    @staticmethod
    def special_tag_representation(seq_output, input_ids, special_tag):
        '''
        这里就是根据special_tag来获取对应的representation
        input_ids就是为了定位位置
        '''
        # nonzero是得到坐标，表示在(input_ids == special_tag)中，值不为0的坐标
        spec_idx = (input_ids == special_tag).nonzero(as_tuple=False)

        temp = []
        for idx in spec_idx:
            temp.append(seq_output[idx[0], idx[1], :])
        tags_rep = torch.stack(temp, dim=0)

        return tags_rep

    def get_ent_position(self, e_mask):
        '''
        获得entity mask的start_index和end_index
        :param e_mask: shape=(bs,seq_len)
        :return:
        '''
        start_idx = e_mask.index(1)
        for i in range(start_idx + 1, len(e_mask)):
            if e_mask[i] == 1 and e_mask[i + 1] == 0:
                return (start_idx, i)
        return start_idx, len(e_mask) - 1

    def get_entity_representation(self, bert_outputs, pooled_output, input_ids, e1_mask=None, e2_mask=None):
        '''
        这里使用两个的bert outputs输出...
        :param bert_outputs:
        :param pool_output:
        :param schema:
            这里主要是MTB的方法，这里的四种方式和ClinicalTransformer保持一直，之后可能会增加
        :return: 直接返回最终的new_pooled_output
        '''
        if self.scheme == 1:
            seq_tags = []  # 论文中的(2),使用[CLS]和实体的start 标记(也就是<e1>,<e2>或者说<s1><s2>)
            for each_tag in [self.config.ent1_start_tag_id, self.config.ent2_start_tag_id]:
                seq_tags.append(self.special_tag_representation(bert_outputs, input_ids, each_tag))
            new_pooled_output = torch.cat((pooled_output, *seq_tags), dim=1)

        elif self.scheme == 2:  # 论文中的(3),使用实体1和实体2的<s1><e1>,<s2><e2>，这个效果在clinicalTransformer论文中效果最好....
            seq_tags = []
            for each_tag in [self.config.ent1_start_tag_id, self.config.ent1_end_tag_id, self.config.ent2_start_tag_id,
                             self.config.ent2_end_tag_id]:
                seq_tags.append(self.special_tag_representation(bert_outputs, input_ids, each_tag))
            new_pooled_output = torch.cat((pooled_output, *seq_tags), dim=1)
        elif self.scheme == 3:  # 论文中的(4),只使用两个实体的开始标志：<s1><s2>...
            seq_tags = []
            for each_tag in [self.config.ent1_start_tag_id, self.config.ent2_start_tag_id]:
                seq_tags.append(self.special_tag_representation(bert_outputs, input_ids, each_tag))
            new_pooled_output = torch.cat(seq_tags, dim=1)
        elif self.scheme == 4:  # 这是论文中介绍的(1),只使用[CLS]的output
            new_pooled_output = pooled_output  # shape=(batch_size,hidden_size*2)
        elif self.scheme == 5:  # 这个是最基本的情况，直接将e1_mask和e2_mask对应的全部拿来

            e1_mask = e1_mask.unsqueeze(1)
            e2_mask = e2_mask.unsqueeze(1)
            ent1_rep = torch.bmm(e1_mask.float(), bert_outputs)
            ent2_rep = torch.bmm(e2_mask.float(), bert_outputs)
            ent1_rep = ent1_rep.squeeze(1)
            ent2_rep = ent2_rep.squeeze(1)
            new_pooled_output = torch.cat([ent1_rep, ent2_rep], dim=1)
        elif self.scheme == 6:  # 只获得真正实体对应的部分，取消掉[s1][e1],[s2][e2]

            # 取消e1_mask,e2_mask在[s1][e1],[s2][e2]的label，也就是直接设为0
            # e1_start_idx, e1_end_idx = self.get_ent_position(e1_mask)
            # e2_start_idx, e2_end_idx = self.get_ent_position(e2_mask)
            # e1_mask[e1_start_idx] = 0
            # e1_mask[e1_end_idx] = 0
            # e2_mask[e2_start_idx] = 0
            # e2_mask[e2_end_idx] = 0
            bs,seq_len = e1_mask.shape
            tmp_e1_mask = e1_mask.cpu().numpy().tolist()
            tmp_e2_mask = e2_mask.cpu().numpy().tolist()
            for i in range(bs):
                tmp_e1 = tmp_e1_mask[i]
                tmp_e2 = tmp_e2_mask[i]
                start_idx_e1 =tmp_e1.index(0)
                end_idx_e1 = start_idx_e1+sum(tmp_e1)-1
                start_idx_e2 =tmp_e2.index(0)
                end_idx_e2 = start_idx_e2 + sum(tmp_e2) - 1
                e1_mask[start_idx_e1][end_idx_e1] = 0
                e2_mask[start_idx_e2][end_idx_e2] = 0

            e1_mask = e1_mask.unsqueeze(1)
            e2_mask = e2_mask.unsqueeze(1)
            ent1_rep = torch.bmm(e1_mask.float(), bert_outputs)
            ent2_rep = torch.bmm(e2_mask.float(), bert_outputs)
            ent1_rep = ent1_rep.squeeze(1)
            ent2_rep = ent2_rep.squeeze(1)
            new_pooled_output = torch.cat([ent1_rep, ent2_rep], dim=1)
        else:
            raise ValueError
        return new_pooled_output

class EntityMarkerBaseModel(BaseBert):

    def __init__(self, config:MyBertConfig):
        """
        这是最基础的BERT模型加载，加载预训练的模型
        :param config:
        :param bert_dir:
        :param dropout_prob:
        """
        super(EntityMarkerBaseModel, self).__init__(config)
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
        elif self.scheme == 9 or self.scheme == -9:
            self.classifier_dim = self.bert_config.hidden_size * 2
        elif self.scheme == 10 or self.scheme == -10:
            self.classifier_dim = self.bert_config.hidden_size * 3
        elif self.scheme == 11 or self.scheme == -11:
            self.classifier_dim = self.bert_config.hidden_size * 2
        elif self.scheme == 12 or self.scheme == -12:
            self.classifier_dim = self.bert_config.hidden_size * 3
        elif self.scheme == 13 or self.scheme == -13:
            # [pooled_output,e1_mask,e2_mask]
            self.classifier_dim = self.bert_config.hidden_size * 3
        elif self.scheme == 113 or self.scheme == -113:
            # [pooled_output,e1_mask,e2_mask]
            self.classifier_dim = self.bert_config.hidden_size * 2
        else:
            raise ValueError("请选择合适的scheme值")

    @staticmethod
    def special_tag_representation(seq_output, input_ids, special_tag):
        '''
        这里就是根据special_tag来获取对应的representation
        input_ids就是为了定位位置
        '''
        # nonzero是得到坐标，表示在(input_ids == special_tag)中，值不为0的坐标
        spec_idx = (input_ids == special_tag).nonzero(as_tuple=False)

        temp = []
        for idx in spec_idx:
            temp.append(seq_output[idx[0], idx[1], :])
        tags_rep = torch.stack(temp, dim=0)

        return tags_rep

    def get_ent_position(self, e_mask):
        """
        获得entity mask的start_index和end_index
        :param e_mask: shape=(bs,seq_len)
        :return:
        """
        start_idx = e_mask.index(1)
        for i in range(start_idx + 1, len(e_mask)):
            if e_mask[i] == 1 and e_mask[i + 1] == 0:
                return (start_idx, i)
        return start_idx, len(e_mask) - 1

    def get_entity_representation(self, bert_outputs, pooled_output, input_ids, e1_mask=None, e2_mask=None):
        '''
        这里使用两个的bert outputs输出...
        :param bert_outputs:
        :param pool_output:
        :param schema:
            这里主要是MTB的方法，这里的四种方式和ClinicalTransformer保持一直，之后可能会增加
        :return: 直接返回最终的new_pooled_output
        '''
        if self.scheme == 1:
            seq_tags = []  # 论文中的(2),使用[CLS]和实体的start 标记(也就是<e1>,<e2>或者说<s1><s2>)
            for each_tag in [self.config.ent1_start_tag_id, self.config.ent2_start_tag_id]:
                seq_tags.append(self.special_tag_representation(bert_outputs, input_ids, each_tag))
            new_pooled_output = torch.cat((pooled_output, *seq_tags), dim=1)

        elif self.scheme == 2:  # 论文中的(3),使用实体1和实体2的<s1><e1>,<s2><e2>，这个效果在clinicalTransformer论文中效果最好....
            seq_tags = []
            for each_tag in [self.config.ent1_start_tag_id, self.config.ent1_end_tag_id, self.config.ent2_start_tag_id,
                             self.config.ent2_end_tag_id]:
                seq_tags.append(self.special_tag_representation(bert_outputs, input_ids, each_tag))
            new_pooled_output = torch.cat((pooled_output, *seq_tags), dim=1)
        elif self.scheme == 3:  # 论文中的(4),只使用两个实体的开始标志：<s1><s2>...
            # 这是matching the blanks文件中的start tag的情况...
            seq_tags = []
            for each_tag in [self.config.ent1_start_tag_id, self.config.ent2_start_tag_id]:
                seq_tags.append(self.special_tag_representation(bert_outputs, input_ids, each_tag))
            new_pooled_output = torch.cat(seq_tags, dim=1)
        elif self.scheme == 4:  # 这是论文中介绍的(1),只使用[CLS]的output
            new_pooled_output = pooled_output  # shape=(batch_size,hidden_size*2)
        elif self.scheme == 5:  # 这个是最基本的情况，直接将e1_mask和e2_mask对应的全部拿来

            e1_mask = e1_mask.unsqueeze(1)
            e2_mask = e2_mask.unsqueeze(1)
            ent1_rep = torch.bmm(e1_mask.float(), bert_outputs)
            ent2_rep = torch.bmm(e2_mask.float(), bert_outputs)
            ent1_rep = ent1_rep.squeeze(1)
            ent2_rep = ent2_rep.squeeze(1)
            new_pooled_output = torch.cat([ent1_rep, ent2_rep], dim=1)
        elif self.scheme == 6:  # 只获得真正实体对应的部分，取消掉[s1][e1],[s2][e2]

            # 取消e1_mask,e2_mask在[s1][e1],[s2][e2]的label，也就是直接设为0
            # e1_start_idx, e1_end_idx = self.get_ent_position(e1_mask)
            # e2_start_idx, e2_end_idx = self.get_ent_position(e2_mask)
            # e1_mask[e1_start_idx] = 0
            # e1_mask[e1_end_idx] = 0
            # e2_mask[e2_start_idx] = 0
            # e2_mask[e2_end_idx] = 0
            bs,seq_len = e1_mask.shape
            tmp_e1_mask = e1_mask.cpu().numpy().tolist()
            tmp_e2_mask = e2_mask.cpu().numpy().tolist()
            for i in range(bs):
                tmp_e1 = tmp_e1_mask[i]
                tmp_e2 = tmp_e2_mask[i]
                start_idx_e1 =tmp_e1.index(0)
                end_idx_e1 = start_idx_e1+sum(tmp_e1)-1
                start_idx_e2 =tmp_e2.index(0)
                end_idx_e2 = start_idx_e2 + sum(tmp_e2) - 1
                e1_mask[start_idx_e1][end_idx_e1] = 0
                e2_mask[start_idx_e2][end_idx_e2] = 0

            e1_mask = e1_mask.unsqueeze(1)
            e2_mask = e2_mask.unsqueeze(1)
            ent1_rep = torch.bmm(e1_mask.float(), bert_outputs)
            ent2_rep = torch.bmm(e2_mask.float(), bert_outputs)
            ent1_rep = ent1_rep.squeeze(1)
            ent2_rep = ent2_rep.squeeze(1)
            new_pooled_output = torch.cat([ent1_rep, ent2_rep], dim=1)
        else:
            raise ValueError
        return new_pooled_output

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

    def get_pool_output(self, sequence_output, sequence_pool_output, input_ids, e1_mask, e2_mask,attention_mask=None):
        """
        这是各种组合entity representation及context representation的方式
        """
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

        elif self.scheme == 10:

            context_mask = ((e1_mask + e2_mask) == 0).long() * attention_mask
            context_mask[:,0] = 0
            e1_h = self.entity_average(sequence_output, e1_mask)
            e2_h = self.entity_average(sequence_output, e2_mask)
            context_h = self.entity_average(sequence_output, context_mask)
            e1_context = e1_h + context_h
            e2_context = e2_h + context_h
            concat_h = torch.cat([sequence_pool_output, e1_context, e2_context], dim=-1)  # torch.Size([16, 2304])
        elif self.scheme == -10:
            context_mask = ((e1_mask + e2_mask) == 0).long() * attention_mask
            context_mask[:, 0] = 0
            e1_h = self.entity_average(sequence_output, e1_mask)
            e2_h = self.entity_average(sequence_output, e2_mask)
            context_h = self.entity_average(sequence_output, context_mask)

            e1_context = e1_h + context_h
            e2_context = e2_h + context_h

            e1_context = self.entity_fc_layer(e1_context)
            e2_context = self.entity_fc_layer(e2_context)
            pooled_output = self.cls_fc_layer(sequence_pool_output)

            concat_h = torch.cat([pooled_output, e1_context, e2_context], dim=-1)  # torch.Size([16, 2304])

        elif self.scheme == 11:

            context_mask = ((e1_mask + e2_mask) == 0).long() * attention_mask
            context_mask[:,0] = 0

            e1_h = self.entity_average(sequence_output, e1_mask)
            e2_h = self.entity_average(sequence_output, e2_mask)
            context_h = self.entity_average(sequence_output, context_mask)
            e1_context = e1_h + context_h
            e2_context = e2_h + context_h
            concat_h = torch.cat([e1_context, e2_context], dim=-1)  # torch.Size([16, 2304])
        elif self.scheme == -11:
            context_mask = ((e1_mask + e2_mask) == 0).long() * attention_mask
            context_mask[:, 0] = 0

            e1_h = self.entity_average(sequence_output, e1_mask)
            e2_h = self.entity_average(sequence_output, e2_mask)
            pooled_output = self.cls_fc_layer(sequence_pool_output)
            context_h = self.entity_average(pooled_output, context_mask)
            e1_context = e1_h + context_h
            e2_context = e2_h + context_h
            e1_context = self.entity_fc_layer(e1_context)
            e2_context = self.entity_fc_layer(e2_context)

            concat_h = torch.cat([e1_context, e2_context], dim=-1)  # torch.Size([16, 2304])
        elif self.scheme == 12:
            # [e1]-[e2]之间的所有范围
            context_mask = torch.zeros_like(e1_mask)

            for idx in range(len(e1_mask)):
                tmp_e1_mask = e1_mask[idx].cpu().numpy().tolist()
                e1_start_idx = tmp_e1_mask.index(1)
                tmp_e2_mask = e2_mask[idx].cpu().numpy().tolist()
                e2_start_idx = tmp_e2_mask.index(1)
                if e1_start_idx>e2_start_idx:

                    e1_start_idx,e2_start_idx = e2_start_idx,e1_start_idx
                    e2_start_idx += torch.sum(e1_mask[idx])
                else:
                    e2_start_idx += torch.sum(e2_mask[idx])


                context_mask[idx][e1_start_idx:e2_start_idx] = 1

            e1_h = self.entity_average(sequence_output, e1_mask)
            e2_h = self.entity_average(sequence_output, e2_mask)
            context_h = self.entity_average(sequence_output, context_mask)
            e1_context = e1_h + context_h
            e2_context = e2_h + context_h
            concat_h = torch.cat([sequence_pool_output,e1_context, e2_context], dim=-1)  # torch.Size([16, 2304])
        elif self.scheme == -12:
            # 这个就是ScRCM方式
            # [e1]-[e2]之间的所有范围
            context_mask = torch.zeros_like(e1_mask)

            for idx in range(len(e1_mask)):
                tmp_e1_mask = e1_mask[idx].cpu().numpy().tolist()
                e1_start_idx = tmp_e1_mask.index(1)
                tmp_e2_mask = e2_mask[idx].cpu().numpy().tolist()
                e2_start_idx = tmp_e2_mask.index(1)
                if e1_start_idx > e2_start_idx:

                    e1_start_idx, e2_start_idx = e2_start_idx, e1_start_idx
                    e2_start_idx += torch.sum(e1_mask[idx])
                else:
                    e2_start_idx += torch.sum(e2_mask[idx])

                context_mask[idx][e1_start_idx:e2_start_idx] = 1

            e1_h = self.entity_average(sequence_output, e1_mask)
            e2_h = self.entity_average(sequence_output, e2_mask)
            context_h = self.entity_average(sequence_output, context_mask)
            e1_context = e1_h + context_h
            e2_context = e2_h + context_h

            e1_context = self.entity_fc_layer(e1_context)
            e2_context = self.entity_fc_layer(e2_context)
            pooled_output = self.cls_fc_layer(sequence_pool_output)

            concat_h = torch.cat([pooled_output, e1_context, e2_context], dim=-1)  # torch.Size([16, 2304])
        elif self.scheme == -113:
            # 这个就是new ScRCM方式
            # [e1]-[e2]之间的所有范围
            context_mask = torch.zeros_like(e1_mask)

            for idx in range(len(e1_mask)):
                tmp_e1_mask = e1_mask[idx].cpu().numpy().tolist()
                e1_start_idx = tmp_e1_mask.index(1)
                tmp_e2_mask = e2_mask[idx].cpu().numpy().tolist()
                e2_start_idx = tmp_e2_mask.index(1)
                if e1_start_idx > e2_start_idx:

                    e1_start_idx, e2_start_idx = e2_start_idx, e1_start_idx
                    e2_start_idx += torch.sum(e1_mask[idx])
                else:
                    e2_start_idx += torch.sum(e2_mask[idx])

                context_mask[idx][e1_start_idx:e2_start_idx] = 1

            e1_h = self.entity_average(sequence_output, e1_mask)
            e2_h = self.entity_average(sequence_output, e2_mask)
            context_h = self.entity_average(sequence_output, context_mask)

            concat_1 = torch.cat([e1_h, context_h, e2_h], dim=-1)


            context1 = self.entity_fc_layer(concat_1)

            pooled_output = self.cls_fc_layer(sequence_pool_output)

            concat_h = torch.cat([pooled_output, context1], dim=-1)  # torch.Size([16, 2304])
        elif self.scheme == 9:
            # 这个和1相反，不适用额外的linear层
            # 这是rbert的方式,[[CLS]],[s1]ent1[e1],[s2]ent2[e2]]

            e1_h = self.entity_average(sequence_output, e1_mask)
            e2_h = self.entity_average(sequence_output, e2_mask)
            e1_cls = e1_h+sequence_output
            e2_cls = e2_h+sequence_output
            concat_h = torch.cat([e1_cls, e2_cls], dim=-1)  # torch.Size([16, 2304])

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
            # MTB方式
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
            # 这是只是用start tag和end tag...
            ent1_start = self.special_tag_representation(sequence_output, input_ids, self.config.ent1_start_tag_id)
            ent2_start = self.special_tag_representation(sequence_output, input_ids, self.config.ent2_start_tag_id)

            ent1_start = self.entity_fc_layer(ent1_start)
            ent2_start = self.entity_fc_layer(ent2_start)

            concat_h = torch.cat([ent1_start, ent2_start], dim=-1)  # torch.Size([16, 2304])
        elif self.scheme == 5:
            # baseline方式
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

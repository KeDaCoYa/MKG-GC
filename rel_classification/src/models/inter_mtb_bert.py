# -*- encoding: utf-8 -*-
"""
@File    :   inter_mtb_bert.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/5/9 15:30   
@Description :   None 

"""
import torch
from ipdb import set_trace
from torch import nn
from torch.nn import CrossEntropyLoss

from src.models.base_layers import FCLayer
from src.models.bert_model import BaseBert


class InterMTBBERT(BaseBert):
    def __init__(self, config):
        super(InterMTBBERT, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.dropout = nn.Dropout(config.dropout_prob)


        self.criterion = CrossEntropyLoss(reduction='none')

        self.scheme = config.scheme

        if self.scheme == 1 or self.scheme == -1:
            # [pooled_output,e1_mask,e2_mask]
            self.classifier_dim = self.bert_config.hidden_size * 4
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

    def get_entity_representation(self,sequence_output1,sequence_output2,sequence_pool_output1,sequence_pool_output2,input_ids1,input_ids2,e1_mask=None,e2_mask=None):

        if self.scheme == 1:
            # 这是rbert的方式,[[CLS]],[s1]ent1[e1],[s2]ent2[e2]]
            # 这个还对pool，ent1,ent2额外使用MLP进行转变...

            e1_h = self.entity_average(sequence_output1, e1_mask)
            e2_h = self.entity_average(sequence_output2, e2_mask)

            # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
            # pooled_output.shape=(batch_size,768)
            pooled_output1 = self.cls_fc_layer(sequence_pool_output1)
            pooled_output2 = self.cls_fc_layer(sequence_pool_output2)

            e1_h = self.entity_fc_layer(e1_h)
            e2_h = self.entity_fc_layer(e2_h)

            # Concat -> fc_layer
            concat_h = torch.cat([pooled_output1, e1_h,pooled_output2, e2_h], dim=-1)  # torch.Size([16, 2304])

        return concat_h

    def forward(self, input_ids1, token_type_ids1,attention_masks1,input_ids2, token_type_ids2,attention_masks2,e1_mask,e2_mask,labels=None):
        '''
        这个应该支持多卡训练
        :param input_ids: (batch_size,seq_len,hidden_size)
        :param token_type_ids:
        :param attention_masks:
        :param labels:
        :param entity_positions:
        :return:
        '''

        bert_outputs1 = self.bert_model(input_ids1, attention_mask=attention_masks1, token_type_ids=token_type_ids1)  #返回 sequence_output, pooled_output, (hidden_states), (attentions)
        bert_outputs2 = self.bert_model(input_ids2, attention_mask=attention_masks2, token_type_ids=token_type_ids2)  #返回 sequence_output, pooled_output, (hidden_states), (attentions)

        bert_output1 = bert_outputs1[0] # shape=(batch_size,seq_len,hidden_size)
        pooled_outputs1 = bert_outputs1[1] # shape=(batch_size,seq_len,hidden_size)

        bert_output2 = bert_outputs2[0]  # shape=(batch_size,seq_len,hidden_size)
        pooled_outputs2 = bert_outputs2[1]  # shape=(batch_size,seq_len,hidden_size)

        #pooled_output.shape = (batch_size,hiddensize*2)

        # 这里提取目标representation...

        pooled_output = self.get_entity_representation(bert_output1,bert_output2,pooled_outputs1,pooled_outputs2,input_ids1,input_ids2,e1_mask=e1_mask,e2_mask=e2_mask)


        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)


        if labels is not None:
            loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))
            return loss,logits

        return logits



# -*- encoding: utf-8 -*-
"""
@File    :   inter_r_bert.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/5/9 15:30   
@Description :   None 

"""
import torch
from ipdb import set_trace

from config import BertConfig
from src.models.base_layers import FCLayer
from src.models.bert_model import BaseBert
import torch.nn as nn




class InterRBERT(BaseBert):
    def __init__(self, config: BertConfig, scheme=1):
        super(InterRBERT, self).__init__(config)

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

        self.final_classifier = FCLayer(
            self.classifier_dim,
            self.config.num_labels,
            self.config.dropout_prob,
            use_activation=False,
        )
        if self.config.freeze_bert:
            self.freeze_parameter(config.freeze_layers)

            # 模型层数的初始化初始化
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

    def get_pool_output(self, sequence_output1, sequence_output2,sequence_pool_output1,sequence_pool_outpu2, input_ids, e1_mask, e2_mask):

        if self.scheme == -1:
            # 这是rbert的方式,[[CLS]],[s1]ent1[e1],[s2]ent2[e2]]
            # 这个还对pool，ent1,ent2额外使用MLP进行转变...
            e1_h = self.entity_average(sequence_output1, e1_mask)
            e2_h = self.entity_average(sequence_output2, e2_mask)

            # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
            # pooled_output.shape=(batch_size,768)
            pooled_output1 = self.cls_fc_layer(sequence_pool_output1)
            pooled_output2 = self.cls_fc_layer(sequence_pool_outpu2)

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
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]  # shape=(batch_size,seq_len,hidden_size)
        pooled_output = outputs[1]  # [CLS],shape = (batch_size,hidden_size)=(16,768)

        concat_h = self.get_pool_output(sequence_output, pooled_output, input_ids, e1_mask, e2_mask)
        logits = self.final_classifier(concat_h)

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

# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2021/11/25
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/11/25: 
-------------------------------------------------
"""
import torch
from ipdb import set_trace

import torch.nn as nn
import torch.nn.functional as F
import logging

from torch.nn.utils.rnn import pad_sequence

from config import MyBertConfig
from src.models.bert_model import BaseBert
from src.ner_predicate import vote, span_predicate
from utils.train_utils import load_model
from utils.loss_utils import LabelSmoothingCrossEntropy, FocalLoss

logger = logging.getLogger('main.bert_span')


class Bert_Span(BaseBert):
    def __init__(self, config: MyBertConfig):
        '''
        :param config:
        :param num_tags:这个为2，表示预测的类别
        :param dropout_prob:
        :param is_train:
        :param loss_type:
        '''

        super(Bert_Span, self).__init__(config)
        # 这个时候numtags=2，因为只有disease一种类别
        self.config = config
        self.num_tags = config.num_span_class
        out_dims = self.bert_config.hidden_size
        mid_linear_dims = 128

        # todo:不使用RElu激活函数的结果，尝试更换激活函数...
        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob),
        )
        out_dims = 128
        self.start_fc = nn.Linear(out_dims, self.num_tags)
        self.end_fc = nn.Linear(out_dims, self.num_tags)

        reduction = 'none'
        self.loss_type = config.span_loss_type
        if self.loss_type == 'ce':
            logger.info('损失函数使用:CrossEntropy')
            self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        elif self.loss_type == 'ls_ce':
            logger.info('损失函数使用:LabelSmoothing CrossEntropy-')

            self.criterion = LabelSmoothingCrossEntropy(reduction=reduction)
        elif self.loss_type == 'focal':  # 这个用于多类别...
            logger.info('损失函数使用:Focal Loss')
            self.criterion = FocalLoss(reduction=reduction)

        init_blocks = [self.mid_linear, self.start_fc, self.end_fc]
        self._init_weights(init_blocks)

    def forward(self, token_ids, attention_masks, token_type_ids, input_token_starts=None, start_ids=None, end_ids=None,
                input_true_length=None):
        """

        :param token_ids: 下面三个，给bert的值
        :param attention_masks:
        :param token_type_ids:
        :param input_token_starts:
        :param start_ids: 这个pad是按照batch的实际长度，并不是按照batch的subword长度，
        :param end_ids: 同上
        :param input_true_length: token_ids的真实长度
        :return:
        """

        if self.config.bert_name in ['scibert','biobert','flash','bert','flash_quad','wwm_bert']:
            bert_outputs = self.bert_model(input_ids=token_ids, attention_mask=attention_masks,
                                           token_type_ids=token_type_ids)
            sequence_output = bert_outputs[0]
        elif self.config.bert_name == 'kebiolm':
            bert_outputs = self.bert_model(input_ids=token_ids, attention_mask=attention_masks,
                                           token_type_ids=token_type_ids, return_dict=False)
            sequence_output = bert_outputs[2]  # shape=(batch_size,seq_len,hidden_dim)=[32, 55, 768]
        else:
            raise ValueError

        origin_sequence_output = []
        for layer, starts in zip(sequence_output, input_token_starts):
            res = layer[starts]  # shape=(seq_len,hidden_size)=(256,768)
            origin_sequence_output.append(res)

        # 这里的max_len和上面的seq_len已经不一样了，因为这里是按照token-level,而不是subword-level
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)

        seq_out = self.mid_linear(padded_sequence_output)

        # start_logits.shape = (batch_size,seq_len)=(8,134)
        start_logits = self.start_fc(seq_out)
        end_logits = self.end_fc(seq_out)

        loss_mask = torch.zeros((start_logits.shape[0], start_logits.shape[1])).to(token_ids.device)

        for i, lens in enumerate(input_true_length):
            loss_mask[i][:lens] = 1
            # 正好修正start_ids,end_ids的情况

        # 由于多GPU，修改start_ids
        out = (start_logits, end_logits,)

        if start_ids is not None and end_ids is not None:  # 这是训练模式，计算loss
            # start_logtis.shape=torch.Size([4096, 14])

            start_logits = start_logits.view(-1, self.num_tags)
            end_logits = end_logits.view(-1, self.num_tags)

            # 去掉 padding 部分的标签，计算真实 loss

            mask = loss_mask.view(-1) == 1

            active_start_logits = start_logits[mask]  # (?,14)这个？的值就并不确定了

            active_end_logits = end_logits[mask]

            active_start_labels = start_ids.view(-1)[mask]

            active_end_labels = end_ids.view(-1)[mask]

            start_loss = self.criterion(active_start_logits, active_start_labels).mean(dim=-1)
            end_loss = self.criterion(active_end_logits, active_end_labels).mean(dim=-1)
            loss = start_loss + end_loss
            out = (loss,) + out

        return out


class EnsembleBertSpan:
    def __init__(self, config, trained_model_path_list, device):
        '''
        可以混合不同的Bert模型来作为Encoder...
        bert_name_list:记录每个模型所使用的bert_name,可以混合不同的模型...,但是decoder保持相同....
        '''
        self.models = []
        self.config = config

        for idx, _path in enumerate(trained_model_path_list):
            logger.info('从{}中加载已训练的模型'.format(_path))

            model = Bert_Span(config)
            model = load_model(model, ckpt_path=_path)
            #model.load_state_dict(torch.load(_path, map_location=torch.device('cpu')))
            model.eval()
            model.to(device)
            self.models.append(model)

    def vote_entities(self, batch_data, device, threshold):
        '''
            集成方法：投票法
            每个模型都会预测得到一系列的实体，然后进行投票选择...

            这个非常简单，就是统计所有的实体的出现个数
        '''
        start_ids = None
        end_ids = None
        if self.config.predicate_flag:
            raw_text_list, token_ids, attention_masks, token_type_ids, origin_to_subword_index,input_true_length = batch_data

        else:
            raw_text_list, token_ids, attention_masks, token_type_ids, start_ids, end_ids, origin_to_subword_index,input_true_length = batch_data
            start_ids = start_ids.to(device)
            end_ids = end_ids.to(device)

        input_true_length = input_true_length.to(device)
        token_ids, attention_masks, token_type_ids = token_ids.to(device), attention_masks.to(
            device), token_type_ids.to(device)
        entities_ls = []
        for idx, model in enumerate(self.models):
            # 使用概率平均  融合

            tmp_start_logits, tmp_end_logits = model(token_ids, attention_masks=attention_masks,
                                                        token_type_ids=token_type_ids,
                                                        input_token_starts=origin_to_subword_index,
                                                        start_ids=start_ids, end_ids=end_ids,
                                                        input_true_length=input_true_length)

            _, tmp_start_logits = torch.max(tmp_start_logits, dim=-1)
            _, tmp_end_logits = torch.max(tmp_end_logits, dim=-1)
            tmp_end_logits = tmp_end_logits.cpu().numpy()
            tmp_start_logits = tmp_start_logits.cpu().numpy()

            decode_entities = span_predicate(tmp_start_logits, tmp_end_logits, raw_text_list, self.config.span_id2label)
            entities_ls.append(decode_entities)

        return vote(entities_ls, threshold)

    def predicate(self, batch_data, device):
        """
        融合法
        """
        start_ids = None
        end_ids = None
        if self.config.predicate_flag:
            raw_text_list, token_ids, attention_masks, token_type_ids, origin_to_subword_index = batch_data
        else:
            raw_text_list, token_ids, attention_masks, token_type_ids, start_ids, end_ids, origin_to_subword_index = batch_data
            start_ids = start_ids.to(device)
            end_ids = end_ids.to(device)

        token_ids, attention_masks, token_type_ids = token_ids.to(device), attention_masks.to(
            device), token_type_ids.to(device)

        start_logits = None
        end_logits = None
        for idx, model in enumerate(self.models):

            # 使用概率平均  融合
            weight = 1 / len(self.models)

            _, tmp_start_logits, tmp_end_logits = model(token_ids, attention_masks=attention_masks,
                                                        token_type_ids=token_type_ids,
                                                        input_token_starts=origin_to_subword_index,
                                                        start_ids=start_ids, end_ids=end_ids)

            tmp_start_logits = tmp_start_logits * weight
            tmp_end_logits = tmp_end_logits * weight

            if start_logits is None:
                start_logits = tmp_start_logits
                end_logits = tmp_end_logits
            else:
                start_logits += tmp_start_logits
                end_logits += tmp_end_logits

        return start_logits, end_logits

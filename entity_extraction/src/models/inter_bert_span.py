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
from src.models.flash import GAU
from src.ner_predicate import vote, span_predicate
from utils.train_utils import load_model
from utils.loss_utils import LabelSmoothingCrossEntropy, FocalLoss

logger = logging.getLogger('main.bert_span')


class InterBertSpan(BaseBert):
    def  __init__(self, config: MyBertConfig):
        """
        这个只能针对普通的二分类
        :param config:
        :param num_tags:这个为2，表示预测的类别
        :param dropout_prob:
        :param is_train:
        :param loss_type:
        """

        super(InterBertSpan, self).__init__(config)
        # 这个时候numtags=2，因为只有disease一种类别
        self.config = config
        self.num_tags = config.num_span_class
        self.scheme = config.inter_scheme

        out_dims = self.bert_config.hidden_size
        mid_linear_dims = 128

        # todo:不使用RElu激活函数的结果，尝试更换激活函数...
        if self.scheme in [1,2,3,5]:
            self.mid_linear = nn.Sequential(
                nn.Linear(out_dims, mid_linear_dims),
                nn.Dropout(config.dropout_prob)
            )
            out_dims = mid_linear_dims * 2

        elif self.scheme == 6:
            # 得到的结果时

            self.mid_linear = nn.LSTM(out_dims, mid_linear_dims, batch_first=True, bidirectional=True,num_layers=2, dropout=0.5)
            self.dropout = nn.Dropout(0.5)
            out_dims = mid_linear_dims * 2
        elif self.scheme == 7:
            # 得到的结果时
            self.mid_linear = GAU(dim=768,dropout=0.4)
            self.dropout = nn.Dropout(0.5)
            out_dims = 768
        elif self.scheme == 8:
            # 输出的shape = (batch_size,seq_len,mid_linear_dims)
            mid_linear_dims = 256
            self.start_mid_linear = nn.Sequential(
                nn.Linear(out_dims, mid_linear_dims),
                nn.Dropout(config.dropout_prob)
            )
            self.end_mid_linear = nn.Sequential(
                nn.Linear(out_dims, mid_linear_dims),
                nn.Dropout(config.dropout_prob)
            )
            out_dims = mid_linear_dims * 2

        if self.scheme == 1 or self.scheme == 5:
            self.inter_linear = nn.Linear(self.num_tags,out_dims)
            self.start_fc = nn.Linear(out_dims, self.num_tags)
            self.end_fc = nn.Linear(out_dims, self.num_tags)
            init_blocks = [self.mid_linear, self.start_fc, self.end_fc, self.inter_linear]

        elif self.scheme == 2:
            self.inter_linear = nn.Linear(self.num_tags, out_dims)
            self.start_fc = nn.Linear(out_dims, self.num_tags)
            self.end_fc = nn.Linear(out_dims*2, self.num_tags)
            init_blocks = [self.mid_linear, self.start_fc, self.end_fc, self.inter_linear]

        elif self.scheme == 3:
            self.mid_linear = nn.Sequential(
                nn.Linear(out_dims, mid_linear_dims),
                nn.Dropout(config.dropout_prob),
                nn.LeakyReLU(),
            )

            self.inter_linear = nn.Linear(self.num_tags, 100)
            self.inter_linear2 = nn.Linear(100, out_dims)
            self.start_fc = nn.Linear(out_dims, self.num_tags)
            self.end_fc = nn.Linear(out_dims * 2, self.num_tags)
            init_blocks = [self.mid_linear, self.start_fc, self.end_fc, self.inter_linear]

        elif self.scheme == 4:
            self.inter_linear = nn.LSTM(self.num_tags, out_dims//2, batch_first=True, bidirectional=True,
                            num_layers=2, dropout=0.5)
            self.start_fc = nn.Linear(out_dims, self.num_tags)
            self.end_fc = nn.Linear(out_dims, self.num_tags)
            init_blocks = [self.mid_linear, self.start_fc, self.end_fc, self.inter_linear]

        elif self.scheme == 6 or self.scheme == 7:
            self.inter_linear = nn.Linear(self.num_tags, out_dims)
            self.start_fc = nn.Linear(out_dims, self.num_tags)
            self.end_fc = nn.Linear(out_dims, self.num_tags)
            init_blocks = [self.mid_linear, self.start_fc, self.end_fc, self.inter_linear]

        elif self.scheme == 8:

            self.inter_linear = nn.Linear(mid_linear_dims, mid_linear_dims)
            self.start_fc = nn.Linear(mid_linear_dims, self.num_tags)
            self.end_fc = nn.Linear(mid_linear_dims, self.num_tags)
            init_blocks = [self.start_mid_linear,self.end_mid_linear, self.start_fc, self.end_fc, self.inter_linear]
        elif self.scheme == 11:
            self.mid_linear = nn.Sequential(
                nn.Linear(out_dims, mid_linear_dims),
                nn.Dropout(config.dropout_prob)
            )
            self.inter_linear = nn.Sequential(
                nn.Linear(self.num_tags,mid_linear_dims),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
            )
            self.start_fc = nn.Linear(mid_linear_dims, self.num_tags)
            self.end_fc = nn.Linear(mid_linear_dims, self.num_tags)
            init_blocks = [self.mid_linear, self.start_fc, self.end_fc, self.inter_linear]
        elif self.scheme == 12:
            self.mid_linear = nn.Sequential(
                nn.Linear(out_dims, mid_linear_dims),
                nn.Dropout(config.dropout_prob)
            )
            self.inter_linear = nn.Sequential(
                nn.Linear(self.num_tags,mid_linear_dims),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
            )
            self.start_fc = nn.Linear(mid_linear_dims, self.num_tags)
            self.end_fc = nn.Linear(mid_linear_dims, self.num_tags)
            init_blocks = [self.mid_linear, self.start_fc, self.end_fc, self.inter_linear]

            self.dynamic_weight = nn.Parameter(torch.empty(1))
            self.dynamic_weight.data.fill_(0.5)  # init sparse_weight
        elif self.scheme == 13:
            # 加权方式是加法
            self.mid_linear = nn.Sequential(
                nn.Linear(out_dims, mid_linear_dims),
                nn.Dropout(config.dropout_prob)
            )
            self.inter_linear = nn.Sequential(
                nn.Linear(self.num_tags,mid_linear_dims),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
            )
            self.start_fc = nn.Linear(mid_linear_dims, self.num_tags)
            self.end_fc = nn.Linear(mid_linear_dims, self.num_tags)
            init_blocks = [self.mid_linear, self.start_fc, self.end_fc, self.inter_linear]
        elif self.scheme == 20:
            """
            20是CNN系列
            """
        elif self.scheme == 30:
            """
            30是BilSTM系列
            """

            self.mid_linear = nn.LSTM(out_dims, mid_linear_dims // 2, batch_first=True, bidirectional=True,num_layers=2, dropout=0.5)
            self.inter_linear = nn.Sequential(
                nn.Linear(self.num_tags, mid_linear_dims),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
            )
            self.start_fc = nn.Linear(mid_linear_dims, self.num_tags)
            self.end_fc = nn.Linear(mid_linear_dims, self.num_tags)
            init_blocks = [self.mid_linear, self.start_fc, self.end_fc, self.inter_linear]


        elif self.scheme == 40:
            """
            40是BiGRU系列的实验
            
            """
            self.mid_linear = nn.GRU(out_dims, mid_linear_dims // 2, batch_first=True, bidirectional=True,
                                      num_layers=2, dropout=0.5)
            self.inter_linear = nn.Sequential(
                nn.Linear(self.num_tags, mid_linear_dims),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
            )

            self.start_fc = nn.Linear(mid_linear_dims, self.num_tags)
            self.end_fc = nn.Linear(mid_linear_dims, self.num_tags)
            init_blocks = [self.mid_linear, self.start_fc, self.end_fc, self.inter_linear]

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

        # 如果是scheme

        if self.scheme == 1:
            # 这是最原始的方式
            seq_out = self.mid_linear(padded_sequence_output)
            start_logits = self.start_fc(seq_out)
            inter_logits = F.relu(self.inter_linear(start_logits))
            seq_out = (seq_out+inter_logits)/2
            end_logits = self.end_fc(seq_out)
        elif self.scheme == 11:
            # 这是最原始的方式

            seq_out = self.mid_linear(padded_sequence_output)
            start_logits = self.start_fc(seq_out)
            inter_logits = self.inter_linear(start_logits)
            seq_out = (seq_out + inter_logits) / 2
            end_logits = self.end_fc(seq_out)
        elif self.scheme == 12:
            # 加权方式是学习参数
            seq_out = self.mid_linear(padded_sequence_output)
            start_logits = self.start_fc(seq_out)
            inter_logits = self.inter_linear(start_logits)

            seq_out = self.dynamic_weight*seq_out + (1-self.dynamic_weight)*inter_logits
            end_logits = self.end_fc(seq_out)
        elif self.scheme == 13:
            # 加权方式是+
            seq_out = self.mid_linear(padded_sequence_output)
            start_logits = self.start_fc(seq_out)
            inter_logits = self.inter_linear(start_logits)

            seq_out = seq_out + inter_logits
            end_logits = self.end_fc(seq_out)
        elif self.scheme == 2:
            seq_out = self.mid_linear(padded_sequence_output)
            start_logits = self.start_fc(seq_out)
            inter_logits = F.relu(self.inter_linear(start_logits))

            seq_out = torch.cat((seq_out,inter_logits),axis=-1)
            end_logits = self.end_fc(seq_out)
        elif self.scheme == 3:
            seq_out = self.mid_linear(padded_sequence_output)

            start_logins = self.inter_linear(seq_out)
            start_logits = self.start_fc(start_logins)
            inter_logits = F.tanh(self.inter_linear2())
            seq_out = torch.cat((seq_out, inter_logits), axis=-1)
            end_logits = self.end_fc(seq_out)
        elif self.scheme == 4:
            seq_out = self.mid_linear(padded_sequence_output)
            start_logits = self.start_fc(seq_out)
            inter_logits = F.relu(self.inter_linear(start_logits)[0])
            seq_out = (seq_out + inter_logits) / 2
            end_logits = self.end_fc(seq_out)
        elif self.scheme == 5:
            seq_out = self.mid_linear(padded_sequence_output)
            start_logits = self.start_fc(seq_out)
            inter_logits = F.relu(self.inter_linear(start_logits))
            seq_out = (seq_out + inter_logits)
            end_logits = self.end_fc(seq_out)
        elif self.scheme == 6:
            seq_out = self.mid_linear(padded_sequence_output)
            seq_out = self.dropout(seq_out[0])
            start_logits = self.start_fc(seq_out)
            inter_logits = F.relu(self.inter_linear(start_logits))
            seq_out = (seq_out + inter_logits)
            end_logits = self.end_fc(seq_out)

        elif self.scheme == 7:

            seq_out = self.mid_linear(padded_sequence_output)
            seq_out = self.dropout(seq_out)

            start_logits = self.start_fc(seq_out)
            inter_logits = F.relu(self.inter_linear(start_logits))
            seq_out = (seq_out + inter_logits)
            end_logits = self.end_fc(seq_out)
        elif self.scheme == 8:

            start_seq_out = self.start_mid_linear(padded_sequence_output)
            end_seq_out = self.end_mid_linear(padded_sequence_output)
            start_logits = self.start_fc(start_seq_out)
            inter_logits = F.relu(self.inter_linear(start_seq_out))
            end_seq_out = (end_seq_out + inter_logits)
            end_logits = self.end_fc(end_seq_out)
        elif self.scheme == 30:
            # bilstm+加权方式是+
            seq_out,_ = self.mid_linear(padded_sequence_output)
            start_logits = self.start_fc(seq_out)
            inter_logits = self.inter_linear(start_logits)

            seq_out = seq_out + inter_logits
            end_logits = self.end_fc(seq_out)
        elif self.scheme == 40:
            # bilstm+加权方式是+

            seq_out, _ = self.mid_linear(padded_sequence_output)
            start_logits = self.start_fc(seq_out)
            inter_logits = self.inter_linear(start_logits)

            seq_out = seq_out + inter_logits
            end_logits = self.end_fc(seq_out)

        else:
            raise ValueError


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

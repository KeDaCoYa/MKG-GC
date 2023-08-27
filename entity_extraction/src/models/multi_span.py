# -*- encoding: utf-8 -*-
"""
@File    :   multi_ner.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/3/27 13:34   
@Description :   这个是多任务学习的NER


"""

import os
import copy
import logging

import torch
import torch.nn.functional as F
from torch import nn
from ipdb import set_trace
from torch.nn.utils.rnn import pad_sequence

from src.models.bert_model import BaseBert
from utils.loss_utils import LabelSmoothingCrossEntropy, FocalLoss

logger = logging.getLogger("main.multiner_bertspan")


class MultiSpanForFive(BaseBert):
    def __init__(self, config):
        """
            这个是
        """
        super(MultiSpanForFive, self).__init__(config)
        # 这个时候numtags=2，因为只有disease一种类别
        self.config = config
        self.num_tags = config.num_span_class
        out_dims = self.bert_config.hidden_size
        mid_linear_dims = 128

        self.protein_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )

        self.cell_line_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )
        self.cell_type_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )
        self.dna_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )
        self.rna_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )

        out_dims = 128

        self.protein_start_fc = nn.Linear(out_dims, self.num_tags)
        self.protein_end_fc = nn.Linear(out_dims, self.num_tags)

        self.cell_line_start_fc = nn.Linear(out_dims, self.num_tags)
        self.cell_line_end_fc = nn.Linear(out_dims, self.num_tags)

        self.cell_type_start_fc = nn.Linear(out_dims, self.num_tags)
        self.cell_type_end_fc = nn.Linear(out_dims, self.num_tags)

        self.dna_start_fc = nn.Linear(out_dims, self.num_tags)
        self.dna_end_fc = nn.Linear(out_dims, self.num_tags)

        self.rna_start_fc = nn.Linear(out_dims, self.num_tags)
        self.rna_end_fc = nn.Linear(out_dims, self.num_tags)

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

        init_blocks = [
            self.protein_mid_linear, self.protein_start_fc, self.protein_end_fc,
            self.cell_line_mid_linear, self.cell_line_start_fc, self.cell_line_end_fc,
            self.cell_type_mid_linear, self.cell_type_start_fc, self.cell_type_end_fc,
            self.dna_mid_linear, self.dna_start_fc, self.dna_end_fc,
            self.rna_mid_linear, self.rna_start_fc, self.rna_end_fc,
        ]
        self._init_weights(init_blocks)

    def forward(self, token_ids, attention_masks, token_type_ids, input_token_starts=None, start_ids=None, end_ids=None,
                input_true_length=None, entity_type_ids=None):
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

        if self.config.bert_name in ['biobert', 'wwm_bert', 'flash_quad', 'scibert', 'bert']:
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
        sequence_output = pad_sequence(origin_sequence_output, batch_first=True)

        if entity_type_ids[0][0].item() == 0:
            '''
            Raw text data with trained parameters
            '''

            protein_sequence_output = F.relu(self.protein_mid_linear(sequence_output))  # gene/protein logit value
            cellline_sequence_output = F.relu(self.cell_line_mid_linear(sequence_output))  # cell line logit value
            dna_sequence_output = F.relu(self.dna_mid_linear(sequence_output))  # dna logit value
            rna_sequence_output = F.relu(self.rna_mid_linear(sequence_output))  # rna logit value
            celltype_sequence_output = F.relu(self.cell_type_mid_linear(sequence_output))  # cell type logit value

            protein_start_logits = self.protein_start_fc(protein_sequence_output)  # gene/protein logit value
            protein_end_logits = self.protein_end_fc(protein_sequence_output)  # gene/protein logit value

            cellline_start_logits = self.cell_line_start_fc(cellline_sequence_output)  # cell line logit value
            cellline_end_logits = self.cell_line_end_fc(cellline_sequence_output)  # cell line logit value

            dna_start_logits = self.dna_start_fc(dna_sequence_output)  # dna logit value
            dna_end_logits = self.dna_end_fc(dna_sequence_output)  # dna logit value

            rna_start_logits = self.rna_start_fc(rna_sequence_output)  # rna logit value
            rna_end_logits = self.rna_end_fc(rna_sequence_output)  # rna logit value

            celltype_start_logits = self.cell_type_start_fc(celltype_sequence_output)  # cell type logit value
            celltype_end_logits = self.cell_type_end_fc(celltype_sequence_output)  # cell type logit value

            # update logit and sequence_output
            sequence_output = dna_sequence_output + protein_sequence_output + celltype_sequence_output + cellline_sequence_output + rna_sequence_output
            start_logits = (
            dna_start_logits, protein_start_logits, celltype_start_logits, cellline_start_logits, rna_start_logits)
            end_logits = (dna_end_logits, protein_end_logits, celltype_end_logits, cellline_end_logits, rna_end_logits)

        else:
            ''' 
            Train, Eval, Test with pre-defined entity type tags
            '''

            protein_idx = copy.deepcopy(entity_type_ids)
            cellline_idx = copy.deepcopy(entity_type_ids)
            dna_idx = copy.deepcopy(entity_type_ids)
            rna_idx = copy.deepcopy(entity_type_ids)
            celltype_idx = copy.deepcopy(entity_type_ids)

            protein_idx[protein_idx != 2] = 0
            cellline_idx[cellline_idx != 4] = 0
            dna_idx[dna_idx != 1] = 0
            rna_idx[rna_idx != 5] = 0
            celltype_idx[celltype_idx != 3] = 0

            protein_sequence_output = protein_idx.unsqueeze(-1) * sequence_output
            cellline_sequence_output = cellline_idx.unsqueeze(-1) * sequence_output
            dna_sequence_output = dna_idx.unsqueeze(-1) * sequence_output
            rna_sequence_output = rna_idx.unsqueeze(-1) * sequence_output
            celltype_sequence_output = celltype_idx.unsqueeze(-1) * sequence_output

            # F.tanh or F.relu

            protein_sequence_output = F.relu(
                self.protein_mid_linear(protein_sequence_output))  # gene/protein logit value
            cellline_sequence_output = F.relu(
                self.cell_line_mid_linear(cellline_sequence_output))  # cell line logit value
            dna_sequence_output = F.relu(self.dna_mid_linear(dna_sequence_output))  # dna logit value
            rna_sequence_output = F.relu(self.rna_mid_linear(rna_sequence_output))  # rna logit value
            celltype_sequence_output = F.relu(
                self.cell_type_mid_linear(celltype_sequence_output))  # cell type logit value

            protein_start_logits = self.protein_start_fc(protein_sequence_output)  # gene/protein logit value
            protein_end_logits = self.protein_end_fc(protein_sequence_output)  # gene/protein logit value

            cellline_start_logits = self.cell_line_start_fc(cellline_sequence_output)  # cell line logit value
            cellline_end_logits = self.cell_line_end_fc(cellline_sequence_output)  # cell line logit value

            dna_start_logits = self.dna_start_fc(dna_sequence_output)  # dna logit value
            dna_end_logits = self.dna_end_fc(dna_sequence_output)  # dna logit value

            rna_start_logits = self.rna_start_fc(rna_sequence_output)  # rna logit value
            rna_end_logits = self.rna_end_fc(rna_sequence_output)  # rna logit value

            celltype_start_logits = self.cell_type_start_fc(celltype_sequence_output)  # cell type logit value
            celltype_end_logits = self.cell_type_end_fc(celltype_sequence_output)  # cell type logit value

            sequence_output = protein_sequence_output + cellline_sequence_output + dna_sequence_output + rna_sequence_output + celltype_sequence_output

            start_logits = protein_start_logits + celltype_start_logits + cellline_start_logits + dna_start_logits + rna_start_logits
            end_logits = protein_end_logits + celltype_end_logits + cellline_end_logits + dna_end_logits + rna_end_logits

        output = (start_logits, end_logits)
        if start_ids is not None and end_ids is not None:  # 这是训练模式，计算loss
            loss_mask = torch.zeros((start_logits.shape[0], start_logits.shape[1])).to(token_ids.device)

            for i, lens in enumerate(input_true_length):
                loss_mask[i][:lens] = 1
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

            return (loss,) + output

        else:
            return output


class MultiSpanForBinary(BaseBert):
    def __init__(self, config):
        super(MultiSpanForBinary, self).__init__(config)
        # 这个时候numtags=2，因为只有disease一种类别
        self.config = config
        self.num_tags = config.num_span_class
        out_dims = self.bert_config.hidden_size
        mid_linear_dims = 128

        self.gene_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )

        self.chemical_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )

        out_dims = 128

        self.gene_start_fc = nn.Linear(out_dims, self.num_tags)
        self.gene_end_fc = nn.Linear(out_dims, self.num_tags)

        self.chemical_start_fc = nn.Linear(out_dims, self.num_tags)
        self.chemical_end_fc = nn.Linear(out_dims, self.num_tags)

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

        init_blocks = [

            self.gene_mid_linear, self.gene_start_fc, self.gene_end_fc,
            self.chemical_mid_linear, self.chemical_start_fc, self.chemical_end_fc,
        ]
        self._init_weights(init_blocks)

    def forward(self, token_ids, attention_masks, token_type_ids, input_token_starts=None, start_ids=None, end_ids=None,
                input_true_length=None, entity_type_ids=None):
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

        if self.config.bert_name in ['biobert', 'wwm_bert', 'flash_quad', 'scibert', 'bert']:
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
        sequence_output = pad_sequence(origin_sequence_output, batch_first=True)

        if entity_type_ids[0][0].item() == 0:
            '''
            Raw text data with trained parameters
            '''

            gene_sequence_output = F.relu(self.gene_mid_linear(sequence_output))  # gene/protein logit value
            chemical_sequence_output = F.relu(self.chemical_mid_linear(sequence_output))  # cell line logit value

            chemical_start_logits = self.chemical_mid_linear(chemical_sequence_output)  # gene/protein logit value
            chemical_end_logits = self.chemical_mid_linear(chemical_sequence_output)  # gene/protein logit value

            gene_start_logits = self.gene_line_start_fc(gene_sequence_output)  # cell line logit value
            gene_end_logits = self.gene_line_end_fc(gene_sequence_output)  # cell line logit value

            sequence_output = chemical_sequence_output + gene_sequence_output
            start_logits = (chemical_start_logits, gene_start_logits)
            end_logits = (chemical_end_logits, gene_end_logits)

        else:
            ''' 
            Train, Eval, Test with pre-defined entity type tags
            '''

            chemical_idx = copy.deepcopy(entity_type_ids)
            gene_idx = copy.deepcopy(entity_type_ids)

            chemical_idx[chemical_idx != 1] = 0
            gene_idx[gene_idx != 2] = 0

            gene_sequence_output = gene_idx.unsqueeze(-1) * sequence_output
            chemical_sequence_output = chemical_idx.unsqueeze(-1) * sequence_output

            # F.tanh or F.relu

            chemical_sequence_output = F.relu(
                self.chemical_mid_linear(chemical_sequence_output))  # gene/protein logit value
            gene_sequence_output = F.relu(self.gene_mid_linear(gene_sequence_output))  # cell line logit value

            gene_start_logits = self.gene_start_fc(gene_sequence_output)  # gene/protein logit value
            gene_end_logits = self.gene_end_fc(gene_sequence_output)  # gene/protein logit value

            chemical_start_logits = self.chemical_start_fc(chemical_sequence_output)  # cell line logit value
            chemical_end_logits = self.chemical_end_fc(chemical_sequence_output)  # cell line logit value

            sequence_output = chemical_sequence_output + gene_sequence_output

            start_logits = chemical_start_logits + gene_start_logits
            end_logits = chemical_end_logits + gene_end_logits

        output = (start_logits, end_logits)

        if start_ids is not None and end_ids is not None:  # 这是训练模式，计算loss
            loss_mask = torch.zeros((start_logits.shape[0], start_logits.shape[1])).to(token_ids.device)

            for i, lens in enumerate(input_true_length):
                loss_mask[i][:lens] = 1
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

            return (loss,) + output

        else:
            return output


class MultiSpanForFour(BaseBert):
    def __init__(self, config):
        super(MultiSpanForFour, self).__init__(config)
        # 针对有四个类别的模型
        self.config = config
        self.num_tags = 5
        out_dims = self.bert_config.hidden_size
        mid_linear_dims = 128

        self.drug_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )

        self.drug_n_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )
        self.group_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )

        self.brand_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )

        out_dims = 128

        self.drug_start_fc = nn.Linear(out_dims, self.num_tags)
        self.drug_end_fc = nn.Linear(out_dims, self.num_tags)

        self.drug_n_start_fc = nn.Linear(out_dims, self.num_tags)
        self.drug_n_end_fc = nn.Linear(out_dims, self.num_tags)

        self.brand_start_fc = nn.Linear(out_dims, self.num_tags)
        self.brand_end_fc = nn.Linear(out_dims, self.num_tags)

        self.group_start_fc = nn.Linear(out_dims, self.num_tags)
        self.group_end_fc = nn.Linear(out_dims, self.num_tags)

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

        init_blocks = [
            self.drug_mid_linear, self.drug_start_fc, self.drug_end_fc,
            self.drug_n_mid_linear, self.drug_n_start_fc, self.drug_n_end_fc,
            self.group_mid_linear, self.group_start_fc, self.group_end_fc,
            self.brand_mid_linear, self.brand_start_fc, self.brand_end_fc,
        ]
        self._init_weights(init_blocks)

    def forward(self, token_ids, attention_masks, token_type_ids, input_token_starts=None, start_ids=None, end_ids=None,
                input_true_length=None, entity_type_ids=None):
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

        if self.config.bert_name in ['biobert', 'wwm_bert', 'flash_quad', 'scibert', 'bert']:
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
        sequence_output = pad_sequence(origin_sequence_output, batch_first=True)

        if entity_type_ids[0][0].item() == 0:
            '''
            Raw text data with trained parameters
            '''

            drug_sequence_output = F.relu(self.drug_mid_linear(sequence_output))  # gene/protein logit value
            drug_n_sequence_output = F.relu(self.drug_n_mid_linear(sequence_output))  # cell line logit value
            brand_sequence_output = F.relu(self.brand_mid_linear(sequence_output))  # gene/protein logit value
            group_sequence_output = F.relu(self.group_mid_linear(sequence_output))  # cell line logit value

            drug_start_logits = self.drug_start_fc(drug_sequence_output)  # gene/protein logit value
            drug_end_logits = self.drug_end_fc(drug_sequence_output)  # gene/protein logit value

            drug_n_start_logits = self.drug_n_start_fc(drug_n_sequence_output)  # cell line logit value
            drug_n_end_logits = self.drug_n_end_fc(drug_n_sequence_output)  # cell line logit value

            brand_start_logits = self.brand_start_fc(brand_sequence_output)  # cell line logit value
            brand_end_logits = self.brand_end_fc(brand_sequence_output)  # cell line logit value

            group_start_logits = self.group_start_fc(group_sequence_output)  # cell line logit value
            group_end_logits = self.group_end_fc(group_sequence_output)  # cell line logit value

            sequence_output = drug_sequence_output + drug_n_sequence_output + brand_sequence_output + group_sequence_output
            start_logits = (drug_start_logits, drug_n_start_logits, brand_start_logits, group_start_logits)
            end_logits = (drug_end_logits, drug_n_end_logits, brand_end_logits, group_end_logits)

        else:
            ''' 
            Train, Eval, Test with pre-defined entity type tags
            '''

            drug_idx = copy.deepcopy(entity_type_ids)
            drug_n_idx = copy.deepcopy(entity_type_ids)
            group_idx = copy.deepcopy(entity_type_ids)
            brand_idx = copy.deepcopy(entity_type_ids)

            drug_idx[drug_idx != 1] = 0
            drug_n_idx[drug_n_idx != 2] = 0
            group_idx[group_idx != 3] = 0
            brand_idx[brand_idx != 4] = 0

            drug_sequence_output = drug_idx.unsqueeze(-1) * sequence_output
            drug_n_sequence_output = drug_n_idx.unsqueeze(-1) * sequence_output
            group_sequence_output = group_idx.unsqueeze(-1) * sequence_output
            brand_sequence_output = brand_idx.unsqueeze(-1) * sequence_output

            drug_sequence_output = F.relu(self.drug_mid_linear(drug_sequence_output))
            drug_n_sequence_output = F.relu(self.drug_n_mid_linear(drug_n_sequence_output))
            group_sequence_output = F.relu(self.group_mid_linear(group_sequence_output))
            brand_sequence_output = F.relu(self.brand_mid_linear(brand_sequence_output))

            drug_start_logits = self.drug_start_fc(drug_sequence_output)  # gene/protein logit value
            drug_end_logits = self.drug_end_fc(drug_sequence_output)  # gene/protein logit value

            drug_n_start_logits = self.drug_n_start_fc(drug_n_sequence_output)  # cell line logit value
            drug_n_end_logits = self.drug_n_end_fc(drug_n_sequence_output)  # cell line logit value

            brand_start_logits = self.brand_start_fc(brand_sequence_output)  # cell line logit value
            brand_end_logits = self.brand_end_fc(brand_sequence_output)  # cell line logit value

            group_start_logits = self.group_start_fc(group_sequence_output)  # cell line logit value
            group_end_logits = self.group_end_fc(group_sequence_output)  # cell line logit value

            sequence_output = drug_sequence_output + drug_n_sequence_output + group_sequence_output + brand_sequence_output

            start_logits = drug_start_logits + drug_n_start_logits + brand_start_logits + group_start_logits
            end_logits = drug_end_logits + drug_n_end_logits + brand_end_logits + group_end_logits

        output = (start_logits, end_logits)

        if start_ids is not None and end_ids is not None:  # 这是训练模式，计算loss
            loss_mask = torch.zeros((start_logits.shape[0], start_logits.shape[1])).to(token_ids.device)

            for i, lens in enumerate(input_true_length):
                loss_mask[i][:lens] = 1
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

            return (loss,) + output

        else:
            return output

class MultiSpanForEight(BaseBert):
    def __init__(self,config):
        super(MultiSpanForEight, self).__init__(config)
        # 这个时候numtags=2，因为只有disease一种类别
        self.config = config
        self.num_tags = config.num_span_class
        out_dims = self.bert_config.hidden_size
        mid_linear_dims = 128

        # 准备的实体类别有:DNA,RNA,Gene/Protein,Disease,Chemical/Durg,cell_type,cell_line,species
        self.chem_drug_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )
        self.gene_protein_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )
        self.disease_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )
        self.cell_line_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )
        self.cell_type_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )
        self.dna_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )
        self.rna_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )
        self.species_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )


        out_dims = 128

        #
        self.chem_drug_start_fc = nn.Linear(out_dims, self.num_tags)
        self.chem_drug_end_fc = nn.Linear(out_dims, self.num_tags)

        self.gene_protein_start_fc = nn.Linear(out_dims, self.num_tags)
        self.gene_protein_end_fc = nn.Linear(out_dims, self.num_tags)

        self.disease_start_fc = nn.Linear(out_dims, self.num_tags)
        self.disease_end_fc = nn.Linear(out_dims, self.num_tags)

        self.cell_line_start_fc = nn.Linear(out_dims, self.num_tags)
        self.cell_line_end_fc = nn.Linear(out_dims, self.num_tags)

        self.cell_type_start_fc = nn.Linear(out_dims, self.num_tags)
        self.cell_type_end_fc = nn.Linear(out_dims, self.num_tags)

        self.dna_start_fc = nn.Linear(out_dims, self.num_tags)
        self.dna_end_fc = nn.Linear(out_dims, self.num_tags)

        self.rna_start_fc = nn.Linear(out_dims, self.num_tags)
        self.rna_end_fc = nn.Linear(out_dims, self.num_tags)

        self.spec_start_fc = nn.Linear(out_dims, self.num_tags)
        self.spec_end_fc = nn.Linear(out_dims, self.num_tags)



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

        init_blocks = [
                self.chem_drug_mid_linear, self.chem_drug_start_fc, self.chem_drug_end_fc,
                self.gene_protein_mid_linear, self.gene_protein_start_fc, self.gene_protein_end_fc,
                self.disease_mid_linear, self.disease_start_fc, self.disease_end_fc,
                self.cell_line_mid_linear, self.cell_line_start_fc, self.cell_line_end_fc,
                self.cell_type_mid_linear, self.cell_type_start_fc, self.cell_type_end_fc,
                self.dna_mid_linear, self.dna_start_fc, self.dna_end_fc,
                self.rna_mid_linear, self.rna_start_fc, self.rna_end_fc,
                self.species_mid_linear, self.spec_start_fc, self.spec_end_fc,
                       ]
        self._init_weights(init_blocks)
    def forward(self, token_ids, attention_masks, token_type_ids, input_token_starts=None, start_ids=None, end_ids=None,input_true_length=None,entity_type_ids=None):
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

        if self.config.bert_name in ['biobert','wwm_bert','flash_quad','bert','scibert']:
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
        sequence_output = pad_sequence(origin_sequence_output, batch_first=True)


        if entity_type_ids[0][0].item() == 0:
            '''
            Raw text data with trained parameters
            '''
            dise_sequence_output = F.relu(self.disease_mid_linear(sequence_output)) # disease logit value
            chem_sequence_output = F.relu(self.chem_drug_mid_linear(sequence_output)) # chemical logit value
            gene_sequence_output = F.relu(self.gene_protein_mid_linear(sequence_output)) # gene/protein logit value
            spec_sequence_output = F.relu(self.species_mid_linear(sequence_output)) # species logit value
            cellline_sequence_output = F.relu(self.cell_line_mid_linear(sequence_output)) # cell line logit value
            dna_sequence_output = F.relu(self.dna_mid_linear(sequence_output)) # dna logit value
            rna_sequence_output = F.relu(self.rna_mid_linear(sequence_output)) # rna logit value
            celltype_sequence_output = F.relu(self.cell_type_mid_linear(sequence_output)) # cell type logit value

            dise_start_logits = self.disease_start_fc(dise_sequence_output) # disease logit value
            dise_end_logits = self.disease_end_fc(dise_sequence_output) # disease logit value

            chem_start_logits = self.chem_drug_start_fc(chem_sequence_output) # chemical logit value
            chem_end_logits = self.chem_drug_end_fc(chem_sequence_output) # chemical logit value

            gene_start_logits = self.gene_protein_start_fc(gene_sequence_output) # gene/protein logit value
            gene_end_logits = self.gene_protein_end_fc(gene_sequence_output) # gene/protein logit value

            spec_start_logits = self.spec_start_fc(spec_sequence_output) # species logit value
            spec_end_logits = self.spec_end_fc(spec_sequence_output) # species logit value

            cellline_start_logits = self.cell_line_start_fc(cellline_sequence_output) # cell line logit value
            cellline_end_logits = self.cell_line_end_fc(cellline_sequence_output) # cell line logit value

            dna_start_logits = self.dna_start_fc(dna_sequence_output) # dna logit value
            dna_end_logits = self.dna_end_fc(dna_sequence_output) # dna logit value

            rna_start_logits = self.rna_start_fc(rna_sequence_output) # rna logit value
            rna_end_logits = self.rna_end_fc(rna_sequence_output) # rna logit value


            celltype_start_logits = self.cell_type_start_fc(celltype_sequence_output) # cell type logit value
            celltype_end_logits = self.cell_type_end_fc(celltype_sequence_output) # cell type logit value


            # update logit and sequence_output
            sequence_output = dise_sequence_output + chem_sequence_output + gene_sequence_output + spec_sequence_output + cellline_sequence_output + dna_sequence_output + rna_sequence_output + celltype_sequence_output
            start_logits = (dise_start_logits,chem_start_logits,gene_start_logits,spec_start_logits,celltype_start_logits,cellline_start_logits,dna_start_logits,rna_start_logits)
            end_logits = (dise_end_logits,chem_end_logits,gene_end_logits,spec_end_logits,celltype_end_logits,cellline_end_logits,dna_end_logits,rna_end_logits)

        else:
            ''' 
            Train, Eval, Test with pre-defined entity type tags
            '''
            # make 1*1 conv to adopt entity type
            dise_idx = copy.deepcopy(entity_type_ids)
            chem_idx = copy.deepcopy(entity_type_ids)
            gene_idx = copy.deepcopy(entity_type_ids)
            spec_idx = copy.deepcopy(entity_type_ids)
            cellline_idx = copy.deepcopy(entity_type_ids)
            dna_idx = copy.deepcopy(entity_type_ids)
            rna_idx = copy.deepcopy(entity_type_ids)
            celltype_idx = copy.deepcopy(entity_type_ids)

            dise_idx[dise_idx != 1] = 0
            chem_idx[chem_idx != 2] = 0
            gene_idx[gene_idx != 3] = 0
            spec_idx[spec_idx != 4] = 0
            cellline_idx[cellline_idx != 5] = 0
            dna_idx[dna_idx != 6] = 0
            rna_idx[rna_idx != 7] = 0
            celltype_idx[celltype_idx != 8] = 0

            dise_sequence_output = dise_idx.unsqueeze(-1) * sequence_output
            chem_sequence_output = chem_idx.unsqueeze(-1) * sequence_output
            gene_sequence_output = gene_idx.unsqueeze(-1) * sequence_output
            spec_sequence_output = spec_idx.unsqueeze(-1) * sequence_output
            cellline_sequence_output = cellline_idx.unsqueeze(-1) * sequence_output
            dna_sequence_output = dna_idx.unsqueeze(-1) * sequence_output
            rna_sequence_output = rna_idx.unsqueeze(-1) * sequence_output
            celltype_sequence_output = celltype_idx.unsqueeze(-1) * sequence_output

            # F.tanh or F.relu
            dise_sequence_output = F.relu(self.disease_mid_linear(dise_sequence_output))  # disease logit value
            chem_sequence_output = F.relu(self.chem_drug_mid_linear(chem_sequence_output))  # chemical logit value
            gene_sequence_output = F.relu(self.gene_protein_mid_linear(gene_sequence_output))  # gene/protein logit value
            spec_sequence_output = F.relu(self.species_mid_linear(spec_sequence_output))  # species logit value
            cellline_sequence_output = F.relu(self.cell_line_mid_linear(cellline_sequence_output))  # cell line logit value
            dna_sequence_output = F.relu(self.dna_mid_linear(dna_sequence_output))  # dna logit value
            rna_sequence_output = F.relu(self.rna_mid_linear(rna_sequence_output))  # rna logit value
            celltype_sequence_output = F.relu(self.cell_type_mid_linear(celltype_sequence_output))  # cell type logit value

            dise_start_logits = self.disease_start_fc(dise_sequence_output)  # disease logit value
            dise_end_logits = self.disease_end_fc(dise_sequence_output)  # disease logit value

            chem_start_logits = self.chem_drug_start_fc(chem_sequence_output)  # chemical logit value
            chem_end_logits = self.chem_drug_end_fc(chem_sequence_output)  # chemical logit value

            gene_start_logits = self.gene_protein_start_fc(gene_sequence_output)  # gene/protein logit value
            gene_end_logits = self.gene_protein_end_fc(gene_sequence_output)  # gene/protein logit value

            spec_start_logits = self.spec_start_fc(spec_sequence_output)  # species logit value
            spec_end_logits = self.spec_end_fc(spec_sequence_output)  # species logit value

            cellline_start_logits = self.cell_line_start_fc(cellline_sequence_output)  # cell line logit value
            cellline_end_logits = self.cell_line_end_fc(cellline_sequence_output)  # cell line logit value

            dna_start_logits = self.dna_start_fc(dna_sequence_output)  # dna logit value
            dna_end_logits = self.dna_end_fc(dna_sequence_output)  # dna logit value

            rna_start_logits = self.rna_start_fc(rna_sequence_output)  # rna logit value
            rna_end_logits = self.rna_end_fc(rna_sequence_output)  # rna logit value

            celltype_start_logits = self.cell_type_start_fc(celltype_sequence_output)  # cell type logit value
            celltype_end_logits = self.cell_type_end_fc(celltype_sequence_output)  # cell type logit value

            sequence_output = dise_sequence_output + chem_sequence_output + gene_sequence_output + spec_sequence_output + cellline_sequence_output + dna_sequence_output + rna_sequence_output + celltype_sequence_output

            start_logits = dise_start_logits + chem_start_logits + gene_start_logits + spec_start_logits + celltype_start_logits + cellline_start_logits + dna_start_logits + rna_start_logits
            end_logits = dise_end_logits + chem_end_logits + gene_end_logits + spec_end_logits + celltype_end_logits + cellline_end_logits + dna_end_logits + rna_end_logits

        output = (start_logits,end_logits)
        if start_ids is not None and end_ids is not None:  # 这是训练模式，计算loss
            loss_mask = torch.zeros((start_logits.shape[0], start_logits.shape[1])).to(token_ids.device)

            for i, lens in enumerate(input_true_length):
                loss_mask[i][:lens] = 1
            # start_logtis.shape=torch.Size([4096, 14])

            start_logits = start_logits.view(-1, self.num_tags)
            end_logits = end_logits.view(-1, self.num_tags)

            # 去掉 padding 部分的标签，计算真实 loss
            mask = loss_mask.view(-1) == 1

            if entity_type_ids[0][0].item() == 0:
                dise_start_logits, chem_start_logits, gene_start_logits, spec_start_logits, celltype_start_logits,cellline_start_logits, dna_start_logits, rna_start_logits = start_logits
                dise_end_logits, chem_end_logits, gene_end_logits, spec_end_logits, celltype_end_logits,cellline_end_logits, dna_end_logits, rna_end_logits = end_logits

                active_start_labels = start_ids.view(-1)[mask]
                active_end_labels = end_ids.view(-1)[mask]

                active_dise_start_logits = dise_start_logits[mask]
                active_dise_end_logits = dise_end_logits[mask]

                active_chem_start_logits = chem_start_logits[mask]
                active_chem_end_logits = chem_end_logits[mask]

                active_gene_start_logits = gene_start_logits[mask]
                active_gene_end_logits = gene_end_logits[mask]

                active_spec_start_logits = spec_start_logits[mask]
                active_spec_end_logits = spec_end_logits[mask]

                active_celltype_start_logits = celltype_start_logits[mask]
                active_celltype_end_logits = celltype_end_logits[mask]

                active_cellline_start_logits = cellline_start_logits[mask]
                active_cellline_end_logits = cellline_end_logits[mask]

                active_dna_start_logits = dna_start_logits[mask]
                active_dna_end_logits = dna_end_logits[mask]

                active_rna_start_logits = rna_start_logits[mask]
                active_rna_end_logits = rna_end_logits[mask]

                dise_start_loss = self.criterion(active_dise_start_logits, active_start_labels).mean(dim=-1)
                dise_end_loss = self.criterion(active_dise_end_logits, active_end_labels).mean(dim=-1)
                dise_loss = dise_start_loss+dise_end_loss

                chem_start_loss = self.criterion(active_chem_start_logits, active_start_labels).mean(dim=-1)
                chem_end_loss = self.criterion(active_chem_end_logits, active_end_labels).mean(dim=-1)
                chem_loss = chem_start_loss+chem_end_loss

                gene_start_loss = self.criterion(active_gene_start_logits, active_start_labels).mean(dim=-1)
                gene_end_loss = self.criterion(active_gene_end_logits, active_end_labels).mean(dim=-1)
                gene_loss= gene_start_loss+gene_end_loss

                spec_start_loss = self.criterion(active_spec_start_logits, active_start_labels).mean(dim=-1)
                spec_end_loss = self.criterion(active_spec_end_logits, active_end_labels).mean(dim=-1)
                spec_loss = spec_start_loss + spec_end_loss

                celltype_start_loss = self.criterion(active_celltype_start_logits, active_start_labels).mean(dim=-1)
                celltype_end_loss = self.criterion(active_celltype_end_logits, active_end_labels).mean(dim=-1)
                celltype_loss = celltype_start_loss + celltype_end_loss

                cellline_start_loss = self.criterion(active_cellline_start_logits, active_start_labels).mean(dim=-1)
                cellline_end_loss = self.criterion(active_cellline_end_logits, active_end_labels).mean(dim=-1)
                cellline_loss = cellline_start_loss + cellline_end_loss

                dna_start_loss = self.criterion(active_dna_start_logits, active_start_labels).mean(dim=-1)
                dna_end_loss = self.criterion(active_dna_end_logits, active_end_labels).mean(dim=-1)
                dna_loss = dna_start_loss + dna_end_loss

                rna_start_loss = self.criterion(active_rna_start_logits, active_start_labels).mean(dim=-1)
                rna_end_loss = self.criterion(active_rna_end_logits, active_end_labels).mean(dim=-1)
                rna_loss = rna_start_loss + rna_end_loss

                loss = dise_loss+chem_loss+gene_loss+cellline_loss+celltype_loss+dna_loss+spec_loss+rna_loss


            else:

                active_start_logits = start_logits[mask]  # (?,14)这个？的值就并不确定了
                active_end_logits = end_logits[mask]
                active_start_labels = start_ids.view(-1)[mask]
                active_end_labels = end_ids.view(-1)[mask]

                start_loss = self.criterion(active_start_logits, active_start_labels).mean(dim=-1)
                end_loss = self.criterion(active_end_logits, active_end_labels).mean(dim=-1)

                loss = start_loss + end_loss

            return (loss,)+output

        else:
            return output


# -*- encoding: utf-8 -*-
"""
@File    :   multi_ner.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :  2022年5月19日
@Description :


"""


import os
import copy
import logging
from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn
from ipdb import set_trace
from torch.nn.utils.rnn import pad_sequence

from src.models.bert_model import BaseBert
from utils.loss_utils import LabelSmoothingCrossEntropy, FocalLoss

logger = logging.getLogger("main.multi_binary_span")

class SelfAttention(nn.Module):
    def __init__(self,input_dim,dim_k,dim_v):
        '''

        :param input_dim: 这个其实就是word_dim
        :param dim_k:  Q,K使用的是相同的dim
        :param dim_v:  V使用的是不同的dim
        '''
        super(SelfAttention,self).__init__()
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.v = nn.Linear(input_dim,dim_v)
        self._norm_fact = 1 / sqrt(dim_k)
    def mask(self,x,mask,mode='mul'):
        '''
        对计算完成的注意力进行mask
        :param mask:
        :param mode:共两种方式'mul','add', 相当于一个是0进行处理
        :return:
        '''
        if mask is None:
            return x
        else:
            #这里进行补充维数，让其和x.shape一致
            for _ in range(len(x.shape) - len(mask.shape)):
                mask = mask.unsqueeze(len(mask.shape))  # 在最后一维加上,变为[batch_size,seq_len,1,1]
            if mode == 'mul':  # mul相当于直接进行掩码，相当于对非mask的地方进行保留，其他地方去掉
                return torch.mul(x, mask)
            else:  # 'add'  这相当于将非mask的地方给变得非常小，
                return x - (1 - mask) * 1e10
    def forward(self,x,mask=None):
        '''
        这里可能也直接输入[]
        :param x:
        :param mask
        :return:output:shape = (batch_size,seq_len,dim_v)
        '''
        Q = self.q(x) #Q,K.shape = (batch_size,seq_len,dim_k)
        K = self.k(x)
        V = self.v(x) # V.shape = (batch_size,seq_len,dim_v)
        #首先计算Q*K的值
        K = K.permute(0,2,1)  #K.shape = (batch_size,dim_k,seq_len)

        res = torch.bmm(Q,K) #shape = (batch_size,seq_len,seq_len)
        res = self.mask(res,mask)
        #开始mask

        attn = nn.Softmax(dim=-1)(res)*self._norm_fact

        #计算最后的输出
        output = torch.matmul(attn,V) #shape=(batch_size,seq_len,dim_v)
        return output

class MultiBinaryBiLSTMSAInterSpanForEight(BaseBert):
    def __init__(self,config):
        super(MultiBinaryBiLSTMSAInterSpanForEight, self).__init__(config)
        # 这个时候numtags=2，因为只有disease一种类别
        self.config = config
        self.num_tags = 2
        out_dims = self.bert_config.hidden_size
        mid_linear_dims = 128

        # 准备的实体类别有:DNA,RNA,Gene/Protein,Disease,Chemical/Durg,cell_type,cell_line,species
        self.chem_drug_mid_linear = nn.LSTM(out_dims, mid_linear_dims, batch_first=True, bidirectional=True, num_layers=2,dropout=0.5)
        self.gene_protein_mid_linear = nn.LSTM(out_dims, mid_linear_dims, batch_first=True, bidirectional=True, num_layers=2,dropout=0.5)
        self.disease_mid_linear = nn.LSTM(out_dims, mid_linear_dims, batch_first=True, bidirectional=True, num_layers=2,dropout=0.5)
        self.cell_line_mid_linear = nn.LSTM(out_dims, mid_linear_dims, batch_first=True, bidirectional=True, num_layers=2,dropout=0.5)
        self.cell_type_mid_linear = nn.LSTM(out_dims, mid_linear_dims, batch_first=True, bidirectional=True, num_layers=2,dropout=0.5)
        self.dna_mid_linear = nn.LSTM(out_dims, mid_linear_dims, batch_first=True, bidirectional=True, num_layers=2,dropout=0.5)
        self.rna_mid_linear = nn.LSTM(out_dims, mid_linear_dims, batch_first=True, bidirectional=True, num_layers=2,dropout=0.5)
        self.species_mid_linear = nn.LSTM(out_dims, mid_linear_dims, batch_first=True, bidirectional=True, num_layers=2,dropout=0.5)

        self.dropout = nn.Dropout(0.5)
        out_dims = 128*2

        self.chem_inter_linear = nn.Linear(self.num_tags, out_dims)
        self.gene_protein_inter_linear = nn.Linear(self.num_tags, out_dims)
        self.disease_inter_linear = nn.Linear(self.num_tags, out_dims)
        self.cell_line_inter_linear = nn.Linear(self.num_tags, out_dims)
        self.cell_type_inter_linear = nn.Linear(self.num_tags, out_dims)
        self.dna_inter_linear = nn.Linear(self.num_tags, out_dims)
        self.rna_inter_linear = nn.Linear(self.num_tags, out_dims)
        self.spec_inter_linear = nn.Linear(self.num_tags, out_dims)


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
                self.chem_drug_mid_linear, self.chem_drug_start_fc, self.chem_drug_end_fc,self.chem_inter_linear,
                self.gene_protein_mid_linear, self.gene_protein_start_fc, self.gene_protein_end_fc,self.gene_protein_inter_linear,
                self.disease_mid_linear, self.disease_start_fc, self.disease_end_fc,self.disease_inter_linear,
                self.cell_line_mid_linear, self.cell_line_start_fc, self.cell_line_end_fc,self.cell_line_inter_linear,
                self.cell_type_mid_linear, self.cell_type_start_fc, self.cell_type_end_fc,self.cell_type_inter_linear,
                self.dna_mid_linear, self.dna_start_fc, self.dna_end_fc,self.dna_inter_linear,
                self.rna_mid_linear, self.rna_start_fc, self.rna_end_fc,self.rna_inter_linear,
                self.species_mid_linear, self.spec_start_fc, self.spec_end_fc,self.spec_inter_linear,

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
        start_ids = (start_ids != 0).long()
        end_ids = (end_ids != 0).long()

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
            dise_sequence_output = F.leaky_relu(self.disease_mid_linear(sequence_output)[0]) # disease logit value
            chem_sequence_output = F.leaky_relu(self.chem_drug_mid_linear(sequence_output)[0]) # chemical logit value
            gene_sequence_output = F.leaky_relu(self.gene_protein_mid_linear(sequence_output)[0]) # gene/protein logit value
            spec_sequence_output = F.leaky_relu(self.species_mid_linear(sequence_output)[0]) # species logit value
            cellline_sequence_output = F.leaky_relu(self.cell_line_mid_linear(sequence_output)[0]) # cell line logit value
            dna_sequence_output = F.leaky_relu(self.dna_mid_linear(sequence_output)[0]) # dna logit value
            rna_sequence_output = F.leaky_relu(self.rna_mid_linear(sequence_output)[0]) # rna logit value
            celltype_sequence_output = F.leaky_relu(self.cell_type_mid_linear(sequence_output)[0]) # cell type logit value

            dise_sequence_output = self.dropout(dise_sequence_output)
            chem_sequence_output = self.dropout(chem_sequence_output)
            gene_sequence_output = self.dropout(gene_sequence_output)
            spec_sequence_output = self.dropout(spec_sequence_output)
            cellline_sequence_output = self.dropout(cellline_sequence_output)
            dna_sequence_output = self.dropout(dna_sequence_output)
            rna_sequence_output = self.dropout(rna_sequence_output)
            celltype_sequence_output = self.dropout(celltype_sequence_output)


            dise_start_logits = self.disease_start_fc(dise_sequence_output) # disease logit value
            disease_inter_logits = self.disease_inter_linear(dise_start_logits)
            dise_sequence_output = (dise_sequence_output + disease_inter_logits) / 2
            dise_end_logits = self.disease_end_fc(dise_sequence_output) # disease logit value

            chem_start_logits = self.chem_drug_start_fc(chem_sequence_output) # chemical logit value
            chem_inter_logits = self.chem_inter_linear(chem_start_logits)
            chem_sequence_output = (chem_sequence_output + chem_inter_logits) / 2
            chem_end_logits = self.chem_drug_end_fc(chem_sequence_output) # chemical logit value

            gene_start_logits = self.gene_protein_start_fc(gene_sequence_output) # gene/protein logit value
            gene_inter_logits = self.gene_protein_inter_linear(gene_start_logits)
            gene_sequence_output = (gene_sequence_output + gene_inter_logits) / 2
            gene_end_logits = self.gene_protein_end_fc(gene_sequence_output) # gene/protein logit value

            spec_start_logits = self.spec_start_fc(spec_sequence_output) # species logit value
            spec_inter_logits = self.spec_inter_linear(spec_start_logits)
            spec_sequence_output = (spec_sequence_output + spec_inter_logits) / 2
            spec_end_logits = self.spec_end_fc(spec_sequence_output) # species logit value


            cellline_start_logits = self.cell_line_start_fc(cellline_sequence_output)  # cell line logit value
            cellline_inter_logits = self.cell_line_inter_linear(cellline_start_logits)
            cellline_sequence_output = (cellline_sequence_output + cellline_inter_logits) / 2
            cellline_end_logits = self.cell_line_end_fc(cellline_sequence_output)  # cell line logit value

            dna_start_logits = self.dna_start_fc(dna_sequence_output) # dna logit value
            dna_inter_logits = self.dna_inter_linear(dna_start_logits)
            dna_sequence_output = (dna_sequence_output + dna_inter_logits) / 2
            dna_end_logits = self.dna_end_fc(dna_sequence_output) # dna logit value

            rna_start_logits = self.rna_start_fc(rna_sequence_output) # rna logit value
            rna_inter_logits = self.rna_inter_linear(rna_start_logits)
            rna_sequence_output = (rna_sequence_output + rna_inter_logits) / 2
            rna_end_logits = self.rna_end_fc(rna_sequence_output) # rna logit value


            celltype_start_logits = self.cell_type_start_fc(celltype_sequence_output) # cell type logit value
            celltype_inter_logits = self.cell_type_inter_linear(celltype_start_logits)
            celltype_sequence_output = (celltype_sequence_output + celltype_inter_logits) / 2
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

            # F.tanh or F.leaky_relu

            dise_sequence_output = F.leaky_relu(self.disease_mid_linear(dise_sequence_output)[0])  # disease logit value
            chem_sequence_output = F.leaky_relu(self.chem_drug_mid_linear(chem_sequence_output)[0])  # chemical logit value
            gene_sequence_output = F.leaky_relu(self.gene_protein_mid_linear(gene_sequence_output)[0])  # gene/protein logit value
            spec_sequence_output = F.leaky_relu(self.species_mid_linear(spec_sequence_output)[0])  # species logit value
            cellline_sequence_output = F.leaky_relu(self.cell_line_mid_linear(cellline_sequence_output)[0])  # cell line logit value
            dna_sequence_output = F.leaky_relu(self.dna_mid_linear(dna_sequence_output)[0])  # dna logit value
            rna_sequence_output = F.leaky_relu(self.rna_mid_linear(rna_sequence_output)[0])  # rna logit value
            celltype_sequence_output = F.leaky_relu(self.cell_type_mid_linear(celltype_sequence_output)[0])  # cell type logit value

            dise_sequence_output = self.dropout(dise_sequence_output)
            chem_sequence_output = self.dropout(chem_sequence_output)
            gene_sequence_output = self.dropout(gene_sequence_output)
            spec_sequence_output = self.dropout(spec_sequence_output)
            cellline_sequence_output = self.dropout(cellline_sequence_output)
            dna_sequence_output = self.dropout(dna_sequence_output)
            rna_sequence_output = self.dropout(rna_sequence_output)
            celltype_sequence_output = self.dropout(celltype_sequence_output)

            dise_start_logits = self.disease_start_fc(dise_sequence_output)  # disease logit value
            disease_inter_logits = self.disease_inter_linear(dise_start_logits)
            dise_sequence_output = (dise_sequence_output + disease_inter_logits) / 2
            dise_end_logits = self.disease_end_fc(dise_sequence_output)  # disease logit value

            chem_start_logits = self.chem_drug_start_fc(chem_sequence_output)  # chemical logit value
            chem_inter_logits = self.chem_inter_linear(chem_start_logits)
            chem_sequence_output = (chem_sequence_output + chem_inter_logits) / 2
            chem_end_logits = self.chem_drug_end_fc(chem_sequence_output)  # chemical logit value

            gene_start_logits = self.gene_protein_start_fc(gene_sequence_output)  # gene/protein logit value
            gene_inter_logits = self.gene_protein_inter_linear(gene_start_logits)
            gene_sequence_output = (gene_sequence_output + gene_inter_logits) / 2
            gene_end_logits = self.gene_protein_end_fc(gene_sequence_output)  # gene/protein logit value

            spec_start_logits = self.spec_start_fc(spec_sequence_output)  # species logit value
            spec_inter_logits = self.spec_inter_linear(spec_start_logits)
            spec_sequence_output = (spec_sequence_output + spec_inter_logits) / 2
            spec_end_logits = self.spec_end_fc(spec_sequence_output)  # species logit value

            cellline_start_logits = self.cell_line_start_fc(cellline_sequence_output)  # cell line logit value
            cell_line_inter_logits = self.cell_line_inter_linear(cellline_start_logits)
            cellline_sequence_output = (cellline_sequence_output + cell_line_inter_logits) / 2
            cellline_end_logits = self.cell_line_end_fc(cellline_sequence_output)  # cell line logit value

            dna_start_logits = self.dna_start_fc(dna_sequence_output)  # dna logit value
            dna_inter_logits = self.dna_inter_linear(dna_start_logits)
            dna_sequence_output = (dna_sequence_output + dna_inter_logits) / 2
            dna_end_logits = self.dna_end_fc(dna_sequence_output)  # dna logit value

            rna_start_logits = self.rna_start_fc(rna_sequence_output)  # rna logit value
            rna_inter_logits = self.rna_inter_linear(rna_start_logits)
            rna_sequence_output = (rna_sequence_output + rna_inter_logits) / 2
            rna_end_logits = self.rna_end_fc(rna_sequence_output)  # rna logit value

            celltype_start_logits = self.cell_type_start_fc(celltype_sequence_output)  # cell type logit value
            celltype_inter_logits = self.cell_type_inter_linear(celltype_start_logits)
            celltype_sequence_output = (celltype_sequence_output + celltype_inter_logits) / 2
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

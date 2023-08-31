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
import torch
from ipdb import set_trace

from config import BertConfig
from src.models.bert_model import BaseBert
import torch.nn as nn
import numpy as np



class MulBERT(BaseBert):
    def __init__(self, config:BertConfig):
        super(MulBERT, self).__init__(config)

        self.num_labels = config.num_labels
        self.config = config
        # 下面这两个dim可以进行修改
        bert_hidden_size = self.bert_config.hidden_size
        self.test1_entity = nn.Linear(bert_hidden_size, bert_hidden_size * config.num_labels)
        self.test2_entity = nn.Linear(bert_hidden_size, bert_hidden_size * config.num_labels)

        nn.init.xavier_normal_(self.test1_entity.weight)
        nn.init.constant_(self.test1_entity.bias, 0.)
        nn.init.xavier_normal_(self.test2_entity.weight)
        nn.init.constant_(self.test2_entity.bias, 0.)



    def forward(self, input_ids, token_type_ids,attention_masks,labels, e1_mask, e2_mask):

        bert_outputs = self.bert_model(
            input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = bert_outputs[0] # shape=(batch_size,seq_len,hidden_size)

        pooled_output = bert_outputs[1]  # [CLS],shape = (batch_size,hidden_size)=(16,768)

        entity1_pool = []
        entity2_pool = []
        batch_size = sequence_output.shape[0]
        for i in range(batch_size):
            # 这是除去##,$$的实际长度...

            ent1_start_index = e1_mask[i].tolist().index(1)+1
            ent1_len = int(torch.sum(e1_mask[i]).cpu()-2)
            ent1_end_index = ent1_start_index+ent1_len-1
            ent2_start_index = e2_mask[i].tolist().index(1) + 1
            ent2_len = int(torch.sum(e2_mask[i]) - 2)
            ent2_end_index = ent2_start_index + ent2_len - 1

            entity1 = torch.mean(sequence_output[i,ent1_start_index:ent1_end_index+1,:],dim=0,keepdim=True)
            entity1_pool.append(entity1)

            entity2 = torch.mean(sequence_output[i, ent2_start_index:ent2_end_index+1, :], dim=0, keepdim=True)
            entity2_pool.append(entity2)
        # 这里并不是pooled_output的结果，很神奇...
        # 这里可以尝试修改一下

        H_clr = sequence_output[:,0]
        entity1_pool = torch.cat(entity1_pool,0)
        entity2_pool = torch.cat(entity2_pool,0)

        test1 = H_clr+entity1_pool
        test2 = H_clr+entity2_pool

        test1 = test1.unsqueeze(1)  # [32, 1, 1024]
        test2 = test2.unsqueeze(1)  # [32, 1, 1024]

        test1 = self.test1_entity(test1)  # bs, 1, 768x19  F.dropout(torch.tanh(test1), 0.1)
        test2 = self.test2_entity(test2)  # bs, 1, 768x19  F.dropout(torch.tanh(test2), 0.1)
        test1 = test1.reshape(H_clr.shape[0], -1, test1.shape[-2],H_clr.shape[-1])  # (batch_size, num_labels = 19, 1, hidden_size=768)
        test2 = test2.reshape(H_clr.shape[0], -1, test1.shape[-2], H_clr.shape[-1])  # [32, 19, 1, 1024]
        attn_score = torch.matmul(test1, test2.permute(0, 1, 3, 2))  # torch.Size([32, 19, 1, 1])
        score = attn_score / np.sqrt(H_clr.shape[-1])  # np.sqrt(768) #score.shape
        score = score.squeeze(-1)
        logits = score.squeeze(-1)
        # Softmax

        if labels is not None:

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            return loss,logits

        return logits  # (loss), logits, (hidden_states), (attentions)



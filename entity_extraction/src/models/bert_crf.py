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
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from config import MyBertConfig
from src.dataset_util.base_dataset import sequence_padding
from src.models.bert_model import BaseBert

from ipdb import set_trace



from torchcrf import CRF

import logging

from src.ner_predicate import crf_predicate, vote

logger = logging.getLogger('main.bert_crf')

class EnsembleBertCRF:
    def __init__(self, trained_model_path_list, config, device,bert_name_list = None):
        '''
        可以混合不同的Bert模型来作为Encoder...
        bert_name_list:记录每个模型所使用的bert_name,可以混合不同的模型...,但是decoder保持相同....
        '''
        self.config = config

        self.models = []

        for idx, _path in enumerate(trained_model_path_list):
            logger.info('从{}中加载已训练的模型'.format(_path))
            if bert_name_list:
                config.bert_name = bert_name_list[idx]
            model = BertCRF(config)

            model.load_state_dict(torch.load(_path, map_location=torch.device('cpu')))

            model.to(device)
            model.eval()
            self.models.append(model)
    def vote_entities(self,batch_data,device,threshold):
        '''
            集成方法：投票法
            每个模型都会预测得到一系列的实体，然后进行投票选择...

            这个非常简单，就是统计所有的实体的出现个数
        '''
        labels = None
        if self.config.predicate_flag: # 面向predicate 无label的text
            raw_text_list, batch_subword_input_ids, batch_subword_token_type_ids, batch_subword_attention_masks, origin_to_subword_indexs, batch_label_mask = batch_data
        else:# 面向验证集....
            raw_text_list, labels, batch_subword_input_ids, batch_subword_token_type_ids, batch_subword_attention_masks, origin_to_subword_indexs, batch_label_mask = batch_data
            labels = labels.to(device)
        token_ids, attention_masks, token_type_ids = batch_subword_input_ids.to(device), batch_subword_attention_masks.to(device), batch_subword_token_type_ids.to(device)

        entities_ls = []
        for idx, model in enumerate(self.models):

            _,tmp_predicate = model(token_ids, attention_masks=attention_masks,
                                  token_type_ids=token_type_ids, labels=labels,
                                  input_token_starts=origin_to_subword_indexs)

            decode_entities = crf_predicate(tmp_predicate, self.config.crf_id2label, raw_text_list)
            entities_ls.append(decode_entities)

        return vote(entities_ls, threshold)


    def predicate(self,batch_data,device):
        '''
        集成方法：模型融合的方法

        '''
        labels = None
        if self.config.predicate_flag:
            raw_text_list, batch_subword_input_ids, batch_subword_token_type_ids, batch_subword_attention_masks, origin_to_subword_indexs, batch_label_mask = batch_data
        else:
            raw_text_list, labels, batch_subword_input_ids, batch_subword_token_type_ids, batch_subword_attention_masks, origin_to_subword_indexs, batch_label_mask = batch_data
            labels = labels.to(device)
        token_ids, attention_masks, token_type_ids = batch_subword_input_ids.to(device), batch_subword_attention_masks.to(device), batch_subword_token_type_ids.to(device)

        logits = None

        for idx, model in enumerate(self.models):

            # 使用概率平均  融合
            weight = 1 / len(self.models)
            self.config.predicate_flag = True # 这里是为了融合logits，所以强行为True
            tmp_predicate = model(token_ids, attention_masks=attention_masks,
                                      token_type_ids=token_type_ids, labels=labels,
                                      input_token_starts=origin_to_subword_indexs)


            tmp_predicate = tmp_predicate * weight
            if logits is None:
                logits = tmp_predicate

            else:
                logits += tmp_predicate
        self.config.predicate_flag = False

        return logits



class BertCRF(BaseBert):
    def __init__(self,config:MyBertConfig):
        super(BertCRF, self).__init__(config=config)
        # 获取Bert模型

        # bert的输出层dim
        self.config = config
        out_dims = self.bert_config.hidden_size


        self.dropout = nn.Dropout(config.dropout_prob)
        self.classifier = nn.Linear(out_dims,config.num_crf_class)
        self.num_crf_class = config.num_crf_class
        # 最后是crf
        mid_linear_dims = 128
        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob),

        )
        self.classifier = nn.Linear(mid_linear_dims, config.num_crf_class)


        self.crf_model = CRF(num_tags=config.num_crf_class,batch_first=True)


        init_blocks = [self.classifier, self.mid_linear]
        self._init_weights(init_blocks)



    def doit(self,token_ids,attention_masks,token_type_ids,labels):
        '''
        这个是token-level的前向传播计算，在中文上比较好，英文由于分词需要进行其他操作
        Returns:
           如果是train或者dev，返回的是loss
           如果是test，则返回预测的标签
        '''
        bert_outputs = self.bert_model(input_ids=token_ids,attention_mask=attention_masks,token_type_ids=token_type_ids)

        seq_out = bert_outputs[0]
        seq_out = self.mid_linear(seq_out)
        emissions = self.classifier(seq_out)


        if labels is not None:

            loss = -1.*self.crf_model(emissions=emissions,tags=labels.long(),mask=attention_masks.byte(),reduction='mean')
            tokens_out = self.crf_model.decode(emissions=emissions, mask=attention_masks.byte())

            return loss,tokens_out
        else:
            tokens_out = self.crf_model.decode(emissions=emissions,mask = attention_masks.byte())
            return tokens_out

    def forward(self, input_ids, token_type_ids=None, attention_masks=None, labels=None,input_token_starts=None):

        '''
            这是tokenizer之后的subword计算
            这里都是动态batch
            input_data.shape=input_token_starts.shape=(batch_size,seq_len)
            labels：是未分词token对应的mask，seq_len和上面的seq_len并不一致
            input_token_starts：这个长度和labels对应

        '''
        if self.config.bert_name in ['scibert','biobert','flash','bert','flash_quad','wwm_bert']:

            bert_outputs = self.bert_model(input_ids=input_ids,attention_mask=attention_masks,token_type_ids=token_type_ids.long())
            sequence_output = bert_outputs[0]  # shape=(batch_size,seq_len,hidden_dim)=[32, 55, 768]
        elif self.config.bert_name == 'kebiolm':
            bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_masks,
                                           token_type_ids=token_type_ids, return_dict=False)
            sequence_output = bert_outputs[2]  # shape=(batch_size,seq_len,hidden_dim)=[32, 55, 768]
        else:
            raise ValueError
        # print("sequence_output", sequence_output.shape)

        # 将subwords的第一个subword作为之前的token representation
        origin_sequence_output = []

        for layer, starts in zip(sequence_output, input_token_starts):
            res = layer[starts]
            origin_sequence_output.append(res)


        # 这里的max_len和上面的seq_len已经不一样了，因为这里是按照token-level,而不是subword-level
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        res = F.relu(self.mid_linear(padded_sequence_output))
        #res = self.dropout(padded_sequence_output)
        emissions = self.classifier(res)
        if self.config.predicate_flag:
            return emissions

        if labels is not None:
            loss_mask = labels.gt(-1)
            loss = -1. * self.crf_model(emissions=emissions, tags=labels.long(), mask=loss_mask.byte(),reduction='mean')
            tokens_out = self.crf_model.decode(emissions=emissions, mask=loss_mask.byte())
            tokens_out = torch.tensor(sequence_padding(tokens_out,length=self.config.max_len+2)).long().to(labels.device)

            return loss,tokens_out
        else:
            loss_mask = torch.ones(1,len(input_token_starts[0]))
            loss_mask = loss_mask.to(device=emissions.device)
            tokens_out = self.crf_model.decode(emissions=emissions, mask=loss_mask.byte())
            return tokens_out



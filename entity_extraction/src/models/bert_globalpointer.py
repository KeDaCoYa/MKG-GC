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

import logging
from ipdb import set_trace

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn



from config import MyBertConfig
from src.decoder import GlobalPointer
from src.models.bert_model import BaseBert
from src.ner_predicate import bert_globalpointer_predicate, vote
from utils.loss_utils import multilabel_categorical_crossentropy

logger = logging.getLogger('main.bert_globalpointer')

class EnsembleBertGlobalPointer:
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
            model = BertGlobalPointer(config)

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
        batch_true_labels = None
        if self.config.predicate_flag: # 面向predicate 无label的text
            raw_text_list, batch_subword_input_ids, batch_subword_token_type_ids, batch_subword_attention_masks, origin_to_subword_indexs = batch_data
        else:# 面向验证集....
            raw_text_list, batch_true_labels, batch_subword_input_ids, batch_subword_token_type_ids, batch_subword_attention_masks, origin_to_subword_indexs, batch_label_mask = batch_data
            batch_true_labels = batch_true_labels.to(device)
        token_ids, attention_masks, token_type_ids = batch_subword_input_ids.to(device), batch_subword_attention_masks.to(device), batch_subword_token_type_ids.to(device)

        entities_ls = []
        for idx, model in enumerate(self.models):

            _, globalpointer_predicate = model(token_ids, attention_masks=attention_masks,token_type_ids=token_type_ids,labels=batch_true_labels,input_token_starts=origin_to_subword_indexs)
            globalpointer_predicate = globalpointer_predicate.cpu().numpy()

            decode_entities = bert_globalpointer_predicate(globalpointer_predicate, self.config.globalpointer_id2label, raw_text_list)
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
            _,globalpointer_predicate  = model(token_ids, attention_masks=attention_masks,
                                      token_type_ids=token_type_ids, labels=labels,
                                      input_token_starts=origin_to_subword_indexs)


            tmp_predicate = globalpointer_predicate * weight
            if logits is None:
                logits = tmp_predicate

            else:
                logits += tmp_predicate
        self.config.predicate_flag = False

        return logits



class BertGlobalPointer(BaseBert):
    def __init__(self,config:MyBertConfig,inner_dim=64,use_RoPE=True):
        '''

        :param config:
        :param num_tags: 实体类别个数，CNeEE为9种类别
        :param inner_dim: 超参数，模型的一个dim
        :param use_RoPE:使用RoPE位置编码
        '''
        super(BertGlobalPointer,self).__init__(config)


        self.config = config
        self.num_tags = config.num_gp_class
        self.inner_dim = inner_dim
        self.hidden_size = self.bert_config.hidden_size
        self.use_RoPE = use_RoPE


        self.dropout = nn.Dropout(config.dropout_prob)
        # 最后的一个全连接层
        self.globalpointer = GlobalPointer(config, hidden_size=self.bert_config.hidden_size,use_RoPE=True)



        self.criterion = multilabel_categorical_crossentropy


    def token_forward(self, input_ids, attention_masks, token_type_ids,labels):
        '''

        :param input_ids:

        param:attention_mask:有bert_tokenizer得到的mask，shape=(batch_size,seq_len)

        :param token_type_ids:
        :return: 无论是训练还是验证，都是返回两个值(loss,logtis)
        '''

        self.device = input_ids.device

        context_outputs = self.bert_model(input_ids, attention_masks, token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size) = torch.Size([16, 128, 1024])
        last_hidden_state = context_outputs[0]

        encoder_output = self.mid_linear(last_hidden_state)

        logits = self.globalpointer(encoder_output=encoder_output, token_ids=input_ids,attention_mask=attention_masks)


        loss = self.criterion(logits, labels)
        return loss,logits



    def forward(self,input_ids,attention_masks,token_type_ids,labels,input_token_starts,input_true_length=None):
        '''
        这是tokenizer之后的subword计算
        这里都是动态batch
        input_data.shape=input_token_starts.shape=(batch_size,seq_len)
        labels：是未分词token对应的mask，seq_len和上面的seq_len并不一致
        input_token_starts：这个长度和labels对应
        :param input_ids:
        :param attention_masks:
        :param token_type_ids:
        :param labels:
        :param input_token_starts:
        :param true_input_lengths: 这是input数据的真实长度
        :return:
        '''
        if self.config.bert_name in ['scibert','biobert','flash','bert','flash_quad','wwm_bert']:
            bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_masks,token_type_ids=token_type_ids.long())
            sequence_output = bert_outputs[0]  # shape=(batch_size,seq_len,hidden_dim)=[32, 55, 768]
        elif self.config.bert_name == 'kebiolm':
            bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_masks,
                                           token_type_ids=token_type_ids, return_dict=False)
            sequence_output = bert_outputs[2]  # shape=(batch_size,seq_len,hidden_dim)=[32, 55, 768]
        else:
            raise ValueError

        # 将subwords的第一个subword作为之前的token representation
        # input_token_starts为相应坐标
        origin_sequence_output = []
        for layer, starts in zip(sequence_output, input_token_starts):
            res = layer[starts]
            origin_sequence_output.append(res)

        # 这里的max_len和上面的seq_len已经不一样了，因为这里是按照token-level,而不是subword-level
        pad_output = pad_sequence(origin_sequence_output, batch_first=True)
        encoder_output = self.dropout(pad_output)

        loss_mask = torch.zeros((encoder_output.shape[0],encoder_output.shape[1])).to(input_ids.device)
        for i, lens in enumerate(input_true_length):
            loss_mask[i][:lens] = 1

        logits = self.globalpointer(encoder_output=encoder_output, token_ids=input_ids, attention_mask=loss_mask)
        if labels is not None:
            loss = self.criterion(logits, labels)
            return loss, logits
        else:
            return logits


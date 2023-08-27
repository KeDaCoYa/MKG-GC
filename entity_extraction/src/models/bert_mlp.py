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



import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence


from ipdb import set_trace

import logging
from src.ner_predicate import crf_predicate, vote
from config import MyBertConfig
from src.models.bert_model import BaseBert


logger = logging.getLogger('main.bert_mlp')


class EnsembleBertMLP:
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
            model = BertMLP(config)

            model.load_state_dict(torch.load(_path, map_location=torch.device('cpu')))
            model.eval()
            model.to(device)
            self.models.append(model)
    def vote_entities(self,batch_data,device,threshold):
        """
            集成方法：投票法
            每个模型都会预测得到一系列的实体，然后进行投票选择...

            这个非常简单，就是统计所有的实体的出现个数
        """
        labels = None
        if self.config.predicate_flag:  # 面向predicate 无label的text
            raw_text_list, batch_subword_input_ids, batch_subword_token_type_ids, batch_subword_attention_masks, origin_to_subword_indexs, batch_label_mask = batch_data
        else:  # 面向验证集....
            raw_text_list, labels, batch_subword_input_ids, batch_subword_token_type_ids, batch_subword_attention_masks, origin_to_subword_indexs, batch_label_mask = batch_data
            labels = labels.to(device)

        token_ids, attention_masks, token_type_ids = batch_subword_input_ids.to(
            device), batch_subword_attention_masks.to(device), batch_subword_token_type_ids.to(device)

        entities_ls = []
        for idx, model in enumerate(self.models):
            _, tmp_predicate = model(token_ids, attention_masks=attention_masks,
                                     token_type_ids=token_type_ids, labels=labels,
                                     input_token_starts=origin_to_subword_indexs)
            # 这里mlp，结果会包含[CLS],[SEP]

            decode_entities = crf_predicate(tmp_predicate, self.config.crf_id2label, raw_text_list)
            entities_ls.append(decode_entities)

        return vote(entities_ls, threshold)


    def predicate(self,batch_data,device):
        '''
        集成方法：模型融合的方法

        '''
        labels = None
        if self.config.predicate_flag:  # 面向predicate 无label的text
            raw_text_list, batch_subword_input_ids, batch_subword_token_type_ids, batch_subword_attention_masks, origin_to_subword_indexs, batch_label_mask = batch_data
        else:  # 面向验证集....
            raw_text_list, labels, batch_subword_input_ids, batch_subword_token_type_ids, batch_subword_attention_masks, origin_to_subword_indexs, batch_label_mask = batch_data
            labels = labels.to(device)

        token_ids, attention_masks, token_type_ids = batch_subword_input_ids.to(device), batch_subword_attention_masks.to(device), batch_subword_token_type_ids.to(device)

        logits = None

        for idx, model in enumerate(self.models):

            # 使用概率平均  融合
            weight = 1 / len(self.models)
            self.config.predicate_flag = True  # 这里是为了融合logits，所以强行为True
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

class BertMLP(BaseBert):
    def __init__(self, config: MyBertConfig):
        super(BertMLP, self).__init__(config=config)
        # 获取Bert模型

        # bert的输出层dim
        self.config = config
        self.num_crf_class = config.num_crf_class
        out_dims = self.bert_config.hidden_size

        self.dropout = nn.Dropout(p=config.dropout_prob)

        mid_linear_dims = 128
        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob),

        )
        self.classifier = nn.Linear(mid_linear_dims, config.num_crf_class)
        init_blocks = [self.classifier,self.mid_linear]
        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)



    def forward(self, input_ids, token_type_ids=None, attention_masks=None, labels=None, input_token_starts=None):

        '''
            这里都是动态batch
            input_data.shape=input_token_starts.shape=(batch_size,seq_len)
            labels：是真实token对应的mask，seq_len和上面的seq_len并不一致
        '''

        if self.config.bert_name in ['scibert','biobert','flash','bert','flash_quad','wwm_bert']:
            bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_masks,
                                       token_type_ids=token_type_ids.long())
            sequence_output = bert_outputs[0]  # shape=(batch_size,seq_len,hidden_dim)=[32, 55, 768]
        elif self.config.bert_name == 'kebiolm':
            bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids,return_dict=False)
            sequence_output = bert_outputs[2]  # shape=(batch_size,seq_len,hidden_dim)=[32, 55, 768]
        else:
            raise ValueError


        # obtain original token representations from sub_words representations (by selecting the first sub_word)
        origin_sequence_output = []
        # 这里筛选得到完成的结果...
        for layer, starts in zip(sequence_output, input_token_starts):
            # 这相当于获得每一个batch的
            # layer.shape=(seq_len,hidden_dim)
            # starts.shape=(seq_len)
            # starts.nonzero()就是获得非零位置的indices
            # 这个res相当于变成之前的sentences[i][-1]的数组

            res = layer[starts]

            origin_sequence_output.append(res)


        # origin_sequence_output = [layer[starts.nonzero().squeeze(1)] for layer, starts in zip(sequence_output, input_token_starts)]
        # 这里的max_len和上面的seq_len已经不一样了，因为这里是按照token-level,而不是subword-level
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)

        res = F.relu(self.mid_linear(padded_sequence_output))
        # res = self.dropout(padded_sequence_output)
        emissions = self.classifier(res)
        # loss_mask = torch.zeros((emissions.shape[0],emissions.shape[1])).to(input_ids.device)
        # for i,lens in enumerate(true_length):
        #     loss_mask[i][:lens] = 1

        if self.config.predicate_flag == True: #这个只在predicate用到的临时参数
            return emissions


        if labels is not None:
            loss_mask = labels.gt(-1)
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='mean')
            # Only keep active parts of the loss

            active_logits = emissions[loss_mask].view(-1, self.num_crf_class)
            active_labels = labels[loss_mask].view(-1)
            loss = loss_fct(active_logits, active_labels)



            return loss, emissions
        else:


            return emissions

    def weight_forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,input_token_starts=None):

        '''
            这是tokenizer之后的subword计算
            这里和forward不同的是不再是选择token的subwords的第一个subword作为token representation，而是进行一个加权平均

            attention_mask = input_data.shape=input_token_starts.shape=(batch_size,seq_len)
            labels：是真实token对应的mask，seq_len和上面的seq_len并不一致

        '''
        # 这三个参数只对bert model有用
        bert_outputs = self.bert_model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids.long())
        sequence_output = bert_outputs[0]  # shape=(batch_size,seq_len,hidden_dim)=[32, 55, 768]
        # print("sequence_output", sequence_output.shape)

        # 将subwords的第一个subword作为之前的token representation
        origin_sequence_output = []
        index = 0
        for layer, starts in zip(sequence_output, input_token_starts):
            subwords_len = int(torch.sum(attention_mask[index]).cpu().numpy())
            res = torch.tensor([],device=input_ids.device)
            for i,start in enumerate(starts):

                if i+1 < len(starts):
                    span_representation = layer[starts[i]:starts[i+1]]
                else:
                    span_representation = layer[starts[i]:subwords_len]

                avg_representation = torch.mean(span_representation,dim=0).reshape(1,-1)

                res = torch.cat((res,avg_representation),dim=0)
            index += 1
            origin_sequence_output.append(res)


        # 这里的max_len和上面的seq_len已经不一样了，因为这里是按照token-level,而不是subword-level
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)


        res = self.mid_linear(padded_sequence_output)
        emissions = self.classifier(res)
        # loss_mask = torch.zeros((emissions.shape[0],emissions.shape[1])).to(input_ids.device)
        # for i,lens in enumerate(true_length):
        #     loss_mask[i][:lens] = 1

        loss_mask = labels.gt(-1)

        if labels is not None:

            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='mean')
            # Only keep active parts of the loss

            active_logits = emissions[loss_mask].view(-1, self.num_crf_class)
            active_labels = labels[loss_mask].view(-1)
            loss = loss_fct(active_logits, active_labels)

            output = np.argmax(emissions.detach().cpu().numpy(), axis=2)
            output_token = []
            for i,j in enumerate(output):
                output_token.append(j[:len(input_token_starts[i])])
            return loss, output_token
        else:
            output = np.argmax(emissions.detach().cpu().numpy(), axis=2)
            output_token = []
            for i, j in enumerate(output):
                output_token.append(j[:len(input_token_starts[i])])
            return output_token


class BertTest(BaseBert):
    def __init__(self, config: MyBertConfig):
        super(BertTest, self).__init__(config=config)
        # 获取Bert模型

        # bert的输出层dim
        self.config = config
        self.num_crf_class = config.num_crf_class
        out_dims = self.bert_config.hidden_size

        self.dropout = nn.Dropout(p=config.dropout_prob)
        self.classifier = nn.Linear(out_dims, config.num_crf_class)

        init_blocks = [self.classifier]
        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)



    def forward(self, input_ids=None, token_type_ids=None, attention_masks=None, labels=None):

        '''
            这里都是动态batch
            input_data.shape=input_token_starts.shape=(batch_size,seq_len)
            labels：是真实token对应的mask，seq_len和上面的seq_len并不一致
        '''
        if self.config.bert_name in ['biobert','flash_quad','wwm_bert']:

            bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_masks,token_type_ids=token_type_ids.long())

            sequence_output = bert_outputs[0]  # shape=(batch_size,seq_len,hidden_dim)=[32, 55, 768]
        elif self.config.bert_name == 'kebiolm':
            bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids,return_dict=False)
            sequence_output = bert_outputs[2]  # shape=(batch_size,seq_len,hidden_dim)=[32, 55, 768]
        else:
            raise ValueError

        res = self.dropout(sequence_output)
        logits = self.classifier(res)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
            # 只计算正常部分的loss，连[cls]和[sep]都不弄,因为[CLS]和[SEP]部分已经被设置为ignore_index

            active_loss = attention_masks.view(-1) == 1  # 这里直接压扁计算...

            active_logits = logits.view(-1, self.config.num_crf_class)  # active_logits.shape=torch.Size([8192, 3])
            # 这里 也是将labels进行压平，这里只是为了确保...
            active_labels = torch.where(active_loss, labels.view(-1),torch.tensor(loss_fct.ignore_index).type_as(labels))
            # active_labels = [-100,    0,    0,    0,    0,    0,    0,    0,...]

            loss = loss_fct(active_logits, active_labels)

            # output = np.argmax(logits.detach().cpu().numpy(), axis=2)
            #
            # output_token = []
            # batch_size,seq_len = logits.shape[:2]
            # for i in range(batch_size):
            #     tmp_token = []
            #     for j in range(seq_len):
            #         if labels[i][j]!=loss_fct.ignore_index:
            #             tmp_token.append(output[i][j])
            #     output_token.append(tmp_token)

            return loss, logits
        else:
            return logits


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

import torch
import torch.nn as nn
from ipdb import set_trace
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torchcrf import CRF

from config import NormalConfig


logger = logging.getLogger('main.bilstm_cnn_crf')


class BiLSTM_CNN_CRF(nn.Module):
    def __init__(self,config:NormalConfig):
        super(BiLSTM_CNN_CRF,self).__init__()
        self.config = config
        self.word_embedding_dim = config.word_embedding_dim
        if config.use_pretrained_embedding:
            logger.info('将预训练的词嵌入加载到nn.Embedding')
            self.word_embedding = nn.Embedding(config.vocab_size, self.word_embedding_dim)
            self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embeddings))

        else:
            self.word_embedding = nn.Embedding(config.vocab_size, config.word_embedding_dim)

        self.char_embedding = nn.Embedding(num_embeddings=config.char_size,embedding_dim=config.char_embedding_dim,padding_idx=0)
        self.dropout = nn.Dropout(config.dropout_prob)

        # 对char embedding的卷积层
        self.char_cnn_layer = nn.Conv1d(in_channels=config.char_embedding_dim,out_channels=config.char_embedding_dim *config.filter_num,kernel_size=config.char_window_size,groups=config.char_embedding_dim)

        # 一般是单层的bilstm
        self.bilstm_layer = nn.LSTM(input_size=self.word_embedding_dim+config.char_embedding_dim *config.filter_num,hidden_size=config.lstm_hidden_dim,num_layers=config.num_bilstm_layers,batch_first=True,bidirectional=True)

        # *2是因为这里是双层结构
        self.classification = nn.Linear(config.lstm_hidden_dim*2, config.num_crf_class)

        self.crf_model = CRF(num_tags=config.num_crf_class, batch_first=True)
    def lstm_custom_init(self):
        '''
        一个初始化，这里
        :return:
        '''
        nn.init.xavier_uniform_(self.bilstm_layer.weight_hh_l0)
        nn.init.xavier_uniform_(self.bilstm_layer.weight_hh_l0_reverse)
        nn.init.xavier_uniform_(self.bilstm_layer.weight_ih_l0)
        nn.init.xavier_uniform_(self.bilstm_layer.weight_ih_l0_reverse)
        self.bilstm_layer.bias_hh_l0.data.fill_(0)
        self.bilstm_layer.bias_hh_l0_reverse.data.fill_(0)
        self.bilstm_layer.bias_ih_l0.data.fill_(0)
        self.bilstm_layer.bias_ih_l0_reverse.data.fill_(0)
        # Init forget gates to 1
        for names in self.bilstm_layer._all_weights:
            for name in filter(lambda n: 'bias' in n, names):
                bias = getattr(self.bilstm_layer, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)
    def sort_by_seq_len_list(self, seq_len_list):
        data_num = len(seq_len_list)
        sort_indices = sorted(range(len(seq_len_list)), key=seq_len_list.__getitem__, reverse=True)
        # 这个reverse_sort_indices表示得是排完序之后的位置对应关系，可以用于还原排序之前的序列
        reverse_sort_indices = [-1 for _ in range(data_num)]
        for i in range(data_num):
            reverse_sort_indices[sort_indices[i]] = i
        sort_index = torch.tensor(sort_indices, dtype=torch.long)
        reverse_sort_index = torch.tensor(reverse_sort_indices, dtype=torch.long)
        return sorted(seq_len_list, reverse=True), sort_index, reverse_sort_index
    def get_seq_len_list_from_mask_tensor(self, mask_tensor):
        batch_size = mask_tensor.shape[0]
        return [int(mask_tensor[k].sum().item()) for k in range(batch_size)]
    def pack(self, input_tensor, mask_tensor):
        # seq_len_list获得每个sequence的真实长度
        seq_len_list = self.get_seq_len_list_from_mask_tensor(mask_tensor)

        sorted_seq_len_list, sort_index, reverse_sort_index = self.sort_by_seq_len_list(seq_len_list)
        # 按照sort_index来获得排序后的文本序列
        sort_index = sort_index.to(mask_tensor.device)
        reverse_sort_index = reverse_sort_index.to(mask_tensor.device)

        input_tensor_sorted = torch.index_select(input_tensor, dim=0, index=sort_index)
        res = pack_padded_sequence(input_tensor_sorted, lengths=sorted_seq_len_list, batch_first=True)
        return res,reverse_sort_index

    def unpack(self, output_packed, max_seq_len, reverse_sort_index):

        output_tensor_sorted, _ = pad_packed_sequence(output_packed, batch_first=True, total_length=max_seq_len)
        # 将数据还原回去
        output_tensor = torch.index_select(output_tensor_sorted, dim=0, index=reverse_sort_index)
        return output_tensor



    def forward(self, token_ids, char_token_ids,labels,attention_masks,pack_unpack=False):
        '''

        :param token_ids:(batch_size,seq_len)
        :param char_token_ids:这是对输入的一个batch中的
        :param labels:
        :param mode:若为0，则是最普通的方式，若为1，则是pack和unpack
        :return:
        '''

        batch_size,seq_len = token_ids.shape[:2]
        word_embedding_out = self.word_embedding(token_ids)
        word_embedding_out_d = self.dropout(word_embedding_out)

        char_embedding_out = self.char_embedding(char_token_ids) # shape=(batch_size,max_seq_len,word_len,char_embedding_dim)
        # 卷积层对char embedding进行卷积和池化
        max_pooling_out = torch.zeros((batch_size,seq_len,self.config.char_embedding_dim*self.config.filter_num)).to(token_ids.device)
        char_embedding_out = char_embedding_out.permute(0,1,3,2) #shape= batch_num x max_seq_len x char_embeddings_dim x word_len=[100, 47, 25, 20]
        for k in range(seq_len):
            # 相当于让一维卷积一次处理一个句子的char feature
            # char_embeddings_feature[:, k, :, :].shape=[100, 25, 20]=(batch_size,character_dim,word_len)
            tmp_cnn = self.char_cnn_layer(char_embedding_out[:,k,:,:])#tmp_cnn.shape=(batch_size,output_channels=newd_im,18(卷积之后的seq_len))=(100,750,18)
            max_pooling_out[:, k, :], _ = torch.max(tmp_cnn, dim=2)
        #max_pooling.shape: batch_num x max_seq_len x filter_num*char_embeddings_dim=[100, 47, 750]

        z = torch.cat((word_embedding_out_d,max_pooling_out),dim=2) #z.shape = (batch_size,batch_max_seq_len,750+100) = (100,47,850)

        # 然后bilstm对z进行encode
        #这里先暂时不使用pack方式，而是直接batch进行
        if pack_unpack:
            input_packed, reverse_sort_index = self.pack(z, attention_masks)
            output_packed, _ = self.bilstm_layer(input_packed)
            rnn_out = self.unpack(output_packed, seq_len, reverse_sort_index)
        else:
            rnn_out,_ = self.bilstm_layer(z) #rnn_out.shape=(batch_size,seq_len,lstm_hidden_dim)=（128，150，256=hidden_size*2）

        # out.shape = [batch.seq_len,num_class)
        emissions = self.classification(rnn_out)

        if labels is not None:

            loss = -1. * self.crf_model(emissions=emissions, tags=labels.long(), mask=attention_masks.byte(),reduction='mean')
            tokens_out = self.crf_model.decode(emissions=emissions, mask=attention_masks.byte())
            return loss, tokens_out
        else:
            tokens_out = self.crf_model.decode(emissions=emissions, mask=attention_masks.byte())
            return tokens_out, attention_masks




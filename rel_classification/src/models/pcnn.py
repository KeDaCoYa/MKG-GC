# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  这是非常基础的关系分类，模型来自PCNN: Relation Classification via Convolutional Deep Neural Network

   Author :        kedaxia
   date：          2021/12/01
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/01: 
-------------------------------------------------
"""
import logging

import torch
import torch.nn as nn
from ipdb import set_trace
import torch.nn.functional as F
from config import NormalConfig


logger = logging.getLogger('main.model_pcnn')

class PCNN(nn.Module):
    def __init__(self,config:NormalConfig):
        super(PCNN,self).__init__()
        self.config = config

        # 加载预训练的Word2vec
        if config.use_pretrained_embedding:
            logger.info('将预训练的词嵌入加载到nn.Embedding')
            self.word_embedding = nn.Embedding(config.vocab_size, config.word_embedding_dim)
            self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embeddings))
            self.word_embedding.requires_grad_=False
        else:
            self.word_embedding = nn.Embedding(config.vocab_size, config.word_embedding_dim)


        self.pos1_embedding = nn.Embedding(config.pos_dis_limit*2+3,config.position_embedding_dim)
        self.pos2_embedding = nn.Embedding(config.pos_dis_limit*2+3,config.position_embedding_dim)

        self.dropout = nn.Dropout(p=config.dropout_prob)

        feature_dim = config.word_embedding_dim+2*config.position_embedding_dim #  50 + 10*2 = 70

        # 四个不同大小的卷积核,(2,70),(3,70),(4,70),(5,70)
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=config.filter_num,
                kernel_size=(k,feature_dim),
                padding=0
            ) for k in config.filters
        ])

        filter_dim = config.filter_num*len(config.filters)

        # 最后的输出层
        self.classifier = nn.Linear(filter_dim,config.relation_labels)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self,token_ids,pos1_feature,pos2_feature,labels):

        word_feature = self.word_embedding(token_ids)
        pos1_feature = self.pos1_embedding(pos1_feature)
        pos2_feature = self.pos2_embedding(pos2_feature)

        # 合并起来,考虑一下特征顺序...
        input_feature = torch.cat([word_feature,pos1_feature,pos2_feature],dim=2)

        # 这里调整一些dim，因为要卷积了
        x = input_feature.unsqueeze(1)
        x = self.dropout(x)

        # 四个卷积层，得到四个结果
        conv_outputs = []
        for conv in self.convs:
            # 这里conv是二维卷积，x.shape=(batch_size,1,seq_len.embedding_dim),相当于输入通道为1，输出通道为num_filters=128
            out = conv(x)
            # 因此out.shape = (batch_size,num_filters(output_channels),(seq_len-kernel_size[0]+2*p)/2,embedding_dim-kernel_size[1]+2*p)/2
            out = torch.tanh(out).squeeze(3)
            conv_outputs.append(out)
        # x = [torch.tanh(conv(x)).squeeze(3) for conv in self.covns]  # x[idx]: batch_size x num_filter x (batch_max_len-knernel_size+1),len(x) = 4
        # x0.shape = (50,128,40),x1.shape = (50,128,39),x2.shape = (50,128,38),x3.shape = (50,128,37)
        # 4个x，都是(50,128)，x.shape=(4,50,128)
        x = [F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2) for i in conv_outputs] ## x[idx]: batch_size x filter_num=(batch_size,128)

        sentence_features = torch.cat(x,dim=1) #sentence_features.shape=(batch_size,4*128)

        x = self.dropout(sentence_features)
        predicate_logits = self.classifier(x)
        if labels is not None:

            loss = self.criterion(predicate_logits,labels)

            _,predicate_token = torch.max(predicate_logits,dim=-1)
            return loss,predicate_token
        else:
            predicate_token = torch.max(predicate_logits, dim=-1)
            return predicate_token






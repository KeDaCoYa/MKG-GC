# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2021/12/02
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/02: 
-------------------------------------------------
"""
import torch
from ipdb import set_trace
from torch.utils.data import Dataset

from src.utils.function_utils import get_pos_feature
from src.dataset_utils.data_process_utils import sequence_padding

class PCNN_Dataset(Dataset):
    def __init__(self,sents,labels,word2id,label2id,max_len,pos_dis_limit):
        super(PCNN_Dataset, self).__init__()
        self.sents = sents
        self.labels = labels
        self.word2id = word2id
        self.label2id = label2id
        self.pos_dis_limit = pos_dis_limit
        self.max_len = max_len



    def __len__(self):
        return len(self.sents)
    def __getitem__(self, item):
        ent1,ent2,sent = self.sents[item]
        label = self.labels[item]
        return ent1,ent2,sent,label
    def collate_fn(self,features):
        '''
        在这里将数据转换为模型需要的数据类别
        features是一个batch_size的结果
        :param features:
        :return:
        '''
        raw_text_li = []
        batch_token_ids = []
        batch_labels = []
        batch_pos1_features = []
        batch_pos2_features = []
        max_batch_len = 0
        for ent1,ent2,sent,label in features:
            if len(sent)>max_batch_len:
                max_batch_len = len(sent)
            raw_text_li.append(sent)

            token_ids = [self.word2id.get(word,self.word2id.get(word.lower(),1)) for word in sent]
            batch_token_ids.append(token_ids)
            label = self.label2id.get(label)
            batch_labels.append(label)

            # 位置特征
            pos1_feature = []
            pos2_feature = []
            # 实体的第一个word
            e1_start = ent1.split(' ')[0] if ' ' in ent1 else ent1
            e2_start = ent2.split(' ')[0] if ' ' in ent2 else ent2

            # 获得两个实体在句子的index
            e1_idx = sent.index(e1_start)  # 实体1在此句子的位置
            e2_idx = sent.index(e2_start)  # 实体2在这个句子上的位置


            for idx, word in enumerate(sent):
                # 这是得到一个word在word2id中的，不过经过了一些变化
                pos1_feature.append(get_pos_feature(idx - e1_idx,limit=self.pos_dis_limit))
                pos2_feature.append(get_pos_feature(idx - e2_idx,limit=self.pos_dis_limit))
            batch_pos1_features.append(pos1_feature)
            batch_pos2_features.append(pos2_feature)


        # 开始补齐...

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids,value=0,length=min(max_batch_len,self.max_len))).long()
        batch_labels = torch.tensor(batch_labels).long()
        batch_pos1_features = torch.tensor(sequence_padding(batch_pos1_features,value=2*self.pos_dis_limit+2,length=min(max_batch_len,self.max_len))).long()
        batch_pos2_features = torch.tensor(sequence_padding(batch_pos2_features,value=2*self.pos_dis_limit+2,length=min(max_batch_len,self.max_len))).long()

        if max_batch_len>self.max_len:
            batch_token_ids = batch_token_ids[:,:self.max_len]
            batch_pos1_features = batch_pos1_features[:,:self.max_len]
            batch_pos2_features = batch_pos2_features[:,:self.max_len]

        return batch_token_ids,batch_pos1_features,batch_pos2_features,batch_labels








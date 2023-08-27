# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2022/01/18
   Copyright:      (c) kedaxia 2022
-------------------------------------------------
   Change Activity:
                   2022/01/18: 
-------------------------------------------------
"""
import logging
import os
import pickle

from ipdb import set_trace
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from huggingface_hub import hf_hub_url, cached_download


import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer, default_data_collator

from src.data_loader import NamesDataset

logger = logging.getLogger('main.sparse_encoder')


class DoubleBioSyn:
    def __init__(self,config,device,initial_sparse_weight=None):
        """
        这个模型就是统筹兼顾sparse encoder和dense encoder
        这个也是关键模型
        :param config:
        """

        self.config = config
        self.device = device
        self.max_len = config.max_len
        self.context_dense_encoder = None
        self.normalize_dense_encoder = None
        self.tokenizer = None

        self.sparse_encoder = None
        self.sparse_weight = None

        if initial_sparse_weight != None:
            self.sparse_weight = self.init_sparse_weight(initial_sparse_weight)

    def init_sparse_weight(self, initial_sparse_weight):
        """
        初始化sparse weight
        ----------
        initial_sparse_weight : float
            initial sparse weight
        """

        self.sparse_weight = nn.Parameter(torch.empty(1).to(self.device))
        self.sparse_weight.data.fill_(initial_sparse_weight) # init sparse_weight

        return self.sparse_weight

    def init_sparse_encoder(self, corpus):
        '''
        初始化使用tf-idf计算sparse representation
        :param corpus:
        :return:
        '''
        self.sparse_encoder = SparseEncoder(self.device).fit(corpus)

        return self.sparse_encoder
    def load_dense_encoder(self, bert_dir):
        '''
        这也就是为了可以自由加载Transformer的model
        '''

        self.normalize_dense_encoder = AutoModel.from_pretrained(bert_dir)
        self.context_dense_encoder = AutoModel.from_pretrained(bert_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)

        self.normalize_dense_encoder = self.normalize_dense_encoder.to(self.device)
        self.context_dense_encoder = self.context_dense_encoder.to(self.device)
        return self.normalize_dense_encoder, self.context_dense_encoder,self.tokenizer

    def load_sparse_encoder(self, sparse_model_path):
        """
        加载已有的sparse model(其实就是weight)
        :param sparse_model_path:
        :return:
        """
        sparse_encoder_path = os.path.join(sparse_model_path,'sparse_encoder.pk')
        # check file exists
        if not os.path.isfile(sparse_encoder_path):
            # download from huggingface hub and cache it
            sparse_encoder_url = hf_hub_url(sparse_model_path, filename="sparse_encoder.pk")
            sparse_encoder_path = cached_download(sparse_encoder_url)

        self.sparse_encoder = SparseEncoder(self.device).load_encoder(path=sparse_encoder_path)

        return self.sparse_encoder

    def get_dense_encoder(self):

        return self.normalize_dense_encoder, self.context_dense_encoder

    def get_dense_tokenizer(self):

        return self.tokenizer

    def get_sparse_encoder(self):

        return self.sparse_encoder

    def get_sparse_weight(self):

        return self.sparse_weight


    def get_sparse_representation(self,mentions,verbose=False):
        '''
        将数据集中的mention，使用sparse encoder进行编码，得到sparse representations
        :param mentions:
        :param verbose:
        :return:
        '''
        batch_size = 1024
        sparse_embeds = []

        # if verbose:
        #     iterations = tqdm(range(0, len(mentions), batch_size))
        # else:
        #     iterations = range(0, len(mentions), batch_size)

        for start in tqdm(range(0, len(mentions), batch_size),disable=not verbose,desc='get sparse embedding'):
            end = min(start + batch_size, len(mentions))
            batch = mentions[start:end]
            # 调用tf-ids的transform，得到
            batch_sparse_embeds = self.sparse_encoder(batch)
            batch_sparse_embeds = batch_sparse_embeds.cpu().numpy()
            sparse_embeds.append(batch_sparse_embeds)
        sparse_embeds = np.concatenate(sparse_embeds, axis=0)

        return sparse_embeds

    def get_dense_representation(self,mentions,verbose=False):
        '''
        这是使用BERT等预训练模型对mentions进行encode
        :param mentions:
        :param verbose:
        :return:
        '''
        # 这时候注意要开启eval，不使用dropout....
        self.context_dense_encoder.eval()  # prevent dropout
        self.normalize_dense_encoder.eval()  # prevent dropout
        batch_size = 1024


        if isinstance(mentions, np.ndarray):
            mentions = mentions.tolist()
        # 添加了这些参数之后，竟然相当于encode_plus，可能是参数return_tensor
        # name_encoding.shape = (1587,25)
        name_encodings = self.tokenizer(mentions, padding="max_length", max_length=self.max_len, truncation=True,
                                        return_tensors="pt")

        name_encodings = name_encodings.to(self.device)

        name_dataset = NamesDataset(name_encodings)
        # default_data_collator这个是非常简单的一个预处理函数
        name_dataloader = DataLoader(name_dataset, shuffle=False, collate_fn=default_data_collator,
                                     batch_size=batch_size)

        # 这里还设置不要进行梯度记录，这里就相当于只是利用模型进行计算，不进行更新
        dense_embeds = []
        with torch.no_grad():
            for batch in tqdm(name_dataloader, disable=not verbose, desc='get dense representation'):

                outputs1 = self.context_dense_encoder(**batch)
                outputs2 = self.normalize_dense_encoder(**batch)
                # 但是不知道为啥不选择outputs[1]作为[CLS]的output...
                batch_context_dense_embeds = outputs1[0][:,0].cpu().detach().numpy()  # [CLS] representations,shape=(batch_size,hidden_size)=(1024,768)
                batch_normalize_dense_embeds = outputs2[0][:,0].cpu().detach().numpy()  # [CLS] representations,shape=(batch_size,hidden_size)=(1024,768)


                # idea:这里选择不同的运算方式
                # 1. 取平均值，相加，合并....
                batch_dense_embeds = batch_normalize_dense_embeds+batch_context_dense_embeds


                dense_embeds.append(batch_dense_embeds)
        dense_embeds = np.concatenate(dense_embeds, axis=0)

        return dense_embeds


    def get_score_matrix(self, query_embeds, dict_embeds):
        """
        返回相似分数矩阵，通过点击
        Parameters
        ----------
        query_embeds :shape = (len(dataset),dim)
        dict_embeds : (len(dict),dim)

        Returns
        -------
        score_matrix : np.array
            2d numpy array of scores
        """
        score_matrix = np.matmul(query_embeds, dict_embeds.T)
        # shape = (len(dataset),len(dict))
        return score_matrix

    def retrieve_candidate(self, score_matrix, topk):
        """
        对相似矩阵检索，得到对于每个mention，得到与其最相似的topk个字典中的数值
        Parameters
        ----------
        score_matrix : 相似分数矩阵 np.array shape=(len(train_dataset),len(dict))
            2d numpy array of scores
        topk : int
            The number of candidates
        Returns
        -------
        topk_idxs : np.array
            2d numpy array of scores [# of query , # of dict] =(1578,71924),就是输入数据和字典中各个实体的相似分数
        """

        def indexing_2d(arr, cols):
            '''
            这是二维取值，根据cols的index来取值
            arr:
            '''
            rows = np.repeat(np.arange(0, cols.shape[0])[:, np.newaxis], cols.shape[1], axis=1)
            # 这个rows结果为
            #       [[   0,    0,    0, ...,    0,    0,    0],
            #        [   1,    1,    1, ...,    1,    1,    1],
            #        [   2,    2,    2, ...,    2,    2,    2],
            #        ...,
            #        [1584, 1584, 1584, ..., 1584, 1584, 1584],
            #        [1585, 1585, 1585, ..., 1585, 1585, 1585],
            #        [1586, 1586, 1586, ..., 1586, 1586, 1586]]

            return arr[rows, cols]


        # 这里是使用argpartion,通过对倒数第20个进行划分，得到最大的20个index
        # 也就是每个单词对应词典中最相似的topk个name，这个时候并没有排序
        topk_idxs = np.argpartition(score_matrix, -topk)[:, -topk:]  # shape=(1587,20)

        # 根据上面得到的index到score_matrix取值
        topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
        # 对index排序
        topk_argidxs = np.argsort(-topk_score_matrix)
        topk_idxs = indexing_2d(topk_idxs, topk_argidxs)

        return topk_idxs

    def save_model(self, path):
        """
        保存所有的模型
            1. dense encoder(BERT等模型)
            2. sparse weight
        :param path:
        :return:
        """
        # save dense encoder,这是transformers自带的方法进行保存模型，
        self.dense_encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # save sparse encoder
        sparse_encoder_path = os.path.join(path, 'sparse_encoder.pk')
        self.sparse_encoder.save_encoder(path=sparse_encoder_path)
        sparse_weight_file = os.path.join(path, 'sparse_weight.pt')

        torch.save(self.sparse_weight, sparse_weight_file)
        logging.info("Sparse weight saved in {}".format(sparse_weight_file))

    def load_model(self, model_name_or_path):
        '''
        这是加载所有的模型
        '''
        self.load_dense_encoder(model_name_or_path)
        self.load_sparse_encoder(model_name_or_path)
        self.load_sparse_weight(model_name_or_path)

        return self



    def load_sparse_weight(self, model_name_or_path):
        sparse_weight_path = os.path.join(model_name_or_path, 'sparse_weight.pt')
        # check file exists
        if not os.path.isfile(sparse_weight_path):
            # download from huggingface hub and cache it
            sparse_weight_url = hf_hub_url(model_name_or_path, filename="sparse_weight.pt")
            sparse_weight_path = cached_download(sparse_weight_url)

        self.sparse_weight = torch.load(sparse_weight_path)

        return self.sparse_weight

class SparseEncoder(object):
    """
        这是Sparse Encoder,
        这就是Tf-idf计算得到spare representation，使用uni，bi-gram...
        直接使用sklearn进行计算，然后转变为tensor类别
    """

    def __init__(self, device):
        self.encoder = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
        self.device = device

    def fit(self, train_corpus):

        self.encoder.fit(train_corpus)
        return self

    def transform(self, mentions):
        '''
        将mentions使用tf-idf进行转变
        :param mentions:
        :return:
        '''

        vec = self.encoder.transform(mentions).toarray()
        vec = torch.FloatTensor(vec)  # return torch float tensor
        vec = vec.to(self.device)
        return vec

    def __call__(self, mentions):
        return self.transform(mentions)

    def vocab(self):
        return self.encoder.vocabulary_

    def save_encoder(self, path):
        with open(path, 'wb') as fout:
            pickle.dump(self.encoder, fout)
            logging.info("Sparse encoder saved in {}".format(path))

    def load_encoder(self, path):
        with open(path, 'rb') as fin:
            self.encoder = pickle.load(fin)
            logging.info("Sparse encoder loaded from {}".format(path))

        return self



class MyRerankNet(nn.Module):
    """
        这个用于多任务的RerankNet
    """
    def __init__(self,config, context_dense_encoder,normalize_encoder,sparse_weight,device):


        super(MyRerankNet, self).__init__()
        self.context_dense_encoder = context_dense_encoder
        self.normalize_encoder = normalize_encoder
        self.config = config

        self.device = device
        self.sparse_weight = sparse_weight
        self.sparse_weight.requires_grad = True
        self.criterion = self.marginal_nll

        # self.optimizer = optim.Adam([
        #     {'params': self.dense_encoder.parameters()},
        #     {'params': self.sparse_weight, 'lr': 0.01, 'weight_decay': 0}],
        #     lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def forward(self, x,type_=0):
        """
        query : (N, h), candidates : (N, topk, h)
        query是输入的mention
        candidate是字典的内容
        output : (N, topk)
        """
        # 这里只预先得到sparse的similiarity scores、
        # dense的similarity scores则是动态得到...
        query_token, candidate_tokens, candidate_s_scores = x
        batch_size, topk, max_length = candidate_tokens['input_ids'].shape


        candidate_s_scores = candidate_s_scores.to(self.device)

        query_token['input_ids'] = query_token['input_ids'].to(self.device)
        query_token['token_type_ids'] = query_token['token_type_ids'].to(self.device)
        query_token['attention_mask'] = query_token['attention_mask'].to(self.device)

        candidate_tokens['input_ids'] = candidate_tokens['input_ids'].to(self.device)
        candidate_tokens['token_type_ids'] = candidate_tokens['token_type_ids'].to(self.device)
        candidate_tokens['attention_mask'] = candidate_tokens['attention_mask'].to(self.device)

        # dense embed for query and candidates
        query_embed = self.dense_encoder(
            input_ids=query_token['input_ids'].squeeze(1),
            token_type_ids=query_token['token_type_ids'].squeeze(1),
            attention_mask=query_token['attention_mask'].squeeze(1),type_=type_
        )
        query_embed = query_embed[0][:, 0].unsqueeze(1)  # query : [batch_size, 1, hidden]

        candidate_embeds = self.dense_encoder(
            input_ids=candidate_tokens['input_ids'].reshape(-1, max_length),
            token_type_ids=candidate_tokens['token_type_ids'].reshape(-1, max_length),
            attention_mask=candidate_tokens['attention_mask'].reshape(-1, max_length),type_=type_
        )
        candidate_embeds = candidate_embeds[0][:, 0].reshape(batch_size, topk, -1)  # [batch_size, topk, hidden]

        # 得到dense similiarity scores,shape=(batch_size,topk)
        candidate_d_score = torch.bmm(query_embed, candidate_embeds.permute(0, 2, 1)).squeeze(1)

        score = self.sparse_weight * candidate_s_scores + candidate_d_score
        return score

    def reshape_candidates_for_encoder(self, candidates):
        """
        reshape candidates for encoder input shape
        [batch_size, topk, max_length] => [batch_size*topk, max_length]
        """
        _, _, max_length = candidates.shape
        candidates = candidates.contiguous().view(-1, max_length)
        return candidates

    def get_loss(self, outputs, targets):
        targets = targets.to(self.device)
        loss = self.criterion(outputs, targets)
        return loss

    def get_embeddings(self, mentions, batch_size=1024):
        """
        Compute all embeddings from mention tokens.
        """
        embedding_table = []
        with torch.no_grad():
            for start in tqdm(range(0, len(mentions), batch_size)):
                end = min(start + batch_size, len(mentions))
                batch = mentions[start:end]
                batch_embedding = self.vectorizer(batch)
                batch_embedding = batch_embedding.cpu()
                embedding_table.append(batch_embedding)
        embedding_table = torch.cat(embedding_table, dim=0)
        return embedding_table


    def marginal_nll(self,score, target):
        """
        sum all scores among positive samples
        损失函数计算
        """
        predict = F.softmax(score, dim=-1)
        loss = predict * target # 只记录target=1的预测分数
        loss = loss.sum(dim=-1)  # sum all positive scores
        loss = loss[loss > 0]  # filter sets with at least one positives
        # 将loss值给限定在[1e-9,1]范围，
        loss = torch.clamp(loss, min=1e-9, max=1)  # for numerical stability
        loss = -torch.log(loss)  # for negative log likelihood
        if len(loss) == 0:
            loss = loss.sum()  # will return zero loss
        else:
            loss = loss.mean()
        return loss


class RerankNet(nn.Module):
    """
        这个是训练模型，用于对模型的训练...
    """
    def __init__(self,config, dense_encoder,sparse_weight,device):

        super(RerankNet, self).__init__()
        self.dense_encoder = dense_encoder
        self.config = config

        self.device = device
        self.sparse_weight = sparse_weight
        self.sparse_weight.requires_grad = True
        self.criterion = self.marginal_nll



    def forward(self, x):
        """
        query : (N, h), candidates : (N, topk, h)
        query是输入的mention
        candidate是字典的内容
        output : (N, topk)
        """
        # 这里只预先得到sparse的similiarity scores、
        # dense的similarity scores则是动态得到...
        query_token, candidate_tokens, candidate_s_scores = x
        batch_size, topk, max_length = candidate_tokens['input_ids'].shape

        candidate_s_scores = candidate_s_scores.to(self.device)

        query_token['input_ids'] = query_token['input_ids'].to(self.device)
        query_token['token_type_ids'] = query_token['token_type_ids'].to(self.device)
        query_token['attention_mask'] = query_token['attention_mask'].to(self.device)

        candidate_tokens['input_ids'] = candidate_tokens['input_ids'].to(self.device)
        candidate_tokens['token_type_ids'] = candidate_tokens['token_type_ids'].to(self.device)
        candidate_tokens['attention_mask'] = candidate_tokens['attention_mask'].to(self.device)

        # dense embed for query and candidates
        query_embed = self.dense_encoder(
            input_ids=query_token['input_ids'].squeeze(1),
            token_type_ids=query_token['token_type_ids'].squeeze(1),
            attention_mask=query_token['attention_mask'].squeeze(1)
        )
        query_embed = query_embed[0][:, 0].unsqueeze(1)  # query : [batch_size, 1, hidden]

        candidate_embeds = self.dense_encoder(
            input_ids=candidate_tokens['input_ids'].reshape(-1, max_length),
            token_type_ids=candidate_tokens['token_type_ids'].reshape(-1, max_length),
            attention_mask=candidate_tokens['attention_mask'].reshape(-1, max_length)
        )
        candidate_embeds = candidate_embeds[0][:, 0].reshape(batch_size, topk, -1)  # [batch_size, topk, hidden]

        # 得到dense similiarity scores,shape=(batch_size,topk)
        candidate_d_score = torch.bmm(query_embed, candidate_embeds.permute(0, 2, 1)).squeeze(1)

        score = self.sparse_weight * candidate_s_scores + candidate_d_score
        return score

    def reshape_candidates_for_encoder(self, candidates):
        """
        reshape candidates for encoder input shape
        [batch_size, topk, max_length] => [batch_size*topk, max_length]
        """
        _, _, max_length = candidates.shape
        candidates = candidates.contiguous().view(-1, max_length)
        return candidates

    def get_loss(self, outputs, targets):
        targets = targets.to(self.device)
        loss = self.criterion(outputs, targets)
        return loss

    def get_embeddings(self, mentions, batch_size=1024):
        """
        Compute all embeddings from mention tokens.
        """
        embedding_table = []
        with torch.no_grad():
            for start in tqdm(range(0, len(mentions), batch_size)):
                end = min(start + batch_size, len(mentions))
                batch = mentions[start:end]
                batch_embedding = self.vectorizer(batch)
                batch_embedding = batch_embedding.cpu()
                embedding_table.append(batch_embedding)
        embedding_table = torch.cat(embedding_table, dim=0)
        return embedding_table


    def marginal_nll(self,score, target):
        """
        sum all scores among positive samples
        损失函数计算
        """
        predict = F.softmax(score, dim=-1)
        loss = predict * target # 只记录target=1的预测分数
        loss = loss.sum(dim=-1)  # sum all positive scores
        loss = loss[loss > 0]  # filter sets with at least one positives
        # 将loss值给限定在[1e-9,1]范围，
        loss = torch.clamp(loss, min=1e-9, max=1)  # for numerical stability
        loss = -torch.log(loss)  # for negative log likelihood
        if len(loss) == 0:
            loss = loss.sum()  # will return zero loss
        else:
            loss = loss.mean()
        return loss
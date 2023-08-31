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

from tqdm import tqdm


from sklearn.metrics.pairwise import cosine_similarity
from pytorch_metric_learning import miners, losses

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast

from transformers import AutoTokenizer, AutoModel


class SapBertModel(nn.Module):
    def __init__(self, encoder, learning_rate, weight_decay, pairwise,loss,device=None, use_miner=True, miner_margin=0.2, type_of_triplets="all", agg_mode="cls"):


        super(SapBertModel, self).__init__()
        # 这个就是BRET
        self.encoder = encoder
        # 表示输入的数据是否是成对的
        self.pairwise = pairwise
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.loss = loss
        self.use_miner = use_miner
        self.miner_margin = miner_margin
        self.agg_mode = agg_mode
        self.device = device

        self.optimizer = optim.AdamW([{'params': self.encoder.parameters()}],lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
        else:
            self.miner = None

        if self.loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(alpha=1, beta=60, base=0.5)  # 1,2,3; 40,50,60
        elif self.loss == "circle_loss":
            self.loss = losses.CircleLoss()
        elif self.loss == "triplet_loss":
            self.loss = losses.TripletMarginLoss()
        elif self.loss == "infoNCE":
            self.loss = losses.NTXentLoss(temperature=0.07)  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        elif self.loss == "lifted_structure_loss":
            self.loss = losses.LiftedStructureLoss()
        elif self.loss == "nca_loss":
            self.loss = losses.NCALoss()

        print("miner:", self.miner)
        print("loss:", self.loss)

    @autocast()
    def forward(self, query_toks1, query_toks2, labels):
        """
        query : (N, h), candidates : (N, topk, h)
        query_toks1和query_toks2 就是bert input的三件套
            shape=(batch_size,seq_len = 25)
            tokenize之后，最多25个单词
        labels就是他们的CUI
        output : (N, topk)
        """
        # 下面得到mention的representation
        last_hidden_state1 = self.encoder(**query_toks1, return_dict=True).last_hidden_state
        last_hidden_state2 = self.encoder(**query_toks2, return_dict=True).last_hidden_state

        if self.agg_mode == "cls":
            query_embed1 = last_hidden_state1[:, 0]  # query : [batch_size, hidden]
            query_embed2 = last_hidden_state2[:, 0]  # query : [batch_size, hidden]
        elif self.agg_mode == "mean_pool":
            query_embed1 = last_hidden_state1.mean(1)  # query : [batch_size, hidden]
            query_embed2 = last_hidden_state2.mean(1)  # query : [batch_size, hidden]
        else:
            raise NotImplementedError()

        # 合并之后shape=(512,hidden_size=768)
        query_embed = torch.cat([query_embed1, query_embed2], dim=0)
        # shape = (512)
        labels = torch.cat([labels, labels], dim=0)

        # 这个就是online mine
        # 这里也就会生成negative/positive pairs,然后计算损失函数
        if self.use_miner:
            # self.miner其实会根据query_embed来生成pairwise，通过相应的算法来生成hard pair/semi-hard pairwise/easy pairwise....
            hard_pairs = self.miner(query_embed, labels)
            loss = self.loss(query_embed, labels, hard_pairs)
            return loss
        else:
            return self.loss(query_embed, labels)

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

        loss, in_topk = self.criterion(outputs, targets)
        return loss, in_topk

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


class SapBERTWrapper(object):
    """
        这是对SapBert的中央控制器
    """

    def __init__(self,device):
        self.device = device
        self.tokenizer = None
        self.encoder = None

    def get_dense_encoder(self):
        assert (self.encoder is not None)

        return self.encoder

    def get_dense_tokenizer(self):
        assert (self.tokenizer is not None)

        return self.tokenizer

    def save_model(self, path, context=False):
        # save bert model, bert config
        self.encoder.save_pretrained(path)

        # save bert vocab
        self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        self.load_bert(path)

        return self

    def load_bert(self, path, lowercase=True):
        self.tokenizer = AutoTokenizer.from_pretrained(path,use_fast=True, do_lower_case=lowercase)
        self.encoder = AutoModel.from_pretrained(path)
        self.encoder = self.encoder.to(self.device)

        return self.encoder, self.tokenizer

    def get_score_matrix(self, query_embeds, dict_embeds, cosine=False, normalise=False):
        """
        Return score matrix

        Parameters
        ----------
        query_embeds : np.array
            2d numpy array of query embeddings
        dict_embeds : np.array
            2d numpy array of query embeddings

        Returns
        -------
        score_matrix : np.array
            2d numpy array of scores
        """
        # 计算相似性分数，cosine 或者是 内积
        if cosine:
            score_matrix = cosine_similarity(query_embeds, dict_embeds)
        else:
            score_matrix = np.matmul(query_embeds, dict_embeds.T)

        if normalise:
            score_matrix = (score_matrix - score_matrix.min()) / (score_matrix.max() - score_matrix.min())

        return score_matrix

    def retrieve_candidate(self, score_matrix, topk):
        """
        Return sorted topk idxes (descending order)

        Parameters
        ----------
        score_matrix : np.array
            2d numpy array of scores
        topk : int
            The number of candidates

        Returns
        -------
        topk_idxs : np.array
            2d numpy array of scores [# of query , # of dict]
        """

        def indexing_2d(arr, cols):
            rows = np.repeat(np.arange(0, cols.shape[0])[:, np.newaxis], cols.shape[1], axis=1)
            return arr[rows, cols]

        # get topk indexes without sorting
        topk_idxs = np.argpartition(score_matrix, -topk)[:, -topk:]

        # get topk indexes with sorting
        topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
        topk_argidxs = np.argsort(-topk_score_matrix)
        topk_idxs = indexing_2d(topk_idxs, topk_argidxs)

        return topk_idxs

    def retrieve_candidate_cuda(self, score_matrix, topk, batch_size=128, show_progress=False):
        """
        Return sorted topk idxes (descending order)

        Parameters
        ----------
        score_matrix : np.array
            2d numpy array of scores
        topk : int
            The number of candidates

        Returns
        -------
        topk_idxs : np.array
            2d numpy array of scores [# of query , # of dict]
        """

        res = None
        for i in tqdm(np.arange(0, score_matrix.shape[0], batch_size), disable=not show_progress):
            score_matrix_tmp = torch.tensor(score_matrix[i:i + batch_size]).to(self.device)
            matrix_sorted = torch.argsort(score_matrix_tmp, dim=1, descending=True)[:, :topk].cpu()
            if res is None:
                res = matrix_sorted
            else:
                res = torch.cat([res, matrix_sorted], axis=0)

        return res.numpy()

    def embed_dense(self, mentions, verbose=False, batch_size=1024, agg_mode="cls"):
        """
        使用BERT等预训练模型来encode

        Parameters
        ----------
        names : np.array
            An array of names

        Returns
        -------
        dense_embeds : list
            A list of dense embeddings
        """
        self.encoder.eval()  # prevent dropout

        batch_size = batch_size
        dense_embeds = []

        # print ("converting names to list...")
        # names = names.tolist()

        with torch.no_grad():

            for start in tqdm(range(0, len(mentions), batch_size),disable= not verbose):
                end = min(start + batch_size, len(mentions))
                batch = mentions[start:end]
                batch_tokenized_names = self.tokenizer.batch_encode_plus(
                    batch, add_special_tokens=True,
                    truncation=True, max_length=25,
                    padding="max_length", return_tensors='pt')
                batch_tokenized_names_cuda = {}
                for k, v in batch_tokenized_names.items():
                    batch_tokenized_names_cuda[k] = v.to(self.device)

                if agg_mode == "cls":
                    batch_dense_embeds = self.encoder(**batch_tokenized_names_cuda)[0][:, 0, :]  # [CLS]
                elif agg_mode == "mean_pool":
                    batch_dense_embeds = self.encoder(**batch_tokenized_names_cuda)[0].mean(1)  # pooling
                else:
                    print("no such agg_mode:", agg_mode)

                batch_dense_embeds = batch_dense_embeds.cpu().detach().numpy()
                dense_embeds.append(batch_dense_embeds)
        dense_embeds = np.concatenate(dense_embeds, axis=0)

        return dense_embeds
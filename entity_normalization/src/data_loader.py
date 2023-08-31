# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2022/01/19
   Copyright:      (c) kedaxia 2022
-------------------------------------------------
   Change Activity:
                   2022/01/19: 
-------------------------------------------------
"""

import numpy as np

import torch
from ipdb import set_trace
from torch.utils.data import Dataset


class NamesDataset(Dataset):
    def __init__(self, encodings):
        '''
        :param encodings: tokenizer之后得到的字典{'input_ids','token_type_ids','attention_mask'}
        '''
        self.encodings = encodings

    def __getitem__(self, idx):
        '''
        处理tokenize之后的字典，所以进行这样的遍历

        :param idx:
        :return:
        '''

        res = {key: val[idx].clone().detach() for key, val in self.encodings.items()}

        return res

    def __len__(self):
        return len(self.encodings.input_ids)

class NormalizationDataset(Dataset):


    def __init__(self, queries, dicts, tokenizer, max_len, topk, d_ratio, s_score_matrix, s_candidate_idxs):
        """
        Retrieve top-k candidates based on sparse/dense embedding
        Parameters
        ----------
        queries : list
            A list of tuples (name, id)
        dicts : list
            A list of tuples (name, id)
        tokenizer : BertTokenizer
            A BERT tokenizer for dense embedding
        topk : int
            The number of candidates
        d_ratio : float
            表示topk中，有d_ratio*topk是根据dense representation来得到的
            (1-d_ration)*topk:则是sparse representation得到的

        s_score_matrix : np.array
        s_candidate_idxs : np.array
        """

        self.query_names, self.query_ids = [row[0] for row in queries], [row[1] for row in queries]
        self.dict_names, self.dict_ids = [row[0] for row in dicts], [row[1] for row in dicts]
        self.topk = topk
        # 使用dense_representation得到的candidate dict name
        self.n_dense = int(topk * d_ratio)
        # 使用sparse_representation得到的candidate dict name
        self.n_sparse = topk - self.n_dense
        self.tokenizer = tokenizer
        self.max_length = max_len
        # 相似分数矩阵
        self.s_score_matrix = s_score_matrix
        self.s_candidate_idxs = s_candidate_idxs
        self.d_candidate_idxs = None

    def set_dense_candidate_idxs(self, d_candidate_idxs):
        self.d_candidate_idxs = d_candidate_idxs

    def __getitem__(self, query_idx):
        """
            这里将sparse representation和dense representation的结合转变...
            label是根据topk candidate idx而发生变化的
            最开始的时候label基本上都是0，经过训练之后才会label有1
        """
        assert (self.s_candidate_idxs is not None)
        assert (self.s_score_matrix is not None)
        assert (self.d_candidate_idxs is not None)

        query_name = self.query_names[query_idx]
        query_token = self.tokenizer(query_name, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')

        #得到query_idx所对应的所有candidate dict name
        # sparse_representation和dense_representation得到
        s_candidate_idx = self.s_candidate_idxs[query_idx]
        d_candidate_idx = self.d_candidate_idxs[query_idx]

        # 先使用spase representaion的相似度来填充topk
        topk_candidate_idx = s_candidate_idx[:self.n_sparse]

        #开始使用dense_idx来补充topk
        for d_idx in d_candidate_idx:
            if len(topk_candidate_idx) >= self.topk:
                break
            if d_idx not in topk_candidate_idx:
                topk_candidate_idx = np.append(topk_candidate_idx, d_idx)

        # sanity check
        assert len(topk_candidate_idx) == self.topk
        assert len(topk_candidate_idx) == len(set(topk_candidate_idx))
        # 获取可能的dict之中的name
        candidate_names = [self.dict_names[candidate_idx] for candidate_idx in topk_candidate_idx]
        # 这是得到topk的candidate dict name对应的相似分数
        candidate_s_scores = self.s_score_matrix[query_idx][topk_candidate_idx]
        labels = self.get_labels(query_idx, topk_candidate_idx).astype(np.float32)

        candidate_tokens = self.tokenizer(candidate_names, max_length=self.max_length, padding='max_length',
                                          truncation=True, return_tensors='pt')
        # query_token,candidate_tokens:就是bert所需要的输入数据格式:input_ids,token_type_ids,attention_masks

        return (query_token, candidate_tokens, candidate_s_scores), labels

    def __len__(self):
        return len(self.query_names)

    def check_label(self, query_id, candidate_id_set):
        """
        所有的queriy的CUI id应该都出现在字典中，不然就是无法进行....
        这个就是表示给定的pair是否是同义词
        :param query_id:
        :param candidate_id_set:
        :return:
        """
        label = 0
        # 因为可能会使用多个字典，所以id有多个
        query_ids = query_id.split("|")

        for q_id in query_ids:
            if q_id in candidate_id_set:
                label = 1
                continue
            else:
                label = 0
                break
        return label

    def get_labels(self, query_idx, candidate_idxs):
        """
        check 当前的query idx与candidate中的有哪些label是一样的，这就是得到negative 和 postive samples...
        """
        labels = np.array([])
        query_id = self.query_ids[query_idx]
        candidate_ids = np.array(self.dict_ids)[candidate_idxs]
        for candidate_id in candidate_ids:
            label = self.check_label(query_id, candidate_id)
            labels = np.append(labels, label)
        return labels


class MetricLearningDataset_pairwise(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """

    def __init__(self, path, tokenizer):  # d_ratio, s_score_matrix, s_candidate_idxs):
        with open(path, 'r') as f:
            lines = f.readlines()
        self.query_ids = []
        self.query_names = []
        # 这里就是读取数据集，pairwise形式
        for line in lines:
            line = line.rstrip("\n")
            query_id, name1, name2 = line.split("||")
            self.query_ids.append(query_id)
            self.query_names.append((name1, name2))
        self.tokenizer = tokenizer
        self.query_id_2_index_id = {k: v for v, k in enumerate(list(set(self.query_ids)))}

    def __getitem__(self, query_idx):
        query_name1 = self.query_names[query_idx][0]
        query_name2 = self.query_names[query_idx][1]
        query_id = self.query_ids[query_idx]
        query_id = int(self.query_id_2_index_id[query_id])

        return query_name1, query_name2, query_id

    def __len__(self):
        return len(self.query_names)


class MetricLearningDataset(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """

    def __init__(self, path, tokenizer):  # d_ratio, s_score_matrix, s_candidate_idxs):

        with open(path, 'r') as f:
            lines = f.readlines()
        self.query_ids = []
        self.query_names = []
        cuis = []
        for line in lines:
            cui, _ = line.split("||")
            cuis.append(cui)

        self.cui2id = {k: v for v, k in enumerate(cuis)}
        for line in lines:
            line = line.rstrip("\n")
            cui, name = line.split("||")
            query_id = self.cui2id[cui]
            # if query_id.startswith("C"):
            #    query_id = query_id[1:]
            # query_id = int(query_id)
            self.query_ids.append(query_id)
            self.query_names.append(name)
        self.tokenizer = tokenizer

    def __getitem__(self, query_idx):

        query_name = self.query_names[query_idx]
        query_id = self.query_ids[query_idx]
        query_token = self.tokenizer.transform([query_name])[0]

        return torch.tensor(query_token), torch.tensor(query_id)

    def __len__(self):
        return len(self.query_names)

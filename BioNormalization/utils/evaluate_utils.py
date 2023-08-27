# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2022/01/20
   Copyright:      (c) kedaxia 2022
-------------------------------------------------
   Change Activity:
                   2022/01/20: 
-------------------------------------------------
"""
import logging

from ipdb import set_trace
from tqdm import tqdm
import numpy as np

from src.models.biosyn import BioSyn

logger = logging.getLogger('main.evaluate')

def check_label(predicted_cui, golden_cui):
    """
    Some composite annotation didn't consider orders
    So, set label '1' if any cui is matched within composite cui (or single cui)
    Otherwise, set label '0'
    """
    return int(len(set(predicted_cui.split("|")).intersection(set(golden_cui.split("|"))))>0)

def check_k(queries):
    '''
    得到当前的一个个数，这个数并不太严谨，随便选了一个
    '''
    return len(queries[0]['mentions'][0]['candidates'])

def evaluate_topk_acc(data):
    """
    evaluate acc@1~acc@k
    """
    queries = data['queries']
    k = check_k(queries)

    for i in range(0, k):
        hit = 0
        for query in queries:
            mentions = query['mentions']
            mention_hit = 0
            for mention in mentions:
                candidates = mention['candidates'][:i + 1]  # to get acc@(i+1)
                mention_hit += np.any([candidate['label'] for candidate in candidates])

            # When all mentions in a query are predicted correctly,
            # we consider it as a hit
            if mention_hit == len(mentions):
                hit += 1

        data['acc{}'.format(i + 1)] = hit / len(queries)

    return data
def predict_topk(biosyn:BioSyn, eval_dictionary, eval_queries, topk, score_mode='hybrid'):
    """
    开始预测模型
    ----------
    score_mode : str
        hybrid, dense, sparse
    """

    sparse_weight = biosyn.get_sparse_weight().item()  # must be scalar value

    # embed dictionary

    dict_sparse_embeds = biosyn.get_sparse_representation(mentions=eval_dictionary[:, 0], verbose=True)
    dict_dense_embeds = biosyn.get_dense_representation(mentions=eval_dictionary[:, 0], verbose=True)

    queries = []
    for eval_query in tqdm(eval_queries, total=len(eval_queries),desc='模型正在预测....'):
        mentions = eval_query[0].replace("+", "|").split("|")
        golden_cui = eval_query[1].replace("+", "|")

        dict_mentions = []
        for mention in mentions:
            mention_sparse_embeds = biosyn.get_sparse_representation(mentions=np.array([mention]))
            mention_dense_embeds = biosyn.get_dense_representation(mentions=np.array([mention]))

            # get score matrix
            sparse_score_matrix = biosyn.get_score_matrix(
                query_embeds=mention_sparse_embeds,
                dict_embeds=dict_sparse_embeds
            )
            dense_score_matrix = biosyn.get_score_matrix(
                query_embeds=mention_dense_embeds,
                dict_embeds=dict_dense_embeds
            )
            if score_mode == 'hybrid':
                score_matrix = sparse_weight * sparse_score_matrix + dense_score_matrix
            elif score_mode == 'dense':
                score_matrix = dense_score_matrix
            elif score_mode == 'sparse':
                score_matrix = sparse_score_matrix
            else:
                raise NotImplementedError()

            # 开始检索，然后返回topk个dict中的index
            candidate_idxs = biosyn.retrieve_candidate(
                score_matrix=score_matrix,
                topk=topk
            )
            np_candidates = eval_dictionary[candidate_idxs].squeeze()
            dict_candidates = []
            for np_candidate in np_candidates:
                dict_candidates.append({
                    'name': np_candidate[0],
                    'cui': np_candidate[1],
                    'label': check_label(np_candidate[1], golden_cui)
                })
            dict_mentions.append({
                'mention': mention,
                'golden_cui': golden_cui,  # golden_cui can be composite cui
                'candidates': dict_candidates
            })
        queries.append({
            'mentions': dict_mentions
        })

    result = {
        'queries': queries
    }

    return result
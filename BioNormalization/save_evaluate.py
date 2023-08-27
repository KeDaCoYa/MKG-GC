# -*- encoding: utf-8 -*-
"""
@File    :   save_evaluate.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/5/10 10:29   
@Description :   None 

"""

import datetime
import json
import os

import numpy as np
import torch
from ipdb import set_trace
from tqdm import tqdm

from config import MyBertConfig
from src.models.biosyn import BioSyn
from src.models.multi_biosyn import MultiBioSyn
from utils.dataset_utils import load_dictionary, load_queries
from utils.evaluate_utils import evaluate_topk_acc, check_label
from utils.function_utils import get_config, get_logger, set_seed


def predict_topk(biosyn: MultiBioSyn, eval_dictionary, eval_queries, topk, score_mode='hybrid',type_=0):
    """
    开始预测模型
    ----------
    score_mode : str
        hybrid, dense, sparse
    """

    sparse_weight = biosyn.get_sparse_weight().item()  # must be scalar value

    # embed dictionary
    if type_ == 0:
        dict_sparse_embeds = biosyn.get_bc5cdr_disease_sparse_representation(mentions=eval_dictionary[:, 0], verbose=True)
        dict_dense_embeds = biosyn.get_dense_representation(mentions=eval_dictionary[:, 0], verbose=True,type_=type_)
    elif type_ == 1:
        dict_sparse_embeds = biosyn.get_bc5cdr_chemical_sparse_representation(mentions=eval_dictionary[:, 0], verbose=True)
        dict_dense_embeds = biosyn.get_dense_representation(mentions=eval_dictionary[:, 0], verbose=True,type_=type_)
    elif type_ == 2:
        dict_sparse_embeds = biosyn.get_ncbi_disease_sparse_representation(mentions=eval_dictionary[:, 0], verbose=True)
        dict_dense_embeds = biosyn.get_dense_representation(mentions=eval_dictionary[:, 0], verbose=True,type_=type_)
    else:
        raise ValueError

    queries = []
    for eval_query in tqdm(eval_queries, total=len(eval_queries), desc='模型正在预测....'):
        mentions = eval_query[0].replace("+", "|").split("|")
        golden_cui = eval_query[1].replace("+", "|")

        dict_mentions = []
        for mention in mentions:
            if type_ == 0:
                mention_sparse_embeds = biosyn.get_bc5cdr_disease_sparse_representation(mentions=np.array([mention]))
                mention_dense_embeds = biosyn.get_dense_representation(mentions=np.array([mention]),type_=type_)
            elif type_ == 1:
                mention_sparse_embeds = biosyn.get_bc5cdr_disease_sparse_representation(mentions=np.array([mention]))
                mention_dense_embeds = biosyn.get_dense_representation(mentions=np.array([mention]), type_=type_)
            elif type_ == 2:
                mention_sparse_embeds = biosyn.get_ncbi_disease_sparse_representation(mentions=np.array([mention]))
                mention_dense_embeds = biosyn.get_dense_representation(mentions=np.array([mention]), type_=type_)
            else:
                raise ValueError

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
def evaluate(biosyn, eval_dictionary, eval_queries, topk, score_mode='hybrid'):
    """
    predict topk and evaluate accuracy

    Parameters
    ----------
    biosyn : BioSyn
        trained biosyn model
    eval_dictionary : str
        dictionary to evaluate
    eval_queries : str
        queries to evaluate
    topk : int
        the number of topk predictions
    score_mode : 选择
        hybrid, dense, sparse
    Returns
    -------
    result : dict
        accuracy and candidates
    """
    result = predict_topk(biosyn, eval_dictionary, eval_queries, topk, score_mode)
    result = evaluate_topk_acc(result)

    return result

def dev(config:MyBertConfig,logger,ckpt_path=None,biosyn=None,device=None):
    """

    :param config:
    :param logger:
    :param ckpt_path:
    :param model:
    :return:
    """
    config.train_dictionary_path = './dataset/{}/train_dictionary.txt'.format(config.dataset_name)
    config.dev_dictionary_path = './dataset/{}/dev_dictionary.txt'.format(config.dataset_name)
    config.test_dictionary_path = './dataset/{}/test_dictionary.txt'.format(config.dataset_name)
    config.train_dir = './dataset/{}/processed_traindev'.format(config.dataset_name)
    config.dev_dir = './dataset/{}/processed_dev'.format(config.dataset_name)
    config.test_dir = './dataset/{}/processed_test'.format(config.dataset_name)

    # 这个针对predicate的时候的path
    config.dictionary_path = './dataset/{}/train_dictionary.txt'.format(config.dataset_name)

    # 加载验证集/测试集 的字典和数据集
    if config.dataset_name in ['bc5cdr-chemical', 'bc5cdr-disease', 'ncbi-disease']:
        eval_dictionary = load_dictionary(dictionary_path=config.dev_dictionary_path)
        eval_queries = load_queries(data_dir=config.dev_dir, filter_composite=True, filter_duplicate=True,
                                    filter_cuiless=False)
    else:
        eval_dictionary = load_dictionary(dictionary_path=config.dictionary_path)
        eval_queries = load_queries(
            config.dev_path,
            filter_composite=True,
            filter_duplicate=True,
            filter_cuiless=True
        )



    if biosyn is None:
        biosyn = BioSyn(config,device)
        # 加载全部的模型
        biosyn.load_model(model_name_or_path=ckpt_path)

    # 得到预测结果
    result_evalset = evaluate(
        biosyn=biosyn,
        eval_dictionary=eval_dictionary,
        eval_queries=eval_queries,
        topk=config.topk
    )


    logger.info("acc@1={}".format(result_evalset['acc1']))
    logger.info("acc@5={}".format(result_evalset['acc5']))

    if config.save_predictions:
        output_file = os.path.join(config.output_dir, "predictions_eval.json")
        with open(output_file, 'w') as f:
            json.dump(result_evalset, f, indent=2)
    return result_evalset['acc1'],result_evalset['acc5']

def test(config:MyBertConfig,logger,ckpt_path=None,biosyn=None,device=None):
    """

    :param config:
    :param logger:
    :param ckpt_path:
    :param model:
    :return:
    """
    config.train_dictionary_path = './dataset/{}/train_dictionary.txt'.format(config.dataset_name)
    config.dev_dictionary_path = './dataset/{}/dev_dictionary.txt'.format(config.dataset_name)
    config.test_dictionary_path = './dataset/{}/test_dictionary.txt'.format(config.dataset_name)
    config.train_dir = './dataset/{}/processed_traindev'.format(config.dataset_name)
    config.dev_dir = './dataset/{}/processed_dev'.format(config.dataset_name)
    config.test_dir = './dataset/{}/processed_test'.format(config.dataset_name)

    # 这个针对predicate的时候的path
    config.dictionary_path = './dataset/{}/test_dictionary.txt'.format(config.dataset_name)

    # 加载验证集/测试集 的字典和数据集
    if config.dataset_name in ['bc5cdr-chemical', 'bc5cdr-disease', 'ncbi-disease']:
        eval_dictionary = load_dictionary(dictionary_path=config.test_dictionary_path)
        eval_queries = load_queries(data_dir=config.test_dir, filter_composite=True, filter_duplicate=True,
                                    filter_cuiless=False)
    else:
        eval_dictionary = load_dictionary(dictionary_path=config.dictionary_path)
        eval_queries = load_queries(
            config.dev_path,
            filter_composite=True,
            filter_duplicate=True,
            filter_cuiless=True
        )



    if biosyn is None:
        biosyn = BioSyn(config,device)
        # 加载全部的模型
        biosyn.load_model(model_name_or_path=ckpt_path)

    # 得到预测结果
    result_evalset = evaluate(
        biosyn=biosyn,
        eval_dictionary=eval_dictionary,
        eval_queries=eval_queries,
        topk=config.topk
    )


    logger.info("acc@1={}".format(result_evalset['acc1']))
    logger.info("acc@5={}".format(result_evalset['acc5']))

    if config.save_predictions:
        output_file = os.path.join(config.output_dir, "predictions_eval.json")
        with open(output_file, 'w') as f:
            json.dump(result_evalset, f, indent=2)
    return result_evalset['acc1'],result_evalset['acc5']

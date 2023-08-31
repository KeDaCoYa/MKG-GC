# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  这是评估数据集，[测试集,验证集],同时要能够支持边训练，边dev
   Author :        kedaxia
   date：          2022/01/19
   Copyright:      (c) kedaxia 2022
-------------------------------------------------
   Change Activity:
                   2022/01/19: 
-------------------------------------------------
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
from src.models.four_multi_biosyn import MultiBioSynFour
from src.models.multi_biosyn import MultiBioSyn
from utils.dataset_utils import load_dictionary, load_queries, load_my_data
from utils.evaluate_utils import predict_topk, evaluate_topk_acc, check_label
from utils.function_utils import get_config, get_logger, set_seed

def multi_predict_topk(biosyn:MultiBioSynFour, eval_dictionary, eval_queries, topk, score_mode='hybrid',type_=0):
    """
    开始预测模型
    ----------
    score_mode : str
        hybrid, dense, sparse
    """
    bc5cdr_disease_sparse_weight, bc5cdr_chem_sparse_weight, ncbi_disease_sparse_weight,bc2gm_sparse_weight = biosyn.get_sparse_weight()
    if type_ == 0:
        sparse_weight = bc5cdr_disease_sparse_weight.item()  # must be scalar value
    elif type_ == 1:
        sparse_weight = bc5cdr_chem_sparse_weight.item()  # must be scalar value
    elif type_ == 2:
        sparse_weight = ncbi_disease_sparse_weight.item()  # must be scalar value
    elif type_ == 3:
        sparse_weight = bc2gm_sparse_weight.item()  # must be scalar value

    # embed dictionary

    if type_ == 0:
        dict_sparse_embeds = biosyn.get_bc5cdr_disease_sparse_representation(mentions=eval_dictionary[:, 0], verbose=True)
    elif type_ == 1:
        dict_sparse_embeds = biosyn.get_bc5cdr_chemical_sparse_representation(mentions=eval_dictionary[:, 0],
                                                                             verbose=True)
    elif type_ == 2:
        dict_sparse_embeds = biosyn.get_ncbi_disease_sparse_representation(mentions=eval_dictionary[:, 0],
                                                                              verbose=True)
    elif type_ == 3:
        dict_sparse_embeds = biosyn.get_bc2gm_sparse_representation(mentions=eval_dictionary[:, 0],verbose=True)

    dict_dense_embeds = biosyn.get_dense_representation(mentions=eval_dictionary[:, 0], verbose=True,type_=type_)


    if type_ == 0:
        desc = "bc5cdr-disease数据集正在评估中..."
    elif type_== 1:
        desc = "bc5cdr-chemical数据集正在评估中..."
    elif type_ == 2:
        desc = "ncbi_disease数据集正在评估中..."
    elif type_ == 3:
        desc = "bc2gm数据集正在评估中..."

    queries = []
    for eval_query in tqdm(eval_queries, total=len(eval_queries),desc=desc):
        mentions = eval_query[0].replace("+", "|").split("|")
        golden_cui = eval_query[1].replace("+", "|")

        dict_mentions = []
        for mention in mentions:
            if type_ == 0:
                mention_sparse_embeds = biosyn.get_bc5cdr_disease_sparse_representation(mentions=np.array([mention]))
            elif type_ == 1:
                mention_sparse_embeds = biosyn.get_bc5cdr_chemical_sparse_representation(mentions=np.array([mention]))
            elif type_ == 2:
                mention_sparse_embeds = biosyn.get_ncbi_disease_sparse_representation(mentions=np.array([mention]))
            elif type_ == 3:
                mention_sparse_embeds = biosyn.get_bc2gm_sparse_representation(mentions=np.array([mention]))


            mention_dense_embeds = biosyn.get_dense_representation(mentions=np.array([mention]),type_=type_)

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
def evaluate(biosyn, eval_dictionary, eval_queries, topk, score_mode='hybrid',type_=0):
    """

    """
    result = multi_predict_topk(biosyn, eval_dictionary, eval_queries, topk, score_mode,type_=type_)
    result = evaluate_topk_acc(result)

    return result

def test(config:MyBertConfig,logger,ckpt_path=None,biosyn=None,device=None,wandb=None,epoch=None,global_step=None):
    '''

    :param config:
    :param logger:
    :param ckpt_path:
    :param model:
    :return:
    '''
    # 加载验证集/测试集 的字典和数据集
    bc5cdr_disease_eval_dictionary = load_dictionary(config.bc5cdr_disease_test_dictionary_path)
    bc5cdr_disease_eval_queries = load_my_data(config.bc5cdr_disease_test_path)

    bc5cdr_chemical_eval_dictionary = load_dictionary(config.bc5cdr_chemical_test_dictionary_path)
    bc5cdr_chemical_eval_queries = load_my_data(config.bc5cdr_chemical_test_path)

    ncbi_disease_eval_dictionary = load_dictionary(config.ncbi_disease_test_dictionary_path)
    ncbi_disease_chemical_eval_queries = load_my_data(config.ncbi_disease_test_path)

    bc2gm_eval_dictionary = load_dictionary(config.bc2gm_train_dictionary_path)
    bc2gm_eval_queries = load_my_data(config.bc2gm_test_path)

    if biosyn is None:
        biosyn = MultiBioSynFour(config,device)
        # 加载全部的模型
        biosyn.load_model(model_name_or_path=ckpt_path)

    # 得到预测结果
    bc5cdr_disease_result_evalset = evaluate(
        biosyn=biosyn,
        eval_dictionary=bc5cdr_disease_eval_dictionary,
        eval_queries=bc5cdr_disease_eval_queries,
        topk=config.topk,type_=0
    )
    bc5cdr_chemical_result_evalset = evaluate(
        biosyn=biosyn,
        eval_dictionary=bc5cdr_chemical_eval_dictionary,
        eval_queries=bc5cdr_chemical_eval_queries,
        topk=config.topk,type_=1
    )

    ncbi_disease_result_evalset = evaluate(
        biosyn=biosyn,
        eval_dictionary=ncbi_disease_eval_dictionary,
        eval_queries=ncbi_disease_chemical_eval_queries,
        topk=config.topk, type_=2
    )

    bc2gm_result_evalset = evaluate(
        biosyn=biosyn,
        eval_dictionary=bc2gm_eval_dictionary,
        eval_queries=bc2gm_eval_queries,
        topk=config.topk, type_=3
    )

    logger.info('-------test-bc5cdr-disease----------')
    logger.info("acc@1={}".format(bc5cdr_disease_result_evalset['acc1']))
    logger.info("acc@5={}".format(bc5cdr_disease_result_evalset['acc5']))

    logger.info('-------test-bc5cdr-chemical----------')
    logger.info("acc@1={}".format(bc5cdr_chemical_result_evalset['acc1']))
    logger.info("acc@5={}".format(bc5cdr_chemical_result_evalset['acc5']))

    logger.info('-------test-ncbi-disease----------')
    logger.info("acc@1={}".format(ncbi_disease_result_evalset['acc1']))
    logger.info("acc@5={}".format(ncbi_disease_result_evalset['acc5']))

    logger.info('-------test-bc2gm----------')
    logger.info("acc@1={}".format(bc2gm_result_evalset['acc1']))
    logger.info("acc@5={}".format(bc2gm_result_evalset['acc5']))

    if config.use_wandb:
        wandb.log(
            {"test-epoch": epoch,
             'test-bc5cdr-disease-hit@1': bc5cdr_disease_result_evalset['acc1'],
             'test-bc5cdr-chemical-hit@1': bc5cdr_chemical_result_evalset['acc1'],
             'test-ncbi-disease-hit@1': ncbi_disease_result_evalset['acc1'],
             'test-bc2gm-hit@1': bc2gm_result_evalset['acc1'],
             'test-bc5cdr-disease-hit@5': bc5cdr_disease_result_evalset['acc5'],
             'test-bc5cdr-chemical-hit@5': bc5cdr_chemical_result_evalset['acc5'],
             'test-ncbi_disease-hit@5': ncbi_disease_result_evalset['acc5'],
             'test-bc2gm-hit@5': bc2gm_result_evalset['acc5'],
             }, step=global_step)
if __name__ == '__main__':

    config = get_config()

    logger = get_logger(config)

    # 设置时间
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff
    # 设置随机种子
    set_seed(config.seed)
    #ckpt_path = '/root/code/bioner/knowledgenormalization/BioNormalization/outputs/save_models/checkpoint_1'
    ckpt_path = '/root/code/bioner/knowledgenormalization/BioNormalization/outputs/save_models/checkpoint_2'
    device = torch.device('cuda') if config.use_gpu else torch.device('cpu')
    dev(config,logger,ckpt_path=ckpt_path,device=device)



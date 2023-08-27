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
from src.models.multi_biosyn import MultiBioSyn
from src.models.my_multi_biosyn import MyMultiBioSynModel
from utils.dataset_utils import load_dictionary, load_queries, load_my_data
from utils.evaluate_utils import predict_topk, evaluate_topk_acc, check_label
from utils.function_utils import get_config, get_logger, set_seed

def multi_predict_topk(biosyn:MyMultiBioSynModel,eval_dictionary, eval_queries, topk, score_mode='hybrid',type_=0,dict_sparse_embeds=None, dict_dense_embeds=None):
    """
    开始预测模型
    ----------
    score_mode : str
        hybrid, dense, sparse
    """

    disease_sparse_weight,chemical_drug_sparse_weight,gene_sparse_weight,cell_type_sparse_weight,cell_line_sparse_weight = biosyn.get_sparse_weight()

    if type_ == 0:
        desc = "disease数据集正在评估中..."
        sparse_weight = disease_sparse_weight.item()
    elif type_== 1:
        desc = "chemical数据集正在评估中..."
        sparse_weight = chemical_drug_sparse_weight.item()
    elif type_ == 2:
        desc = "gene_protein数据集正在评估中..."
        sparse_weight = gene_sparse_weight.item()
    elif type_ == 3:
        desc = "cell type数据集正在评估中..."
        sparse_weight = cell_type_sparse_weight.item()

    elif type_ == 4:
        desc = "cell_line数据集正在评估中..."
        sparse_weight = cell_line_sparse_weight.item()
    #
    # elif type_ == 5:
    #     desc = "species数据集正在评估中..."
    #     sparse_weight = disease_sparse_weight

    queries = []
    for eval_query in tqdm(eval_queries, total=len(eval_queries),desc=desc):
        mentions = eval_query[0].replace("+", "|").split("|")
        golden_cui = eval_query[1].replace("+", "|")

        dict_mentions = []
        for mention in mentions:
            if type_ == 0:
                mention_sparse_embeds = biosyn.get_disease_sparse_representation(mentions=np.array([mention]))
            elif type_ == 1:
                mention_sparse_embeds = biosyn.get_chemical_drug_sparse_representation(mentions=np.array([mention]))
            elif type_ == 2:
                mention_sparse_embeds = biosyn.get_gene_protein_sparse_representation(mentions=np.array([mention]))
            elif type_ == 3:
                mention_sparse_embeds = biosyn.get_cell_type_sparse_representation(mentions=np.array([mention]))
            elif type_ == 4:
                mention_sparse_embeds = biosyn.get_cell_line_sparse_representation(mentions=np.array([mention]))


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
def evaluate(biosyn, eval_dictionary, eval_queries, topk, score_mode='hybrid',type_=0,dict_sparse_embeds=None,dict_dense_embeds=None):
    """

    """
    result = multi_predict_topk(biosyn, eval_dictionary, eval_queries, topk, score_mode,type_=type_,dict_sparse_embeds=dict_sparse_embeds,dict_dense_embeds=dict_dense_embeds)
    result = evaluate_topk_acc(result)

    return result
def dev(config:MyBertConfig,logger,ckpt_path=None,biosyn=None,device=None,wandb=None,epoch=None,global_step=None,all_dictionary=None,all_dict_sparse_embeds=None,all_dict_dense_embeds=None):
    '''

    :param config:
    :param logger:
    :param ckpt_path:
    :param model:
    :return:
    '''
    # 加载验证集/测试集 的字典和数据集
    disease_dictionary,chemical_drug_dictionary,gene_dictionary,cell_type_dictionary,cell_line_dictionary=all_dictionary
    disease_sparse_dict_embeds,chemical_drug_sparse_dict_embeds,gene_sparse_dict_embeds,cell_type_sparse_dict_embeds,cell_line_sparse_dict_embeds = all_dict_sparse_embeds
    disease_dense_dict_embeds,chemical_drug_dense_dict_embeds,gene_dense_dict_embeds,cell_type_dense_dict_embeds,cell_line_dense_dict_embeds = all_dict_dense_embeds


    disease_eval_queries = load_my_data(config.mesh_disease_dev_path)
    chemical_eval_queries = load_my_data(config.chemical_drug_dev_path)
    gene_eval_queries = load_my_data(config.gene_protein_dev_path)
    cell_type_eval_queries = load_my_data(config.cell_type_dev_path)
    cell_line_eval_queries = load_my_data(config.cell_line_dev_path)
    # species_eval_queries = load_my_data(config.species_dev_path)


    if biosyn is None:
        biosyn = MultiBioSyn(config,device)
        # 加载全部的模型
        biosyn.load_model(model_name_or_path=ckpt_path)

    # 得到预测结果
    disease_result_evalset = evaluate(
        biosyn=biosyn,
        eval_dictionary=disease_dictionary,
        eval_queries=disease_eval_queries,
        topk=config.topk,
        type_=0,
        dict_sparse_embeds=disease_sparse_dict_embeds,
        dict_dense_embeds=disease_dense_dict_embeds

    )
    chemical_result_evalset = evaluate(
        biosyn=biosyn,
        eval_dictionary=chemical_drug_dictionary,
        eval_queries=chemical_eval_queries,
        topk=config.topk,
        type_=1,
        dict_sparse_embeds=chemical_drug_sparse_dict_embeds,
        dict_dense_embeds=chemical_drug_dense_dict_embeds
    )

    gene_result_evalset = evaluate(
        biosyn=biosyn,
        eval_dictionary=gene_dictionary,
        eval_queries=gene_eval_queries,
        topk=config.topk,
        type_=2,
        dict_sparse_embeds=gene_sparse_dict_embeds,
        dict_dense_embeds=gene_dense_dict_embeds
    )

    cell_type_result_evalset = evaluate(
        biosyn=biosyn,
        eval_dictionary=cell_type_dictionary,
        eval_queries=cell_type_eval_queries,
        topk=config.topk,
        type_=3,
        dict_sparse_embeds=cell_type_sparse_dict_embeds,
        dict_dense_embeds=cell_type_dense_dict_embeds
    )

    cell_line_result_evalset = evaluate(
        biosyn=biosyn,
        eval_dictionary=cell_line_dictionary,
        eval_queries=cell_line_eval_queries,
        topk=config.topk,
        type_=4,
        dict_sparse_embeds=cell_line_sparse_dict_embeds,
        dict_dense_embeds=cell_line_dense_dict_embeds
    )

    # species_result_evalset = evaluate(
    #     biosyn=biosyn,
    #     eval_dictionary=species_dictionary,
    #     eval_queries=species_eval_queries,
    #     topk=config.topk,
    #     type_=5,
    #     dict_sparse_embeds=species_sparse_dict_embeds,
    #     dict_dense_embeds=species_dense_dict_embeds
    # )

    logger.info('-------disease----------')
    logger.info("acc@1={}".format(disease_result_evalset['acc1']))
    logger.info("acc@5={}".format(disease_result_evalset['acc5']))

    logger.info('-------chemical_drug----------')
    logger.info("acc@1={}".format(chemical_result_evalset['acc1']))
    logger.info("acc@5={}".format(chemical_result_evalset['acc5']))

    logger.info('-------gene_protein----------')
    logger.info("acc@1={}".format(gene_result_evalset['acc1']))
    logger.info("acc@5={}".format(gene_result_evalset['acc5']))

    logger.info('-------cell_type----------')
    logger.info("acc@1={}".format(cell_type_result_evalset['acc1']))
    logger.info("acc@5={}".format(cell_type_result_evalset['acc5']))

    logger.info('-------cell_line----------')
    logger.info("acc@1={}".format(cell_line_result_evalset['acc1']))
    logger.info("acc@5={}".format(cell_line_result_evalset['acc5']))

    # logger.info('-------species----------')
    # logger.info("acc@1={}".format(species_result_evalset['acc1']))
    # logger.info("acc@5={}".format(species_result_evalset['acc5']))


    if config.use_wandb:
        wandb.log(
            {"dev-epoch": epoch,
             'dev-disease-hit@1': disease_result_evalset['acc1'],
             'dev-chemical-hit@1': chemical_result_evalset['acc1'],
             'dev-gene-hit@1': gene_result_evalset['acc1'],
             'dev-cell_type-hit@1': cell_type_result_evalset['acc1'],
             'dev-cell_line-hit@1': cell_line_result_evalset['acc1'],

             'dev-disease-hit@5': disease_result_evalset['acc5'],
             'dev-chemical-hit@5': chemical_result_evalset['acc5'],
             'dev-gene-hit@5': gene_result_evalset['acc5'],
             'dev-cell_type-hit@5': cell_type_result_evalset['acc5'],
             'dev-cell_line-hit@5': cell_line_result_evalset['acc5'],

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



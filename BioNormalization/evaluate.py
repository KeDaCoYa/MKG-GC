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

import torch

from config import MyBertConfig
from src.models.biosyn import BioSyn
from utils.dataset_utils import load_dictionary, load_queries
from utils.evaluate_utils import predict_topk, evaluate_topk_acc
from utils.function_utils import get_config, get_logger, set_seed


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
    '''

    :param config:
    :param logger:
    :param ckpt_path:
    :param model:
    :return:
    '''
    # 加载验证集/测试集 的字典和数据集
    eval_dictionary = load_dictionary(dictionary_path=config.dev_dictionary_path)
    eval_queries = load_queries(data_dir=config.dev_dir,filter_composite=True,filter_duplicate=True,filter_cuiless=False)
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



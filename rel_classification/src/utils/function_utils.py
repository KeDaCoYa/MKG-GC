# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2021/12/01
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/01: 
-------------------------------------------------
"""

import os
import random
import datetime
import logging
import time

from collections import defaultdict

import numpy as np
import torch
from ipdb import set_trace
from prettytable import PrettyTable




logger = logging.getLogger("main.function_utils")


def set_cv_config(config,idx):
    config.logs_dir = './outputs/logs/{}/{}/cv5/cv_{}/'.format(config.model_name, config.dataset_name,idx)
    config.tensorboard_dir = './outputs/tensorboard/{}/{}/cv5/cv_{}/'.format(config.model_name, config.dataset_name,idx)
    config.output_dir = './outputs/save_models/{}/{}/cv5/cv_{}/'.format(config.model_name, config.dataset_name,idx)
    # 最后模型对文本预测的结果存放点
    config.predicate_dir = './outputs/predicate_outputs/{}/{}/cv5/cv_{}/'.format(config.model_name, config.dataset_name,idx)


    config.train_labels_path = './{}/{}/mid_dataset/labels.txt'.format(config.dataset_type, config.dataset_name)
    config.train_mtb_path = './{}/{}/cv5/cv_{}/train/mtb_train.txt'.format(config.dataset_type, config.dataset_name,idx)
    config.train_normal_path = './{}/{}/cv5/cv_{}/train/normal_train.txt'.format(config.dataset_type, config.dataset_name,idx)
    

    config.dev_labels_path = config.train_labels_path
    config.dev_mtb_path = './{}/{}/cv5/cv_{}/dev/mtb_dev.txt'.format(config.dataset_type, config.dataset_name,
                                                                           idx)
    config.dev_normal_path = './{}/{}/cv5/cv_{}/dev/normal_dev.txt'.format(config.dataset_type,
                                                                                 config.dataset_name, idx)


    config.relation_labels = config.train_labels_path

def load_model_and_parallel(model, gpu_ids, ckpt_path=None, strict=True, load_type='one2one'):
    """
    加载模型 & 放置到 GPU 中（单卡 / 多卡）
    主要是多卡模型训练和保存的问题...
    load_type:表示加载模型的类别，one2one,one2many,many2one,many2many
    """
    gpu_ids = gpu_ids.split(',')
    device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])

    if load_type == 'one2one':
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            try:

                model.load_state_dict(checkpoint)
            except:

                # embedding_shape = checkpoint['bert_model.embeddings.word_embeddings.weight'].shape
                # model.bert_model.resize_token_embeddings(embedding_shape[0])
                model.load_state_dict(checkpoint)
                logger.warning("发生异常，重新调整word embedding size加载成功")



    elif load_type == 'one2many':
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint)
            gpu_ids = [int(x) for x in gpu_ids]
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)

        else:
            # 如果是在一个卡上训练，多个卡上加载，那么
            gpu_ids = [int(gpu_ids[0])]
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    elif load_type == 'many2one':
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)

            gpu_ids = [int(gpu_ids[0])]
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)

            model.load_state_dict(checkpoint)
        else:
            gpu_ids = [int(x) for x in gpu_ids]

            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            gpu_ids = [int(x) for x in gpu_ids]

            model = torch.nn.DataParallel(model, device_ids=gpu_ids)

            model.load_state_dict(checkpoint)
        else:
            gpu_ids = [int(x) for x in gpu_ids]
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.to(device)
    if ckpt_path:
        logger.info("加载{}成功".format(ckpt_path))
    return model, device

def get_pos_feature(x,limit):
    """
        :param x = idx - entity_idx
        这个方法就是不管len(sentence)多长，都限制到这个位置范围之内

        x的范围就是[-len(sentence),len(sentence)] 转换到都是正值范围
        -limit ~ limit => 0 ~ limit * 2+2
        将范围转换一下，为啥
    """
    if x < -limit:
        return 0
    elif x >= -limit and x <= limit:
        return x + limit + 1
    else:
        return limit * 2 + 2

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def correct_datetime(sec,what):
    beijing_time = datetime.datetime.now() +datetime.timedelta(hours=8)
    return beijing_time.timetuple()

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def get_type_weights(labels):
    '''
    计算每种实体类别对应的权重，方便之后的评估
    :param labels:如果是crf，则只需要根据BIO进行统计
    :return:
    '''
    type_weights = defaultdict(int)
    count = len(labels)
    for label in labels:
        type_weights[label] += 1


    for key,value in type_weights.items():
        type_weights[key] = value/count
    return type_weights

def count_parameters(model):
    '''
    统计参数量，统计需要梯度更新的参数量，并且参数不共享
    :param model:
    :return:
    '''
    requires_grad_nums = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameter_nums = sum(p.numel() for p in model.parameters())
    return requires_grad_nums,parameter_nums

def gentl_print(gentl_li):

    tb = PrettyTable()
    tb.field_names = ['实体类别','f1', 'precision', 'recall']
    for a,f1,p,r in gentl_li:
        tb.add_row([a,round(f1,5),round(p,5),round(r,5)])

    logger.info(tb)


def save_model(config, model, epoch=100,mode='best_model'):
    '''
        无论是多卡保存还是单卡保存，所有的保存都是一样的
    '''
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    if mode == 'best_model':
        output_dir = os.path.join(config.output_dir, 'best_model')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        logger.info('-----将模型保存到 {}------'.format(output_dir))

        torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    else:

        output_dir =  os.path.join(config.output_dir,str(epoch))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)


        logger.info('-----将模型保存到 {}------'.format(output_dir))

        torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))


def load_model(model, ckpt_path=None):
    """
    加载模型 & 放置到 GPU 中（单卡 / 多卡）
    load_type:表示加载模型的类别，one2one,one2many,many2one,many2many
    """

    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint)

    return model

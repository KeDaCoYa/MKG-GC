# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description : 这是模型训练过程中需要的各种trick
                例如 学习率调整器...
   Author :        kedaxia
   date：          2021/11/08
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/11/08:
-------------------------------------------------
"""

import os
import logging

import torch
from ipdb import set_trace

import torch.optim as optim
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger('main.train_utils')

def load_model(model, ckpt_path=None):
    """
    加载模型 & 放置到 GPU 中（单卡 / 多卡）
    load_type:表示加载模型的类别，one2one,one2many,many2one,many2many
    """

    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path)

        try:
            model.load_state_dict(checkpoint)

        except:
            logger.info('这是多GPU训练得到的模型，重新修改')

            from collections import OrderedDict
            new_checkpoint = OrderedDict()
            for key,value in checkpoint.items():

                new_key = ".".join(key.split('.')[1:])
                new_checkpoint[new_key] = value

            model.load_state_dict(new_checkpoint)
    logger.info("加载{}成功".format(ckpt_path))
    return model

def build_optimizer_and_scheduler(config,model,t_toal):
    if config.which_model == 'bert':
        return build_bert_optimizer_and_scheduler(config,model,t_toal)
    else:

        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(config.warmup_proportion * t_toal), num_training_steps=t_toal
        )
        return optimizer,scheduler
def build_bert_optimizer_and_scheduler(config,model,t_toal):
    '''
    使用warmup学习器,这个是用于基于BERT模型的学习器和优化器
    :param config:
    :param model:
    :param t_total:
    :return:
    '''
    module = (
        model.module if hasattr(model, "module") else model
    )

    bert_param_optimizer = []
    other_param_optimizer = []

    # 差分学习率
    no_decay = ['bias', "LayerNorm.weight"]  # 对bias和LayerNorm.weight的学习率进行修改...

    model_param = list(module.named_parameters())  # 所有的模型的参数...

    for name, param in model_param:
        space = name.split('.')
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, param))
        else:
            other_param_optimizer.append((name, param))
    if config.inter_scheme == 12:

        optimizer_grouped_parameters = [
            # bert module
            {
                "params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
                # 这个表示不包含在no_decay的param进行weight decay
                "weight_decay": config.weight_decay,
                'lr': config.learning_rate
            },
            {'params': module.dynamic_weight, 'lr': 0.01, 'weight_decay': 0.0}
            ,

            {
                "params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
                # 这个表示只要是包括no_decay的就不进行weight_decay
                "weight_decay": 0.0,
                'lr': config.learning_rate
            },
            # 除了bert的其他模块，差分学习率
            {
                "params": [p for n, p in other_param_optimizer if n!='dynamic_weight' and not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
                'lr': config.other_lr
            },
            {
                "params": [p for n, p in other_param_optimizer if n!='dynamic_weight' and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                'lr': config.other_lr
            },
        ]
    else:
        optimizer_grouped_parameters = [
            # bert module
            {
                "params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
                # 这个表示不包含在no_decay的param进行weight decay
                "weight_decay": config.weight_decay,
                'lr': config.learning_rate
            },

            {
                "params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
                # 这个表示只要是包括no_decay的就不进行weight_decay
                "weight_decay": 0.0,
                'lr': config.learning_rate
            },
            # 除了bert的其他模块，差分学习率
            {
                "params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
                'lr': config.other_lr
            },
            {
                "params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                'lr': config.other_lr
            },
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon,betas=(0.9,0.999))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(config.warmup_proportion * t_toal), num_training_steps=t_toal
    )

    return optimizer,scheduler


def build_optimizer(config,model):
    '''
        创建optimizer
        这里采用差分学习率的方法，对不同层采用不同的学习率
    '''
    #这里是存储bert的参数
    bert_param_optimizer = []
    # 这里存储其他网络层的参数
    other_param_optimizer = []

    #差分学习率
    no_decay = ['bias','LayerNorm.weight']
    model_pram = list(model.named_parameters())

    for name,param in model_pram:
        space = name.split('.')
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name,param))
        else:
            other_param_optimizer.append((name,param))
    
    optimizer_grouped_parameters = [
        # bert module
        {
            "params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], # 这个表示不包含在no_decay的param进行weight decay
            "weight_decay": config.weight_decay,
            'lr': config.learning_rate
            }, 
        
        {
            "params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],# 这个表示只要是包括no_decay的就不进行weight_decay
            "weight_decay": 0.0,
            'lr': config.learning_rate
            },
        # 除了bert的其他模块，差分学习率
        {
            "params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay, 
            'lr': config.other_lr
            },
        {
            "params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, 
            'lr': config.other_lr
            },
    ]

    optimizer = AdamW(optimizer_grouped_parameters,lr=config.learning_rate,eps=config.adam_epsilon)

    return optimizer




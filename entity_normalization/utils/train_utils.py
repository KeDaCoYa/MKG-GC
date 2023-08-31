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
import os
import logging

import torch
from ipdb import set_trace
from torch import optim
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F

logger = logging.getLogger('main.train_utils')

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

    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(config.warmup_proportion * t_toal), num_training_steps=t_toal
    )

    return optimizer,scheduler

def build_optimizer(config,model):
    if config.use_n_gpu:
        optimizer = optim.Adam([
            {'params': model.module.dense_encoder.parameters()},
            {'params': model.module.sparse_weight, 'lr': 0.01, 'weight_decay': 0},
            ],
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        optimizer = optim.Adam([
            {'params': model.dense_encoder.parameters()},
            {'params': model.sparse_weight, 'lr': 0.01, 'weight_decay': 0}
            ],
            lr=config.learning_rate,
            weight_decay=config.weight_decay)
    return optimizer

def multi_build_optimizer(config,model):
    if config.use_n_gpu:
        optimizer = optim.Adam([
            {'params': model.module.dense_encoder.parameters()},
            {'params': model.module.disease_sparse_weight, 'lr': 0.01, 'weight_decay': 0},
            {'params': model.module.chemical_drug_sparse_weight, 'lr': 0.01, 'weight_decay': 0},
            {'params': model.module.gene_sparse_weight, 'lr': 0.01, 'weight_decay': 0},
            {'params': model.module.cell_type_sparse_weight, 'lr': 0.01, 'weight_decay': 0},
            {'params': model.module.cell_line_sparse_weight, 'lr': 0.01, 'weight_decay': 0},
            ],
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:

        optimizer = optim.Adam([
            {'params': model.dense_encoder.parameters()},
            {'params': model.disease_sparse_weight, 'lr': 0.01, 'weight_decay': 0},
            {'params': model.chemical_drug_sparse_weight, 'lr': 0.01, 'weight_decay': 0},
            {'params': model.gene_sparse_weight, 'lr': 0.01, 'weight_decay': 0},
            {'params': model.cell_type_sparse_weight, 'lr': 0.01, 'weight_decay': 0},
            {'params': model.cell_line_sparse_weight, 'lr': 0.01, 'weight_decay': 0},
            ],
            lr=config.learning_rate,
            weight_decay=config.weight_decay)
    return optimizer

def build_optimizer_and_scheduler(config,model,t_toal):
    '''
    使用warmup学习器,这个是用于基于BERT模型的学习器和优化器
    :param config:
    :param model:
    :param t_total:
    :return:
    '''
    # 这个作用就是当是多GPU训练的模型，则...
    no_decay = ["bias", "LayerNorm.weight"]
    # 这里将参数进行分组，然后分别使用不同的参数进行更新，之后在optimizer.param_groups就会分为了两组
    # 默认是只有一组
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(config.warmup_proportion * t_toal), num_training_steps=t_toal
    )

    return optimizer,scheduler

def marginal_nll(score, target):
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

def load_model_and_parallel(model, gpu_ids, ckpt_path=None, load_type='one2one'):
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

            model.load_state_dict(checkpoint)


    elif load_type == 'one2many':
        # 单卡训练的模型加载到多卡上
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint)
            gpu_ids = [int(x) for x in gpu_ids]
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        else:

            gpu_ids = [int(gpu_ids[0])]
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    elif load_type == 'many2one':
        # 多卡训练的模型加载到单卡上
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
            # set_trace()
            model.load_state_dict(checkpoint)
        else:
            gpu_ids = [int(x) for x in gpu_ids]
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.to(device)

    return model, device




def save_model(config, model, epoch=100,mode='best_model'):
    """
        无论是多卡保存还是单卡保存，所有的保存都是一样的
    """
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    if mode == 'best_model':
        output_dir = os.path.join(config.output_dir,'multi_task_five_dataset', 'best_model')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        logger.info('-----将模型保存到 {}------'.format(output_dir))

        torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    else:

        output_dir = os.path.join(config.output_dir,'multi_task_five_dataset',str(epoch))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)


        logger.info('-----将模型保存到 {}------'.format(output_dir))

        torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))

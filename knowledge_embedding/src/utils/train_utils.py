# -*- encoding: utf-8 -*-
"""
@File    :   train_utils.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/17 20:04   
@Description :   主要是训练的时候相关内容

"""
import datetime
import os
import logging

from ipdb import set_trace

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger("main.train_utils")


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """
    这个方法是对inputs按照length进行对齐
        length长，则补充value
        length短，则删除过长的

    Numpy函数，将序列padding到同一长度
    按照一个batch的最大长度进行padding
    :param inputs:(batch_size,None),每个序列的长度不一样
    :param seq_dim: 表示对哪些维度进行pad，默认为1，只有当对label进行pad的时候，seq_dim=3,因为labels.shape=(batch_size,entity_type,seq_len,seq_len)
        因为一般都是对(batch_size,seq_len)进行pad，，，
    :param length: 这个是设置补充之后的长度，一般为None，根据batch的实际长度进行pad
    :param value:
    :param mode:
    :return:
    """
    pad_length = length
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)  # length=np.array([max_batch_length])
    elif not hasattr(length, '__getitem__'):  # 如果这个length的类别不是列表....,就进行转变

        length = [length]
    # logger.info('这个batch下面的最长长度为{}'.format(length[0]))
    if seq_dims == 3:  # 这个只针对globalpointer的情况

        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        length[1] = pad_length
        length[2] = pad_length
        slices = [np.s_[:length[i]] for i in
                  range(seq_dims)]  # 获得针对针对不同维度的slice，对于seq_dims=0,slice=[None:max_len:None],max_len是seq_dims的最大值
        slices = tuple(slices) if len(slices) > 1 else slices[0]
    else:
        slices = [np.s_[:length[i]] for i in
                  range(seq_dims)]  # 获得针对针对不同维度的slice，对于seq_dims=0,slice=[None:max_len:None],max_len是seq_dims的最大值
        slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]  # 有多少个维数，就需要多少个(0,0),一般是一个

    outputs = []
    for x in inputs:
        # X为一个列表
        # 这里就是截取长度
        x = x[slices]
        for i in range(seq_dims):  # 对不同的维度逐步进行扩充
            if mode == 'post':
                # np.shape(x)[i]是获得当前的实际长度
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)


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
            model.load_state_dict(checkpoint)


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
            # set_trace()
            model.load_state_dict(checkpoint)
        else:
            gpu_ids = [int(x) for x in gpu_ids]
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.to(device)

    return model, device


def save_model(config, model, epoch=100, mode='best_model'):
    '''
        无论是多卡保存还是单卡保存，所有的保存都是一样的
    '''
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    if config.use_n_gpu:
        model = model.module
    if mode == 'best_model':
        output_dir = os.path.join(config.output_dir, 'best_model')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        logger.info('-----将模型保存到 {}------'.format(output_dir))

        torch.save(model.state_dict(), os.path.join(output_dir, '{}.pt'.format(config.model_name)))
    else:

        output_dir = os.path.join(config.output_dir, str(epoch))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        logger.info('-----将模型保存到 {}------'.format(output_dir))

        torch.save(model.state_dict(), os.path.join(output_dir, '{}.pt'.format(config.model_name)))




def build_bert_optimizer_and_scheduler(config, model, t_toal):
    """
    使用warmup学习器,这个是用于基于BERT模型的学习器和优化器
    :param config:
    :param model:
    :param t_total:
    :return:
    """

    # 这里是存储bert的参数
    bert_param_optimizer = []
    # 这里存储其他网络层的参数
    other_param_optimizer = []

    # 差分学习率

    model_pram = list(model.named_parameters())

    for name, param in model_pram:
        space = name.split('.')
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, param))
        else:
            other_param_optimizer.append((name, param))

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=1e-6,betas=(0.9,0.98))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(config.warmup_proportion * t_toal), num_training_steps=t_toal
    )

    return optimizer, scheduler


def build_optimizer(config, model):
    # 这里是存储bert的参数
    bert_param_optimizer = []
    # 这里存储其他网络层的参数
    other_param_optimizer = []

    # 差分学习率

    model_pram = list(model.named_parameters())

    for name, param in model_pram:
        space = name.split('.')
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, param))
        else:
            other_param_optimizer.append((name, param))

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=1e-6, betas=(0.9, 0.98))
    return optimizer


def choose_dataset_dataloader(config):
    pass


def choose_model(config):
    pass


def get_metric_writer(config):
    now = datetime.datetime.now()
    if not os.path.exists(config.tensorboard_dir):
        os.makedirs(config.tensorboard_dir)
    metric_writer = SummaryWriter(os.path.join(config.tensorboard_dir,
                                               "metric_{} {}-{} {}-{}-{}".format(config.model_name, now.month,
                                                                                 now.day,
                                                                                 now.hour, now.minute,
                                                                                 now.second)))
    return metric_writer


def get_parameter_writer(config):
    if not os.path.exists(config.tensorboard_dir):
        os.makedirs(config.tensorboard_dir)
    now = datetime.datetime.now()
    parameter_writer = SummaryWriter(
        os.path.join(config.tensorboard_dir, "parameter_{} {}-{} {}-{}-{}".format(config.model_name, now.month,
                                                                                  now.day,
                                                                                  now.hour, now.minute,
                                                                                  now.second)))
    return parameter_writer

# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  这是所有模型的配置文件
   Author :        kedaxia
   date：          2021/11/08
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/11/08: 今天是个好日期
-------------------------------------------------
"""
import json

from ipdb import set_trace
from transformers import BertConfig




class MyBertConfig(BertConfig):
    def __init__(self, **kwargs):
        '''
        开始使用argparse控制config的参数

        :param model_name:
        :param ner_type:
        '''

        super(MyBertConfig, self).__init__(**kwargs)

        self.dataset_name = kwargs['dataset_name']

        self.use_n_gpu = kwargs['use_n_gpu']  # 这里只在bert模型下使用，其他模型一般不需要...
        self.model_name = kwargs['model_name']

        self.seed = kwargs['seed']

        self.over_fitting_rate = kwargs['over_fitting_rate']  # 这个用于表示训练集的f1和验证集f1之间的差距，如果为1表示不会限制
        self.over_fitting_epoch = kwargs['over_fitting_epoch']
        self.early_stop = kwargs['early_stop']
        self.use_scheduler = kwargs['use_scheduler']

        self.use_gpu = kwargs['use_gpu']
        self.gpu_id = kwargs['gpu_id']

        self.warmup_proportion = kwargs['warmup_proportion']  # 学习率调整
        self.weight_decay = 0.01
        self.ema_decay = 0.999

        self.gradient_accumulation_steps = kwargs['gradient_accumulation_steps']
        self.max_grad_norm = kwargs['max_grad_norm']

        # 这个参数如果为True则
        self.fixed_batch_length = kwargs['fixed_batch_length']  # 这个参数控制batch的长度是否固定

        self.logfile_name = kwargs['logfile_name']
        self.logs_dir = './outputs/logs/{}/{}/'.format(self.model_name, self.dataset_name)
        self.output_dir = './outputs/save_models/{}/{}/'.format(self.model_name, self.dataset_name)
        self.tensorboard_dir = './outputs/tensorboard/{}/{}/tensorboard/'.format(self.model_name, self.dataset_name)
        self.predicate_dir = './outputs/predicate_outputs/{}_{}/{}/'.format(kwargs["bert_name"], self.model_name,
                                                                            self.dataset_name)

        self.data_dir = './dataset/{}/'.format(self.dataset_name)

        # 对数据集进行排序

        self.use_fp16 = kwargs['use_fp16']
        # 评价方式，micro或者macro

        self.num_epochs = kwargs['num_epochs']

        self.use_ema = kwargs['use_ema']

        self.verbose = kwargs['verbose']
        self.metric_verbose = kwargs['metric_verbose']
        self.use_wandb = kwargs['use_wandb']

        self.metric_summary_writer = kwargs['metric_summary_writer']
        self.parameter_summary_writer = kwargs['parameter_summary_writer']
        self.print_step = kwargs['print_step']
        self.save_model = kwargs['save_model']

        self.predicate_flag = False  # 在训练和验证的时候都是False，只有面对无标签的测试集才会打开
        self.debug = kwargs['debug']

        self.bert_dir = kwargs['bert_dir']
        self.bert_name = kwargs['bert_name']

        self.batch_size = kwargs['batch_size']
        self.eval_batch_size = kwargs['eval_batch_size']

        self.max_len = kwargs['max_len']

        self.other_lr = kwargs['other_lr']

        self.adam_epsilon = 1e-8

        self.dropout_prob = kwargs['dropout_prob']  # 设置bert的dropout
        # self.freeze_layers = ['layer.1.', 'layer.3','layer.4', 'layer.5', 'layer.7','layer.8','layer.9']
        # self.freeze_layers = ['layer.2','layer.3','layer.4', 'layer.5', 'layer.6','layer.7','layer.8','layer.9','layer.10']

        # 这是之前的层
        # self.freeze_layers = ['layer.1.', 'layer.3', 'layer.4', 'layer.5', 'layer.7', 'layer.9', 'layer.10']

        # 微调最后四层
        self.freeze_bert = kwargs['freeze_bert']
        if self.bert_name == 'flash_quad':
            self.freeze_layers = ['layer.0', 'layer.1.', 'layer.2', 'layer.3', 'layer.4', 'layer.5', 'layer.6',
                                  'layer.7', 'layer.8' 'layer.9', 'layer.10', 'layer.11']
        else:
            self.freeze_layers = ['layer.0', 'layer.1.', 'layer.2', 'layer.3', 'layer.4', 'layer.5', 'layer.6',
                                  'layer.7']
        self.do_lower_case = True # 将所有的文本进行小写

        self.num_labels = 2
        self.cls_method = kwargs['cls_method']
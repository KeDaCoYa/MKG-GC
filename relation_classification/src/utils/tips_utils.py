# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2021/12/22
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/22: 
-------------------------------------------------
"""

import os
import datetime
import argparse
import logging

from ipdb import set_trace

from config import NormalConfig, BertConfig, MyKebioConfig, MyBertConfig
from src.utils.function_utils import correct_datetime


def get_normal_config():
    parser = argparse.ArgumentParser()


    parser.add_argument('--dataset_name', type=str, help='选择re数据集名称')
    parser.add_argument('--model_name', type=str, default='single_entity_marker',help='给model一个名字')

    parser.add_argument('--gpu_id', type=str, default='-1', help='选择哪块gpu使用，0,1,2...')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10, help='')

    parser.add_argument('--fixed_batch_length', type=bool, default=False, help='对数据集按照顺序进行排序')

    parser.add_argument('--embedding_type', type=str, default='word2vec')
    parser.add_argument('--use_pretrained_embedding', type=bool, default=True)
    parser.add_argument('--attention_mechanism', type=str, default='sa')

    parser.add_argument('--schema', type=int, default=1, help='选择何种entity representation用于关系分类，主要是参考MTB的方式')
    parser.add_argument('--dropout_prob', type=float, default=0.2,help="除了bert之外的dropout 概率")
    parser.add_argument('--other_lr', type=float, default=1e-3)

    parser.add_argument('--lstm_pack_unpack', type=bool, default=False, help='对bilstm的前向传播是否使用此方式')
    parser.add_argument('--num_bilstm_layers', type=int, default=2, help='bilstm的层数')

    parser.add_argument('--use_ema', type=bool, default=False, help='是否使用EMA')
    parser.add_argument('--over_fitting_rate', type=float, default=0.3, help='验证集和训练集的f1差别在多大的时候停止')
    parser.add_argument('--logfile_name', type=str, default='', help='给logfile起个名字')
    parser.add_argument('--over_fitting_epoch', type=int, default=10, help='表示有几个epoch没有超过最大f1则停止')
    parser.add_argument('--train_verbose', type=bool, default=False, help='是否在训练过程中每个batch显示各种值')

    parser.add_argument('--max_len', type=int, default=256, help='序列长度')

    parser.add_argument('--print_step', type=int, default=1, help='多少个step打印当前step的score')
    parser.add_argument('--evaluate_mode', type=str, default='micro', help='对数据集按照顺序进行排序')

    parser.add_argument('--use_parameter_summary_writer', type=bool, default=False, help='是否记录metric参数')
    parser.add_argument('--use_metric_summary_writer', type=bool, default=False, help='是否记录parameter参数')


    args = parser.parse_args()
    gpu_id = args.gpu_id

    model_name = args.model_name
    dataset_name = args.dataset_name
    embedding_type = args.embedding_type
    attention_mechanism = args.attention_mechanism
    use_pretrained_embedding = args.use_pretrained_embedding
    entity_type = args.entity_type
    batch_size = args.batch_size
    seed = args.seed
    dropout_prob = args.dropout_prob
    other_lr = args.other_lr
    use_sort = args.use_sort
    evaluate_mode = args.evaluate_mode
    use_ema = args.use_ema
    over_fitting_rate = args.over_fitting_rate
    fixed_batch_length = args.fixed_batch_length
    logfile_name = args.logfile_name
    over_fitting_epoch = args.over_fitting_epoch
    train_verbose = args.train_verbose

    use_parameter_summary_writer = args.use_parameter_summary_writer
    use_metric_summary_writer = args.use_metric_summary_writer

    lstm_pack_unpack = args.lstm_pack_unpack
    max_len = args.max_len
    num_bilstm_layers = args.num_bilstm_layers
    print_step = args.print_step
    num_epochs = args.num_epochs

    dataset_name = dataset_name.strip()
    embedding_type = embedding_type.strip()
    attention_mechanism = attention_mechanism.strip()

    config = NormalConfig(gpu_ids=gpu_id,model_name=model_name,
                          dataset_name=dataset_name,use_metric_summary_writer=use_metric_summary_writer,
                          embedding_type=embedding_type, attention_mechanism=attention_mechanism,
                          use_pretrained_embedding=use_pretrained_embedding,use_parameter_summary_writer=use_parameter_summary_writer,
                          batch_size=batch_size, seed=seed, dropout_prob=dropout_prob,
                          other_lr=other_lr, use_sort=use_sort, evaluate_mode=evaluate_mode, use_ema=use_ema,
                          over_fitting_rate=over_fitting_rate,
                          logfile_name=logfile_name, fixed_batch_length=fixed_batch_length,
                          over_fitting_epoch=over_fitting_epoch, train_verbose=train_verbose,
                          max_len=max_len, lstm_pack_unpack=lstm_pack_unpack,
                          print_step=print_step, num_epochs=num_epochs, entity_type=entity_type,
                          num_bilstm_layers=num_bilstm_layers,
                          )
    return config


def get_bert_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=1, help='选择哪块gpu使用，0,1,2...')
    parser.add_argument('--dataset_type', type=str, default='general_domain_dataset', help='选择哪个领域的数据集进行处理',)
    parser.add_argument('--dataset_name', type=str, default='semeval2010', help='使用的具体数据集名称')
    parser.add_argument('--run_type', type=str, default='normal', help='选择训练模式', choices=['normal', 'cv5'])
    parser.add_argument('--class_type', type=str, default='other', help='选择数据的关系类别，数据集为AllData的时候才在意',
                        choices=['other', 'single', 'multi'])

    parser.add_argument('--bert_name', type=str, default='r_bert', help='选择使用的模型是')
    parser.add_argument('--bert_dir', type=str, help='预训练模型的存放地址')
    parser.add_argument('--logfile_name', type=str, default='', help='给logfile起个名字')
    parser.add_argument('--model_name', type=str, default='', help='给model一个名字')


    #
    parser.add_argument('--data_format', type=str, default='single', help='设置数据输入格式，single表示单据，cross表示双句子', choices=['single', 'cross','inter'])
    # 这里是通过不同的方式来获取entity representaion，然后用于关系分类...
    # 最后三个是matching the blanks里的方式

    parser.add_argument('--scheme', type=int, default=1, help='采用不同的entity representation表示方式')

    parser.add_argument('--num_epochs', type=int, default=30, help='')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=512, help='最大长度')
    parser.add_argument('--fixed_batch_length', type=bool, default=False, help='动态batch或者根据batch修改')
    parser.add_argument('--num_labels', type=int, default=2, help='二分类或者多分类')
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='scheduler的预测步骤')

    parser.add_argument('--use_n_gpu', type=bool, default=False, help='是否使用多个GPU同时训练...')

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--use_sort', type=bool, default=False, help='对数据集按照顺序进行排序')
    parser.add_argument('--use_gpu', type=bool, default=False, help='')
    parser.add_argument('--use_fp16', type=bool, default=False, help='')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='')
    parser.add_argument('--freeze_bert', type=bool, default=False, help='是否冻结bert层')
    parser.add_argument('--use_scheduler', type=bool, default=False, help='是否使用学习率调整期')

    # all就是使用所有的指标进行评测////
    parser.add_argument('--evaluate_mode', type=str, default='micro', help='performance的度量类别',
                        choices=['micro', 'macro', 'weight', 'all', 'binary'])

    parser.add_argument('--over_fitting_rate', type=float, default=0.15, help='验证集和训练集的f1差别在多大的时候停止')
    parser.add_argument('--over_fitting_epoch', type=int, default=5, help='表示有几个epoch没有超过最大f1则停止')

    parser.add_argument('--dropout_prob', type=float, default=0.2, help='BERT之外的dropout prob, bert使用的由默认的config决定')

    parser.add_argument('--other_lr', type=float, default=2e-4, help='BERT之外的网络学习率')
    parser.add_argument('--bert_lr', type=float, default=2e-5, help='预训练模型的学习率')

    parser.add_argument('--use_ema', type=bool, default=False, help='是否使用EMA')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='裁剪梯度')

    parser.add_argument('--subword_weight_mode', type=str, default='first',
                        help='选择第一个subword作为token representation；或者是平均值', choices=['first', 'avg'])  # 一般就是first word

    parser.add_argument('--train_verbose', type=bool, default=False, help='是否在训练过程中每个batch显示各种值')
    parser.add_argument('--dev_verbose', type=bool, default=True, help='是否在验证过程中每个batch显示各种值')
    parser.add_argument('--use_wandb', type=bool, default=False, help='是否使用wandb存储所有performance')
    parser.add_argument('--save_model', type=bool, default=False, help='是否保存最佳模型....')
    parser.add_argument('--print_step', type=int, default=1, help='打印频次')
    parser.add_argument('--debug', type=bool, default=False, help='debug模式的开启')

    parser.add_argument('--use_parameter_summary_writer', type=bool, default=False, help='是否记录metric参数')
    parser.add_argument('--use_metric_summary_writer', type=bool, default=False, help='是否记录parameter参数')

    args = parser.parse_args()
    gpu_id = args.gpu_id
    dataset_type = args.dataset_type
    class_type = args.class_type
    run_type = args.run_type
    dataset_name = args.dataset_name
    use_gpu = args.use_gpu
    use_n_gpu = args.use_n_gpu
    bert_dir = args.bert_dir
    bert_name = args.bert_name
    save_model = args.save_model
    gradient_accumulation_steps = args.gradient_accumulation_steps
    use_fp16 = args.use_fp16

    model_name = args.model_name
    num_epochs = args.num_epochs
    num_labels = args.num_labels
    batch_size = args.batch_size
    use_sort = args.use_sort
    evaluate_mode = args.evaluate_mode
    use_ema = args.use_ema
    use_scheduler = args.use_scheduler

    use_parameter_summary_writer = args.use_parameter_summary_writer
    use_metric_summary_writer = args.use_metric_summary_writer


    other_lr = args.other_lr
    bert_lr = args.bert_lr
    max_grad_norm = args.max_grad_norm
    warmup_proportion = args.warmup_proportion

    freeze_bert = args.freeze_bert
    seed = args.seed
    over_fitting_rate = args.over_fitting_rate
    dropout_prob = args.dropout_prob
    logfile_name = args.logfile_name

    data_format = args.data_format
    over_fitting_epoch = args.over_fitting_epoch
    fixed_batch_length = args.fixed_batch_length
    train_verbose = args.train_verbose


    max_len = args.max_len
    subword_weight_mode = args.subword_weight_mode
    print_step = args.print_step
    scheme = args.scheme
    use_wandb = args.use_wandb
    debug = args.debug

    dataset_type = dataset_type.strip()
    bert_dir = bert_dir.strip()
    bert_name = bert_name.strip()
    dataset_type = dataset_type.strip()
    dataset_name = dataset_name.strip()
    subword_weight_mode = subword_weight_mode.strip()
    if bert_name == 'kebiolm':

        config = MyKebioConfig(model_name=model_name, gpu_id=gpu_id, dataset_type=dataset_type,
                               dataset_name=dataset_name,
                               num_labels=num_labels,
                               num_epochs=num_epochs, batch_size=batch_size, use_sort=use_sort, bert_name=bert_name,
                               bert_dir=bert_dir, warmup_proportion=warmup_proportion, use_scheduler=use_scheduler,
                               evaluate_mode=evaluate_mode, use_ema=use_ema, over_fitting_rate=over_fitting_rate,
                               seed=seed,
                               use_gpu=use_gpu, debug=debug,
                               other_lr=other_lr, dropout_prob=dropout_prob, logfile_name=logfile_name, bert_lr=bert_lr,
                               max_grad_norm=max_grad_norm, run_type=run_type,
                               fixed_batch_length=fixed_batch_length, over_fitting_epoch=over_fitting_epoch,
                               scheme=scheme,use_parameter_summary_writer=use_parameter_summary_writer,
                               save_model=save_model, class_type=class_type,use_metric_summary_writer=use_metric_summary_writer,
                               train_verbose=train_verbose, max_len=max_len,
                               use_n_gpu=use_n_gpu, use_wandb=use_wandb,
                               subword_weight_mode=subword_weight_mode, print_step=print_step, data_format=data_format,
                               freeze_bert=freeze_bert)
    else:
        config = MyBertConfig(model_name=model_name, gpu_id=gpu_id, dataset_type=dataset_type,
                              dataset_name=dataset_name,
                              num_labels=num_labels, class_type=class_type,
                              num_epochs=num_epochs, batch_size=batch_size, use_sort=use_sort, bert_name=bert_name,
                              bert_dir=bert_dir, warmup_proportion=warmup_proportion, use_scheduler=use_scheduler,
                              evaluate_mode=evaluate_mode, use_ema=use_ema, over_fitting_rate=over_fitting_rate,
                              seed=seed,use_fp16=use_fp16,gradient_accumulation_steps=gradient_accumulation_steps,
                              use_gpu=use_gpu, debug=debug,
                              other_lr=other_lr, dropout_prob=dropout_prob, logfile_name=logfile_name, bert_lr=bert_lr,
                              max_grad_norm=max_grad_norm, run_type=run_type,
                              fixed_batch_length=fixed_batch_length, over_fitting_epoch=over_fitting_epoch,
                              scheme=scheme,use_parameter_summary_writer=use_parameter_summary_writer,
                              save_model=save_model,use_metric_summary_writer=use_metric_summary_writer,
                              train_verbose=train_verbose, max_len=max_len,
                              use_n_gpu=use_n_gpu, use_wandb=use_wandb,
                              subword_weight_mode=subword_weight_mode, print_step=print_step, data_format=data_format,
                              freeze_bert=freeze_bert)

    return config


def get_logger(config):
    logger = logging.getLogger('main')
    logging.Formatter.converter = correct_datetime

    logger.setLevel(level=logging.INFO)

    if not os.path.exists(config.logs_dir):
        os.makedirs(config.logs_dir)

    now = datetime.datetime.now() + datetime.timedelta(hours=8)
    year, month, day, hour, minute, secondas = now.year, now.month, now.day, now.hour, now.minute, now.second
    handler = logging.FileHandler(os.path.join(config.logs_dir,
                                               '{} {}_{}_{} {}:{}:{}.txt'.format(config.logfile_name, year, month, day,
                                                                                 hour,
                                                                                 minute, secondas)))

    handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def print_hyperparameters(config, logger):
    hyper_parameters = vars(config)

    for key, value in hyper_parameters.items():

        if key == 'bert_dir':
            value = value.split('/')[-1]
            logger.info('预训练模型：{}'.format(value))
        elif key == 'use_pretrained_embedding':
            if value:
                logger.info('预训练的词嵌入{}'.format(config.embedding_type))
        elif key == 'model_name':
            logger.info('模型名称:{}'.format(value))
        elif key == 'evaluate_mode':
            logger.info('performance评价方式:{}'.format(value))
        elif key == 'seed':
            logger.info('随机种子:{}'.format(value))
        elif key == 'batch_size':
            logger.info('batch_size:{}'.format(value))
        elif key == 'logs_dir':
            logger.info('日志保存路径:{}'.format(value))
        elif key == 'tensorboard_dir':
            logger.info('tensorboard的存储文件在:{}'.format(value))
        elif key == 'output_dir':
            logger.info('预训练模型的保存地址:{}'.format(value))
        elif 'lstm' in config.model_name:
            if key == 'num_bilstm_layers':
                logger.info('BiLSTM的层数为{}'.format(value))
            elif key == 'lstm_pack_unpack':
                if value:
                    logger.info('这里BilSTM的计算采用pad_pack方式')
        elif key == 'use_gpu':
            if value:
                logger.info('显卡使用的:{}'.format((config.gpu_id)))
        elif key == 'use_n_gpu':
            if value:
                logger.info('使用多卡训练模型')
        elif key == 'fixed_batch_length':
            if value:
                logger.info("sequence的最大长度：{}".format(config.max_len))
            else:
                if config.use_sort:
                    logger.info('sequence长度：{}'.format(config.max_len))
                else:
                    logger.info('sequence最大长度：{}'.format(config.max_len))
        elif key == 'other_lr':

            logger.info('BERT的学习率:{}'.format(config.bert_lr))
            logger.info('其他网络的学习率:{}'.format(value))
        elif key == 'freeze_bert':
            if value:
                logger.info('冻结的bert层为:{}'.format(config.freeze_layers))
        elif key == 'attention_mechanism' and 'att' in config.model_name:
            if value == 'sa':
                logger.info('注意力机制：Self-Attention')
            if value == 'mha':
                logger.info('注意力机制：Multi-Head Attention')
        elif key == 'use_ema':
            if value:
                logger.info('使用滑动加权平均模型')
        elif key == 'scheme':
            logger.info('scheme:{}'.format(value))
            if value == 1:
                logger.info('entity representation: [CLS]+[s1]ent1[e1]+[s2]ent2[e2]')
            elif value == 2:
                logger.info('entity representation: [CLS]+[s1]+[e1]+[s2]+[e2]')
            elif value == 3:
                logger.info('entity representation: [CLS]+[s1]+[s2]')
            elif value == 4:
                logger.info('entity representation:[s1]+s[2]')
            elif value == 5:
                logger.info('entity representation: [CLS]')
            elif value == 6:
                logger.info('entity representation: [s1]ent1[e1]+[s2]ent2[e2]')
            elif value == 7:
                logger.info('entity representation: ent1+ent2')
            elif value == 8:
                logger.info('entity representation: [CLS]+ent1+ent2')


def show_log(logger, idx, len_dataloader, t_total, epoch, global_step, loss, p, r, f1, acc, evaluate_mode, type_='train',
             scheme=0):
    if scheme == 0:
        if type_ == 'train':
            logger.info(
                '训练集训练中...:  Epoch {} | Step:{}/{}|{}/{}'.format(epoch, idx, len_dataloader, global_step, t_total))
        elif type_ == 'dev':
            logger.info(
                '验证集评估中...:  Epoch {} | Step:{}/{}|{}/{}'.format(epoch, idx, len_dataloader, global_step, t_total))
        else:
            logger.info(
                '测试集评估中...:  Epoch {} | Step:{}/{}|{}/{}'.format(epoch, idx, len_dataloader, global_step, t_total))

    else:
        if type_ == 'train':
            logger.info('********Epoch {} [训练集完成]********'.format(epoch))
        elif type_ == 'dev':
            logger.info('********Epoch {} [验证集完成]********'.format(epoch))
        else:
            logger.info('********Epoch {} [测试集完成]********'.format(epoch))

    logger.info('---------------{}--------------'.format(evaluate_mode))
    logger.info('Loss:{:.5f}'.format(loss))
    logger.info('Accuracy:{:.5f}'.format(acc))

    logger.info('Precision:{:.5f}'.format(p))
    logger.info('Recall:{:.5f}'.format(r))
    logger.info('F1:{:.5f}'.format(f1))


def wandb_log(wandb, epoch, global_step, f1, p, r, acc, loss, type_,evaluate_mode, **kwargs):
    f1_key = '{}_{}_f1'.format(type_,evaluate_mode)
    p_key = '{}_{}_p'.format(type_,evaluate_mode)
    r_key = '{}_{}_r'.format(type_,evaluate_mode)
    acc_key = '{}_acc'.format(type_)
    epoch_key = '{}_epoch'.format(type_)
    loss_type = '{}_loss'.format(type_)
    lr = 'lr'
    if type_ == 'train':
        wandb.log({epoch_key: epoch, f1_key: f1, p_key: p, r_key: r, loss_type: loss, acc_key: acc,
                   lr: kwargs['learning_rate']}, step=global_step)
    else:
        wandb.log({epoch_key: epoch, f1_key: f1, p_key: p, r_key: r, loss_type: loss, acc_key: acc}, step=global_step)

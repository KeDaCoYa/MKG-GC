# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2022/01/13
   Copyright:      (c) kedaxia 2022
-------------------------------------------------
   Change Activity:
                   2022/01/13: 
-------------------------------------------------
"""

import argparse
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
from torch.utils.data import DataLoader


logger = logging.getLogger("main.function_utils")


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
                                                                                 hour, minute, secondas)))

    handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def get_config():
    from config import MyBertConfig
    parser = argparse.ArgumentParser()



    parser.add_argument('--dataset_dir', type=str, default='', help='数据集所在的文件夹')
    parser.add_argument('--dataset_name', type=str, default='', help='数据集名称')



    parser.add_argument('--bert_name', type=str, default='scibert', help='这是使用的哪个预训练模型')
    parser.add_argument('--bert_dir', type=str, default='', help='预训练模型的路径')


    parser.add_argument('--model_name', type=str, default='biosyn', help='正式的模型名称，非常标准的名称，用于之后的程序运行')
    parser.add_argument('--task_name', type=str, default='没有名字的task', help='给这次的任务取个名字')

    parser.add_argument('--gpu_id', type=str, default='1', help='选择哪块gpu使用，0,1,2...')
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--use_fp16', type=bool, default=False)
    parser.add_argument('--use_n_gpu', type=bool, default=False, help='是否使用多个GPU同时训练...')

    parser.add_argument('--freeze_bert', type=bool, default=False, help='是否冻结部分参数...')
    parser.add_argument('--use_scheduler', type=bool, default=False, help='是否使用学习器...')
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='...')


    parser.add_argument('--use_amp', type=bool, default=False,help='混合精度加速模型训练')

    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_len', type=int, default=25, help='每个entity mention的最大长度为25')

    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--dropout_prob', type=float, default=0.1, help='BERT使用的dropout')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪...')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='梯度累加次数...')

    parser.add_argument('--task_encoder_nums', type=int, default=4, help='对于每个task_specific encoder的个数')



    parser.add_argument('--over_fitting_rate', type=float, default=0.15, help='验证集和训练集的f1差别在多大的时候停止')
    parser.add_argument('--over_fitting_epoch', type=int, default=5, help='表示有几个epoch没有超过最大f1则停止')
    parser.add_argument('--early_stop', type=bool, default=False, help='采用早停机制,防止过拟合')

    parser.add_argument('--use_metric_summary_writer', type=bool, default=False, help='是否使用SummaryWriter记录参数')
    parser.add_argument('--use_parameter_summary_writer', type=bool, default=False, help='是否使用SummaryWriter记录参数')

    parser.add_argument('--logfile_name', type=str, default='', help='给logfile起个名字')
    parser.add_argument('--save_model', type=bool, default=False, help='是否保存最佳模型....')
    parser.add_argument('--save_predictions', type=bool, default=False, help='是否保存预测的结果....')
    parser.add_argument('--print_step', type=int, default=1, help='打印频次')
    parser.add_argument('--verbose', type=bool, default=True, help='是否在训练过程中每个batch显示各种值')
    parser.add_argument('--use_wandb', type=bool, default=False, help='是否使用wandb来记录训练结果')
    parser.add_argument('--debug', type=bool, default=False, help='debug模式则只会使用一小部分的数据进行训练')

# ---------- 针对BioSyn的超参数 -------------------
    parser.add_argument('--dense_ratio', type=float,default=0.5)
    parser.add_argument('--topk', type=int,default=20)
    parser.add_argument('--encoder_type', type=str,default='bert')

    # 针对SapBERT的超参数
    parser.add_argument('--type_of_triplets', type=str, choices=['all','hard','easy'],default='all')
    parser.add_argument('--miner_margin', type=float,default=0.2)
    parser.add_argument('--agg_mode', type=str,default='cls')
    parser.add_argument('--loss', type=str,default='ms_loss',help='选择哪个损失函数')
    parser.add_argument('--use_miner', type=bool,default=True,help='是否使用miner来产生pairwise')
    parser.add_argument('--pairwise', type=bool,default=True,help='if loading pairwise formatted datasets')






    args = parser.parse_args()


    bert_name = args.bert_name
    bert_dir = args.bert_dir
    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name
    task_name = args.task_name
    gradient_accumulation_steps = args.gradient_accumulation_steps

    model_name = args.model_name
    seed = args.seed

    logfile_name = args.logfile_name
    task_encoder_nums = args.task_encoder_nums
    warmup_proportion = args.warmup_proportion

    save_model = args.save_model
    save_predictions = args.save_predictions

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    learning_rate = args.learning_rate
    use_scheduler = args.use_scheduler
    max_len = args.max_len

    gpu_id = args.gpu_id
    use_gpu = args.use_gpu
    use_amp = args.use_amp
    use_n_gpu = args.use_n_gpu

    dropout_prob = args.dropout_prob
    max_grad_norm = args.max_grad_norm

    over_fitting_rate = args.over_fitting_rate
    over_fitting_epoch = args.over_fitting_epoch
    early_stop = args.early_stop


    use_metric_summary_writer = args.use_metric_summary_writer
    use_parameter_summary_writer = args.use_parameter_summary_writer
    print_step = args.print_step
    verbose = args.verbose
    use_wandb = args.use_wandb
    debug = args.debug


    # ---------针对bert model--------------
    freeze_bert = args.freeze_bert
    bert_dir = bert_dir.strip()
    bert_name = bert_name.strip()
    # 针对BioNormalization的超参数
    dense_ratio = args.dense_ratio
    topk = args.topk

    # 针对SapBERT的超参数
    type_of_triplets = args.type_of_triplets
    miner_margin = args.miner_margin
    agg_mode = args.agg_mode
    loss = args.loss
    use_miner = args.use_miner
    pairwise = args.pairwise

    encoder_type = args.encoder_type
    use_fp16 = args.use_fp16



    config = MyBertConfig(model_name=model_name, gpu_id=gpu_id, num_epochs=num_epochs, batch_size=batch_size,
                  use_amp=use_amp, early_stop=early_stop,task_name=task_name,dataset_name=dataset_name,dataset_dir=dataset_dir,
                  gradient_accumulation_steps=gradient_accumulation_steps,dense_ratio=dense_ratio,type_of_triplets=type_of_triplets,
                  over_fitting_rate=over_fitting_rate,learning_rate=learning_rate,loss=loss,miner_margin=miner_margin,
                  save_model=save_model,topk=topk,use_miner=use_miner,agg_mode=agg_mode,
                  seed=seed, dropout_prob=dropout_prob, logfile_name=logfile_name,warmup_proportion=warmup_proportion,
                  over_fitting_epoch=over_fitting_epoch,save_predictions=save_predictions,
                  max_grad_norm=max_grad_norm,pairwise=pairwise,use_scheduler=use_scheduler,
                  use_metric_summary_writer=use_metric_summary_writer,task_encoder_nums=task_encoder_nums,
                  verbose=verbose, use_parameter_summary_writer=use_parameter_summary_writer,
                  max_len=max_len,debug=debug,use_fp16=use_fp16,
                  use_wandb=use_wandb,weight_decay=weight_decay,
                  print_step=print_step, use_n_gpu=use_n_gpu,
                  bert_dir=bert_dir,
                  bert_name=bert_name,encoder_type=encoder_type,
                  use_gpu=use_gpu,
                  freeze_bert=freeze_bert)

    return config


def print_hyperparameters(config):
    hyper_parameters = vars(config)
    logger.info('-------此次任务的名称为:{}---------'.format(config.task_name))
    for key, value in hyper_parameters.items():

        if key == 'bert_dir':
            logger.info('预训练模型：{}'.format(value))
        elif key == 'use_pretrained_embedding':
            if value:
                logger.info('预训练的词嵌入{}'.format(config.embedding_type))
        elif key == 'model_name':
            logger.info('模型名称:{}'.format(value))
        elif key == 'evaluate_mode':
            logger.info('评价标准:{}'.format(value))
        elif key == 'seed':
            logger.info('随机种子:{}'.format(value))
        elif key == 'batch_size':
            logger.info('batch_size:{}'.format(value))
        elif key == 'logs_dir':
            logger.info('日志保存路径:{}'.format(value))
        elif key == 'tensorboard_dir':
            logger.info('tensorboard的存储文件在:{}'.format(value))
        elif key == 'output_dir':
            logger.info('模型的保存地址:{}'.format(value))
        elif key == 'num_bilstm_layers':
            logger.info('BiLSTM的层数为{}'.format(value))
        elif key == 'use_fp16':
            if value:
                logger.info('使用fp16加速模型训练...')
        elif key == 'lstm_pack_unpack':
            if value:
                logger.info('这里BilSTM的计算采用pad_pack方式')
        elif key == 'use_gpu':
            if value:
                logger.info('显卡使用的:{}'.format(torch.cuda.get_device_name(int(config.gpu_id))))
                logger.info('显卡使用的:{}'.format((config.gpu_id)))

        elif key == 'lr':

            logger.info('BERT的学习率:{}'.format(value))
            logger.info('其他网络的学习率:{}'.format(config.other_lr))

        elif key == 'attention_mechanism' and 'att' in config.model_name:
            if value == 'sa':
                logger.info('注意力机制：Self-Attention')
            if value == 'mha':
                logger.info('注意力机制：Multi-Head Attention')
        elif key == 'use_ema':
            if value:
                logger.info('使用滑动加权平均模型')

        elif key == 'freeze_bert':
            if value:
                logger.info('冻结BERT层:{}'.format(config.freeze_layers))
        # else:
        #     logger.info("{}:{}".format(key,value))


# 初始参数设置

def choose_model(config, type='train'):
    '''
    根据输入的条件，选择合适的模型
    '''
    pass


def choose_dataset_and_loader(config, train_data, train_labels, tokenizer=None, word2id=None, type='train'):
    pass


def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed)  # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def correct_datetime(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def count_parameters(model):
    '''
    统计参数量，统计需要梯度更新的参数量，并且参数不共享
    :param model:
    :return:
    '''
    requires_grad_nums = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameter_nums = sum(p.numel() for p in model.parameters())
    return requires_grad_nums,parameter_nums



def gentl_print(gentl_li, average='micro'):
    tb = PrettyTable()
    tb.field_names = ['实体类别', '{}-f1'.format(average), '{}-precision'.format(average), '{}-recall'.format(average),
                      'support']
    for a, f1, p, r, support in gentl_li:
        tb.add_row([a, round(f1, 5), round(p, 5), round(r, 5), support])

    logger.info(tb)


def final_print_score(train_res_li, dev_res_li):
    for i in range(len(train_res_li)):
        logger.info('epoch{} 训练集 f1:{:.5f},p:{:.5f},r:{:.5f}'.format(i + 1, train_res_li[i]['f1'],
                                                                     train_res_li[i]['p'],
                                                                     train_res_li[i]['r']))
        logger.info('        验证集 f1:{:.5f},p:{:.5f},r:{:.5f}'.format(dev_res_li[i]['f1'],
                                                                     dev_res_li[i]['p'],
                                                                     dev_res_li[i]['r']))


def get_type_weights(labels):
    '''
    计算每种实体类别对应的权重，方便之后的评估
    :param labels:如果是crf，则只需要根据BIO进行统计
    :return:
    '''
    type_weights = defaultdict(int)
    count = 0
    for label in labels:
        for word in label:
            if word != 'O':
                BIO_format, entity_type = word.split('-')
                if BIO_format == 'B':
                    type_weights[entity_type] += 1
                    count += 1
    for key, value in type_weights.items():
        type_weights[key] = value / count
    return type_weights


def argsort(seq):
    '''
    这里seq传入的是一个列表，其值为列表的长度，但是都是负数
    '''
    # 这里也可以修改为如下代码
    res = sorted(range(len(seq)), key=lambda x: seq[x])
    return sorted(range(len(seq)), key=seq.__getitem__)


def argsort_sequences_by_lens(list_in):
    '''
    这个函数是对list_in按照序列长度进行排序,这个list_in的值就是序列的长度
    sort_indices为从大到小，对应句子index的list
    '''
    data_num = len(list_in)
    # 这里得到的就是从大到小的indice
    sort_indices = argsort([-len(item) for item in list_in])
    reverse_sort_indices = [-1 for _ in range(data_num)]
    for i in range(data_num):
        reverse_sort_indices[sort_indices[i]] = i
    # 这个reverse_sort_indices就是sort_indices的排序，正好相反
    return sort_indices, reverse_sort_indices



def load_model(model, ckpt_path=None):
    """
    加载模型 & 放置到 GPU 中（单卡 / 多卡）
    load_type:表示加载模型的类别，one2one,one2many,many2one,many2many
    """

    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint)

    return model


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
    if mode == 'best_model':
        output_dir = os.path.join(config.output_dir, 'best_model')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        logger.info('-----将模型保存到 {}------'.format(output_dir))

        torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    else:

        output_dir = os.path.join(config.output_dir, str(epoch))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        logger.info('-----将模型保存到 {}------'.format(output_dir))

        torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))


def save_metric_writer(metric_writer, loss, global_step, p, r, f1, type='Training'):
    '''
    这个主要是记录模型的performance
    '''
    metric_writer.add_scalar("{}/loss".format(type), loss, global_step)
    metric_writer.add_scalar("{}/precision".format(type), p, global_step)
    metric_writer.add_scalar("{}/recall".format(type), r, global_step)
    metric_writer.add_scalar("{}/f1".format(type), f1, global_step)


def save_parameter_writer(model_writer, model, step):
    '''
    这个是记录模型的参数、梯度等信息
    '''
    for name, param in model.named_parameters():
        model_writer.add_histogram('model_param_' + name, param.clone().cpu().data.numpy(), step)
        if param.grad is not None:
            model_writer.add_histogram('model_grad_' + name, param.grad.clone().cpu().numpy(), step)


def show_log(logger, idx, len_dataloader, t_total, epoch, global_step, loss,mlm_loss,nsp_loss, mlm_acc,nsp_acc, type='train'):
    if type == 'train':
        logger.info('训练集训练中...:  Epoch {} | Step:{}/{}|{}/{}'.format(epoch, idx, len_dataloader, global_step, t_total))
    else:
        logger.info('验证集评估中...:  Epoch {} | Step:{}/{}|{}/{}'.format(epoch, idx, len_dataloader, global_step, t_total))
    logger.info('Loss:{:.5f}'.format(loss))
    logger.info(' mlm loss:{:.5f}'.format(mlm_loss))
    logger.info(' nsp loss:{:.5f}'.format(nsp_loss))

    logger.info('mlm accuracy:{:.5f}'.format(mlm_acc))
    logger.info('nsp accuracy:{:.5f}'.format(nsp_acc))




def print_hyperparameters(config):
    hyper_parameters = vars(config)
    logger.info('-------此次任务的名称为:{}---------'.format(config.task_name))
    for key, value in hyper_parameters.items():

        if key == 'bert_dir':
            logger.info('预训练模型：{}'.format(value))

        elif key == 'model_name':
            logger.info('模型名称:{}'.format(value))

        elif key == 'seed':
            logger.info('随机种子:{}'.format(value))
        elif key == 'batch_size':
            logger.info('batch_size:{}'.format(value))
        elif key == 'logs_dir':
            logger.info('日志保存路径:{}'.format(value))
        elif key == 'tensorboard_dir':

            logger.info('tensorboard的存储文件在:{}'.format(value))
        elif key == 'output_dir':
            logger.info('模型的保存地址:{}'.format(value))

        elif key == 'use_fp16':
            if value:
                logger.info('使用fp16加速模型训练...')

        elif key == 'use_gpu':
            if value:
                logger.info('显卡使用的:{}'.format(torch.cuda.get_device_name(int(config.gpu_id))))
                logger.info('显卡使用的:{}'.format((config.gpu_id)))
        elif key == 'max_len':
            logger.info("固定batch data的最大长度为：{}".format(config.max_len))
        elif key == 'lr':
            logger.info('BERT的学习率:{}'.format(value))

        elif key == 'freeze_bert':
            if value:
                logger.info('冻结BERT层:{}'.format(config.freeze_layers))
        elif key == 'encoder_type':

            logger.info("多任务使用的encoder为{},共{}层".format(config.encoder_type,config.task_encoder_nums))
        # else:
        #     logger.info("{}:{}".format(key,value))

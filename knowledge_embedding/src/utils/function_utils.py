# -*- encoding: utf-8 -*-
"""
@File    :   function_utils.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/17 20:04   
@Description :   None 

"""
import argparse
import os
import random
import datetime
import logging


import numpy as np
import json, pickle
import os
import csv
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from ipdb import set_trace
from prettytable import PrettyTable
from torch.utils.data import DataLoader

from config import MyBertConfig

logger = logging.getLogger("main.function_utils")


def correct_datetime(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False, help='是否开启debug模式，debug模式使用少量数据运行')

    parser.add_argument('--dataset_name', type=str, default='umls', help='选择合适的数据集')

    parser.add_argument('--bert_name', type=str, default='biobert', help='这是使用的哪个预训练模型')
    parser.add_argument('--bert_dir', type=str, default='', help='预训练模型的路径')
    parser.add_argument('--model_name', type=str, default='', help='正式的模型名称，非常标准的名称')

    parser.add_argument('--gpu_id', type=str, default='1', help='选择哪块gpu使用，0,1,2...')
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--use_n_gpu', type=bool, default=False, help='是否使用多个GPU同时训练...')
    parser.add_argument('--use_fp16', type=bool, default=False)

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--max_len', type=int, default=256, help='最大长度')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--fixed_batch_length', type=bool, default=False, help='动态batch或者根据batch修改')
    parser.add_argument('--dropout_prob', type=float, default=0.1, help='BERT使用的dropout')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="学习率的设置")
    parser.add_argument('--other_lr', type=float, default=1e-4, help='BERT之外的网络学习率')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪...')

    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='warm_lienar的学习率调整期')
    parser.add_argument('--freeze_bert', type=bool, default=False, help='是否冻结bert的部分层数')
    parser.add_argument('--use_scheduler', type=bool, default=False, help='是否使用学习率调整期')

    parser.add_argument('--use_ema', type=bool, default=False, help='是否使用EMA')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='梯度累加次数...')

    parser.add_argument('--over_fitting_rate', type=float, default=0.15, help='验证集和训练集的f1差别在多大的时候停止')
    parser.add_argument('--over_fitting_epoch', type=int, default=5, help='表示有几个epoch没有超过最大f1则停止')
    parser.add_argument('--early_stop', type=bool, default=False, help='采用早停机制,防止过拟合')

    parser.add_argument('--metric_summary_writer', type=bool, default=False, help='是否使用SummaryWriter记录参数')
    parser.add_argument('--parameter_summary_writer', type=bool, default=False, help='是否使用SummaryWriter记录参数')

    parser.add_argument('--logfile_name', type=str, default='', help='给logfile起个名字')
    parser.add_argument('--save_model', type=bool, default=False, help='是否保存最佳模型....')
    parser.add_argument('--print_step', type=int, default=1, help='打印频次')
    parser.add_argument('--verbose', type=bool, default=True, help='是否在训练过程中每个batch显示各种值')
    parser.add_argument('--use_wandb', type=bool, default=False, help='是否使用wandb来记录训练结果')
    parser.add_argument('--metric_verbose', type=bool, default=False, help='是否验证集查看详细的评价指标，开启之后会非常慢')
    parser.add_argument('--cls_method', type=str, default='cls')

    args = parser.parse_args()

    dataset_name = args.dataset_name
    debug = args.debug

    bert_name = args.bert_name
    bert_dir = args.bert_dir

    model_name = args.model_name
    seed = args.seed

    logfile_name = args.logfile_name
    save_model = args.save_model
    cls_method = args.cls_method

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    max_len = args.max_len

    gpu_id = args.gpu_id
    use_gpu = args.use_gpu
    use_fp16 = args.use_fp16
    use_n_gpu = args.use_n_gpu

    dropout_prob = args.dropout_prob
    gradient_accumulation_steps = args.gradient_accumulation_steps
    max_grad_norm = args.max_grad_norm
    warmup_proportion = args.warmup_proportion
    use_ema = args.use_ema
    use_scheduler = args.use_scheduler

    over_fitting_rate = args.over_fitting_rate
    over_fitting_epoch = args.over_fitting_epoch
    early_stop = args.early_stop
    fixed_batch_length = args.fixed_batch_length

    metric_summary_writer = args.metric_summary_writer
    parameter_summary_writer = args.parameter_summary_writer
    print_step = args.print_step

    learning_rate = args.learning_rate

    verbose = args.verbose
    use_wandb = args.use_wandb
    metric_verbose = args.metric_verbose

    # ---------针对bert model--------------
    freeze_bert = args.freeze_bert
    other_lr = args.other_lr

    dataset_name = dataset_name.strip()

    bert_dir = bert_dir.strip()
    bert_name = bert_name.strip()

    config = MyBertConfig(model_name=model_name, gpu_id=gpu_id,
                          dataset_name=dataset_name, num_epochs=num_epochs, batch_size=batch_size,eval_batch_size=eval_batch_size,
                          use_fp16=use_fp16, use_gpu=use_gpu,cls_method=cls_method,
                          early_stop=early_stop,metric_verbose=metric_verbose,
                          gradient_accumulation_steps=gradient_accumulation_steps,
                          use_ema=use_ema, over_fitting_rate=over_fitting_rate,
                          save_model=save_model, debug=debug, learning_rate=learning_rate,
                          seed=seed, other_lr=other_lr, dropout_prob=dropout_prob, logfile_name=logfile_name,
                          fixed_batch_length=fixed_batch_length, over_fitting_epoch=over_fitting_epoch,
                          max_grad_norm=max_grad_norm, metric_summary_writer=metric_summary_writer,
                          verbose=verbose, parameter_summary_writer=parameter_summary_writer, max_len=max_len,
                          use_wandb=use_wandb, use_scheduler=use_scheduler, warmup_proportion=warmup_proportion,
                          print_step=print_step, use_n_gpu=use_n_gpu,
                          bert_dir=bert_dir, bert_name=bert_name,
                          freeze_bert=freeze_bert)

    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_parameter_writer(model_writer, model, step):
    """
    这个是记录模型的参数、梯度等信息
    """
    for name, param in model.named_parameters():
        model_writer.add_histogram('model_param_' + name, param.clone().cpu().data.numpy(), step)
        if param.grad is not None:
            model_writer.add_histogram('model_grad_' + name, param.grad.clone().cpu().numpy(), step)


def print_hyperparameters(config):
    hyper_parameters = vars(config)

    for key, value in hyper_parameters.items():
        if key == 'dataset_name':
            logger.info('数据集：{}'.format(value))
        elif key == 'bert_dir':
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
        elif key == 'fixed_batch_length':
            if value:
                logger.info("固定batch data的最大长度为：{}".format(config.max_len))
            else:
                logger.info('动态batch data长度，并不固定')
        elif key == 'lr':

            logger.info('BERT的学习率:{}'.format(value))
            logger.info('其他网络的学习率:{}'.format(config.other_lr))

        elif key == 'use_ema':
            if value:
                logger.info('使用滑动加权平均模型')

        elif key == 'freeze_bert':
            if value:
                logger.info('冻结BERT层:{}'.format(config.freeze_layers))
        # else:
        #     logger.info("{}:{}".format(key,value))


# 初始参数设置

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
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def dir_exists(path):
    return os.path.exists(path) and os.path.isdir(path)

def file_exists(path):
    return os.path.exists(path) and os.path.isfile(path)


def show_log(epo_idx, num_epochs, cur_step, lens, t_total, global_step, type='train', scheme=0, metric=None,verbose=False):
    """

    :param epo_idx:
    :param num_epochs:
    :param cur_step:
    :param lens: 表示一个trainloader的数量
    :param t_total:
    :param global_step:
    :param loss:
    :param type:
    :param scheme: 0表示是训练过程中的参数展示，1表示是训练完成之后的展示
    :param metric:
    :return:
    """
    if scheme == 0:
        if type == 'train':
            logger.info(
                '训练集训练中...:  Epoch {}/{} | Step:{}/{}|{}/{}'.format(epo_idx, num_epochs, cur_step, lens, global_step,
                                                                    t_total))
            logger.info('Loss:{:.5f}'.format(metric['loss'] if 'loss' in metric else 0.))
#            logger.info('Acc:{:.5f}'.format(metric['acc']))



        else:
            logger.info('验证集评估中...:  Epoch {}'.format(epo_idx))
            #logger.info('Acc:{:.5f}'.format(metric['acc']))
            logger.info('Loss:{:.5f}'.format(metric['loss'] if 'loss' in metric else 0.))
            if verbose:

                logger.info('Hit@1:{:.5f}'.format(metric['hit@1']))
                logger.info('Hit@3:{:.5f}'.format(metric['hit@3']))
                logger.info('Hit@10:{:.5f}'.format(metric['hit@10']))
                logger.info('MR:{:.5f}'.format(metric['MR']))
                logger.info('MRR:{:.5f}'.format(metric['MRR']))


    else:
        if type == 'train':
            logger.info('********Epoch {} [训练完成]********'.format(epo_idx))
            logger.info('Loss:{:.5f}'.format(metric['loss']))
            #logger.info('Acc:{:.5f}'.format(metric['acc']))
        else:
            logger.info('********Epoch {} [验证完成]********'.format(epo_idx))

            logger.info('Loss:{:.5f}'.format(metric['loss'] if 'loss' in metric else 0.))
            #logger.info('Acc:{:.5f}'.format(metric['acc']))
            if verbose:
                logger.info('Hit@1:{:.5f}'.format(metric['hit@1']))
                logger.info('Hit@3:{:.5f}'.format(metric['hit@3']))
                logger.info('Hit@10:{:.5f}'.format(metric['hit@10']))
                logger.info('MR:{:.5f}'.format(metric['MR']))
                logger.info('MRR:{:.5f}'.format(metric['MRR']))



def wandb_log(wandb, epoch, global_step, type='train',verbose=False, metric=None):
    if type == 'train':
        lr = metric['lr'] if 'lr' in metric else 0
        wandb.log(
            {"train-epoch": epoch, "train_loss": metric['loss'], 'lr': lr},step=global_step)
    else:
        if verbose:
            wandb.log(
                {
                    "dev-epoch": epoch,
                    "dev_loss": metric['loss'] if 'loss' in metric else 0.,
                    'dev-Hit@1': metric['hit@1'],
                    'dev-Hit@3': metric['hit@3'],
                    'Hit@10': metric['hit@10'],
                    'dev-MR': metric['MR'],
                    'dev-MRR': metric['MRR'],
                    'dev-Loss': metric['loss'] if 'loss' in metric else 0.,

                 },
                step=global_step)
        else:
            wandb.log(
                {
                    "dev-epoch": epoch,
                    "dev_loss":  metric['loss'] if 'loss' in metric else 0.,
                    "dev_acc":  metric['Acc'] if 'Acc' in metric else 0.,

                },
                step=global_step)


StAR_FILE_PATH = None  # Your own file path

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


# def select_field(features, field):
#     return [
#         [
#             choice[field]
#             for choice in feature.choices_features
#         ]
#         for feature in features
#     ]

def save_json(obj, path):

    file_dir = "/".join(path.split('/')[:-1])

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp)


def load_json(input_file):
    with open(input_file, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data


def load_jsonl(input_file):
    data_list = []
    with open(input_file, "r", encoding="utf-8") as fp:
        for line in fp:
            data_list.append(json.loads(line))
    return data_list


def save_jsonl_with_offset(obj, path):
    assert isinstance(obj, list)
    offset_list = []
    with open(path, "w", encoding="utf-8") as fp:
        for _elem in obj:
            offset_list.append(fp.tell())
            dump_str = json.dumps(_elem) + os.linesep
            fp.write(dump_str)
    assert len(obj) == len(offset_list)
    return offset_list


def load_jsonl_with_offset(offset, path):
    with open(path, encoding="utf-8") as fp:
        fp.seek(offset)
        return json.loads(fp.readline())


def save_pkll_with_offset(obj, path):
    assert isinstance(obj, list)
    offset_list = []
    with open(path, "wb") as fp:
        for _elem in obj:
            offset_list.append(fp.tell())
            fp.write(pickle.dumps(_elem))
        last_offset = fp.tell()
    assert len(obj) == len(offset_list)
    pair_offset_list = []
    for _idx in range(len(offset_list)):
        if _idx < len(offset_list) - 1:
            pair_offset_list.append([offset_list[_idx], offset_list[_idx+1] - offset_list[_idx]])
        else:
            pair_offset_list.append([offset_list[_idx], last_offset - offset_list[_idx]])
    return pair_offset_list


def load_pkll_with_offset(offset, path):
    assert len(offset) == 2  # 1. seek 2. for size
    with open(path, "rb") as fp:
        fp.seek(offset[0])
        line = fp.read(offset[1])
        return pickle.loads(line)


def load_pickle(path):
    with open(path, "rb") as fp:
        data = pickle.load(fp)
        return data


def save_pickle(obj, path):
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def save_list_to_file(str_list, file_path, use_basename=False):
    with open(file_path, "w", encoding="utf-8") as fp:
        for path_str in str_list:
            fp.write(os.path.basename(path_str) if use_basename else path_str)
            fp.write(os.linesep)


def load_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines


def load_list_from_file(file_path):
    data = []
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as fp:
            for line in fp:
                data.append(line.strip())
    return data


def get_data_path_list(data_dir, suffix=None):  # , recursive=False
    path_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            temp = os.path.join(root, file)
            if isinstance(suffix, str) and not temp.endswith(suffix):
                continue
            path_list.append(temp)
    return path_list


def file_exists(path):
    return os.path.exists(path) and os.path.isfile(path)


def dir_exists(path):
    return os.path.exists(path) and os.path.isdir(path)


# =============
# ==STR Part ==
def is_lower(text):
    return all(ord("a") <= ord(c) <= ord("z") for c in text)


def is_capital(text):
    return all(ord("A") <= ord(c) <= ord("Z") for c in text)


def is_word(text):
    return all(ord("A") <= ord(c) <= ord("Z") or ord("a") <= ord(c) <= ord("z") for c in text)


def get_val_str_from_dict(val_dict):
    # sort
    sorted_list = list(sorted(val_dict.items(), key=lambda item: item[0]))
    str_return = ""
    for key, val in sorted_list:
        if len(str_return) > 0:
            str_return += ", "
        str_return += "%s: %.4f" % (key, val)
    return str_return


def parse_span_str(_span_str, min_val=0, max_val=10000):

    if isinstance(_span_str, (int, float)):
        return int(_span_str), max_val
    elif isinstance(_span_str, type(None)):
        return None
    elif not isinstance(_span_str, str):
        return min_val, max_val
    lst = _span_str.split(",")
    if len(lst) == 0:
        return min_val, max_val
    elif len(lst) == 1:
        if len(lst[0]) > 0:
            return int(lst[0]), max_val
        else:
            return min_val, max_val
    elif len(lst) == 2:
        _minv, _maxv  = min_val, max_val
        if len(lst[0]) > 0:
            _minv = int(lst[0])
        if len(lst[1]) > 0:
            _maxv = int(lst[1])
        return _minv, _maxv
    else:
        raise AttributeError("got invalid {} as {}".format(_span_str, _span_str.split(",")))


# statistics
def get_statistics_for_num_list(num_list):
    res = {}
    arr = np.array(num_list, dtype="float32")
    res["mean"] = np.mean(arr)
    res["median"] = np.median(arr)
    res["min"] = np.min(arr)
    res["max"] = np.max(arr)
    res["std"] = np.std(arr)
    res["len"] = len(num_list)
    return res



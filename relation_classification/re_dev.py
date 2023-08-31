# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  这是使用标准数据集进行单独评估,
   Author :        kedaxia
   date：          2021/12/03
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/03: 
-------------------------------------------------
"""

import os
import datetime
import time
import random
from collections import defaultdict
import copy
import logging

from ipdb import set_trace
from tqdm import tqdm

import wandb

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, precision_score, recall_score, \
    classification_report, confusion_matrix

import torch
from transformers import BertTokenizer

from src.dataset_utils.data_process_utils import read_data, get_label2id

from src.utils.function_utils import correct_datetime, set_seed, save_model, load_model_and_parallel
from src.utils.tips_utils import get_bert_config, get_logger, print_hyperparameters, wandb_log, show_log
from src.utils.train_utils import build_optimizer_and_scheduler, relation_classification_decode, batch_to_device, \
    set_tokenize_special_tag, choose_dataloader, choose_model


def dev(model, config, tokenizer, label2id, device, ckpt_path=None, epoch=None, global_step=None, logger=None,
        type_='dev'):
    if model is None:
        model = choose_model(config)
        model, device = load_model_and_parallel(model, config.gpu_ids, ckpt_path=ckpt_path, strict=True,
                                                load_type='one2one')

    examples = read_data(config, type_=type_)
    if config.debug:
        examples = examples[:config.batch_size * 2]
    dev_dataloader = choose_dataloader(config, examples, label2id, tokenizer, device)

    batch_loss = 0.
    batch_dev_f1 = 0.
    batch_dev_p = 0.
    batch_dev_acc = 0.
    batch_dev_r = 0.

    all_dev_labels = []
    all_dev_tokens = []

    model.eval()
    with torch.no_grad():
        for step, batch_data in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader),
                                     desc="{}数据集正在进行评估".format(type_)):
            if config.data_format == 'single':

                input_ids, token_type_ids, attention_masks, e1_mask, e2_mask, labels = batch_data

                loss, logits = model(input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask)
                predicate_token = relation_classification_decode(logits)

            elif config.data_format == 'cross':
                input_ids, token_type_ids, attention_masks, e1_mask, e2_mask, labels = batch_data

                loss, logits = model(input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask)
                predicate_token = relation_classification_decode(logits)


            else:
                raise ValueError
            # if config.use_n_gpu:
            loss = loss.mean()
            labels = labels.cpu().numpy()

            all_labels = [id for _, id in label2id.items()]

            all_dev_labels.extend(labels)
            all_dev_tokens.extend(predicate_token)


            p_r_f1_s = precision_recall_fscore_support(labels, predicate_token, labels=all_labels,
                                                           average=config.evaluate_mode)

            tmp_dev_acc = accuracy_score(labels, predicate_token)
            tmp_dev_p = p_r_f1_s[0]
            tmp_dev_r = p_r_f1_s[1]
            tmp_dev_f1 = p_r_f1_s[2]
            batch_dev_f1 += tmp_dev_f1
            batch_dev_p += tmp_dev_p
            batch_dev_acc += tmp_dev_acc
            batch_dev_r += tmp_dev_r
            batch_loss += loss.item()
            if config.use_wandb:
                wandb_log(wandb, epoch, global_step, tmp_dev_f1, tmp_dev_p, tmp_dev_r, tmp_dev_acc, 0.,
                          evaluate_mode=config.evaluate_mode, type_=type_)

    count = len(dev_dataloader)
    dev_loss = batch_loss / count

    p_r_f1_s = precision_recall_fscore_support(all_dev_labels, all_dev_tokens, labels=all_labels,
                                               average=config.evaluate_mode)

    dev_p = p_r_f1_s[0]
    dev_r = p_r_f1_s[1]
    dev_f1 = p_r_f1_s[2]
    dev_acc = accuracy_score(all_dev_labels, all_dev_tokens)

    show_log(logger, -1, 0, 0, epoch, global_step, dev_loss, dev_p, dev_r, dev_f1, 0.00, config.evaluate_mode,
             type_=type_, scheme=1)
    if config.use_wandb:
        wandb_log(wandb, epoch, global_step, dev_f1, dev_p, dev_r, dev_acc, loss.item(),
                  evaluate_mode=config.evaluate_mode, type_=type_)
    reports = classification_report(all_dev_labels, all_dev_tokens, labels=all_labels, digits=4)

    logger.info("-----------验证集epoch:{} 报告-----------".format(epoch))
    logger.info(reports)
    return dev_p, dev_r, dev_f1


if __name__ == '__main__':
    config = get_bert_config()
    # 设置时间
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff

    set_seed(config.seed)

    # 测试wandb

    # project表示这次项目，entity:表示提交人，config为超参数

    logger = get_logger(config)
    if config.use_wandb:
        wandb.init(project="关系分类-{}".format(config.dataset_name), entity="kedaxia", config=vars(config))
    logger.info('----------------本次模型运行的参数--------------------')
    print_hyperparameters(config, logger)
    ckpt_path = ''
    device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(config.bert_dir)

    set_tokenize_special_tag(config, tokenizer)

    label2id, id2label = get_label2id(config.relation_labels)
    model = choose_model(config)
    dev(model, config, tokenizer, label2id, device, ckpt_path=ckpt_path, epoch=0, global_step=0, logger=logger,
        type_='dev')

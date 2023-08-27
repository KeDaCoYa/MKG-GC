# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
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
import copy

from ipdb import set_trace
import wandb
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, precision_score, recall_score, \
    classification_report, confusion_matrix
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BertTokenizer

from re_dev import dev
from src.dataset_utils.data_process_utils import read_data, get_label2id
from src.dataset_utils.multi_dataset import read_multi_data
from src.utils.function_utils import set_seed, save_model, load_model_and_parallel, set_cv_config
from src.utils.tips_utils import get_bert_config, get_logger, print_hyperparameters, wandb_log, show_log
from src.utils.train_utils import build_optimizer_and_scheduler, relation_classification_decode, batch_to_device, \
    set_tokenize_special_tag, choose_model, choose_dataloader, build_optimizer


def dev(model,config,tokenizer,label2id,device,ckpt_path=None,epoch=None,global_step=None,logger=None,type_='dev'):


    if model is None:

        model = choose_model(config)
        model,device = load_model_and_parallel(model, config.gpu_ids, ckpt_path=ckpt_path, strict=True, load_type='one2one')

    examples = read_data(config, type_=type_)

    if config.debug:
        examples = examples[:config.batch_size * 3]

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
        for step, batch_data in tqdm(enumerate(dev_dataloader),desc="{}数据集正在进行评估".format(type_)):
            input_ids1, token_type_ids1, attention_masks1, input_ids2, token_type_ids2, attention_masks2, e1_mask, e2_mask, labels = batch_data
            with torch.no_grad():
                loss,logits = model(input_ids1, token_type_ids1, attention_masks1, input_ids2, token_type_ids2,attention_masks2, e1_mask, e2_mask, labels)



            labels = labels.cpu().numpy()
            predicate_token = relation_classification_decode(logits)
            all_labels = [id for _, id in label2id.items()]
            all_dev_labels.extend(labels)
            all_dev_tokens.extend(predicate_token)

            if config.num_labels == 2:

                p_r_f1_s = precision_recall_fscore_support(labels, predicate_token, labels=all_labels,average='binary')
            else:
                p_r_f1_s = precision_recall_fscore_support(labels, predicate_token,labels=all_labels,average=config.evaluate_mode)
            tmp_dev_acc = accuracy_score(labels,predicate_token)
            tmp_dev_p = p_r_f1_s[0]
            tmp_dev_r = p_r_f1_s[1]
            tmp_dev_f1 = p_r_f1_s[2]
            batch_dev_f1 += tmp_dev_f1
            batch_dev_p += tmp_dev_p
            batch_dev_acc += tmp_dev_acc
            batch_dev_r += tmp_dev_r

            loss = loss.mean()
            batch_loss += loss.item()
            if config.use_wandb:
                wandb_log(wandb, epoch, global_step,  tmp_dev_f1, tmp_dev_p, tmp_dev_r, tmp_dev_acc, 0., type_=type_)


    count = len(dev_dataloader)
    dev_loss = batch_loss / count
    dev_p = batch_dev_p / count
    dev_r = batch_dev_r / count
    dev_f1 = batch_dev_f1 / count
    dev_acc = batch_dev_acc / count

    show_log(logger, -1, 0, 0, epoch, global_step, dev_loss, dev_p, dev_r, dev_f1,0.00, config.evaluate_mode, type='dev', scheme=1)
    if config.use_wandb:
        wandb_log(wandb, epoch, global_step, dev_f1, dev_p, dev_r, dev_acc, loss.item(), type_=type_)
    reports = classification_report(all_dev_labels, all_dev_tokens, labels=all_labels,digits=4)
    logger.info("-----------验证集epoch:{} 报告-----------".format(epoch))
    logger.info(reports)
    return   dev_p,dev_r,dev_f1



if __name__ == '__main__':
    config = get_bert_config()

    # 设置时间
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff

    set_seed(config.seed)
    logger = get_logger(config)
    # 测试wandb

    # project表示这次项目，entity:表示提交人，config为超参数
    if config.run_type == 'normal':
        ckpt_path = ''

        if config.use_wandb:
            wandb_name = f'{config.bert_name}_{config.model_name}_bs{config.batch_size}_schema{config.scheme}_maxlen{config.max_len}'
            wandb.init(project="关系分类-{}".format(config.dataset_name), config=vars(config),
                       name=wandb_name)
        logger.info('----------------本次模型运行的参数--------------------')
        print_hyperparameters(config, logger)

        train(config, logger)
    elif config.run_type == 'cv5':

        for i in range(1, 6):
            logger.info('-----------CV:{}-----------'.format(i))
            ckpt_path = ''
            set_cv_config(config, i)

            if config.use_wandb:
                wandb_name = f'{config.bert_name}_{config.bert_name}_{config.model_name}_bs{config.batch_size}_schema{config.scheme}_maxlen{config.max_len}_cv_{i}'
                wandb.init(project="关系分类-{}".format(config.dataset_name), entity="kedaxia", config=vars(config),
                           name=wandb_name)
            logger.info('----------------本次模型运行的参数--------------------')

            train(config, logger)
            print_hyperparameters(config, logger)
    else:
        raise ValueError

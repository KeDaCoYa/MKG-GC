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
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from sklearn.metrics import precision_recall_fscore_support,accuracy_score,f1_score,precision_score,recall_score,classification_report,confusion_matrix

import torch
from transformers import BertTokenizer

from src.dataset_utils.data_process_utils import read_data, get_label2id
from src.dataset_utils.entity_marker import MultiMTBDataset
from src.dataset_utils.entity_type_marker import MultiNormalDataset
from src.dataset_utils.multi_dataset import read_multi_data
from src.models.multi_mtb_bert import MultiMtbBertForBC6, MultiMtbBertForBC7
from src.models.multi_entitymarker_model import *

from src.utils.function_utils import correct_datetime, set_seed, save_model, load_model_and_parallel
from src.utils.tips_utils import get_bert_config, get_logger, print_hyperparameters, wandb_log, show_log
from src.utils.train_utils import build_optimizer_and_scheduler, relation_classification_decode, batch_to_device, \
    set_tokenize_special_tag, choose_dataloader, choose_model, error_analysis


def dev(model,config,tokenizer,label2id,device,ckpt_path=None,epoch=None,global_step=None,logger=None,type_='dev'):


    if model is None:
        model = choose_model(config)
        model,device = load_model_and_parallel(model, config.gpu_ids, ckpt_path=ckpt_path, strict=True, load_type='one2one')

    examples = read_multi_data(config,type_=type_)

    if config.debug:
        examples = examples[:config.batch_size * 50]
    if config.data_format == 'single':  # 这个针对sentence-level的关系分类
        dev_dataset = MultiNormalDataset(examples, config=config, label2id=label2id, tokenizer=tokenizer,
                                           device=device)
    elif config.data_format == 'cross':
        dev_dataset = MultiMTBDataset(examples, config=config, label2id=label2id, tokenizer=tokenizer, device=device)
    else:
        raise ValueError

    dev_dataloader = DataLoader(dataset=dev_dataset, shuffle=True, collate_fn=dev_dataset.collate_fn,
                                  num_workers=0, batch_size=config.batch_size)

    batch_loss = 0.
    batch_dev_f1 = 0.
    batch_dev_p = 0.
    batch_dev_acc = 0.
    batch_dev_r = 0.



    all_dev_labels = []
    all_predicate_tokens = []
    binary_all_dev_labels = []
    binary_all_predicate_tokens = []

    dev_raw_text = []
    model.eval()
    with torch.no_grad():
        for step, batch_data in tqdm(enumerate(dev_dataloader),total=len(dev_dataloader),desc="{}数据集正在进行评估".format(type_)):
            if config.data_format == 'cross':

                input_ids, token_type_ids, attention_masks, e1_mask, e2_mask, labels,rel_type = batch_data

                loss, logits = model(input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask,rel_type)
                predicate_token = relation_classification_decode(logits)

            elif config.data_format == 'single':
                input_ids, token_type_ids, attention_masks, e1_mask, e2_mask, labels,rel_type = batch_data

                loss, logits = model(input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask,rel_type)
                predicate_token = relation_classification_decode(logits)


            else:
                raise ValueError
            #if config.use_n_gpu:
            loss = loss.mean()
            input_ids = input_ids.cpu().numpy().tolist()
            for idx in range(len(batch_data[0])):
                end_idx = input_ids[idx].index(102)
                dev_raw_text.append(tokenizer.decode(input_ids[idx][1:end_idx]))



            binary_labels = (labels > 0).long()
            binary_labels = binary_labels.cpu().numpy()
            multi_labels = labels.cpu().numpy()
            all_labels = [id for _, id in label2id.items()]

            # 分类二分类和三分类的结果
            binary_all_dev_labels.extend(binary_labels)
            all_dev_labels.extend(multi_labels)

            new_predicate_token = []
            rel_type = rel_type.cpu().numpy()
            for idx, pred in enumerate(predicate_token):
                if pred:
                    new_predicate_token.append(rel_type[idx])
                else:
                    new_predicate_token.append(0)

            binary_all_predicate_tokens.extend(predicate_token)
            all_predicate_tokens.extend(new_predicate_token)

            # if config.num_labels == 2:
            #     tmp_dev_acc = accuracy_score(binary_labels, predicate_token)
            #     p_r_f1_s = precision_recall_fscore_support(binary_labels, predicate_token, average='binary')
            # else:
                # p_r_f1_s_ = precision_recall_fscore_support(labels, predicate_token, labels=all_labels,
                #                                            average=config.evaluate_mode)
            tmp_dev_acc = accuracy_score(multi_labels, new_predicate_token)
            # all_labels = all_labels[1:]
            p_r_f1_s = precision_recall_fscore_support(multi_labels, new_predicate_token, labels=all_labels,
                                                       average=config.evaluate_mode)

            tmp_dev_p = p_r_f1_s[0]
            tmp_dev_r = p_r_f1_s[1]
            tmp_dev_f1 = p_r_f1_s[2]
            batch_dev_f1 += tmp_dev_f1
            batch_dev_p += tmp_dev_p
            batch_dev_acc += tmp_dev_acc
            batch_dev_r += tmp_dev_r
            batch_loss += loss.item()
            if config.use_wandb:
                wandb_log(wandb, epoch, global_step,  tmp_dev_f1, tmp_dev_p, tmp_dev_r, tmp_dev_acc, 0.,evaluate_mode=config.evaluate_mode, type_=type_)


    count = len(dev_dataloader)
    dev_loss = batch_loss / count
    dev_p = batch_dev_p / count
    dev_r = batch_dev_r / count
    dev_f1 = batch_dev_f1 / count
    dev_acc = batch_dev_acc / count

    show_log(logger, -1, 0, 0, epoch, global_step, dev_loss, dev_p, dev_r, dev_f1,0.00, config.evaluate_mode, type_='dev', scheme=1)
    if config.use_wandb:
        wandb_log(wandb, epoch, global_step, dev_f1, dev_p, dev_r, dev_acc, loss.item(), evaluate_mode=config.evaluate_mode,type_=type_)


    reports1 = classification_report(binary_all_dev_labels, binary_all_predicate_tokens, labels=[0,1],digits=4)
    # error_analysis(all_dev_labels, all_predicate_tokens,dev_raw_text)
    reports2 = classification_report(all_dev_labels, all_predicate_tokens, labels=[0,1,2,3,4,5],digits=4)
    logger.info("-----------验证集epoch:{} 报告-----------".format(epoch))
    logger.info(reports1)
    logger.info(reports2)
    return   dev_p,dev_r,dev_f1




if __name__ == '__main__':
    config = get_bert_config()
    # 设置时间
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff

    set_seed(config.seed)


    logger = get_logger(config)
    if config.use_wandb:
        wandb.init(project="关系分类-{}".format(config.dataset_name), entity="kedaxia", config=vars(config))
    logger.info('----------------本次模型运行的参数--------------------')
    print_hyperparameters(config, logger)
    device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(config.bert_dir)

    label2id, id2label = get_label2id(config.relation_labels)
    config.num_labels = len(label2id)

    config.vocab_size = len(tokenizer)

    # ckpt_path = '/opt/data/private/luyuwei/code/bioner/re/outputs/save_models/sing_entity_marker_schema_-12/multi/AllDataset/best_model/model.pt'
    for i in range(1,16):

        ckpt_path = '/opt/data/private/luyuwei/code/bioner/re/outputs/save_models/2022-06-17/multi_task_biobert_sing_entity_marker_free_nums8_scheduler0.1_bs64_schema-12_lr1e-05/sing_entity_marker/AllDataset/13/model.pt'
        logger.info(ckpt_path)
        if config.dataset_name == 'BC6ChemProt':
            if config.data_format == 'single':
                model = MultiSingleEntityMarkerForBC6(config)
            elif config.data_format == 'cross':
                model = MultiMtbBertForBC6(config)
        elif config.dataset_name == 'BC7DrugProt':
            if config.data_format == 'single':
                model = MultiSingleEntityMarkerForBC7(config)
            elif config.data_format == 'cross':
                model = MultiMtbBertForBC7(config)
        elif config.dataset_name == 'AllDataset':
            if config.data_format == 'single':
                model = MultiSingleEntityMarkerForAlldata(config)
            elif config.data_format == 'cross':
                pass
        elif config.dataset_name in ['DDI2013','AIMed','BioInfer','euadr','GAD','HPRD-50','LLL','IEPA']:
            if config.data_format == 'single':
                model = MultiSingleEntityMarkerForBinary(config)
            elif config.data_format == 'cross':
                pass
        else:
            raise ValueError("输入正确的多任务关系分类数据集")

        model.to(device)
        set_tokenize_special_tag(config, tokenizer)
        model.bert_model.resize_token_embeddings(len(tokenizer))

        model, device = load_model_and_parallel(model, '0', ckpt_path=ckpt_path, strict=True, load_type='many2one')

        dev(model, config, tokenizer, label2id, device, epoch=0, global_step=0, logger=logger,
            type_='dev')
        break
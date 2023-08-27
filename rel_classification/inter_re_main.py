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
import warnings
from ipdb import set_trace
import wandb
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, precision_score, recall_score, \
    classification_report, confusion_matrix
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BertTokenizer

from inter_re_dev import dev
from src.dataset_utils.data_process_utils import read_data, get_label2id
from src.utils.function_utils import set_seed, save_model, load_model_and_parallel, set_cv_config
from src.utils.tips_utils import get_bert_config, get_logger, print_hyperparameters, wandb_log, show_log
from src.utils.train_utils import build_optimizer_and_scheduler, relation_classification_decode, batch_to_device, \
    set_tokenize_special_tag, choose_model, choose_dataloader, build_optimizer


def train(config=None, logger=None):
    # 这里初始化device，为了在Dataset时加载到device之中
    device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(config.bert_dir)

    set_tokenize_special_tag(config, tokenizer)

    label2id, id2label = get_label2id(config.relation_labels)
    config.num_labels = len(label2id)
    examples = read_data(config)

    if config.debug:
        examples = examples[:config.batch_size * 3]
    train_dataloader = choose_dataloader(config, examples, label2id, tokenizer, device)
    model = choose_model(config)

    # 当添加新的token之后，就要重新调整embedding_size...
    model.bert_model.resize_token_embeddings(len(tokenizer))

    if config.use_n_gpu and torch.cuda.device_count() > 1:
        model, device = load_model_and_parallel(model, '0,1', ckpt_path=None, load_type='many2one')
    else:
        model.to(device)
    t_total = config.num_epochs * len(train_dataloader)
    if config.use_scheduler:
        optimizer, scheduler = build_optimizer_and_scheduler(config, model, t_toal=t_total)
    else:
        optimizer = build_optimizer(config, model)
    # optimizer = build_optimizer(config, model)

    if config.summary_writer:
        metric_writer = SummaryWriter(
            os.path.join(config.tensorboard_dir,
                         "metric_{} {}-{} {}-{}-{}".format(config.model_name, now.month, now.day,
                                                           now.hour, now.minute,
                                                           now.second)))
    best_model = None
    global_step = 0
    best_p = best_r = best_f1 = 0.
    best_epoch = 0
    # 使用wandb来记录模型训练的时候各种参数....
    # wandb.watch(model, torch.nn.CrossEntropyLoss, log="all", log_freq=2)

    for epoch in range(1, config.num_epochs + 1):
        batch_loss = 0.
        batch_train_f1 = 0.
        batch_train_p = 0.
        batch_train_r = 0.

        all_train_labels = []
        all_predicate_tokens = []

        model.train()
        for idx, batch_data in tqdm(enumerate(train_dataloader),desc="数据集:{},{}_{}....".format(config.dataset_name,config.bert_name,config.model_name)):
            input_ids1, token_type_ids1, attention_masks1, input_ids2, token_type_ids2, attention_masks2, e1_mask,e2_mask,labels = batch_data

            loss, logits = model(input_ids1, token_type_ids1,attention_masks1,input_ids2, token_type_ids2,attention_masks2,e1_mask,e2_mask,labels)
            predicate_token = relation_classification_decode(logits)

            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if config.use_scheduler:
                scheduler.step()  # Update learning rate schedule

            learning_rate = optimizer.param_groups[0]['lr']
            labels = labels.cpu().numpy()

            all_labels = [id for _, id in label2id.items()]

            all_train_labels.extend(labels)
            all_predicate_tokens.extend(predicate_token)

            if config.num_labels == 2:
                p_r_f1_s = precision_recall_fscore_support(labels, predicate_token, labels=all_labels, average='binary')
            else:
                p_r_f1_s = precision_recall_fscore_support(labels, predicate_token, labels=all_labels,
                                                           average=config.evaluate_mode)

            acc = accuracy_score(labels, predicate_token)

            tmp_train_p = p_r_f1_s[0]
            tmp_train_r = p_r_f1_s[1]
            tmp_train_f1 = p_r_f1_s[2]
            batch_train_f1 += tmp_train_f1
            batch_train_p += tmp_train_p
            batch_train_r += tmp_train_r
            batch_loss += loss.item()
            if config.use_wandb:
                wandb_log(wandb, epoch, global_step, tmp_train_f1, tmp_train_p, tmp_train_r, acc, loss.item(),
                          type_='train', learning_rate=learning_rate)
            if config.train_verbose and global_step % config.print_step == 0:
                show_log(logger, idx, len(train_dataloader), t_total, epoch, global_step, loss, tmp_train_p,
                         tmp_train_r, tmp_train_f1, acc, config.evaluate_mode, type='train')

            global_step += 1

        count = len(train_dataloader)
        batch_loss = batch_loss / count
        train_p = batch_train_p / count
        train_r = batch_train_r / count
        train_f1 = batch_train_f1 / count

        show_log(logger, -1, len(train_dataloader), t_total, epoch, global_step, batch_loss, train_p, train_r, train_f1,
                 0.00, config.evaluate_mode, type='train', scheme=1)

        reports = classification_report(all_train_labels, all_predicate_tokens, labels=all_labels,digits=4)
        logger.info("-------训练集epoch:{} 报告----------".format(epoch))
        logger.info(reports)

        dev_p, dev_r, dev_f1 = dev(model, config, tokenizer, label2id, device, epoch=epoch, global_step=global_step,
                                   logger=logger,type_='dev')
        if config.dataset_name in ['BC6ChemProt']:
            test_p, test_r, test_f1 = dev(model, config, tokenizer, label2id, device, epoch=epoch, global_step=global_step,
                                       logger=logger,type_='test')
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_p = dev_p
            best_r = dev_r
            best_epoch = epoch
            if config.save_model:
                best_model = copy.deepcopy(model)
        if config.save_model:
            save_model(config, model, epoch=epoch, mode='other')
        # if (
        #         epoch >= 3 and train_f1 - dev_f1 > config.over_fitting_rate) or best_epoch - epoch >= config.over_fitting_epoch:  # 如果训练集的f1超过验证集9个百分点，自动停止
        #     logger.info('.............过拟合，提前停止训练...............')
        #     logger.info(
        #         '{}任务中{}模型下，在第{}epoch中，最佳的是{}-f1:{:.5f},{}-p:{:.5f},{}-r:{:.5f},将模型存储在{}'.format(config.dataset_name,
        #                                                                                          config.model_name,
        #                                                                                          best_epoch,
        #                                                                                          config.evaluate_mode,
        #                                                                                          best_f1,
        #                                                                                          config.evaluate_mode,
        #                                                                                          best_p,
        #                                                                                          config.evaluate_mode,
        #                                                                                          best_r,
        #                                                                                          config.output_dir))
        #     if config.save_model:
        #         save_model(config, best_model, mode='best_model')
        #     if config.summary_writer:
        #         metric_writer.close()
        #
        #     logger.info('----------------本次模型运行的参数------------------')
        #     print_hyperparameters(config, logger)
        #     return

    logger.info('{}任务中{}模型下，在第{}epoch中，最佳的是{}-f1:{:.5f},{}-p:{:.5f},{}-r:{:.5f},将模型存储在{}'.format(config.dataset_name,
                                                                                                 config.model_name,
                                                                                                 best_epoch,
                                                                                                 config.evaluate_mode,
                                                                                                 best_f1,
                                                                                                 config.evaluate_mode,
                                                                                                 best_p,
                                                                                                 config.evaluate_mode,
                                                                                                 best_r,
                                                                                                 config.output_dir))

    if config.save_model:
        save_model(config, best_model, mode='best_model')
    if config.summary_writer:
        metric_writer.close()

    logger.info('----------------本次模型运行的参数------------------')
    print_hyperparameters(config, logger)
    # Optional


if __name__ == '__main__':
    config = get_bert_config()
    warnings.filterwarnings("ignore")
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

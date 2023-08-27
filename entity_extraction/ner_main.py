# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  这里的运行是长度以每个batch的最大长度作为长度，而不是一个固定长度
   Author :        kedaxia
   date：          2021/11/08
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/11/08: 
-------------------------------------------------
"""

import logging
import os
import datetime
import time
from collections import defaultdict
import copy

from ipdb import set_trace
import torch
import numpy as np
import wandb
from torch import nn
from tqdm import tqdm

from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast as ac
from torch.nn.utils import clip_grad_norm_
from seqeval.metrics import classification_report, f1_score, accuracy_score, recall_score, precision_score

from ner_dev import dev

from src.evaluate.evaluate_span import evaluate_span_fpr
from src.models.inter_bert_span import InterBertSpan

from utils.function_utils import correct_datetime, set_seed, print_hyperparameters, gentl_print, get_type_weights, \
    get_logger, save_metric_writer, save_parameter_writer, choose_model, choose_dataset_and_loader, final_print_score, \
    get_config, list_find, get_wandb_name, count_parameters
from utils.data_process_utils import read_data, convert_example_to_span_features, convert_example_to_crf_features, \
    sort_by_lengths

from utils.function_utils import load_model_and_parallel, save_model
from utils.train_utils import build_optimizer_and_scheduler, build_optimizer

from utils.trick_utils import EMA

from utils.function_utils import show_log, wandb_log


def train(config=None, logger=None):
    train_data, train_labels = read_data(config.train_file_path)
    if config.debug:  # 开启debug，则试验一个batch的数据
        train_data = train_data[:config.batch_size * 5]
        train_labels = train_labels[:config.batch_size * 5]
    if config.evaluate_mode == 'weight':
        type_weight = get_type_weights(train_labels)
    else:
        type_weight = None

    # 加载预训练模型的分词器
    tokenizer = None
    if config.which_model == 'bert':
        if config.bert_name in ['scibert', 'biobert', 'flash', 'flash_quad', 'wwm_bert','bert']:
            tokenizer = BertTokenizer(os.path.join(config.bert_dir, 'vocab.txt'))

        elif config.bert_name == 'kebiolm':
            tokenizer = AutoTokenizer.from_pretrained(config.bert_dir)
        else:
            raise ValueError

    word2id, id2word = None, None
    metric = None

    model, metric = choose_model(config)  # 这个metric没啥用，只在globalpointer有用


    train_dataset, train_loader = choose_dataset_and_loader(config, train_data, train_labels, tokenizer, word2id)
    #  加载模型是否是多GPU或者
    if config.use_n_gpu and torch.cuda.device_count() > 1:
        model, device = load_model_and_parallel(model, '0,1', ckpt_path=None, load_type='many2one')
    else:
        # model, device = load_model_and_parallel(model, '0,1', ckpt_path=None, load_type='one2one')
        device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')
        model.to(device)

    t_total = config.num_epochs * len(train_loader)

    logger.info('--------神经网络模型架构------')
    logger.info(model)

    if config.use_scheduler:
        optimizer, scheduler = build_optimizer_and_scheduler(config, model, t_toal=t_total)
        logger.info('学习率调整器:{}'.format(scheduler))
    else:
        optimizer = build_optimizer(config, model)

    logger.info('优化器:{}'.format(optimizer))

    if config.use_ema:
        ema = EMA(model, config.ema_decay)
        ema.register()

    best_epoch = 0
    best_model = None
    if config.metric_summary_writer:
        if not os.path.exists(config.tensorboard_dir):
            os.makedirs(config.tensorboard_dir)
        metric_writer = SummaryWriter(os.path.join(config.tensorboard_dir,
                                                   "metric_{} {}-{} {}-{}-{}".format(config.model_name, now.month,
                                                                                     now.day,
                                                                                     now.hour, now.minute,
                                                                                     now.second)))
    if config.parameter_summary_writer:
        if not os.path.exists(config.tensorboard_dir):
            os.makedirs(config.tensorboard_dir)
        parameter_writer = SummaryWriter(
            os.path.join(config.tensorboard_dir, "parameter_{} {}-{} {}-{}-{}".format(config.model_name, now.month,
                                                                                      now.day,
                                                                                      now.hour, now.minute,
                                                                                      now.second)))
    dev_res_li = []
    train_res_li = []
    best_f1 = 0.
    best_r = 0.
    best_p = 0.
    dev_f1_decay_count = 0
    if config.use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    global_step = 1
    # requires_grad_nums, parameter_nums = count_parameters(model)
    # set_trace()
    for epoch in range(config.num_epochs):
        epoch += 1
        epoch_loss = 0.  # 统计一个epoch的平均loss
        model.train()
        train_p, train_r, train_f1, train_loss = 0., 0., 0., 0.
        # 准备这里存储model parameters,以epoch为单位存储参数

        for step, batch_data in tqdm(enumerate(train_loader),total=len(train_loader),desc="模型:{}在数据集:{}训练中....".format(config.wandb_name,config.ner_dataset_name)):

            train_predicate = []
            train_callback_info = []

            if config.model_name in ['bert_crf', 'bert_bilstm_crf', 'bert_mlp']:
                raw_text_list, batch_true_labels, batch_subword_input_ids, batch_subword_token_type_ids, batch_subword_attention_masks, origin_to_subword_indexs, batch_label_mask = batch_data
                token_ids, attention_masks, token_type_ids = batch_subword_input_ids.to(
                    device), batch_subword_attention_masks.to(device), batch_subword_token_type_ids.to(device)
                true_labels = batch_true_labels.to(device)
                origin_to_subword_indexs = origin_to_subword_indexs.to(device)

                if config.subword_weight_mode == 'first':
                    loss, train_predicate = model(token_ids,
                                                  attention_masks=attention_masks,
                                                  token_type_ids=token_type_ids,
                                                  labels=true_labels,
                                                  input_token_starts=origin_to_subword_indexs)

                elif config.subword_weight_mode == 'avg':
                    loss, train_predicate = model.weight_forward(token_ids, attention_mask=attention_masks,
                                                                 token_type_ids=token_type_ids, labels=true_labels,
                                                                 input_token_starts=origin_to_subword_indexs)
                else:
                    raise ValueError
                # 这里由于多GPU运算，因此对train_predicate进行额外处理

                if config.use_n_gpu or config.fixed_batch_length:
                    if config.decoder_layer == 'crf':
                        train_predicate = [
                            train_predicate[i][:list_find(origin_to_subword_indexs[i], 0)].cpu().numpy().tolist()
                            for i in range(token_ids.shape[0])]
                    else:

                        train_predicate = np.argmax(train_predicate.detach().cpu().numpy(), axis=2)

                        train_predicate = [
                            train_predicate[i][:list_find(origin_to_subword_indexs[i], 0)].tolist()
                            for i in range(token_ids.shape[0])]

                true_train_labels = true_labels.cpu().numpy()
                train_callback_info.extend(raw_text_list)

            elif config.model_name in ['bert_span','inter_bert_span','interbertbilstmspan','interbergruspan']:

                raw_text_list, token_ids, attention_masks, token_type_ids, start_ids, end_ids, origin_to_subword_index, input_true_length = batch_data
                token_ids, attention_masks, token_type_ids, start_ids, end_ids = token_ids.to(
                    device), attention_masks.to(device), token_type_ids.to(device), start_ids.to(
                    device), end_ids.to(device)
                input_true_length = input_true_length.to(device)
                origin_to_subword_index = origin_to_subword_index.to(device)

                train_callback_info.extend(raw_text_list)

                loss, tmp_start_logits, tmp_end_logits = model(token_ids, attention_masks=attention_masks,
                                                               token_type_ids=token_type_ids,
                                                               start_ids=start_ids,
                                                               end_ids=end_ids,
                                                               input_token_starts=origin_to_subword_index,
                                                               input_true_length=input_true_length)

                _, span_start_logits = torch.max(tmp_start_logits, dim=-1)
                _, span_end_logits = torch.max(tmp_end_logits, dim=-1)

                span_start_logits = span_start_logits.cpu().numpy()
                span_end_logits = span_end_logits.cpu().numpy()
                start_ids = start_ids.cpu().numpy()
                end_ids = end_ids.cpu().numpy()
            elif config.model_name == 'bert_globalpointer':

                raw_text_list, batch_true_labels, batch_subword_input_ids, batch_subword_token_type_ids, batch_subword_attention_masks, origin_to_subword_indexs, batch_label_mask, input_true_length = batch_data

                token_ids, attention_masks, token_type_ids = batch_subword_input_ids.to(
                    device), batch_subword_attention_masks.to(device), batch_subword_token_type_ids.to(device)
                global_pointer_true_labels = batch_true_labels.to(device)
                input_true_length = input_true_length.to(device)
                origin_to_subword_indexs = origin_to_subword_indexs.to(device)

                train_callback_info.extend(raw_text_list)

                loss, globalpointer_predicate = model(token_ids, attention_masks=attention_masks,
                                                      token_type_ids=token_type_ids,
                                                      labels=global_pointer_true_labels,
                                                      input_token_starts=origin_to_subword_indexs,
                                                      input_true_length=input_true_length)

            # 这一部分是为了测试和debug，测试各种改进
            elif config.model_name == 'bert_test':
                input_ids, token_type_ids, attention_masks, subword_labels, true_train_labels = batch_data
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_masks = attention_masks.to(device)
                subword_labels = subword_labels.to(device)

                if config.use_fp16:
                    with ac():
                        loss, logits = model(input_ids, attention_masks=attention_masks,
                                             token_type_ids=token_type_ids, labels=subword_labels)
                else:
                    loss, logits = model(input_ids, attention_masks=attention_masks, token_type_ids=token_type_ids,
                                         labels=subword_labels)
                # logits是模型的预测结果,shape=(batch_size,seq_len,num_labels)
                # 这里需要进行decode
                output = np.argmax(logits.detach().cpu().numpy(), axis=2)
                subword_labels = subword_labels.detach().cpu().numpy()
                train_predicate = []
                batch_size, seq_len = logits.shape[:2]
                for i in range(batch_size):
                    tmp_token = []
                    for j in range(seq_len):
                        if subword_labels[i][j] != -1:
                            tmp_token.append(output[i][j])
                    train_predicate.append(tmp_token)
            else:
                raise ValueError


            loss = loss.mean()
            if config.use_fp16:
                scaler.scale(loss).backward()
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    if config.use_scheduler:
                        scheduler.step()
                    if config.parameter_summary_writer:
                        save_parameter_writer(parameter_writer, model, global_step)
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    if config.use_scheduler:
                        scheduler.step()
                    if config.parameter_summary_writer:
                        save_parameter_writer(parameter_writer, model, global_step)
                    optimizer.zero_grad()

            if config.use_ema:
                ema.update()
            if config.metric_summary_writer and config.use_scheduler:
                metric_writer.add_scalar('Training/learning_rate', scheduler.get_lr()[0], global_step)
            epoch_loss += loss.item()
            global_step += 1

            # 这是开始进行训练的评估
            if config.verbose:
                if config.decoder_layer in ['crf', 'mlp', 'test']:
                    # crf的结果中，无论是pred还是label里面都有特殊token,[CLS],[SEP],所以需要去除
                    # crf自动将<start><end>给去除
                    predicate_label_BIO = []
                    true_label_BIO = []
                    for i in range(len(train_predicate)):
                        predicate_label_BIO.append([config.crf_id2label[x] for x in train_predicate[i]])
                        true_label_BIO.append(
                            [config.crf_id2label[x] for x in true_train_labels[i][:len(train_predicate[i])]])

                    # 对于BIO，只能是micro和macro，不可以使用weight评估
                    tmp_train_p = precision_score(true_label_BIO, predicate_label_BIO)
                    tmp_train_r = recall_score(true_label_BIO, predicate_label_BIO)
                    tmp_train_f1 = f1_score(true_label_BIO, predicate_label_BIO)

                elif config.decoder_layer == 'span':
                    tmp_train_f1, tmp_train_p, tmp_train_r = evaluate_span_fpr(span_start_logits, span_end_logits,
                                                                               start_ids, end_ids, train_callback_info,
                                                                               config.span_label2id, type_weight,
                                                                               average=config.evaluate_mode,
                                                                               verbose=config.verbose)

                elif config.decoder_layer == 'globalpointer':
                    tmp_train_f1, tmp_train_p, tmp_train_r = metric.get_evaluate_fpr(globalpointer_predicate,
                                                                                     global_pointer_true_labels,
                                                                                     config.globalpointer_label2id,
                                                                                     type_weight,
                                                                                     average=config.evaluate_mode,
                                                                                     verbose=config.verbose)

                else:
                    raise ValueError

                train_f1 += tmp_train_f1
                train_r += tmp_train_r
                train_p += tmp_train_p
                lr = scheduler.get_lr()[0] if config.use_scheduler else config.learning_rate
                if config.use_wandb:
                    wandb_log(wandb, epoch, global_step, tmp_train_f1, tmp_train_p, tmp_train_r, 0., loss.item(),
                              type='train', lr=lr)

                # logger.info("学习率{}".format(model.dynamic_weight.item()))
                show_log(logger, step, len(train_loader), t_total, epoch, global_step, loss, tmp_train_p,
                             tmp_train_r,
                             tmp_train_f1, 0., config.evaluate_mode, type='train', scheme=0)
                if config.metric_summary_writer:
                    save_metric_writer(metric_writer, loss.item(), global_step, tmp_train_p, tmp_train_r, tmp_train_f1,
                                       type='Training')


            train_loss += loss.item()


        if config.save_model:
            save_model(config, model, epoch=epoch, mode='other')
        # 开始验证,一个epoch进行一次验证，或者根据global step进行验证
        if config.verbose:
            train_p = train_p / len(train_loader)
            train_r = train_r / len(train_loader)
            train_f1 = train_f1 / len(train_loader)
            train_loss = train_loss / len(train_loader)
            if config.metric_summary_writer:
                save_metric_writer(metric_writer, epoch_loss / len(train_loader), global_step, train_p, train_r, train_f1,
                                   type='Training_epoch')

            # 这里这个记录，用于最后输出每个epoch的结果
            train_res_li.append({'f1': train_f1, 'p': train_p, 'r': train_r})
            show_log(logger, 0, len(train_loader), t_total, epoch, global_step, train_loss, train_p, train_r, train_f1, 0.,
                     config.evaluate_mode, type='train', scheme=1)

        if config.use_ema:
            ema.apply_shadow()

        #  开始Dev，验证集格式....
        logger.info('-----------开始验证集-------------')
        if config.ner_dataset_name in ['origin_jnlpba']:
            # 这个数据集只有训练集和测试集，没有验证集

            dev_p, dev_r, dev_f1, dev_loss = dev(model=model, config=config, device=device, type_weight=type_weight,
                                                 metric=metric, logger=logger, epoch=epoch, global_step=global_step,
                                                 word2id=word2id, tokenizer=tokenizer,type_='test')
            if config.use_wandb:
                wandb_log(wandb, epoch, global_step, dev_f1, dev_p, dev_r, 0., dev_loss, lr=0, type='test')
            show_log(logger, 0, len(train_loader), t_total, epoch, global_step, dev_loss, dev_p, dev_r,
                     dev_f1, 0., config.evaluate_mode, type='test', scheme=1)
        elif config.ner_dataset_name in ['linnaeus', 'jnlpba', 's800', 'BC4CHEMD', 'BC2GM', 'BC5CDR-chem',
                                       'BC5CDR-disease', 'BC6ChemProt', 'NCBI-disease','BC5CDR']:
            # 这些数据集既有验证集也有训练集
            dev_p, dev_r, dev_f1, dev_loss = dev(model=model, config=config, device=device, type_weight=type_weight,
                                                 metric=metric, logger=logger, epoch=epoch, global_step=global_step,
                                                 word2id=word2id, tokenizer=tokenizer, type_='dev')
            test_p, test_r, test_f1, test_loss = dev(model=model, config=config, device=device, type_weight=type_weight,
                                                     metric=metric, logger=logger, epoch=epoch, global_step=global_step,
                                                     word2id=word2id, tokenizer=tokenizer, type_='test')
            show_log(logger, 0, len(train_loader), t_total, epoch, global_step, dev_loss, dev_p, dev_r,
                     dev_f1, 0., config.evaluate_mode, type='dev', scheme=1)
            show_log(logger, 0, len(train_loader), t_total, epoch, global_step, test_loss, test_p, test_r,
                     test_f1, 0., config.evaluate_mode, type='test', scheme=1)
            if config.use_wandb:
                wandb_log(wandb, epoch, global_step, dev_f1, dev_p, dev_r, 0., dev_loss, lr=0, type='dev')
                wandb_log(wandb, epoch, global_step, test_f1, test_p, test_r, 0., test_loss, lr=0, type='test')
        else:
            dev_p, dev_r, dev_f1, dev_loss = dev(model=model, config=config, device=device, type_weight=type_weight,
                                                 metric=metric, logger=logger, epoch=epoch, global_step=global_step,
                                                 word2id=word2id, tokenizer=tokenizer, type_='dev')
            if config.use_wandb:
                wandb_log(wandb, epoch, global_step, dev_f1, dev_p, dev_r, 0., dev_loss, lr=0, type='dev')
            show_log(logger, 0, len(train_loader), t_total, epoch, global_step, dev_loss, dev_p, dev_r,
                     dev_f1, 0., config.evaluate_mode, type='dev', scheme=1)
        if config.use_ema:
            ema.restore()

        dev_res_li.append({'f1': dev_f1, 'p': dev_p, 'r': dev_r})

        if config.metric_summary_writer:
            save_metric_writer(metric_writer, dev_loss, global_step, dev_p, dev_r, dev_f1,
                               type='Dev')

        if best_f1 < dev_f1:
            best_f1 = dev_f1
            best_p = dev_p
            best_r = dev_r
            best_epoch = epoch

            if config.save_model:
                best_model = copy.deepcopy(model)
            dev_f1_decay_count = 0
        else:
            dev_f1_decay_count += 1
        # 开始过拟合检测
        if config.early_stop and ((
                                          epoch >= 3 and train_f1 > 0.95 and train_f1 - dev_f1 > config.over_fitting_rate) or best_epoch - epoch >= config.over_fitting_epoch or dev_f1_decay_count >= 3):  # 如果训练集的f1超过验证集9个百分点，自动停止
            logger.info('.............过拟合，提前停止训练...............')
            logger.info('{}任务中{}模型下，在第{}epoch中，最佳的是f1:{:.5f},p:{:.5f},r:{:.5f},将模型存储在{}'.format(config.ner_dataset_name,
                                                                                                config.model_name,
                                                                                                best_epoch, best_f1,
                                                                                                best_p, best_r,
                                                                                                config.output_dir))
            if config.save_model:
                save_model(config, best_model, mode='best_model')
            if config.parameter_summary_writer:
                parameter_writer.close()
            if config.metric_summary_writer:
                metric_writer.close()
            for i in range(len(train_res_li)):
                logger.info('epoch{} 训练集 f1:{:.5f},p:{:.5f},r:{:.5f}'.format(i + 1, train_res_li[i]['f1'],
                                                                             train_res_li[i]['p'],
                                                                             train_res_li[i]['r']))
                logger.info('        验证集 f1:{:.5f},p:{:.5f},r:{:.5f}'.format(dev_res_li[i]['f1'],
                                                                             dev_res_li[i]['p'],
                                                                             dev_res_li[i]['r']))
            logger.info('----------------本次模型运行的参数------------------')
            print_hyperparameters(config)
            final_print_score(train_res_li, dev_res_li)
            return

    logger.info('{}任务中{}模型下，在第{}epoch中，最佳的是f1:{:.5f},p:{:.5f},r:{:.5f},将模型存储在{}'.format(config.ner_dataset_name,
                                                                                        config.model_name, best_epoch,
                                                                                        best_f1, best_p, best_r,
                                                                                        config.output_dir))

    if config.save_model and best_model:
        save_model(config, best_model, mode='best_model')
    if config.parameter_summary_writer:
        parameter_writer.close()
    if config.metric_summary_writer:
        metric_writer.close()
    logger.info('----------------本次模型运行的参数------------------')
    print_hyperparameters(config)
    final_print_score(train_res_li, dev_res_li)
    return best_f1, best_p, best_r


if __name__ == '__main__':

    config = get_config()



    # 设置时间
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff
    # 设置随机种子
    set_seed(config.seed)
    wandb_name,project_name = get_wandb_name(config)
    config.output_dir = './outputs/save_models/{}/{}/{}/{}/'.format(str(datetime.date.today()), wandb_name,
                                                                        config.model_name, config.ner_dataset_name)

    config.logs_dir = './outputs/logs/{}/{}/{}/{}/'.format(str(datetime.date.today()), wandb_name, config.model_name,
                                                           config.ner_dataset_name)
    logger = get_logger(config)
    config.wandb_name = wandb_name
    if config.run_type == 'normal':
        if config.use_wandb:
            wandb.init(project=project_name, config=vars(config),name=wandb_name,notes=config.wandb_notes)

        logger.info('----------------本次模型运行的参数------------------')
        print_hyperparameters(config)

        train(config, logger)
    elif config.run_type == 'cv5':
        avg_f1, avg_p, avg_r = 0., 0., 0.
        for i in range(1, 6):
            if config.use_wandb:
                wandb.init(project="实体抽取-{}_cv{}".format(config.ner_dataset_name, i), entity="kedaxia",
                           config=vars(config))
            logger.info('.............cv:{}.................'.format(i))
            config.train_file_path = './NERdata/{}/cv5/{}/train.txt'.format(config.ner_dataset_name, i)
            config.dev_file_path = './NERdata/{}/cv5/{}/dev.txt'.format(config.ner_dataset_name, i)
            config.output_dir = './outputs/save_models/{}/{}/cv5/cv_{}/'.format(config.model_name,
                                                                                config.ner_dataset_name, i)
            if not os.path.exists(config.output_dir):
                os.makedirs(config.output_dir)
            print_hyperparameters(config)
            f1, p, r = train(config, logger)
            avg_f1 += f1
            avg_p += p
            avg_r += r
        logger.info('五折交叉验证结果:f1:{},p:{},r:{}'.format(avg_f1 / 5, avg_p / 5, avg_r / 5))



    elif config.run_type == 'cv10':
        avg_f1, avg_p, avg_r = 0., 0., 0.
        for i in range(1, 11):
            logger.info('.............cv:{}.................'.format(i))
            config.train_file_path = './NERdata/{}/cv10/{}/train.txt'.format(config.ner_dataset_name, i)
            config.dev_file_path = './NERdata/{}/cv10/{}/dev.txt'.format(config.ner_dataset_name, i)
            config.output_dir = './outputs/save_models/{}/{}/cv10/cv_{}/'.format(config.model_name,
                                                                                 config.ner_dataset_name, i)

            f1, p, r = train(config, logger)
            avg_f1 += f1
            avg_p += p
            avg_r += r
        logger.info('十折交叉验证结果:f1:{},p:{},r:{}'.format(avg_f1 / 10, avg_p / 10, avg_r / 10))

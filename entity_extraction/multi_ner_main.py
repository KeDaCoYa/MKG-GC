# -*- encoding: utf-8 -*-
"""
@File    :   multi_ner_main.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/3/27 15:27   
@Description :   这个是多任务学习的NER

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
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from multi_ner_dev import dev
from src.evaluate.evaluate_span import evaluate_span_fpr
from src.models.multi_binary_inter_bilstm_sainter_span import MultiBinaryBiLSTMSAInterSpanForEight
from src.models.multi_binary_inter_bilstm_span import MultiBinaryInterBiLSTMSpanForEight
from src.models.multi_binary_inter_doublemid_bilstm_span import MultiBinaryDoubleMidInterBiLSTMSpanForEight
from src.models.multi_binary_inter_gru_span import MultiBinaryInterGRUSpanForEight
from src.models.multi_binary_inter_linear_span import MultiBinaryInterLinearSpanForEight
from src.models.multi_binary_inter_span import MultiBinaryInterSpanForEight
from src.models.multi_inter_span import MultiInterSpanForEight
from utils.function_utils import correct_datetime, set_seed, print_hyperparameters, gentl_print, get_type_weights, \
    get_logger, save_metric_writer, save_parameter_writer, choose_model, choose_dataset_and_loader, final_print_score, \
    get_config, list_find, choose_multi_dataset_and_loader, choose_multi_ner_model, count_parameters
from utils.data_process_utils import read_data, convert_example_to_span_features, convert_example_to_crf_features, \
    sort_by_lengths

from utils.function_utils import load_model_and_parallel, save_model
from utils.train_utils import build_optimizer_and_scheduler, build_optimizer

from utils.trick_utils import EMA

from utils.function_utils import show_log, wandb_log


def train(config=None, logger=None):

    tokenizer = None
    if config.which_model == 'bert':
        if config.bert_name in ['scibert', 'biobert', 'flash', 'flash_quad', 'wwm_bert', 'bert']:
            tokenizer = BertTokenizer(os.path.join(config.bert_dir, 'vocab.txt'))
        elif config.bert_name == 'kebiolm':
            tokenizer = AutoTokenizer.from_pretrained(config.bert_dir)
        else:
            raise ValueError


    metric = None
    if config.model_name == 'binary_bert_span' or config.model_name == 'bert_span':
        model, metric = choose_multi_ner_model(config)  # 这个metric没啥用，只在globalpointer有用
    elif config.model_name == 'inter_binary_bert_span':
        model = MultiBinaryInterSpanForEight(config)
    elif config.model_name == 'inter_bert_span':
        model = MultiInterSpanForEight(config)
    elif config.model_name == 'inter_binary_bert_bilstm_span':
        model = MultiBinaryInterBiLSTMSpanForEight(config)
    elif config.model_name == 'inter_binary_bert_bilstm_double_mid_span':
        model = MultiBinaryDoubleMidInterBiLSTMSpanForEight(config)
    elif config.model_name == 'inter_binary_bert_sa_mid_span':
        model = MultiBinaryBiLSTMSAInterSpanForEight(config)
    elif config.model_name == 'inter_binary_bert_gru_mid_span':
        model = MultiBinaryInterGRUSpanForEight(config)
    elif config.model_name == 'inter_binary_bert_linear_mid_span':
        model = MultiBinaryInterLinearSpanForEight(config)
    else:
        raise ValueError

    train_dataset, train_loader = choose_multi_dataset_and_loader(config, tokenizer, type_='train')

    if config.use_n_gpu and torch.cuda.device_count() > 1:
        model, device = load_model_and_parallel(model, '0,1', ckpt_path=None, load_type='many2one')
    else:
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
    requires_grad_nums, parameter_nums = count_parameters(model)

    global_step = 1
    for epoch in range(config.num_epochs):
        epoch += 1
        epoch_loss = 0.  # 统计一个epoch的平均loss
        model.train()
        train_p, train_r, train_f1, train_loss = 0., 0., 0., 0.
        # 准备这里存储model parameters,以epoch为单位存储参数

        for step, batch_data in tqdm(enumerate(train_loader), total=len(train_loader),
                                     desc="模型:{} 数据集:{} 训练中".format(config.wandb_name, config.ner_dataset_name)):

            train_callback_info = []

            if config.model_name in ['binary_bert_span','inter_binary_bert_span','bert_span','inter_bert_span','inter_binary_bert_bilstm_span','inter_binary_bert_bilstm_double_mid_span','inter_binary_bert_sa_mid_span','inter_binary_bert_linear_mid_span','inter_binary_bert_gru_mid_span']:

                raw_text_list, token_ids, attention_masks, token_type_ids, start_ids, end_ids, origin_to_subword_index, input_true_length, entity_type_ids = batch_data
                token_ids, attention_masks, token_type_ids, start_ids, end_ids = token_ids.to(
                    device), attention_masks.to(device), token_type_ids.to(device), start_ids.to(device), end_ids.to(
                    device)
                entity_type_ids = entity_type_ids.to(device)
                input_true_length = input_true_length.to(device)
                origin_to_subword_index = origin_to_subword_index.to(device)

                train_callback_info.extend(raw_text_list)

                loss, tmp_start_logits, tmp_end_logits = model(token_ids, attention_masks=attention_masks,
                                                               token_type_ids=token_type_ids,
                                                               start_ids=start_ids,
                                                               end_ids=end_ids,
                                                               input_token_starts=origin_to_subword_index,
                                                               input_true_length=input_true_length,
                                                               entity_type_ids=entity_type_ids)

                _, span_start_logits = torch.max(tmp_start_logits, dim=-1)
                _, span_end_logits = torch.max(tmp_end_logits, dim=-1)

                span_start_logits = span_start_logits.cpu().numpy()
                span_end_logits = span_end_logits.cpu().numpy()

                if 'binary' in config.model_name:
                    # 如果这是多个二分类实体抽取模型，那么需要将预测的结果根据rel type进行修改...
                    # span_start_logits的值都是0,1
                    # 这里根据entity type进行还原

                    new_start_logits = []
                    new_end_logits = []
                    for bs in range(len(span_start_logits)):
                        tmp_start = []
                        tmp_end = []
                        for idx in range(len(span_start_logits[0])):

                            if span_start_logits[bs][idx] == 1:
                                tmp_start.append(entity_type_ids[bs][0].item())
                            else:
                                tmp_start.append(0)

                            if span_end_logits[bs][idx] == 1:
                                tmp_end.append(entity_type_ids[bs][0].item())
                            else:
                                tmp_end.append(0)
                            # if entity_type_ids[bs][0].item() == 1:
                            #
                            #     if span_start_logits[bs][idx] == 1:
                            #         tmp_start.append(1)
                            #     else:
                            #         tmp_start.append(0)
                            #
                            #     if span_end_logits[bs][idx] == 1:
                            #         tmp_end.append(1)
                            #     else:
                            #         tmp_end.append(0)
                            # elif entity_type_ids[bs][0].item() == 2:
                            #     if span_start_logits[bs][idx] == 1:
                            #         tmp_start.append(2)
                            #     else:
                            #         tmp_start.append(0)
                            #
                            #     if span_end_logits[bs][idx] == 1:
                            #         tmp_end.append(2)
                            #     else:
                            #         tmp_end.append(0)
                        new_start_logits.append(tmp_start)
                        new_end_logits.append(tmp_end)
                    span_start_logits = new_start_logits
                    span_end_logits = new_end_logits


                start_ids = start_ids.cpu().numpy()
                end_ids = end_ids.cpu().numpy()
                tmp_train_f1, tmp_train_p, tmp_train_r = evaluate_span_fpr(span_start_logits, span_end_logits,
                                                                           start_ids, end_ids, train_callback_info,
                                                                           config.span_label2id, None,
                                                                           average=config.evaluate_mode,
                                                                           verbose=config.verbose)
            else:
                raise ValueError
            train_f1 += tmp_train_f1
            train_r += tmp_train_r
            train_p += tmp_train_p

            loss = loss.mean()
            train_loss += loss.item()

            lr = scheduler.get_lr()[0] if config.use_scheduler else config.learning_rate

            if config.use_wandb:

                wandb_log(wandb, epoch, global_step, tmp_train_f1, tmp_train_p, tmp_train_r, 0., loss.item(),
                          type='train', lr=lr)
            if config.verbose:
                show_log(logger, step, len(train_loader), t_total, epoch, global_step, loss, tmp_train_p, tmp_train_r,
                         tmp_train_f1, 0., config.evaluate_mode, type='train', scheme=0)
            if config.metric_summary_writer:
                save_metric_writer(metric_writer, loss.item(), global_step, tmp_train_p, tmp_train_r, tmp_train_f1,
                                   type='Training')

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
            global_step += 1


        if config.save_model:
            save_model(config, model, epoch=epoch, mode='other')
        # 开始验证,一个epoch进行一次验证，或者根据global step进行验证

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

        dev_p, dev_r, dev_f1, dev_loss = dev(model=model, config=config, device=device,
                                             metric=metric, logger=logger, epoch=epoch, global_step=global_step,
                                             tokenizer=tokenizer,wandb=wandb)
        if config.ner_dataset_name in ['multi_jnlpba', 'multi_BC6', 'multi_BC5']:
            # 额外使用测试集
            test_p, test_r, test_f1, test_loss = dev(model=model, config=config, device=device,
                                                     metric=metric, logger=logger, epoch=epoch, global_step=global_step,
                                                     tokenizer=tokenizer, type_='test')
            if config.use_wandb:
                wandb_log(wandb, epoch, global_step, test_f1, test_p, test_r, 0., test_loss, lr=0, type='test')
        if config.use_ema:
            ema.restore()
        dev_res_li.append({'f1': dev_f1, 'p': dev_p, 'r': dev_r})
        if config.use_wandb:
            wandb_log(wandb, epoch, global_step, dev_f1, dev_p, dev_r, 0., dev_loss, lr=0, type='dev')

        show_log(logger, 0, len(train_loader), t_total, epoch, global_step, dev_loss, dev_p, dev_r,
                 dev_f1, 0., config.evaluate_mode, type='dev', scheme=1)

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
        logger.info(
            '{}任务中{}模型下，在第{}epoch中，最佳的是f1:{:.5f},p:{:.5f},r:{:.5f},将模型存储在{}'.format(config.ner_dataset_name,
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
                                                                                        config.model_name,
                                                                                        best_epoch,
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

    if config.use_scheduler:
        if config.freeze_layers:
            wandb_name = f'MT_{config.bert_name}_{config.model_name}_epochs{config.num_epochs}_free{config.freeze_layer_nums}_scheduler{config.warmup_proportion}_bs{config.batch_size}_lr{config.learning_rate}'
        else:
            wandb_name = f'MT_{config.bert_name}_{config.model_name}_epochs{config.num_epochs}_nofree_scheduler{config.warmup_proportion}_bs{config.batch_size}_lr{config.learning_rate}'
    else:
        if config.freeze_bert:
            wandb_name = f'MT_{config.bert_name}_{config.model_name}_epochs{config.num_epochs}_free{config.freeze_layer_nums}_bs{config.batch_size}_lr{config.learning_rate}'
        else:
            wandb_name = f'MT_{config.bert_name}_{config.model_name}_epochs{config.num_epochs}_no_freeze_bs{config.batch_size}_lr{config.learning_rate}'

    if 'CV' in config.ner_dataset_name:
        if 'lite' in config.ner_dataset_name:

            project_name = "实体抽取-multi_all_dataset_v1_lite"
        else:
            project_name = "实体抽取-multi_all_dataset_large"
        wandb_name = config.logfile_name+'_' + wandb_name
    else:
        project_name = "实体抽取-{}".format(config.ner_dataset_name)

    config.output_dir = './outputs/save_models/{}/{}/{}/{}/'.format( str(datetime.date.today()),wandb_name,config.model_name, config.ner_dataset_name)
    config.logs_dir = './outputs/logs/{}/{}/{}/{}/'.format(str(datetime.date.today()), wandb_name, config.model_name,
                                                           config.ner_dataset_name)
    logger = get_logger(config)
    config.wandb_name = wandb_name

    if config.run_type == 'normal':
        if config.use_wandb:
            wandb.init(project=project_name, config=vars(config), name=wandb_name)
        logger.info('----------------本次模型运行的参数------------------')
        print_hyperparameters(config)

        train(config, logger)
    elif config.run_type == 'cv5':
        avg_f1, avg_p, avg_r = 0., 0., 0.
        for i in range(1, 6):
            wandb_name += '_cv_{}'.format(i)


            wandb.init(project="实体抽取-{}".format(config.ner_dataset_name, i),config=vars(config), name=wandb_name)
            logger.info('.............cv:{}.................'.format(i))
            config.data_dir = './NERdata/{}/cv5/cv{}/'.format(config.ner_dataset_name, i)
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




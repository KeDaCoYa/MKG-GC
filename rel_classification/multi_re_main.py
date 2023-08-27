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
import random
import warnings
import os
import datetime
import copy

from ipdb import set_trace
import wandb
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, precision_score, recall_score, \
    classification_report, confusion_matrix
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BertTokenizer

from multi_re_dev import dev
from src.dataset_utils.data_process_utils import read_data, get_label2id
from src.dataset_utils.entity_marker import MultiMTBDataset
from src.dataset_utils.entity_type_marker import MultiNormalDataset
from src.dataset_utils.multi_dataset import read_multi_data
from src.models.multi_mtb_bert import MultiMtbBertForBC6, MultiMtbBertForBC7
from src.models.multi_entitymarker_model import MultiSingleEntityMarkerForAlldata,MultiSingleEntityMarkerForBC7,MultiSingleEntityMarkerForBC6,MultiSingleEntityMarkerForBinary
from src.models.multi_rbert_large import MultiRBERTForAlldataLarge
from src.utils.function_utils import set_seed, save_model, load_model_and_parallel, set_cv_config, count_parameters
from src.utils.tips_utils import get_bert_config, get_logger, print_hyperparameters, wandb_log, show_log
from src.utils.train_utils import build_optimizer_and_scheduler, relation_classification_decode, batch_to_device, \
    set_tokenize_special_tag, choose_model, choose_dataloader, build_optimizer, save_parameter_writer


def train(config=None, logger=None):
    # 这里初始化device，为了在Dataset时加载到device之中
    device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(config.bert_dir)

    set_tokenize_special_tag(config, tokenizer)

    label2id, id2label = get_label2id(config.relation_labels)
    config.num_labels = len(label2id)
    examples = read_multi_data(config)
    # 随机打乱一下顺序
    random.shuffle(examples)

    if config.debug:
        examples = examples[:config.batch_size * 3]
    if config.data_format == 'single':  # 这个针对sentence-level的关系分类
        train_dataset = MultiNormalDataset(examples, config=config, label2id=label2id, tokenizer=tokenizer,
                                           device=device)
    elif config.data_format == 'cross':
        train_dataset = MultiMTBDataset(examples, config=config, label2id=label2id, tokenizer=tokenizer, device=device)
    else:
        raise ValueError

    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, collate_fn=train_dataset.collate_fn,
                                  num_workers=0, batch_size=config.batch_size)
    if config.dataset_name == 'BC6ChemProt':
        if config.data_format == 'single':
            model = MultiSingleEntityMarkerForBC6(config)
        elif config.data_format == 'cross':
            model = MultiMtbBertForBC6(config)
    elif config.dataset_name == 'BC7DrugProt':
        if config.data_format == 'normal':
            model = MultiSingleEntityMarkerForBC7(config)
        elif config.data_format == 'mtb':
            model = MultiMtbBertForBC7(config)
    elif config.dataset_name == 'AllDataset' or 'CV' in config.dataset_name:
        config.num_labels = 2
        if config.data_format == 'single':
            if 'large' in config.model_name:
                model = MultiRBERTForAlldataLarge(config)
            else:
                model = MultiSingleEntityMarkerForAlldata(config)
        elif config.data_format == 'cross':
            raise NotImplementedError
    elif config.dataset_name in ['DDI2013', 'AIMed', 'BioInfer', 'euadr', 'GAD', 'HPRD-50', 'LLL', 'IEPA']:
        if config.data_format == 'normal':
            model = MultiSingleEntityMarkerForBinary(config)
        elif config.data_format == 'mtb':
            pass
    else:
        raise ValueError("输入正确的多任务关系分类数据集")

    if config.use_fp16:
        scaler = torch.cuda.amp.GradScaler()
    # 当添加新的token之后，就要重新调整embedding_size...
    model.bert_model.resize_token_embeddings(len(tokenizer))
    if config.use_n_gpu and torch.cuda.device_count() > 1:
        ckpt_path = '/root/code/bioner/re/outputs/save_models/r_bert_schema_1/multi/AllDataset/1/model.pt'
        model, device = load_model_and_parallel(model, '0,1', ckpt_path=None, load_type='many2one')
    else:
        model.to(device)

    t_total = config.num_epochs * len(train_dataloader)
    if config.use_scheduler:
        optimizer, scheduler = build_optimizer_and_scheduler(config, model, t_toal=t_total)
    else:
        optimizer = build_optimizer(config, model)
    # optimizer = build_optimizer(config, model)

    if config.use_metric_summary_writer:
        metric_writer = SummaryWriter(
            os.path.join(config.tensorboard_dir,
                         "metric_{} {}-{} {}-{}-{}".format(config.model_name, now.month, now.day,
                                                           now.hour, now.minute,
                                                           now.second)))
    if config.use_parameter_summary_writer:
        if not os.path.exists(config.tensorboard_dir):
            os.makedirs(config.tensorboard_dir)
        parameter_writer = SummaryWriter(
            os.path.join(config.tensorboard_dir, "parameter_{} {}-{} {}-{}-{}".format(config.model_name, now.month,
                                                                                      now.day,
                                                                                      now.hour, now.minute,
                                                                                      now.second)))

    best_model = None
    global_step = 0
    best_p = best_r = best_f1 = 0.
    best_epoch = 0
    # 使用wandb来记录模型训练的时候各种参数....
    # wandb.watch(model, torch.nn.CrossEntropyLoss, log="all", log_freq=2)
    # requires_grad_nums, parameter_nums = count_parameters(model)
    # set_trace()
    for epoch in range(1, config.num_epochs + 1):
        batch_loss = 0.
        batch_train_f1 = 0.
        batch_train_p = 0.
        batch_train_r = 0.
        batch_train_acc = 0.

        all_train_labels = []
        all_predicate_tokens = []
        binary_all_train_labels = []
        binary_all_predicate_tokens = []

        model.train()
        for idx, batch_data in tqdm(enumerate(train_dataloader),total=len(train_dataloader),
                                    desc="数据集:{},{}_{}....".format(config.dataset_name, config.bert_name,
                                                                   config.model_name)):

            if config.data_format == 'single':

                input_ids, token_type_ids, attention_masks, e1_mask, e2_mask, labels, rel_type = batch_data

                loss, logits = model(input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask,
                                     rel_type=rel_type)
                predicate_token = relation_classification_decode(logits)

            elif config.data_format == 'cross':
                input_ids, token_type_ids, attention_masks, e1_mask, e2_mask, labels, rel_type = batch_data

                loss, logits = model(input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask,
                                     rel_type=rel_type)
                predicate_token = relation_classification_decode(logits)
                # rel_type = rel_type.cpu().numpy()
                # if config.dataset_name == 'AllDataset':
                #     # 将1转变为1,2,3,4,5
                #     tmp_predicate = []
                #     for idx in range(len(labels)):
                #         if predicate_token[idx] == 0:
                #             tmp_predicate.append(0)
                #         else:
                #             tmp_predicate.append(rel_type[idx])
                #     predicate_token = tmp_predicate
            else:
                raise ValueError

            loss = loss.mean()

            if config.use_fp16:
                scaler.scale(loss).backward()
                if (idx + 1) % config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    if config.use_scheduler:
                        scheduler.step()
                    if config.use_metric_summary_writer:
                        save_parameter_writer(parameter_writer, model, global_step)
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (idx + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    if config.use_scheduler:
                        scheduler.step()
                    if config.use_metric_summary_writer:
                        save_parameter_writer(parameter_writer, model, global_step)
                    optimizer.zero_grad()

            learning_rate = optimizer.param_groups[0]['lr']
            binary_labels = (labels > 0).long()
            binary_labels = binary_labels.cpu().numpy()
            multi_labels = labels.cpu().numpy()
            all_labels = [id for _, id in label2id.items()]

            # 分类二分类和三分类的结果
            binary_all_train_labels.extend(binary_labels)
            all_train_labels.extend(multi_labels)

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
            #     acc = accuracy_score(binary_labels, predicate_token)
            #     p_r_f1_s = precision_recall_fscore_support(binary_labels, predicate_token, average='binary')
            # else:
                # p_r_f1_s_ = precision_recall_fscore_support(labels, predicate_token, labels=all_labels,
                #                                            average=config.evaluate_mode)
            acc = accuracy_score(multi_labels, new_predicate_token)
            # all_labels = all_labels[1:]

            p_r_f1_s = precision_recall_fscore_support(multi_labels, new_predicate_token, labels=all_labels,
                                                       average=config.evaluate_mode)

            tmp_train_p = p_r_f1_s[0]
            tmp_train_r = p_r_f1_s[1]
            tmp_train_f1 = p_r_f1_s[2]
            batch_train_f1 += tmp_train_f1
            batch_train_p += tmp_train_p
            batch_train_r += tmp_train_r
            batch_train_acc += acc
            batch_loss += loss.item()
            if config.use_wandb:
                wandb_log(wandb, epoch, global_step, tmp_train_f1, tmp_train_p, tmp_train_r, acc, loss.item(),
                          type_='train',evaluate_mode=config.evaluate_mode, learning_rate=learning_rate)
            if config.train_verbose and global_step % config.print_step == 0:
                show_log(logger, idx, len(train_dataloader), t_total, epoch, global_step, loss, tmp_train_p,
                         tmp_train_r, tmp_train_f1, acc, config.evaluate_mode, type_='train')

                # tmp_train_p = p_r_f1_s_[0]
                # tmp_train_r = p_r_f1_s_[1]
                # tmp_train_f1 = p_r_f1_s_[2]
                # show_log(logger, idx, len(train_dataloader), t_total, epoch, global_step, loss, tmp_train_p,
                #          tmp_train_r, tmp_train_f1, acc, config.evaluate_mode, type='train')
                # logger.info('average_Loss:{:.5f}'.format(batch_train_f1/idx))
                # logger.info('average_Accuracy:{:.5f}'.format(batch_train_acc/idx))
                #
                # logger.info('average_Precision:{:.5f}'.format(batch_train_p/idx))
                # logger.info('average_Recall:{:.5f}'.format(batch_train_r/idx))
                # logger.info('average_F1:{:.5f}'.format(batch_train_f1/idx))
            global_step += 1

        count = len(train_dataloader)
        batch_loss = batch_loss / count
        train_p = batch_train_p / count
        train_r = batch_train_r / count
        train_f1 = batch_train_f1 / count

        show_log(logger, -1, len(train_dataloader), t_total, epoch, global_step, batch_loss, train_p, train_r, train_f1,
                 0.00, config.evaluate_mode, type_='train', scheme=1)

        reports = classification_report(binary_all_train_labels, binary_all_predicate_tokens, labels=[0, 1],digits=4)
        reports1 = classification_report(all_train_labels, all_predicate_tokens, labels=[0, 1, 2, 3, 4, 5],digits=4)
        logger.info("-------训练集epoch:{} 报告----------".format(epoch))
        logger.info(reports)
        logger.info(reports1)

        dev_p, dev_r, dev_f1 = dev(model, config, tokenizer, label2id, device, epoch=epoch, global_step=global_step,
                                   logger=logger, type_='dev')
        if config.dataset_name in ['BC6ChemProt']:
            test_p, test_r, test_f1 = dev(model, config, tokenizer, label2id, device, epoch=epoch,
                                          global_step=global_step,
                                          logger=logger, type_='test')
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
    if config.use_metric_summary_writer:
        metric_writer.close()
    if config.use_parameter_summary_writer:
        parameter_writer.close()
    logger.info('----------------本次模型运行的参数------------------')
    print_hyperparameters(config, logger)
    # Optional


if __name__ == '__main__':
    config = get_bert_config()

    # 设置时间
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff

    set_seed(config.seed)


    # 测试wandb
    warnings.filterwarnings("ignore")
    if config.freeze_bert:
        if config.use_scheduler:
            wandb_name = f'multi_task_{config.bert_name}_{config.model_name}_free_nums{config.freeze_layer_nums}_scheduler{config.warmup_proportion}_bs{config.batch_size}_schema{config.scheme}_lr{config.bert_lr}'
        else:
            wandb_name = f'multi_task_{config.bert_name}_{config.model_name}_free_nums{config.freeze_layer_nums}_bs{config.batch_size}_schema{config.scheme}_lr{config.bert_lr}'
    else:

        if config.use_scheduler:
            wandb_name = f'multi_task_{config.bert_name}_{config.model_name}_no_freeze_scheduler{config.warmup_proportion}_bs{config.batch_size}_schema{config.scheme}_lr{config.bert_lr}'
        else:
            wandb_name = f'multi_task_{config.bert_name}_{config.model_name}_no_freeze_bs{config.batch_size}_schema{config.scheme}_lr{config.bert_lr}'
    config.output_dir = './outputs/save_models/{}/{}/{}/{}/'.format(str(datetime.date.today()), wandb_name,
                                                                    config.model_name, config.dataset_name)
    config.logs_dir = './outputs/logs/{}/{}/{}/{}/'.format(str(datetime.date.today()), wandb_name, config.model_name,
                                                           config.dataset_name)
    logger = get_logger(config)
    if 'CV' in config.dataset_name:
        project_name = '关系分类-AllDataset'
        wandb_name = config.logfile_name+wandb_name
    else:
        project_name = "关系分类-{}".format(config.dataset_name)
    # project表示这次项目，entity:表示提交人，config为超参数

    if config.run_type == 'normal':
        ckpt_path = ''

        if config.use_wandb:
            warnings.filterwarnings("ignore")
            wandb.init(project=project_name, config=vars(config), name=wandb_name)
        logger.info('----------------本次模型运行的参数--------------------')
        print_hyperparameters(config, logger)

        train(config, logger)
    elif config.run_type == 'cv5':

        for i in range(1, 6):
            if config.freeze_bert:
                if config.use_scheduler:
                    wandb_name = f'cv_{i}_multi_task_{config.bert_name}_{config.model_name}_free_nums{config.freeze_layer_nums}_scheduler{config.warmup_proportion}_bs{config.batch_size}_schema{config.scheme}_lr{config.bert_lr}'
                else:
                    wandb_name = f'cv_{i}_multi_task_{config.bert_name}_{config.model_name}_free_nums{config.freeze_layer_nums}_bs{config.batch_size}_schema{config.scheme}_lr{config.bert_lr}'
            else:
                if config.use_scheduler:
                    wandb_name = f'cv_{i}_multi_task_{config.bert_name}_{config.model_name}_no_freeze_scheduler{config.warmup_proportion}_bs{config.batch_size}_schema{config.scheme}_lr{config.bert_lr}'
                else:
                    wandb_name = f'cv_{i}_multi_task_{config.bert_name}_{config.model_name}_no_freeze_bs{config.batch_size}_schema{config.scheme}_lr{config.bert_lr}'

            if config.use_wandb:
                warnings.filterwarnings("ignore")
                wandb.init(project="关系分类-{}".format(config.dataset_name), config=vars(config), name=wandb_name)

            logger.info('-----------CV:{}-----------'.format(i))
            ckpt_path = ''
            set_cv_config(config, i)
            logger.info('----------------本次模型运行的参数--------------------')

            train(config, logger)
            print_hyperparameters(config, logger)
    # else:
    #     raise ValueError

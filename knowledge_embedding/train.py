# -*- encoding: utf-8 -*-
"""
@File    :   train.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/17 20:25   
@Description :   训练文件

"""
import datetime
import os
from os.path import join

import numpy as np
import torch
import wandb
from ipdb import set_trace

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

from dev import link_predication_dev
from drug_target_dev import DT_link_predication_dev
from predicate import star_predict
from src.models.kg_bert import KGBertModel
from src.models.kgbert_convkb import KGBertConvKBModel
from src.models.star_model import BertForPairScoring
from src.utils.dataset_utils import KGProcessor, KGBertDataset
from src.utils.function_utils import get_config, get_logger, set_seed, save_parameter_writer, wandb_log, show_log, \
    print_hyperparameters, save_json
from src.utils.kb_dataset import DatasetForPairwiseRankingLP
from src.utils.metric_utils import calculate_metrics_for_link_prediction
from src.utils.train_utils import load_model_and_parallel, build_bert_optimizer_and_scheduler, get_metric_writer, get_parameter_writer, build_optimizer


def link_predication_train(config, logger):
    if config.bert_name in ['bert', 'biobert', 'wwm_bert', 'flash', 'flash_quad', 'scibert']:
        tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
    else:
        raise ValueError("bert_name的值应该为['biobert','wwm_bert','flash_quad','flash']")

    # 数据读取阶段
    if config.model_name == 'kgbert':
        processor = KGProcessor(debug=config.debug)
        label_list = processor.get_labels()
        num_labels = len(label_list)
        entity_list = processor.get_entities(config.data_dir)
        train_examples = processor.get_train_examples(config.data_dir)
        if config.debug:  # 开启debug，则试验一个batch的数据
            train_examples = train_examples[:config.batch_size * 5]

        train_dataset = KGBertDataset(config, train_examples, tokenizer)

        train_loader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=0,
                                  batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    elif config.model_name == 'star':
        train_dataset = DatasetForPairwiseRankingLP(
            config.dataset_name, "train", None, "./dataset/",
            'bert', tokenizer, config.do_lower_case,
            config.max_len, neg_times=5, neg_weights=[1., 1., 0.],
            type_cons_neg_sample=True, type_cons_ratio=0.
        )
        batch_size = config.batch_size
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size,collate_fn=train_dataset.data_collate_fn,num_workers=0)

        # 进行删除
        # dev_dataset = DatasetForPairwiseRankingLP(
        #     config.dataset_name, "dev", None, "./dataset/",
        #     'bert', tokenizer, config.do_lower_case,
        #     config.max_len,
        # )
        test_dataset = DatasetForPairwiseRankingLP(
            config.dataset_name, "test", None, "./dataset/",
            'bert', tokenizer, config.do_lower_case,
            config.max_len,
        )


    else:
        raise ValueError("model name错误")

    if config.model_name == 'kgbert':
        model = KGBertModel(config, 2)
    elif config.model_name == 'star':

        config.vocab_size = len(tokenizer)
        model = BertForPairScoring(config)

    elif config.model_name == 'kgbert_convkb':
        model = KGBertConvKBModel(config, 2)
    else:
        raise ValueError("请选择正确的model name")
    #  加载模型是否是多GPU或者
    if config.use_n_gpu and torch.cuda.device_count() > 1:
        model, device = load_model_and_parallel(model, '0,1', ckpt_path=None, load_type='many2one')
    else:
        device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')
        model.to(device)

    if config.use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    if config.metric_summary_writer:
        metric_writer = get_metric_writer(config)
    if config.parameter_summary_writer:
        parameter_writer = get_parameter_writer(config)

    t_total = config.num_epochs * len(train_loader)
    best_epoch = 0

    best_model = None
    best_hit1 = 0.
    best_hit3 = 0.
    best_hit10 = 0.
    best_MR = 0.
    best_MRR = 0.

    if config.use_scheduler:
        optimizer, scheduler = build_bert_optimizer_and_scheduler(config, model, t_toal=t_total)
        logger.info('学习率调整器:{}'.format(scheduler))
    else:
        optimizer = build_optimizer(config, model)

    logger.info('优化器:{}'.format(optimizer))
    global_step = 0

    # --------------------------
    # dataset_list = [train_dataset, test_dataset]
    #
    # tuple_ranks, metric_res = star_predict(
    #     config, test_dataset.raw_examples, dataset_list, model, device, verbose=True)
    # output_str = calculate_metrics_for_link_prediction(tuple_ranks, verbose=True)
    # save_json(tuple_ranks, join(config.output_dir, "tuple_ranks.json"))
    # --------------------------



    for epo_idx, epoch in enumerate(range(config.num_epochs)):
        epoch += 1
        model.train()
        epoch_acc = 0.
        epoch_loss = 0.
        for step, batch_data in enumerate(train_loader):
            if config.model_name == 'kgbert':
                input_ids, attention_masks, token_type_ids, label_ids = batch_data
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                token_type_ids = token_type_ids.to(device)
                label_ids = label_ids.to(device)

                logits, loss = model(input_ids, token_type_ids, attention_masks, label_ids)
            elif config.model_name == 'kgbert_convkb':
                input_ids, attention_masks, token_type_ids, label_ids, head_mask, rel_mask, tail_mask = batch_data
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                token_type_ids = token_type_ids.to(device)
                label_ids = label_ids.to(device)

                head_mask = head_mask.to(device)
                rel_mask = rel_mask.to(device)
                tail_mask = tail_mask.to(device)

                logits, loss = model(input_ids, token_type_ids, attention_masks, label_ids, head_mask, rel_mask,tail_mask)

            elif config.model_name == 'star':

                batch = tuple(t.to(device) for t in batch_data)
                inputs = train_dataset.batch2feed_dict(batch)
                outputs = model(**inputs)

                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            else:
                raise ValueError()
            loss = loss.mean()

            lr = optimizer.param_groups[0]['lr']
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
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    if config.use_scheduler:
                        scheduler.step()
                    if config.parameter_summary_writer:
                        save_parameter_writer(parameter_writer, model, global_step)
                    optimizer.zero_grad()

            # label_ids = label_ids.cpu().numpy()
            # logits = logits.detach().cpu().numpy()
            # preds = np.argmax(logits, axis=1)
            # acc = (preds == label_ids).mean()
            # epoch_acc += acc
            epoch_loss += loss


            global_step += 1

            show_log(epo_idx, config.num_epochs, step, len(train_loader), t_total, global_step, type='train', scheme=0,
                     metric={"loss": epoch_loss/step, 'lr': lr})
            if config.use_wandb:
                wandb_log(wandb, epoch, global_step, type='train', metric={"loss": epoch_loss/step, 'lr': lr})

        train_loss = epoch_loss / len(train_loader)

        show_log(epo_idx, config.num_epochs, 0, len(train_loader), t_total, global_step, type='train', scheme=1,
                 metric={"loss": train_loss})
        # 每五次进行
        if epoch % 4 == 0:
            if config.model_name in ['kgbert','kgbert_convkb']:
                if config.dataset_name in ['umls', 'WN18RR', 'FB15K-237']:
                    metric_res = link_predication_dev(config, logger, model, tokenizer, device, train_type='dev',
                                                      entity_list=entity_list)
                elif config.dataset_name in ['drugbank', 'drug_central', 'drug_target_dataset']:
                    metric_res = DT_link_predication_dev(config, logger, model, tokenizer, device, train_type='dev')
                else:
                    raise ValueError("dataset_name错误，选择合适的dataset")


            elif config.model_name == 'star':

                #dataset_list = [train_dataset, dev_dataset, test_dataset]
                dataset_list = [train_dataset, test_dataset]

                tuple_ranks,metric_res = star_predict(
                    config, test_dataset.raw_examples, dataset_list, model,device, verbose=True)
                output_str = calculate_metrics_for_link_prediction(tuple_ranks, verbose=True)
                save_json(tuple_ranks, join(config.output_dir, "tuple_ranks.json"))
                with open(join(config.output_dir, "link_prediction_metrics.txt"), "w", encoding="utf-8") as fp:
                    fp.write(output_str)\

            if config.use_wandb:
                wandb_log(wandb, epoch, global_step, type='dev', verbose=config.metric_verbose, metric=metric_res)
            show_log(epo_idx, config.num_epochs, 0, len(train_loader), t_total, global_step, type='dev', scheme=1,
                     metric=metric_res, verbose=config.metric_verbose)
            if metric_res['hit@10'] > best_hit10:
                best_epoch = epoch
                best_hit1 = metric_res['hit@1']
                best_hit3 = metric_res['hit@3']
                best_hit10 = metric_res['hit@10']
                best_MR = metric_res['MR']
                best_MRR = metric_res['MRR']
    # # 将最佳结果放入到最后一个epoch
    wandb.log({
        'best-Hit@1': best_hit1,
        'best-Hit@3': best_hit3,
        'best-Hit@10': best_hit10,
        'best-MR': best_MR,
        'dev-MRR': best_MRR})
    logger.info('*******此次运行结果完成，共:{}个epoch********'.format(config.num_epochs))
    logger.info('best-epoch:{}'.format(best_epoch))
    logger.info('best-hit@1'.format(best_hit1))
    logger.info('best-hit@3'.format(best_hit3))
    logger.info('best-hit@10'.format(best_hit10))
    logger.info('best-MR'.format(best_MR))
    logger.info('best-MRR'.format(best_MRR))


if __name__ == '__main__':
    config = get_config()
    logger = get_logger(config)
    if config.use_wandb:
        wandb_name = f'{config.bert_name}_{config.model_name}_bs{config.batch_size}_max_lens_{config.max_len}'
        set_trace()
        wandb.init(project="知识嵌入-{}".format(config.dataset_name), config=vars(config),
                   name=wandb_name)
    # 设置时间
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff
    # 设置随机种子
    set_seed(config.seed)
    print_hyperparameters(config)
    link_predication_train(config, logger)

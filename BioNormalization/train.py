# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :   在单个数据集上训练模型

   Author :        kedaxia
   date：          2022/01/18
   Copyright:      (c) kedaxia 2022
-------------------------------------------------
   Change Activity:
                   2022/01/18: 
-------------------------------------------------
"""
import os
import datetime
from copy import deepcopy

import wandb
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from ipdb import set_trace

from tqdm import tqdm

from biosyn_evaluate import dev, test
from sapbert_train import sapbert_train
from src.data_loader import NormalizationDataset
from src.models.biosyn import BioSyn
from src.models.biosyn_lite import BioSynLite,RerankNet
from src.models.multi_biosyn import MultiBioSyn
from utils.dataset_utils import load_dictionary, load_queries
from utils.function_utils import get_config, get_logger, set_seed, save_model, count_parameters
from utils.train_utils import build_optimizer


def biosyn_train(config, logger):
    # prepare for output
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # 加载字典，shape=(dictioanry_size,2)
    # 第一列为name，第二类为ID

    if config.dataset_name in ['bc5cdr-chemical','bc5cdr-disease','ncbi-disease']:
        train_dictionary = load_dictionary(dictionary_path=config.train_dictionary_path)
        train_queries = load_queries(
            data_dir=config.train_dir,
            filter_composite=True,
            filter_duplicate=True,
            filter_cuiless=True
        )
        eval_dictionary=None
    else:
        train_dictionary = load_dictionary(dictionary_path=config.dictionary_path)
        train_queries = load_queries(
            config.train_path,
            filter_composite=True,
            filter_duplicate=True,
            filter_cuiless=True
        )
        eval_dictionary = train_dictionary
    if config.debug:  # 开启debug，使用小部分数据集进行测试
        train_dictionary = train_dictionary[:100]
        train_queries = train_queries[:10]
        config.output_dir = config.output_dir + "_draft"

    device = torch.device('cuda') if config.use_gpu else torch.device('cpu')
    # 获取词典中的所有name
    train_dictionary_names = train_dictionary[:, 0]
    train_mentions = train_queries[:, 0]

    # biosyn = BioSyn(
    #     config=config,
    #     device=device,
    #     initial_sparse_weight=0
    # )

    biosyn = BioSynLite(
        config=config,
        device=device,
        initial_sparse_weight=0
    )

    # 使用tf-idf来计算得到sparse vector，先对字典中的name进行encode
    biosyn.init_sparse_encoder(corpus=train_dictionary_names)
    # 加载BioBERT等预训练模型...
    biosyn.load_dense_encoder(config.bert_dir)

    # 这个是训练模型训练的核心
    model = RerankNet(
        config,
        dense_encoder=biosyn.get_dense_encoder(),
        sparse_weight=biosyn.get_sparse_weight(),
        device=device
    )

    logger.info("开始计算得到Sparse embedding")
    # 这是对训练集的entity和字典进行编码
    # 因为sparse representation是不会发生改变的，因此最开始就继续宁计算
    train_query_sparse_embeds = biosyn.get_sparse_representation(
        mentions=train_mentions)  # train.shape = (1587, 1122),实体数目为1587
    train_dict_sparse_embeds = biosyn.get_sparse_representation(
        mentions=train_dictionary_names)  # shape=(71924, 1122),字典数目为71924
    # 计算sparse similiarity scores，采用inner dot计算相似分数
    train_sparse_score_matrix = biosyn.get_score_matrix(
        query_embeds=train_query_sparse_embeds,
        dict_embeds=train_dict_sparse_embeds
    )

    # 然后这里根据sprase representation来选择出top k=20
    train_sparse_candidate_idxs = biosyn.retrieve_candidate(
        score_matrix=train_sparse_score_matrix,
        topk=config.topk
    )

    train_set = NormalizationDataset(
        queries=train_queries,
        dicts=train_dictionary,
        tokenizer=biosyn.get_dense_tokenizer(),
        s_score_matrix=train_sparse_score_matrix,
        s_candidate_idxs=train_sparse_candidate_idxs,
        topk=config.topk,
        d_ratio=config.dense_ratio,
        max_len=config.max_len
    )
    if config.use_fp16:
        scaler = torch.cuda.amp.GradScaler()


    train_dataloader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    t_total = config.num_epochs * len(train_dataloader)
    global_step = 0
    optimizer = build_optimizer(config, model)
    # # model.eval()
    # # acc1, acc5 = dev(config=config, logger=logger, biosyn=biosyn, device=device, eval_dictionary=eval_dictionary)
    #
    # if config.use_wandb:
    #     wandb.log(
    #         {"untrained-epoch": 0, 'untrained-hit@1': acc1, 'untrained-hit@5': acc5}, step=global_step)

    best_model = None
    best_epoch = 0
    best_acc1=0.
    best_acc5=0.

    for epoch in range(1, config.num_epochs + 1):
        logger.info('>>>>>>>>>>>>>>>>>进入训练状态<<<<<<<<<<<<<<<<<<<<<<<<<<')
        # 得到dense representation....
        # 直接将所有的train_mentions得到其dense representation
        train_query_dense_embeds = biosyn.get_dense_representation(mentions=train_mentions, verbose=True)
        train_dict_dense_embeds = biosyn.get_dense_representation(mentions=train_dictionary_names, verbose=True)
        # 计算dense scores，就是inner dot计算得到
        # score_matrix.shape = (len(train_dataset),len(dict_len))
        # 相当于得到对于每个单词其其与词典中每个单词的相似度
        train_dense_score_matrix = biosyn.get_score_matrix(
            query_embeds=train_query_dense_embeds,
            dict_embeds=train_dict_dense_embeds
        )
        # 得到最终的top index(已经经过排序)
        # 这是得到每个train_data,其最可能在dict中对应的word的idx
        train_dense_candidate_idxs = biosyn.retrieve_candidate(
            score_matrix=train_dense_score_matrix,
            topk=config.topk
        )
        # 在训练的过程中，由dense representation得到的结果会不停地发生改变，不断地更新最有可能candidate
        train_set.set_dense_candidate_idxs(d_candidate_idxs=train_dense_candidate_idxs)

        train_loss = 0.
        train_steps = 0

        model.train()
        for step, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):


            # batch_y就是label，为CUI
            batch_x, batch_y = data

            batch_pred = model(batch_x)
            if config.use_n_gpu and torch.cuda.device_count()>1:
                loss = model.module.get_loss(batch_pred, batch_y)
            else:
                loss = model.get_loss(batch_pred, batch_y)
            train_loss += loss.item()
            if config.use_wandb:
                wandb.log(
                    {"train-epoch": epoch, 'margin_loss': loss.item(),
                     'train_lr': optimizer.param_groups[0]['lr']}, step=global_step)
            if config.use_fp16:
                scaler.scale(loss).backward()
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    # if config.use_scheduler:
                    #     scheduler.step()

                    optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    # if config.use_scheduler:
                    #     scheduler.step()

                    optimizer.zero_grad()

            logger.info(
                'Epoch:{} 训练中>>>>>> {}/{} loss:{:5f} ,lr={},sparse_weight={:.5f}'.format(epoch, global_step, t_total,
                                                                                         loss.item(),
                                                                                         optimizer.param_groups[0]['lr'],
                                                                                         model.sparse_weight.item()))
            train_steps += 1
            global_step += 1

        train_loss /= (train_steps + 1e-9)
        logger.info(
            'Epoch:{} 训练中完成 {}/{} loss:{:5f} ,lr={},sparse_weight={:.5f}'.format(epoch, global_step, t_total,
                                                                                     train_loss,
                                                                                     optimizer.param_groups[0][
                                                                                         'lr'],
                                                                                     model.sparse_weight.item()))
        if config.use_wandb:
            wandb.log(
                {"train-epoch": epoch, 'margin_loss': train_loss, 'sparse_weight': model.sparse_weight.item()},
                step=global_step)
        logger.info('>>>>>>>>>>>>>>>>>进入验证装填<<<<<<<<<<<<<<<<<<<<<<<<<<')
        model.eval()
        acc1, acc5 = dev(config=config, logger=logger, biosyn=biosyn, device=device,eval_dictionary=eval_dictionary)

        if config.use_wandb:
            wandb.log(
                {"dev-epoch": epoch, 'dev-hit@1': acc1, 'dev-hit@5': acc5}, step=global_step)

        if best_acc1<acc1:
            best_epoch = epoch
            best_acc1 = acc1
            best_acc5 = acc5
            best_model = deepcopy(model)
        if config.save_model:
            path = os.path.join(config.output_dir, config.dataset_name,str(epoch))
            logger.info("模型保存到:{}".format(path))
            model.save_model()
    if config.save_model:
        path = os.path.join(config.output_dir,config.dataset_name,'best_model')
        logger.info("模型保存到:{}".format(path))
        best_model.save_model()
        # acc1, acc5 = test(config=config, logger=logger, biosyn=biosyn, device=device)
        # if config.use_wandb:
        #     wandb.log(
        #         {"test-epoch": epoch, 'test-hit@1': acc1, 'test-hit@5': acc5}, step=global_step)

if __name__ == '__main__':

    config = get_config()

    logger = get_logger(config)

    # 设置时间
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff
    # 设置随机种子
    set_seed(config.seed)
    if config.freeze_bert:
        wandb_name = f'freeze_{config.freeze_layer_nums}_{config.model_name}_bs{config.batch_size}_maxlen{config.max_len}'
    else:
        wandb_name = f'{config.model_name}_bs{config.batch_size}_maxlen{config.max_len}'

    config.output_dir = './outputs/save_models/{}/{}/{}/{}/'.format(str(datetime.date.today()), wandb_name,
                                                                    config.model_name, config.dataset_name)
    config.logs_dir = './outputs/logs/{}/{}/{}/{}/'.format(str(datetime.date.today()), wandb_name, config.model_name,
                                                           config.dataset_name)

    if config.model_name == 'biosyn':
        logger.info('<<<<<<<<<<<<<<<<<<对BioSyn进行预训练>>>>>>>>>>>>>>>>>>>>')
        if config.use_wandb:

            wandb.init(project="实体标准化-{}".format(config.dataset_name),config=vars(config),
                       name=wandb_name)
        biosyn_train(config, logger)
    elif config.model_name == 'sapbert':
        logger.info('<<<<<<<<<<<<<<<<<<对SapBERT进行预训练>>>>>>>>>>>>>>>>>>>>')
        if config.use_wandb:
            wandb_name = f'{config.model_name}_bs{config.batch_size}_maxlen{config.max_len}'
            wandb.init(project="实体标准化-{}".format(config.dataset_name), config=vars(config),
                       name=wandb_name)
        sapbert_train(config, logger)
    else:
        raise NotImplementedError("暂时没有")

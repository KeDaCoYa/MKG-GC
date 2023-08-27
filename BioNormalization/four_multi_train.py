# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :   这是同时训练三个数据集:bc5cdr-disease,bc5cdr-chemical,ncbi-disease

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
import random

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

import wandb
from ipdb import set_trace
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from four_biosyn_evaluate import  test
from sapbert_train import sapbert_train
from src.data_loader import NormalizationDataset
from src.models.four_multi_biosyn import MultiBioSynFour, MyRerankNetFour
from utils.dataset_utils import load_dictionary, load_queries, load_my_data
from utils.function_utils import get_config, get_logger, set_seed, save_model, load_model_and_parallel, \
    print_hyperparameters
from utils.train_utils import marginal_nll, build_optimizer


def biosyn_train(config, logger):
    # prepare for output

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # 加载字典，shape=(dictioanry_size,2)
    # 第一列为name，第二类为ID

    bc5cdr_disease_train_dictionary = load_dictionary(config.bc5cdr_disease_train_dictionary_path)
    bc5cdr_chemical_train_dictionary = load_dictionary(config.bc5cdr_chemical_train_dictionary_path)
    ncbi_disease_train_dictionary = load_dictionary(config.ncbi_disease_train_dictionary_path)
    bc2gm_train_dictionary = load_dictionary(config.bc2gm_train_dictionary_path)

    bc5cdr_disease_train_queries = load_my_data(config.bc5cdr_disease_train_path)
    bc5cdr_chemical_train_queries = load_my_data(config.bc5cdr_chemical_train_path)
    ncbi_disease_train_queries = load_my_data(config.ncbi_disease_train_path)
    bc2gm_train_queries = load_my_data(config.bc2gm_train_path)


    if config.debug:  # 开启debug，使用小部分数据集进行测试
        bc5cdr_disease_train_dictionary = bc5cdr_disease_train_dictionary[:100]
        bc5cdr_chemical_train_dictionary = bc5cdr_chemical_train_dictionary[:100]
        ncbi_disease_train_dictionary = ncbi_disease_train_dictionary[:100]
        bc5cdr_disease_train_queries = bc5cdr_disease_train_queries[:100]
        bc5cdr_chemical_train_queries = bc5cdr_chemical_train_queries[:100]
        ncbi_disease_train_queries = ncbi_disease_train_queries[:100]
        bc2gm_train_queries = bc2gm_train_queries[:100]
        config.output_dir = config.output_dir + "_draft"

    device = torch.device('cuda') if config.use_gpu else torch.device('cpu')
    # 获取词典中的所有name
    bc5cdr_disease_train_dictionary_names = bc5cdr_disease_train_dictionary[:, 0]
    bc5cdr_chemical_train_dictionary_names = bc5cdr_chemical_train_dictionary[:, 0]
    ncbi_disease_train_dictionary_names = ncbi_disease_train_dictionary[:, 0]
    bc2gm_train_dictionary_names = bc2gm_train_dictionary[:, 0]

    ncbi_disease_train_mentions = ncbi_disease_train_queries[:, 0]
    bc5cdr_disease_train_mentions = bc5cdr_disease_train_queries[:, 0]
    bc5cdr_chemical_train_mentions = bc5cdr_chemical_train_queries[:, 0]
    bc2gm_train_mentions = bc2gm_train_queries[:, 0]

    # 在这里将会对数据进行统一个数，这里设定为一个平均个数
    ncbi_disease_train_mentions = ncbi_disease_train_mentions.tolist()
    bc5cdr_disease_train_mentions = bc5cdr_disease_train_mentions.tolist()
    bc5cdr_chemical_train_mentions = bc5cdr_chemical_train_mentions.tolist()
    bc2gm_train_mentions = bc2gm_train_mentions.tolist()

    bc5cdr_disease_train_queries = bc5cdr_disease_train_queries.tolist()
    bc5cdr_chemical_train_queries = bc5cdr_chemical_train_queries.tolist()
    ncbi_disease_train_queries = ncbi_disease_train_queries.tolist()
    bc2gm_train_queries = bc2gm_train_queries.tolist()

    avg_count = max(len(ncbi_disease_train_mentions),len(bc5cdr_chemical_train_mentions),len(bc5cdr_chemical_train_mentions),len(bc2gm_train_mentions))
    #avg_count = (len(ncbi_disease_train_mentions)+len(bc5cdr_chemical_train_mentions)+len(bc5cdr_chemical_train_mentions)+len(bc2gm_train_mentions))//4
    if len(ncbi_disease_train_mentions)>avg_count:
        logger.info("ncbi_disease数据集从{}缩减到{}".format(len(ncbi_disease_train_mentions),avg_count))
        choise_idx = random.sample(range(len(ncbi_disease_train_mentions)),avg_count)
        ncbi_disease_train_mentions = [ncbi_disease_train_mentions[idx] for idx in choise_idx]
        ncbi_disease_train_queries = [ncbi_disease_train_queries[idx] for idx in choise_idx]

    else:
        # 随机重复已有的训练集
        logger.info("ncbi_disease数据集从{}扩展到{}".format(len(ncbi_disease_train_mentions), avg_count))
        for i in range(avg_count-len(ncbi_disease_train_mentions)):
            idx = random.randint(0,len(ncbi_disease_train_mentions)-1)

            ncbi_disease_train_mentions.append(ncbi_disease_train_mentions[idx])
            ncbi_disease_train_queries.append(ncbi_disease_train_queries[idx])

    if len(bc5cdr_disease_train_mentions)>avg_count:
        logger.info("bc5cdr_disease数据集从{}缩减到{}".format(len(bc5cdr_disease_train_mentions), avg_count))

        choise_idx = random.sample(range(len(bc5cdr_disease_train_mentions)), avg_count)
        bc5cdr_disease_train_mentions = [bc5cdr_disease_train_mentions[idx] for idx in choise_idx]
        bc5cdr_disease_train_queries = [bc5cdr_disease_train_queries[idx] for idx in choise_idx]
    else:
        # 随机重复已有的训练集
        logger.info("bc5cdr_disease数据集从{}扩展到{}".format(len(bc5cdr_disease_train_mentions), avg_count))
        for i in range(avg_count-len(bc5cdr_disease_train_mentions)):
            idx = random.randint(0, len(bc5cdr_disease_train_mentions)-1)

            bc5cdr_disease_train_mentions.append(bc5cdr_disease_train_mentions[idx])
            bc5cdr_disease_train_queries.append(bc5cdr_disease_train_queries[idx])
    if len(bc5cdr_chemical_train_mentions)>avg_count:
        logger.info("bc5cdr_chemical数据集从{}缩减到{}".format(len(bc5cdr_chemical_train_mentions), avg_count))
        choise_idx = random.sample(range(len(bc5cdr_chemical_train_mentions)), avg_count)
        bc5cdr_chemical_train_mentions = [bc5cdr_chemical_train_mentions[idx] for idx in choise_idx]
        bc5cdr_chemical_train_queries = [bc5cdr_chemical_train_queries[idx] for idx in choise_idx]
    else:
        # 随机重复已有的训练集
        logger.info("bc5cdr_chemical数据集从{}扩展到{}".format(len(bc5cdr_chemical_train_mentions), avg_count))
        for i in range(avg_count-len(bc5cdr_chemical_train_mentions)):
            idx = random.randint(0, len(bc5cdr_chemical_train_mentions)-1)
            bc5cdr_chemical_train_mentions.append(bc5cdr_chemical_train_mentions[idx])
            bc5cdr_chemical_train_queries.append(bc5cdr_chemical_train_queries[idx])

    if len(bc2gm_train_mentions)>avg_count:
        logger.info("bc2gm数据集从{}缩减到{}".format(len(bc2gm_train_mentions), avg_count))
        choise_idx = random.sample(range(len(bc2gm_train_mentions)), avg_count)
        bc2gm_train_mentions = [bc2gm_train_mentions[idx] for idx in choise_idx]
        bc2gm_train_queries = [bc2gm_train_queries[idx] for idx in choise_idx]
    else:
        # 随机重复已有的训练集
        logger.info("bc2gm数据集从{}扩展到{}".format(len(bc2gm_train_mentions), avg_count))
        for i in range(avg_count-len(bc2gm_train_mentions)):
            idx = random.randint(0, len(bc2gm_train_mentions)-1)
            try:
                bc2gm_train_mentions.append(bc2gm_train_mentions[idx])
            except:
                set_trace()
            bc2gm_train_queries.append(bc2gm_train_queries[idx])

    ncbi_disease_train_mentions = np.array(ncbi_disease_train_mentions)
    bc5cdr_disease_train_mentions = np.array(bc5cdr_disease_train_mentions)
    bc5cdr_chemical_train_mentions = np.array(bc5cdr_chemical_train_mentions)
    bc2gm_train_mentions = np.array(bc2gm_train_mentions)

    bc5cdr_disease_train_queries = np.array(bc5cdr_disease_train_queries)
    bc5cdr_chemical_train_queries = np.array(bc5cdr_chemical_train_queries)
    ncbi_disease_train_queries = np.array(ncbi_disease_train_queries)
    bc2gm_train_queries = np.array(bc2gm_train_queries)

    biosyn = MultiBioSynFour(
        config=config,
        device=device,
        initial_sparse_weight=0
    )

    # 使用tf-idf来计算得到sparse vector，先对字典中的name进行encode
    biosyn.init_bc5cdr_disease_sparse_encoder(corpus=bc5cdr_disease_train_dictionary_names)
    biosyn.init_bc5cdr_chem_sparse_encoder(corpus=bc5cdr_chemical_train_dictionary_names)
    biosyn.init_ncbi_disease_sparse_encoder(corpus=ncbi_disease_train_dictionary_names)
    biosyn.init_bc2gm_sparse_encoder(corpus=bc2gm_train_dictionary_names)

    # 加载BioBERT等预训练模型...
    biosyn.load_dense_encoder(config.bert_dir)

    # 这个是训练模型训练的核心
    model = MyRerankNetFour(
        config,
        dense_encoder=biosyn.get_dense_encoder(),
        sparse_weight=biosyn.get_sparse_weight(),
        device=device
    )

    if config.use_n_gpu and torch.cuda.device_count() > 1:
        model, device = load_model_and_parallel(model, '0,1', ckpt_path=None, load_type='many2one')
    else:
        device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')
        model.to(device)

    logger.info("开始计算得到Sparse embedding")
    # 这是对训练集的entity和字典进行编码
    # 因为sparse representation是不会发生改变的，因此最开始就继续宁计算
    bc5cdr_disease_train_query_sparse_embeds = biosyn.get_bc5cdr_disease_sparse_representation(
        mentions=bc5cdr_disease_train_mentions)  # train.shape = (1587, 1122),实体数目为1587
    bc5cdr_disease_train_dict_sparse_embeds = biosyn.get_bc5cdr_disease_sparse_representation(
        mentions=bc5cdr_disease_train_dictionary_names)  # shape=(71924, 1122),字典数目为71924
    # 计算sparse similiarity scores，采用inner dot计算相似分数
    bc5cdr_disease_train_sparse_score_matrix = biosyn.get_score_matrix(
        query_embeds=bc5cdr_disease_train_query_sparse_embeds,
        dict_embeds=bc5cdr_disease_train_dict_sparse_embeds
    )
    # 然后这里根据sprase representation来选择出top k=20
    bc5cdr_disease_train_sparse_candidate_idxs = biosyn.retrieve_candidate(
        score_matrix=bc5cdr_disease_train_sparse_score_matrix,
        topk=config.topk
    )
    bc5cdr_disease_train_set = NormalizationDataset(
        queries=bc5cdr_disease_train_queries,
        dicts=bc5cdr_disease_train_dictionary,
        tokenizer=biosyn.get_dense_tokenizer(),
        s_score_matrix=bc5cdr_disease_train_sparse_score_matrix,
        s_candidate_idxs=bc5cdr_disease_train_sparse_candidate_idxs,
        topk=config.topk,
        d_ratio=config.dense_ratio,
        max_len=config.max_len
    )
    bc5cdr_disease_train_dataloader = DataLoader(bc5cdr_disease_train_set, batch_size=config.batch_size, shuffle=True)


    bc5cdr_chemical_train_query_sparse_embeds = biosyn.get_bc5cdr_chemical_sparse_representation(
        mentions=bc5cdr_chemical_train_mentions)  # train.shape = (1587, 1122),实体数目为1587
    bc5cdr_chemical_train_dict_sparse_embeds = biosyn.get_bc5cdr_chemical_sparse_representation(
        mentions=bc5cdr_chemical_train_dictionary_names)  # shape=(71924, 1122),字典数目为71924
    # 计算sparse similiarity scores，采用inner dot计算相似分数
    bc5cdr_chemical_train_sparse_score_matrix = biosyn.get_score_matrix(
        query_embeds=bc5cdr_chemical_train_query_sparse_embeds,
        dict_embeds=bc5cdr_chemical_train_dict_sparse_embeds
    )
    # 然后这里根据sprase representation来选择出top k=20
    bc5cdr_chemical_train_sparse_candidate_idxs = biosyn.retrieve_candidate(
        score_matrix=bc5cdr_chemical_train_sparse_score_matrix,
        topk=config.topk
    )
    bc5cdr_chemical_train_set = NormalizationDataset(
        queries=bc5cdr_chemical_train_queries,
        dicts=bc5cdr_chemical_train_dictionary,
        tokenizer=biosyn.get_dense_tokenizer(),
        s_score_matrix=bc5cdr_chemical_train_sparse_score_matrix,
        s_candidate_idxs=bc5cdr_chemical_train_sparse_candidate_idxs,
        topk=config.topk,
        d_ratio=config.dense_ratio,
        max_len=config.max_len
    )
    bc5cdr_chemical_train_dataloader = DataLoader(bc5cdr_chemical_train_set, batch_size=config.batch_size, shuffle=True)


    ncbi_disease_train_query_sparse_embeds = biosyn.get_ncbi_disease_sparse_representation(
        mentions=ncbi_disease_train_mentions)  # train.shape = (1587, 1122),实体数目为1587
    ncbi_disease_train_dict_sparse_embeds = biosyn.get_ncbi_disease_sparse_representation(
        mentions=ncbi_disease_train_dictionary_names)  # shape=(71924, 1122),字典数目为71924
    # 计算sparse similiarity scores，采用inner dot计算相似分数
    ncbi_disease_train_sparse_score_matrix = biosyn.get_score_matrix(
        query_embeds=ncbi_disease_train_query_sparse_embeds,
        dict_embeds=ncbi_disease_train_dict_sparse_embeds
    )
    # 然后这里根据sprase representation来选择出top k=20
    ncbi_disease_train_sparse_candidate_idxs = biosyn.retrieve_candidate(
        score_matrix=ncbi_disease_train_sparse_score_matrix,
        topk=config.topk
    )

    ncbi_disease_train_set = NormalizationDataset(
        queries=ncbi_disease_train_queries,
        dicts=ncbi_disease_train_dictionary,
        tokenizer=biosyn.get_dense_tokenizer(),
        s_score_matrix=ncbi_disease_train_sparse_score_matrix,
        s_candidate_idxs=ncbi_disease_train_sparse_candidate_idxs,
        topk=config.topk,
        d_ratio=config.dense_ratio,
        max_len=config.max_len
    )
    ncbi_disease_train_dataloader = DataLoader(ncbi_disease_train_set, batch_size=config.batch_size, shuffle=True)

    bc2gm_train_query_sparse_embeds = biosyn.get_bc2gm_sparse_representation(
        mentions=bc2gm_train_mentions)  # train.shape = (1587, 1122),实体数目为1587
    bc2gm_train_dict_sparse_embeds = biosyn.get_bc2gm_sparse_representation(
        mentions=bc2gm_train_dictionary_names)  # shape=(71924, 1122),字典数目为71924
    # 计算sparse similiarity scores，采用inner dot计算相似分数
    bc2gm_train_sparse_score_matrix = biosyn.get_score_matrix(
        query_embeds=bc2gm_train_query_sparse_embeds,
        dict_embeds=bc2gm_train_dict_sparse_embeds
    )
    # 然后这里根据sprase representation来选择出top k=20
    bc2gm_train_sparse_candidate_idxs = biosyn.retrieve_candidate(
        score_matrix=bc2gm_train_sparse_score_matrix,
        topk=config.topk
    )

    bc2gm_train_set = NormalizationDataset(
        queries=bc2gm_train_queries,
        dicts=bc2gm_train_dictionary,
        tokenizer=biosyn.get_dense_tokenizer(),
        s_score_matrix=bc2gm_train_sparse_score_matrix,
        s_candidate_idxs=bc2gm_train_sparse_candidate_idxs,
        topk=config.topk,
        d_ratio=config.dense_ratio,
        max_len=config.max_len
    )
    bc2gm_train_dataloader = DataLoader(bc2gm_train_set, batch_size=config.batch_size, shuffle=True)



    if config.use_fp16:
        scaler = torch.cuda.amp.GradScaler()
    # todo: 这里进行修改，3sparse weight

    if config.use_n_gpu and torch.cuda.device_count() > 1:
        optimizer = optim.Adam([
            {'params': model.module.dense_encoder.parameters()},
            {'params': model.module.bc5cdr_disease_sparse_weight, 'lr': 0.01, 'weight_decay': 0},
            {'params': model.module.bc5cdr_chem_sparse_weight, 'lr': 0.01, 'weight_decay': 0},
            {'params': model.module.ncbi_disease_sparse_weight, 'lr': 0.01, 'weight_decay': 0},
            {'params': model.module.bc2gm_sparse_weight, 'lr': 0.01, 'weight_decay': 0},
            ],
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

    else:
        optimizer = optim.Adam([
            {'params': model.dense_encoder.parameters()},
            {'params': model.bc5cdr_disease_sparse_weight, 'lr': 0.01, 'weight_decay': 0},
            {'params': model.bc5cdr_chem_sparse_weight, 'lr': 0.01, 'weight_decay': 0},
            {'params': model.ncbi_disease_sparse_weight, 'lr': 0.01, 'weight_decay': 0},
            {'params': model.bc2gm_sparse_weight, 'lr': 0.01, 'weight_decay': 0},

            ],
            lr=config.learning_rate,
            weight_decay=config.weight_decay)
    if config.use_scheduler:
        t_total = config.num_epochs * len(ncbi_disease_train_dataloader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(config.warmup_proportion * t_total), num_training_steps=t_total
        )
    # if config.use_n_gpu and torch.cuda.device_count() > 1:
    #     optimizer = optim.Adam([
    #         {'params': model.module.dense_encoder.parameters()},
    #         {'params': model.module.bc5cdr_disease_sparse_weight, 'lr': 0.01, 'weight_decay': 0},
    #         ],
    #         lr=config.learning_rate,
    #         weight_decay=config.weight_decay
    #     )
    # else:
    #     optimizer = optim.Adam([
    #         {'params': model.dense_encoder.parameters()},
    #         {'params': model.bc5cdr_disease_sparse_weight, 'lr': 0.01, 'weight_decay': 0},
    #
    #         ],
    #         lr=config.learning_rate,
    #         weight_decay=config.weight_decay)

    t_total = config.num_epochs * len(bc5cdr_disease_train_dataloader)
    global_step = 1

    criterion = marginal_nll

    model.eval()
    #dev(config=config, logger=logger, biosyn=biosyn, device=device, wandb=wandb,epoch=0,global_step=0)
    # test(config=config, logger=logger, biosyn=biosyn, device=device, wandb=wandb,epoch=0,global_step=0)


    for epoch in range(1, config.num_epochs + 1):
        logger.info('>>>>>>>>>>>>>>>>>进入训练状态<<<<<<<<<<<<<<<<<<<<<<<<<<')
        # 得到dense representation....
        # 直接将所有的train_mentions得到其dense representation
        bc5cdr_disease_train_query_dense_embeds = biosyn.get_dense_representation(mentions=bc5cdr_disease_train_mentions, verbose=True)
        bc5cdr_disease_train_dict_dense_embeds = biosyn.get_dense_representation(mentions=bc5cdr_disease_train_dictionary_names, verbose=True)
        bc5cdr_disease_train_dense_score_matrix = biosyn.get_score_matrix(
            query_embeds=bc5cdr_disease_train_query_dense_embeds,
            dict_embeds=bc5cdr_disease_train_dict_dense_embeds
        )
        bc5cdr_disease_train_dense_candidate_idxs = biosyn.retrieve_candidate(
            score_matrix=bc5cdr_disease_train_dense_score_matrix,
            topk=config.topk
        )
        # 在训练的过程中，由dense representation得到的结果会不停地发生改变，不断地更新最有可能candidate
        bc5cdr_disease_train_set.set_dense_candidate_idxs(d_candidate_idxs=bc5cdr_disease_train_dense_candidate_idxs)

        bc5cdr_chemical_train_query_dense_embeds = biosyn.get_dense_representation(
            mentions=bc5cdr_chemical_train_mentions, verbose=True,type_=1)
        bc5cdr_chemical_train_dict_dense_embeds = biosyn.get_dense_representation(
            mentions=bc5cdr_chemical_train_dictionary_names, verbose=True,type_=1)
        bc5cdr_chemical_train_dense_score_matrix = biosyn.get_score_matrix(
            query_embeds=bc5cdr_chemical_train_query_dense_embeds,
            dict_embeds=bc5cdr_chemical_train_dict_dense_embeds
        )
        bc5cdr_chemical_train_dense_candidate_idxs = biosyn.retrieve_candidate(
            score_matrix=bc5cdr_chemical_train_dense_score_matrix,
            topk=config.topk
        )
        # 在训练的过程中，由dense representation得到的结果会不停地发生改变，不断地更新最有可能candidate
        bc5cdr_chemical_train_set.set_dense_candidate_idxs(d_candidate_idxs=bc5cdr_chemical_train_dense_candidate_idxs)

        ncbi_disease_train_query_dense_embeds = biosyn.get_dense_representation(
            mentions=ncbi_disease_train_mentions, verbose=True, type_=2)
        ncbi_disease_train_dict_dense_embeds = biosyn.get_dense_representation(
            mentions=ncbi_disease_train_dictionary_names, verbose=True, type_=2)
        ncbi_disease_train_dense_score_matrix = biosyn.get_score_matrix(
            query_embeds=ncbi_disease_train_query_dense_embeds,
            dict_embeds=ncbi_disease_train_dict_dense_embeds
        )
        ncbi_disease_train_dense_candidate_idxs = biosyn.retrieve_candidate(
            score_matrix=ncbi_disease_train_dense_score_matrix,
            topk=config.topk
        )
        # 在训练的过程中，由dense representation得到的结果会不停地发生改变，不断地更新最有可能candidate
        ncbi_disease_train_set.set_dense_candidate_idxs(d_candidate_idxs=ncbi_disease_train_dense_candidate_idxs)

        bc2gm_train_query_dense_embeds = biosyn.get_dense_representation(
            mentions=bc2gm_train_mentions, verbose=True, type_=3)
        bc2gm_train_dict_dense_embeds = biosyn.get_dense_representation(
            mentions=bc2gm_train_dictionary_names, verbose=True, type_=3)
        bc2gm_train_dense_score_matrix = biosyn.get_score_matrix(
            query_embeds=bc2gm_train_query_dense_embeds,
            dict_embeds=bc2gm_train_dict_dense_embeds
        )
        bc2gm_train_dense_candidate_idxs = biosyn.retrieve_candidate(
            score_matrix=bc2gm_train_dense_score_matrix,
            topk=config.topk
        )
        # 在训练的过程中，由dense representation得到的结果会不停地发生改变，不断地更新最有可能candidate
        bc2gm_train_set.set_dense_candidate_idxs(d_candidate_idxs=bc2gm_train_dense_candidate_idxs)

        train_loss = 0.
        train_steps = 0
        model.train()


        for step, data in tqdm(enumerate(zip(bc5cdr_disease_train_dataloader, bc5cdr_chemical_train_dataloader,ncbi_disease_train_dataloader,bc2gm_train_dataloader)),desc="正在训练模型...."):

            #model.optimizer.zero_grad()
            optimizer.zero_grad()
            # batch_y就是label，为CUI
            bc5cdr_disease_data,bc5cdr_chemcial_data,ncbi_disease_data,bc2gm_data = data
            batch_x, batch_y = bc5cdr_disease_data
            batch_pred = model(batch_x,type_=0)
            if config.use_n_gpu:
                targets = batch_y.to(device)
                bc5cdr_disease_loss = criterion(batch_pred, targets)

                batch_x, batch_y = bc5cdr_chemcial_data
                batch_pred = model(batch_x, type_=1)
                targets = batch_y.to(device)
                bc5cdr_chemical_loss = criterion(batch_pred, targets)


                batch_x, batch_y = ncbi_disease_data
                batch_pred = model(batch_x, type_=2)
                targets = batch_y.to(device)
                ncbi_disease_loss = criterion(batch_pred, targets)

                batch_x, batch_y = bc2gm_data
                batch_pred = model(batch_x, type_=3)
                targets = batch_y.to(device)
                bc2gm_loss = criterion(batch_pred, targets)

            else:
                bc5cdr_disease_loss = model.get_loss(batch_pred, batch_y)

                batch_x, batch_y = bc5cdr_chemcial_data
                batch_pred = model(batch_x, type_=1)
                bc5cdr_chemical_loss = model.get_loss(batch_pred, batch_y)

                batch_x, batch_y = ncbi_disease_data
                batch_pred = model(batch_x, type_=2)
                ncbi_disease_loss = model.get_loss(batch_pred, batch_y)

                batch_x, batch_y = bc2gm_data
                batch_pred = model(batch_x, type_=3)
                bc2gm_loss = model.get_loss(batch_pred, batch_y)



            loss = bc5cdr_disease_loss + bc5cdr_chemical_loss+ncbi_disease_loss+bc2gm_loss
            lr = optimizer.param_groups[0]['lr']

            # todo: 这里进行修改，3 sparse weight
            if config.use_n_gpu and torch.cuda.device_count() > 1:

                bc5cdr_disease_sparse_weight = model.module.bc5cdr_disease_sparse_weight.item()
                bc5cdr_chem_sparse_weight = model.module.bc5cdr_chem_sparse_weight.item()
                ncbi_disease_sparse_weight = model.module.ncbi_disease_sparse_weight.item()

                bc2gm_sparse_weight = model.module.bc2gm_sparse_weight.item()
            else:
                bc5cdr_disease_sparse_weight = model.bc5cdr_disease_sparse_weight.item()
                bc5cdr_chem_sparse_weight = model.bc5cdr_chem_sparse_weight.item()
                ncbi_disease_sparse_weight = model.ncbi_disease_sparse_weight.item()
                bc2gm_sparse_weight = model.bc2gm_sparse_weight.item()
            # todo:这是1 sprase weight
            # if config.use_n_gpu and torch.cuda.device_count() > 1:
            #
            #     bc5cdr_disease_sparse_weight = model.module.bc5cdr_disease_sparse_weight.item()
            #
            # else:
            #     bc5cdr_disease_sparse_weight = model.bc5cdr_disease_sparse_weight.item()

            logger.info(
                'Epoch:{} 训练中>>>>>> {}/{} loss:{:5f},bc5dis_loss:{:.5f},bc5chem_loss:{:.5f},ncbi_loss:{:.5f} bc2gm_loss:{:.5f},lr={}'.format(epoch, global_step, t_total,
                                                                                         loss.item(),
                                                                                         bc5cdr_disease_loss.item(),
                                                                                         bc5cdr_chemical_loss.item(),
                                                                                         ncbi_disease_loss.item(),
                                                                                         bc2gm_loss.item(),
                                                                                         lr))
            logger.info("   bc5cdr-disease sparse weight:{:.5f}".format(bc5cdr_disease_sparse_weight))
            logger.info("   bc5cdr-chem sparse weight:{:.5f}".format(bc5cdr_chem_sparse_weight))
            logger.info("   ncbi-disease sparse weight:{:.5f}".format(ncbi_disease_sparse_weight))
            logger.info("   bc2gm sparse weight:{:.5f}".format(bc2gm_sparse_weight))

            # logger.info("   chem sparse weight:{:.5f}".format(bc5cdr_chem_sparse_weight))

            if config.use_wandb:
                # todo: 这里是3 sparse weight的修改注释
                wandb.log(
                    {"train-epoch": epoch,
                     'total_margin_loss': loss.item(),
                     'bc5cdr_disease_margin_loss': bc5cdr_disease_loss.item(),
                     'bc5cdr_chemical_total_margin_loss': bc5cdr_chemical_loss.item(),
                     'disease sparse weight': bc5cdr_disease_sparse_weight,
                     'bc2gm sparse weight': bc2gm_sparse_weight,
                     'chem sparse weight': bc5cdr_chem_sparse_weight,
                     'ncbi-disease sparse weight': ncbi_disease_sparse_weight,
                     'train_lr': lr},
                    step=global_step)
                # wandb.log(
                #     {"train-epoch": epoch,
                #      'total_margin_loss': loss.item(),
                #      'bc5cdr_disease_margin_loss': bc5cdr_disease_loss.item(),
                #      'bc5cdr_chemical_total_margin_loss': bc5cdr_chemical_loss.item(),
                #      'disease sparse weight': bc5cdr_disease_sparse_weight,
                #
                #      'train_lr': lr},
                #     step=global_step)
            # loss.backward()
            # optimizer.step()
            if config.use_fp16:
                scaler.scale(loss).backward()
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    if config.use_scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    if config.use_scheduler:
                        scheduler.step()

                    optimizer.zero_grad()
            train_loss += loss.item()

            train_steps += 1
            global_step += 1



        train_loss /= (train_steps + 1e-9)
        logger.info(
            'Epoch:{} 训练Epoch完成>>>>>> {}/{} loss:{:5f} ,lr={}'.format(epoch, global_step, t_total,
                                                                                     train_loss,
                                                                                     lr,))

        if config.use_wandb:
            wandb.log(
                {"train-epoch": epoch, 'margin_loss': train_loss},
                step=global_step)
        logger.info('>>>>>>>>>>>>>>>>>进入验证装填<<<<<<<<<<<<<<<<<<<<<<<<<<')

        model.eval()
       # dev(config=config, logger=logger, biosyn=biosyn, device=device,wandb=wandb)
        test(config=config, logger=logger, biosyn=biosyn, device=device,wandb=wandb)

        # if config.save_model:
        #
        #     checkpoint_dir = os.path.join(config.output_dir, "checkpoint_{}".format(epoch))
        #     logger.info('将模型保存到:{}'.format(checkpoint_dir))
        #     if not os.path.exists(checkpoint_dir):
        #         os.makedirs(checkpoint_dir)
        #     biosyn.save_model(checkpoint_dir)


if __name__ == '__main__':

    config = get_config()

    logger = get_logger(config)

    # 设置时间
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff
    # 设置随机种子
    set_seed(config.seed)
    print_hyperparameters(config)
    if config.model_name == 'biosyn':
        logger.info('<<<<<<<<<<<<<<<<<<对BioSyn进行预训练>>>>>>>>>>>>>>>>>>>>')
        if config.use_wandb:
            if config.use_scheduler:
                if config.freeze_bert:
                    wandb_name = f'四任务_{config.model_name}_epochs{config.num_epochs}_encoder_{config.encoder_type}_{config.task_encoder_nums}_scheduler{config.warmup_proportion}_lr{config.learning_rate}_freeze_{len(config.freeze_layers)}_bs{config.batch_size}_maxlen{config.max_len}'
                else:
                    wandb_name = f'四任务_{config.model_name}_epochs{config.num_epochs}_encoder_{config.encoder_type}_{config.task_encoder_nums}_scheduler{config.warmup_proportion}_lr{config.learning_rate}_nofreeze_bs{config.batch_size}_maxlen{config.max_len}'
            else:
                if config.freeze_bert:
                    wandb_name = f'四任务_{config.model_name}_epochs{config.num_epochs}_encoder_{config.encoder_type}_{config.task_encoder_nums}_lr{config.learning_rate}_freeze_{len(config.freeze_layers)}_bs{config.batch_size}_maxlen{config.max_len}'
                else:
                    wandb_name = f'四任务_{config.model_name}_epochs{config.num_epochs}_encoder_{config.encoder_type}_{config.task_encoder_nums}_lr{config.learning_rate}_nofreeze_bs{config.batch_size}_maxlen{config.max_len}'

            wandb.init(project="多任务实体标准化",  config=vars(config),
                       name=wandb_name)

            config.output_dir = './outputs/save_models/{}/{}/{}/{}/'.format(str(datetime.date.today()), wandb_name,
                                                                        config.model_name, config.dataset_name)
            config.logs_dir = './outputs/logs/{}/{}/{}/{}/'.format(str(datetime.date.today()), wandb_name,
                                                                   config.model_name, config.dataset_name)

        else:
            config.output_dir = './outputs/save_models/{}/{}/{}/{}/'.format(str(datetime.date.today()), 'no_wandb',
                                                                            config.model_name, config.dataset_name)
            config.logs_dir = './outputs/logs/{}/{}/{}/{}/'.format(str(datetime.date.today()), 'no_wandb',
                                                                   config.model_name, config.dataset_name)

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

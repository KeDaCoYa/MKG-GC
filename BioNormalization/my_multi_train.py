# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :   开始同时加载多个数据集

   Author :        kedaxia
   date：          2022/01/18
   Copyright:      (c) kedaxia 2022
-------------------------------------------------
   Change Activity: 这是我的最终标准化模型，用于webservice...


-------------------------------------------------
"""
import os

import datetime
import random

import numpy as np
import torch
import wandb
from ipdb import set_trace
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import MyBertConfig
from my_multi_evaluate import dev
from sapbert_train import sapbert_train
from src.data_loader import NormalizationDataset

from src.models.my_multi_biosyn import MyMultiBioSynModel, MultiRerankNet
from utils.dataset_utils import load_dictionary, load_queries, load_my_data
from utils.function_utils import get_config, get_logger, set_seed, save_model, count_parameters
from utils.train_utils import build_optimizer, load_model_and_parallel, multi_build_optimizer


def my_multi_biosyn_train(config:MyBertConfig, logger):
    # prepare for output
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # 加载字典，shape=(dictioanry_size,2)
    # 第一列为name，第二类为ID
    disease_train_dictionary = load_dictionary(config.disease_dictionary_path)
    chemical_drug_train_dictionary = load_dictionary(config.chemical_drug_dictionary_path)
    gene_protein_train_dictionary = load_dictionary(config.gene_protein_dictionary_path)
    cell_type_train_dictionary = load_dictionary(config.cell_type_dictionary_path)
    cell_line_train_dictionary = load_dictionary(config.cell_line_dictionary_path)
    #species_train_dictionary = load_dictionary(config.species_dictionary_path)

    disease_train_queries = load_my_data(config.mesh_disease_train_path)
    chemical_drug_train_queries = load_my_data(config.chemical_drug_train_path)
    gene_protein_train_queries = load_my_data(config.gene_protein_train_path)
    cell_type_train_queries = load_my_data(config.cell_type_train_path)
    cell_line_train_queries = load_my_data(config.cell_line_train_path)
    #species_train_queries = load_my_data(config.species_train_path)


    if config.debug:  # 开启debug，使用小部分数据集进行测试
        disease_train_dictionary = disease_train_dictionary[:1000]
        chemical_drug_train_dictionary = chemical_drug_train_dictionary[:1000]
        gene_protein_train_dictionary = gene_protein_train_dictionary[:1000]
        cell_type_train_dictionary = cell_type_train_dictionary[:1000]
        cell_line_train_dictionary = cell_line_train_dictionary[:1000]
        #species_train_dictionary = species_train_dictionary[:10]

        disease_train_queries = disease_train_queries[:100]
        chemical_drug_train_queries = chemical_drug_train_queries[:100]
        gene_protein_train_queries = gene_protein_train_queries[:100]
        cell_type_train_queries = cell_type_train_queries[:100]
        cell_line_train_queries = cell_line_train_queries[:100]
        #species_train_queries = species_train_queries[:100]

        config.output_dir = config.output_dir + "_draft"

    device = torch.device('cuda') if config.use_gpu else torch.device('cpu')
    # 获取词典中的所有name
    disease_train_dictionary_names = disease_train_dictionary[:, 0]
    chemical_drug_train_dictionary_names = chemical_drug_train_dictionary[:, 0]
    gene_protein_train_dictionary_names = gene_protein_train_dictionary[:, 0]
    cell_type_train_dictionary_names = cell_type_train_dictionary[:, 0]
    cell_line_train_dictionary_names = cell_line_train_dictionary[:, 0]
    #species_train_dictionary_names = species_train_dictionary[:, 0]

    disease_train_mentions = disease_train_queries[:, 0]
    chemical_drug_train_mentions = chemical_drug_train_queries[:, 0]
    gene_protein_train_mentions = gene_protein_train_queries[:, 0]
    cell_type_train_mentions = cell_type_train_queries[:, 0]
    cell_line_train_mentions = cell_line_train_queries[:, 0]
    #species_train_mentions = species_train_queries[:, 0]

    # 在这里将会对数据进行统一个数，这里设定为一个平均个数
    # disease_train_mentions = disease_train_mentions.tolist()
    # chemical_drug_train_mentions = chemical_drug_train_mentions.tolist()
    # gene_protein_train_mentions = gene_protein_train_mentions.tolist()
    # cell_type_train_mentions = cell_type_train_mentions.tolist()
    # cell_line_train_mentions = cell_line_train_mentions.tolist()
    #species_train_mentions = species_train_mentions.tolist()

    # disease_train_queries = disease_train_queries.tolist()
    # chemical_drug_train_queries = chemical_drug_train_queries.tolist()
    # gene_protein_train_queries = gene_protein_train_queries.tolist()
    # cell_type_train_queries = cell_type_train_queries.tolist()
    # cell_line_train_queries = cell_line_train_queries.tolist()
    #species_train_queries = species_train_queries.tolist()

    # if not config.debug:
    #     # 数据集个数的调整
    #     sample_count = 10000
    #     # 训练方法，随机sample 10000个，然后进行训练
    # else:
    #     sample_count = 100
    # disease_train_mentions = [random.choice(disease_train_mentions) for _ in range(sample_count)]
    # disease_train_queries = [random.choice(disease_train_queries) for _ in range(sample_count)]
    #
    # chemical_drug_train_mentions = [random.choice(chemical_drug_train_mentions) for _ in range(sample_count)]
    # chemical_drug_train_queries = [random.choice(chemical_drug_train_queries) for _ in range(sample_count)]
    #
    # gene_protein_train_mentions = [random.choice(gene_protein_train_mentions) for _ in range(sample_count)]
    # gene_protein_train_queries = [random.choice(gene_protein_train_queries) for _ in range(sample_count)]
    #
    # cell_type_train_mentions = [random.choice(cell_type_train_mentions) for _ in range(sample_count)]
    # cell_type_train_queries = [random.choice(cell_type_train_queries) for _ in range(sample_count)]
    #
    # cell_line_train_mentions = [random.choice(cell_line_train_mentions) for _ in range(sample_count)]
    # cell_line_train_queries = [random.choice(cell_line_train_queries) for _ in range(sample_count)]

    # species_train_mentions = [random.choice(species_train_mentions) for _ in range(sample_count)]
    # species_train_queries = [random.choice(species_train_queries) for _ in range(sample_count)]

    # disease_train_mentions = np.array(disease_train_mentions)
    # chemical_drug_train_mentions = np.array(chemical_drug_train_mentions)
    # gene_protein_train_mentions = np.array(gene_protein_train_mentions)
    # cell_type_train_mentions = np.array(cell_type_train_mentions)
    # cell_line_train_mentions = np.array(cell_line_train_mentions)
    #species_train_mentions = np.array(species_train_mentions)

    # disease_train_queries = np.array(disease_train_queries)
    # chemical_drug_train_queries = np.array(chemical_drug_train_queries)
    # gene_protein_train_queries = np.array(gene_protein_train_queries)
    # cell_type_train_queries = np.array(cell_type_train_queries)
    # cell_line_train_queries = np.array(cell_line_train_queries)
    #species_train_queries = np.array(species_train_queries)

    biosyn = MyMultiBioSynModel(
        config=config,
        device=device,
        initial_sparse_weight=0
    )

    # 使用tf-idf来计算得到sparse vector，先对字典中的name进行encode
    biosyn.init_disease_sparse_encoder(corpus=disease_train_dictionary_names)
    biosyn.init_chemical_drug_sparse_encoder(corpus=chemical_drug_train_dictionary_names)
    biosyn.init_gene_protein_sparse_encoder(corpus=gene_protein_train_dictionary_names)
    biosyn.init_cell_type_sparse_encoder(corpus=cell_type_train_dictionary_names)
    biosyn.init_cell_line_sparse_encoder(corpus=cell_line_train_dictionary_names)
    #biosyn.init_species_sparse_encoder(corpus=species_train_dictionary_names)
    # 加载BioBERT等预训练模型...
    biosyn.load_dense_encoder(config.bert_dir)

    # 这个是训练模型训练的核心
    model = MultiRerankNet(
        config,
        dense_encoder=biosyn.get_dense_encoder(),
        sparse_weight=biosyn.get_sparse_weight(),
        device=device
    )
    # requires_grad_nums, parameter_nums = count_parameters(model)
    # set_trace()

    if config.use_n_gpu and torch.cuda.device_count() > 1:
        model, device = load_model_and_parallel(model, '0,1', ckpt_path=None, load_type='many2one')
    else:
        device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')
        model.to(device)
    logger.info("开始计算得到Disease Sparse embedding...")
    # 这是对训练集的entity和字典进行编码
    # 因为sparse representation是不会发生改变的，因此最开始就继续宁计算
    disease_train_query_sparse_embeds = biosyn.get_disease_sparse_representation(
        mentions=disease_train_mentions)
    disease_train_dict_sparse_embeds = biosyn.get_disease_sparse_representation(
        mentions=disease_train_dictionary_names)
    disease_train_sparse_score_matrix = biosyn.get_score_matrix(
        query_embeds=disease_train_query_sparse_embeds,
        dict_embeds=disease_train_dict_sparse_embeds
    )
    disease_train_sparse_candidate_idxs = biosyn.retrieve_candidate(
        score_matrix=disease_train_sparse_score_matrix,
        topk=config.topk
    )

    disease_train_dataset = NormalizationDataset(
        queries=disease_train_queries,
        dicts=disease_train_dictionary,
        tokenizer=biosyn.get_dense_tokenizer(),
        s_score_matrix=disease_train_sparse_score_matrix,
        s_candidate_idxs=disease_train_sparse_candidate_idxs,
        topk=config.topk,
        d_ratio=config.dense_ratio,
        max_len=config.max_len
    )

    disease_train_dataloader = DataLoader(disease_train_dataset, batch_size=config.batch_size, shuffle=True)

    logger.info("Chemical Sparse embedding...")
    chemical_drug_train_query_sparse_embeds = biosyn.get_chemical_drug_sparse_representation(
        mentions=chemical_drug_train_mentions)
    chemical_drug_train_dict_sparse_embeds = biosyn.get_chemical_drug_sparse_representation(
        mentions=chemical_drug_train_dictionary_names)
    chemical_drug_train_sparse_score_matrix = biosyn.get_score_matrix(
        query_embeds=chemical_drug_train_query_sparse_embeds,
        dict_embeds=chemical_drug_train_dict_sparse_embeds
    )
    chemical_drug_train_sparse_candidate_idxs = biosyn.retrieve_candidate(
        score_matrix=chemical_drug_train_sparse_score_matrix,
        topk=config.topk
    )

    chemical_drug_train_dataset = NormalizationDataset(
        queries=chemical_drug_train_queries,
        dicts=chemical_drug_train_dictionary,
        tokenizer=biosyn.get_dense_tokenizer(),
        s_score_matrix=chemical_drug_train_sparse_score_matrix,
        s_candidate_idxs=chemical_drug_train_sparse_candidate_idxs,
        topk=config.topk,
        d_ratio=config.dense_ratio,
        max_len=config.max_len
    )
    chemical_drug_train_dataloader = DataLoader(chemical_drug_train_dataset, batch_size=config.batch_size, shuffle=True)

    logger.info("Gene Sparse embedding...")
    gene_protein_train_query_sparse_embeds = biosyn.get_gene_protein_sparse_representation(
        mentions=gene_protein_train_mentions)
    gene_protein_train_dict_sparse_embeds = biosyn.get_gene_protein_sparse_representation(
        mentions=gene_protein_train_dictionary_names)
    gene_protein_train_sparse_score_matrix = biosyn.get_score_matrix(
        query_embeds=gene_protein_train_query_sparse_embeds,
        dict_embeds=gene_protein_train_dict_sparse_embeds
    )
    gene_protein_train_sparse_candidate_idxs = biosyn.retrieve_candidate(
        score_matrix=gene_protein_train_sparse_score_matrix,
        topk=config.topk
    )

    gene_protein_train_dataset = NormalizationDataset(
        queries=gene_protein_train_queries,
        dicts=gene_protein_train_dictionary,
        tokenizer=biosyn.get_dense_tokenizer(),
        s_score_matrix=gene_protein_train_sparse_score_matrix,
        s_candidate_idxs=gene_protein_train_sparse_candidate_idxs,
        topk=config.topk,
        d_ratio=config.dense_ratio,
        max_len=config.max_len
    )
    gene_protein_train_dataloader = DataLoader(gene_protein_train_dataset, batch_size=config.batch_size, shuffle=True)

    logger.info("Cell Type Sparse embedding...")
    cell_type_train_query_sparse_embeds = biosyn.get_cell_type_sparse_representation(
        mentions=cell_type_train_mentions)
    cell_type_train_dict_sparse_embeds = biosyn.get_cell_type_sparse_representation(
        mentions=cell_type_train_dictionary_names)

    cell_type_train_sparse_score_matrix = biosyn.get_score_matrix(
        query_embeds=cell_type_train_query_sparse_embeds,
        dict_embeds=cell_type_train_dict_sparse_embeds
    )
    cell_type_train_sparse_candidate_idxs = biosyn.retrieve_candidate(
        score_matrix=cell_type_train_sparse_score_matrix,
        topk=config.topk
    )

    cell_type_train_dataset = NormalizationDataset(
        queries=cell_type_train_queries,
        dicts=cell_type_train_dictionary,
        tokenizer=biosyn.get_dense_tokenizer(),
        s_score_matrix=cell_type_train_sparse_score_matrix,
        s_candidate_idxs=cell_type_train_sparse_candidate_idxs,
        topk=config.topk,
        d_ratio=config.dense_ratio,
        max_len=config.max_len
    )
    cell_type_train_dataloader = DataLoader(cell_type_train_dataset, batch_size=config.batch_size, shuffle=True)

    logger.info("Cell line Sparse embedding...")
    cell_line_train_query_sparse_embeds = biosyn.get_cell_line_sparse_representation(
        mentions=cell_line_train_mentions)
    cell_line_train_dict_sparse_embeds = biosyn.get_cell_line_sparse_representation(
        mentions=cell_line_train_dictionary_names)
    cell_line_train_sparse_score_matrix = biosyn.get_score_matrix(
        query_embeds=cell_line_train_query_sparse_embeds,
        dict_embeds=cell_line_train_dict_sparse_embeds
    )
    cell_line_train_sparse_candidate_idxs = biosyn.retrieve_candidate(
        score_matrix=cell_line_train_sparse_score_matrix,
        topk=config.topk
    )

    cell_line_train_dataset = NormalizationDataset(
        queries=cell_line_train_queries,
        dicts=cell_line_train_dictionary,
        tokenizer=biosyn.get_dense_tokenizer(),
        s_score_matrix=cell_line_train_sparse_score_matrix,
        s_candidate_idxs=cell_line_train_sparse_candidate_idxs,
        topk=config.topk,
        d_ratio=config.dense_ratio,
        max_len=config.max_len
    )
    cell_line_train_dataloader = DataLoader(cell_line_train_dataset, batch_size=config.batch_size, shuffle=True)

    # species_train_query_sparse_embeds = biosyn.get_species_sparse_representation(
    #     mentions=species_train_mentions)
    # species_train_dict_sparse_embeds = biosyn.get_species_sparse_representation(
    #     mentions=species_train_dictionary_names)
    # species_train_sparse_score_matrix = biosyn.get_score_matrix(
    #     query_embeds=species_train_query_sparse_embeds,
    #     dict_embeds=species_train_dict_sparse_embeds
    # )
    # species_train_sparse_candidate_idxs = biosyn.retrieve_candidate(
    #     score_matrix=species_train_sparse_score_matrix,
    #     topk=config.topk
    # )
    #
    # species_train_dataset = NormalizationDataset(
    #     queries=species_train_queries,
    #     dicts=species_train_dictionary,
    #     tokenizer=biosyn.get_dense_tokenizer(),
    #     s_score_matrix=species_train_sparse_score_matrix,
    #     s_candidate_idxs=species_train_sparse_candidate_idxs,
    #     topk=config.topk,
    #     d_ratio=config.dense_ratio,
    #     max_len=config.max_len
    # )
    # species_train_dataloader = DataLoader(species_train_dataset, batch_size=config.batch_size, shuffle=True)


    t_total = config.num_epochs * len(disease_train_dataloader)
    if config.use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    optimizer = multi_build_optimizer(config, model)
    global_step = 1

    best_model = None
    best_epoch = 0
    best_hit1 = 0
    best_hit5 = 0




    for epoch in range(1, config.num_epochs + 1):
        logger.info('>>>>>>>>>>>>>>>>>进入训练状态<<<<<<<<<<<<<<<<<<<<<<<<<<')
        # 得到dense representation....
        # 直接将所有的train_mentions得到其dense representation
        disease_train_query_dense_embeds = biosyn.get_dense_representation(mentions=disease_train_mentions, verbose=True)
        disease_train_dict_dense_embeds = biosyn.get_dense_representation(mentions=disease_train_dictionary_names, verbose=True)
        disease_train_dense_score_matrix = biosyn.get_score_matrix(
            query_embeds=disease_train_query_dense_embeds,
            dict_embeds=disease_train_dict_dense_embeds
        )
        disease_train_dense_candidate_idxs = biosyn.retrieve_candidate(
            score_matrix=disease_train_dense_score_matrix,
            topk=config.topk
        )
        disease_train_dataset.set_dense_candidate_idxs(d_candidate_idxs=disease_train_dense_candidate_idxs)

        chemical_drug_train_query_dense_embeds = biosyn.get_dense_representation(mentions=chemical_drug_train_mentions,verbose=True,type_=1)
        chemical_drug_train_dict_dense_embeds = biosyn.get_dense_representation(mentions=chemical_drug_train_dictionary_names,verbose=True,type_=1)
        chemical_drug_train_dense_score_matrix = biosyn.get_score_matrix(
            query_embeds=chemical_drug_train_query_dense_embeds,
            dict_embeds=chemical_drug_train_dict_dense_embeds
        )
        chemical_drug_train_dense_candidate_idxs = biosyn.retrieve_candidate(
            score_matrix=chemical_drug_train_dense_score_matrix,
            topk=config.topk
        )
        chemical_drug_train_dataset.set_dense_candidate_idxs(d_candidate_idxs=chemical_drug_train_dense_candidate_idxs)

        gene_protein_train_query_dense_embeds = biosyn.get_dense_representation(mentions=gene_protein_train_mentions,verbose=True,type_=2)
        gene_protein_train_dict_dense_embeds = biosyn.get_dense_representation(mentions=gene_protein_train_dictionary_names,verbose=True,type_=2)
        gene_protein_train_dense_score_matrix = biosyn.get_score_matrix(
            query_embeds=gene_protein_train_query_dense_embeds,
            dict_embeds=gene_protein_train_dict_dense_embeds
        )
        gene_protein_train_dense_candidate_idxs = biosyn.retrieve_candidate(
            score_matrix=gene_protein_train_dense_score_matrix,
            topk=config.topk
        )
        gene_protein_train_dataset.set_dense_candidate_idxs(d_candidate_idxs=gene_protein_train_dense_candidate_idxs)

        cell_type_train_query_dense_embeds = biosyn.get_dense_representation(mentions=cell_type_train_mentions,
                                                                           verbose=True,type_=3)
        cell_type_train_dict_dense_embeds = biosyn.get_dense_representation(mentions=cell_type_train_dictionary_names,
                                                                          verbose=True,type_=3)
        cell_type_train_dense_score_matrix = biosyn.get_score_matrix(
            query_embeds=cell_type_train_query_dense_embeds,
            dict_embeds=cell_type_train_dict_dense_embeds
        )
        cell_type_train_dense_candidate_idxs = biosyn.retrieve_candidate(
            score_matrix=cell_type_train_dense_score_matrix,
            topk=config.topk
        )
        cell_type_train_dataset.set_dense_candidate_idxs(d_candidate_idxs=cell_type_train_dense_candidate_idxs)

        cell_line_train_query_dense_embeds = biosyn.get_dense_representation(mentions=cell_line_train_mentions,
                                                                           verbose=True,type_=4)
        cell_line_train_dict_dense_embeds = biosyn.get_dense_representation(mentions=cell_line_train_dictionary_names,
                                                                          verbose=True,type_=4)
        cell_line_train_dense_score_matrix = biosyn.get_score_matrix(
            query_embeds=cell_line_train_query_dense_embeds,
            dict_embeds=cell_line_train_dict_dense_embeds
        )
        cell_line_train_dense_candidate_idxs = biosyn.retrieve_candidate(
            score_matrix=cell_line_train_dense_score_matrix,
            topk=config.topk
        )
        cell_line_train_dataset.set_dense_candidate_idxs(d_candidate_idxs=cell_line_train_dense_candidate_idxs)

        # species_train_query_dense_embeds = biosyn.get_dense_representation(mentions=species_train_mentions,
        #                                                                    verbose=True)
        # species_train_dict_dense_embeds = biosyn.get_dense_representation(mentions=species_train_dictionary_names,
        #                                                                   verbose=True)
        # species_train_dense_score_matrix = biosyn.get_score_matrix(
        #     query_embeds=species_train_query_dense_embeds,
        #     dict_embeds=species_train_dict_dense_embeds
        # )
        # species_train_dense_candidate_idxs = biosyn.retrieve_candidate(
        #     score_matrix=species_train_dense_score_matrix,
        #     topk=config.topk
        # )
        # species_train_dataset.set_dense_candidate_idxs(d_candidate_idxs=species_train_dense_candidate_idxs)
        # if epoch == 1:
        #     # 未训练之前的模型评估
        #     all_dictionary = (
        #         disease_train_dictionary, chemical_drug_train_dictionary, gene_protein_train_dictionary,
        #         cell_type_train_dictionary,
        #         cell_line_train_dictionary)
        #     all_dict_sparse_embeds = (
        #         disease_train_dict_sparse_embeds, chemical_drug_train_dict_sparse_embeds,
        #         gene_protein_train_dict_sparse_embeds,
        #         cell_type_train_dict_sparse_embeds, cell_line_train_dict_sparse_embeds)
        #     all_dict_dense_embeds = (
        #         disease_train_dict_dense_embeds, chemical_drug_train_dict_dense_embeds,
        #         gene_protein_train_dict_dense_embeds,
        #         cell_type_train_dict_dense_embeds, cell_line_train_dict_dense_embeds)
        #
        #     model.eval()
        #     dev(config=config, logger=logger, biosyn=biosyn, device=device, wandb=wandb, all_dictionary=all_dictionary,
        #         all_dict_sparse_embeds=all_dict_sparse_embeds, epoch=0,global_step=0,all_dict_dense_embeds=all_dict_dense_embeds)

        train_loss = 0.
        train_steps = 0


        model.train()
        for step, data in tqdm(enumerate(zip(disease_train_dataloader,chemical_drug_train_dataloader,gene_protein_train_dataloader,cell_type_train_dataloader,cell_line_train_dataloader)),desc="正在训练模型...."):

            #optimizer.zero_grad()
            # batch_y就是label，为CUI
            disease_data,chemical_drug_data,gene_protein_data,cell_type_data,cell_line_data = data

            batch_x, batch_y = disease_data
            batch_pred = model(batch_x,type_=0)
            if config.use_n_gpu and torch.cuda.device_count()>1:
                disease_loss = model.module.get_loss(batch_pred, batch_y)
            else:
                disease_loss = model.get_loss(batch_pred, batch_y)
            del disease_data

            batch_x, batch_y = chemical_drug_data
            batch_pred = model(batch_x,type_=1)
            if config.use_n_gpu and torch.cuda.device_count()>1:
                chemical_loss = model.module.get_loss(batch_pred, batch_y)
            else:
                chemical_loss = model.get_loss(batch_pred, batch_y)
            del chemical_drug_data

            batch_x, batch_y = gene_protein_data
            batch_pred = model(batch_x, type_=2)
            if config.use_n_gpu and torch.cuda.device_count()>1:
                gene_loss = model.module.get_loss(batch_pred, batch_y)
            else:
                gene_loss = model.get_loss(batch_pred, batch_y)
            del gene_protein_data

            batch_x, batch_y = cell_type_data
            batch_pred = model(batch_x, type_=3)
            if config.use_n_gpu and torch.cuda.device_count()>1:
                cell_type_loss = model.module.get_loss(batch_pred, batch_y)
            else:
                cell_type_loss = model.get_loss(batch_pred, batch_y)
            del cell_type_data

            batch_x, batch_y = cell_line_data
            batch_pred = model(batch_x, type_=4)
            if config.use_n_gpu and torch.cuda.device_count()>1:
                cell_line_loss = model.module.get_loss(batch_pred, batch_y)
            else:
                cell_line_loss = model.get_loss(batch_pred, batch_y)
            del cell_line_data
            # batch_x, batch_y = gene_protein_data
            # batch_pred = model(batch_x, type_=5)
            # species_loss = model.get_loss(batch_pred, batch_y)
            if config.use_n_gpu and torch.cuda.device_count() > 1:

                disease_sparse_weight = model.module.disease_sparse_weight.item()
                chemical_drug_sparse_weight = model.module.chemical_drug_sparse_weight.item()
                gene_sparse_weight = model.module.gene_sparse_weight.item()
                cell_type_sparse_weight = model.module.cell_type_sparse_weight.item()
                cell_line_sparse_weight = model.module.cell_line_sparse_weight.item()
            else:
                disease_sparse_weight = model.disease_sparse_weight.item()
                chemical_drug_sparse_weight = model.chemical_drug_sparse_weight.item()
                gene_sparse_weight = model.gene_sparse_weight.item()
                cell_type_sparse_weight = model.cell_type_sparse_weight.item()
                cell_line_sparse_weight = model.cell_line_sparse_weight.item()

            loss = disease_loss + chemical_loss+gene_loss+cell_type_loss+cell_line_loss
            logger.info(
                'Epoch:{} 训练中>>>>>> {}/{} loss:{:.5f},lr={}'.format(epoch, global_step, t_total,
                                                                                         loss.item(),
                                                                                         optimizer.param_groups[0]['lr']))

            logger.info("   disease loss:{:.5f}".format(disease_loss.item()))
            logger.info("   disease sparse weight:{:.5f}".format(disease_sparse_weight))
            logger.info("   chemical_drug loss:{:.5f}".format(chemical_loss.item()))
            logger.info("   chemical_drug sparse weight:{:.5f}".format(chemical_drug_sparse_weight))
            logger.info("   gene_protein loss:{:.5f}".format(gene_loss.item()))
            logger.info("   gene_sparse weight:{:.5f}".format(gene_sparse_weight))
            logger.info("   cell type loss:{:.5f}".format(cell_type_loss.item()))
            logger.info("   cell type sparse weight:{:.5f}".format(cell_type_sparse_weight))
            logger.info("   cell line loss:{:.5f}".format(cell_line_loss.item()))
            logger.info("   cell line sparse weight:{:.5f}".format(cell_line_sparse_weight))
            # logger.info("   species loss:{:.5f}".format(species_loss.item()))

            if config.use_wandb:
                wandb.log(
                    {"train-epoch": epoch,
                     'train-total_loss': loss.item(),
                     'train-disease_loss': disease_loss.item(),
                     'train-chemical_loss': chemical_loss.item(),
                     'train-gene_loss': gene_loss.item(),
                     'train-cell_type_loss': cell_type_loss.item(),
                     'train-cell_line_loss': cell_line_loss.item(),
                     'train-disease_sparse_weight': disease_sparse_weight,
                     'train-chemical_drug_sparse_weight': chemical_drug_sparse_weight,
                     'train-gene_sparse_weight': gene_sparse_weight,
                     'train-cell_type_sparse_weight': cell_type_sparse_weight,
                     'train-cell_line_sparse_weight': cell_line_sparse_weight,
                     'train_lr': optimizer.param_groups[0]['lr']},
                    step=global_step)

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
            # loss.backward()
            # optimizer.step()
            train_loss += loss.item()


            train_steps += 1
            global_step += 1


        train_loss /= (train_steps + 1e-9)
        logger.info(
            'Epoch:{} 训练Epoch完成>>>>>> {}/{} loss:{:5f} ,lr={}'.format(epoch, global_step, t_total,
                                                                                     train_loss,
                                                                                     optimizer.param_groups[0]['lr'],))

        if config.save_model:
            path = os.path.join(config.output_dir, 'multi_task_five_dataset',str(epoch))
            logger.info("模型保存到:{}".format(path))


            biosyn.save_model(path,config,epoch)


        if config.use_wandb:
            wandb.log(
                {"train-epoch": epoch, 'margin_loss': train_loss},
                step=global_step)
        logger.info('>>>>>>>>>>>>>>>>>进入验证装填<<<<<<<<<<<<<<<<<<<<<<<<<<')

        all_dictionary = (disease_train_dictionary, chemical_drug_train_dictionary, gene_protein_train_dictionary, cell_type_train_dictionary, cell_line_train_dictionary)
        all_dict_sparse_embeds = (disease_train_dict_sparse_embeds, chemical_drug_train_dict_sparse_embeds, gene_protein_train_dict_sparse_embeds, cell_type_train_dict_sparse_embeds, cell_line_train_dict_sparse_embeds)
        all_dict_dense_embeds = (disease_train_dict_dense_embeds, chemical_drug_train_dict_dense_embeds, gene_protein_train_dict_dense_embeds, cell_type_train_dict_dense_embeds, cell_line_train_dict_dense_embeds)


        model.eval()
        dev(config=config, logger=logger,  epoch=epoch,global_step=global_step,biosyn=biosyn, device=device,wandb=wandb,all_dictionary=all_dictionary,all_dict_sparse_embeds=all_dict_sparse_embeds,all_dict_dense_embeds=all_dict_dense_embeds)


        if config.save_model:

            checkpoint_dir = os.path.join(config.output_dir, 'multi_task_five_dataset','best_model')
            logger.info('将模型保存到:{}'.format(checkpoint_dir))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            biosyn.save_model(checkpoint_dir, config, epoch,mode='best_model')


if __name__ == '__main__':

    config = get_config()

    logger = get_logger(config)

    # 设置时间
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff
    # 设置随机种子
    set_seed(config.seed)

    if config.model_name == 'biosyn':
        logger.info('<<<<<<<<<<<<<<<<<<对BioSyn进行预训练>>>>>>>>>>>>>>>>>>>>')
        if config.use_wandb:
            if config.freeze_bert:
                wandb_name = f'Five_task_{config.model_name}_bs{config.batch_size}_free{config.freeze_layer_nums}_lr{config.learning_rate}_{config.task_encoder_nums}_encoder_type_{config.encoder_type}_maxlen{config.max_len}'
            else:
                wandb_name = f'Five_Task_{config.model_name}_bs{config.batch_size}_no_freeze_lr{config.learning_rate}_{config.task_encoder_nums}_encoder_type_{config.encoder_type}_maxlen{config.max_len}'

            wandb.init(project="多任务实体标准化",  config=vars(config),name=wandb_name)
            config.output_dir = './outputs/save_models/{}/{}/{}/{}/'.format(str(datetime.date.today()), wandb_name,
                                                                            config.model_name, config.dataset_name)
            config.logs_dir = './outputs/logs/{}/{}/{}/{}/'.format(str(datetime.date.today()), wandb_name,
                                                                   config.model_name, config.dataset_name)

        else:
            config.output_dir = './outputs/save_models/{}/{}/{}/{}/'.format(str(datetime.date.today()), 'no_wandb',
                                                                            config.model_name, config.dataset_name)
            config.logs_dir = './outputs/logs/{}/{}/{}/{}/'.format(str(datetime.date.today()), 'no_wandb',
                                                                   config.model_name, config.dataset_name)

        my_multi_biosyn_train(config, logger)
    elif config.model_name == 'sapbert':
        logger.info('<<<<<<<<<<<<<<<<<<对SapBERT进行预训练>>>>>>>>>>>>>>>>>>>>')
        if config.use_wandb:
            wandb_name = f'{config.model_name}_bs{config.batch_size}_maxlen{config.max_len}'
            wandb.init(project="实体标准化-{}".format(config.dataset_name), config=vars(config),
                       name=wandb_name)
        sapbert_train(config, logger)
    else:
        raise NotImplementedError("暂时没有")

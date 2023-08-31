# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :   这是将数据集

   Author :        kedaxia
   date：          2022/01/18
   Copyright:      (c) kedaxia 2022
-------------------------------------------------
   Change Activity:
                   2022年5月10日：这个训练方式是冻结所有bert，然后训练其他ceng的训练方式，因为内存不够用了

-------------------------------------------------
"""
import os

import datetime
import random

import numpy as np
import torch
import wandb
from ipdb import set_trace
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from save_evaluate import dev, test
from config import MyBertConfig
from sapbert_train import sapbert_train
from src.data_loader import NormalizationDataset
from src.models.biosyn import BioSyn, RerankNet
from src.models.multi_biosyn import MultiBioSyn
from utils.dataset_utils import load_dictionary, load_queries, load_my_data
from utils.function_utils import get_config, get_logger, set_seed, save_model, load_model_and_parallel, \
    print_hyperparameters
from utils.train_utils import marginal_nll, build_optimizer
global_step = 0

def biosyn_train(config, logger,biosyn,model,device,optimizer,type_=0,epoch=0,global_step=0):
    """

    :param config:
    :param logger:
    :param train_dictionary_path:
    :param train_dir:
    :param train_path:
    :param type_: 0,1,2 三种不同的数据集
    :return:
    """
    # prepare for output
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # 加载字典，shape=(dictioanry_size,2)
    # 第一列为name，第二类为ID

    if config.dataset_name == 'bc5cdr-chemical':
        train_dictionary_path = config.bc5cdr_chemical_train_dictionary_path
        train_dir = "./dataset/bc5cdr-chemical/train.txt"
    elif config.dataset_name == 'bc5cdr-disease':
        train_dictionary_path = config.bc5cdr_disease_train_dictionary_path
        train_dir = "./dataset/bc5cdr-disease/train.txt"
    elif config.dataset_name == 'ncbi-disease':
        train_dictionary_path = config.ncbi_disease_train_dictionary_path
        train_dir = "./dataset/ncbi-disease/train.txt"
    else:
        raise ValueError
    train_dictionary = load_dictionary(dictionary_path=train_dictionary_path)
    train_queries = load_queries(
        data_dir=train_dir,
        filter_composite=True,
        filter_duplicate=True,
        filter_cuiless=True
    )



    if config.debug:  # 开启debug，使用小部分数据集进行测试
        train_dictionary = train_dictionary[:100]
        train_queries = train_queries[:10]
        config.output_dir = config.output_dir + "_draft"


    # 获取词典中的所有name
    train_dictionary_names = train_dictionary[:, 0]
    train_mentions = train_queries[:, 0]

    # 使用tf-idf来计算得到sparse vector，先对字典中的name进行encode
    if type_ == 0:
        biosyn.init_bc5cdr_disease_sparse_encoder(corpus=train_dictionary_names)
        train_query_sparse_embeds = biosyn.get_bc5cdr_disease_sparse_representation(
            mentions=train_mentions)  # train.shape = (1587, 1122),实体数目为1587
        train_dict_sparse_embeds = biosyn.get_bc5cdr_disease_sparse_representation(
            mentions=train_dictionary_names)  # shape=(71924, 1122),字典数目为71924
        # 计算sparse similiarity scores，采用inner dot计算相似分数
        train_sparse_score_matrix = biosyn.get_score_matrix(
            query_embeds=train_query_sparse_embeds,
            dict_embeds=train_dict_sparse_embeds
        )
    elif type_ == 1:
        biosyn.init_bc5cdr_chem_sparse_encoder(corpus=train_dictionary_names)
        train_query_sparse_embeds = biosyn.get_bc5cdr_chemical_sparse_representation(
            mentions=train_mentions)  # train.shape = (1587, 1122),实体数目为1587
        train_dict_sparse_embeds = biosyn.get_bc5cdr_chemical_sparse_representation(
            mentions=train_dictionary_names)  # shape=(71924, 1122),字典数目为71924
        # 计算sparse similiarity scores，采用inner dot计算相似分数
        train_sparse_score_matrix = biosyn.get_score_matrix(
            query_embeds=train_query_sparse_embeds,
            dict_embeds=train_dict_sparse_embeds
        )
    elif type_ == 2:
        biosyn.init_ncbi_disease_sparse_encoder(corpus=train_dictionary_names)
        train_query_sparse_embeds = biosyn.get_ncbi_disease_sparse_representation(
            mentions=train_mentions)  # train.shape = (1587, 1122),实体数目为1587
        train_dict_sparse_embeds = biosyn.get_ncbi_disease_sparse_representation(
            mentions=train_dictionary_names)  # shape=(71924, 1122),字典数目为71924
        # 计算sparse similiarity scores，采用inner dot计算相似分数
        train_sparse_score_matrix = biosyn.get_score_matrix(
            query_embeds=train_query_sparse_embeds,
            dict_embeds=train_dict_sparse_embeds
        )
    else:
        raise ValueError
    logger.info("开始计算得到Sparse embedding")
    # 这是对训练集的entity和字典进行编码
    # 因为sparse representation是不会发生改变的，因此最开始就继续宁计算



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

    train_dataloader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)

    t_total = config.num_epochs * len(train_dataloader)

    logger.info('>>>>>>>>>>>>>>>>>{}:进入训练状态<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(config.dataset_name))
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
    for step, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader),desc="正在更新模型参数..."):
        lr = optimizer.param_groups[0]['lr']
        optimizer.zero_grad()
        # batch_y就是label，为CUI
        batch_x, batch_y = data
        batch_pred = model(batch_x)

        loss = model.get_loss(batch_pred, batch_y)
        if config.use_wandb:
            wandb.log(
                {"train-epoch": epoch,
                 'margin_loss': loss.item(),
                 'train_lr': lr
                 },
                step=global_step)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        train_steps += 1
        global_step += 1

    train_loss /= (train_steps + 1e-9)
    logger.info(
        'Epoch:{} 训练中>>>>>> {}/{} loss:{:5f} ,lr={},sparse_weight={:.5f}'.format(epoch, global_step, t_total,
                                                                                 train_loss,
                                                                                 optimizer.param_groups[0][
                                                                                     'lr'],
                                                                                 model.sparse_weight.item()))
    if config.use_wandb:
        wandb.log(
            {"train-epoch": epoch, 'margin_loss': train_loss, 'sparse_weight': model.sparse_weight.item()},
            step=global_step)
    logger.info('>>>>>>>>>>>>>>>>>{}:开始进行验证<<<<<<<<<<<<<<<<<<<<<<<'.format(config.dataset_name))
    #
    # acc1, acc5 = dev(config=config, logger=logger, biosyn=biosyn, device=device)
    #
    # if config.use_wandb:
    #     wandb.log(
    #         {"dev-epoch": epoch, 'dev-{}-hit@1'.format(config.dataset_name): acc1, 'dev-{}-hit@5'.format(config.dataset_name): acc5}, step=global_step)
    logger.info('>>>>>>>>>>>>>>>>>{}:开始进行测试集<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(config.dataset_name))
    model.eval()
    acc1, acc5 = test(config=config, logger=logger, biosyn=biosyn, device=device)
    if config.use_wandb:
        wandb.log(
        {"test-epoch": epoch, 'test-{}-hit@1'.format(config.dataset_name): acc1, 'test-{}-hit@5'.format(config.dataset_name): acc5}, step=global_step)


def train(config:MyBertConfig):
    device = torch.device('cuda') if config.use_gpu else torch.device('cpu')

    biosyn = MultiBioSyn(
        config=config,
        device=device,
        initial_sparse_weight=0
    )


    # 加载BioBERT等预训练模型...
    biosyn.load_dense_encoder(config.bert_dir)

    # 这个是训练模型训练的核心
    model = RerankNet(
        config,
        dense_encoder=biosyn.get_dense_encoder(),
        sparse_weight=biosyn.get_sparse_weight(),
        device=device
    )
    optimizer = build_optimizer(config, model)
    global_step = 0
    idxx_ = 0
    for epoch in range(config.num_epochs):
        for idx,dataset_name in enumerate(['bc5cdr-disease','bc5cdr-chemical','ncbi-disease']):
            idxx_ += 1
            global_step = 10000*(idxx_)
            config.dataset_name = dataset_name
            biosyn_train(config, logger, biosyn, model, device,optimizer,type_=idx,epoch=epoch,global_step=global_step)

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

    if config.use_wandb:
        wandb_name = f'三任务_{config.model_name}_bs{config.batch_size}_maxlen{config.max_len}'
        wandb.init(project="多任务实体标准化",  config=vars(config),
                   name=wandb_name)

    train(config)


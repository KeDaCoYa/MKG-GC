# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :   sapbert的训练其实就像bert模型一样的训练
                   使用UNLS中的所有数据进行训练，一次一般不一会对sqpbert进行训练
                   除非找到大量的数据...
   Author :        kedaxia
   date：          2022/01/20
   Copyright:      (c) kedaxia 2022
-------------------------------------------------
   Change Activity:
                   2022/01/20: 
-------------------------------------------------
"""
import os
import time

import torch
from ipdb import set_trace
from pytorch_metric_learning import samplers
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import MyBertConfig
from src.data_loader import MetricLearningDataset_pairwise, MetricLearningDataset
from src.models.sapbert import SapBertModel, SapBERTWrapper


def sapbert_train(config:MyBertConfig,logger):

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    device = torch.device('cuda') if config.use_gpu else torch.device('cpu')
    # 加载分词器和BERT model(就是Pubmed BERT、BioBERT、SciBERT等各种模型)
    # 这个就是训练模型
    model_wrapper = SapBERTWrapper(device)
    # encoder就是各种bert
    encoder, tokenizer = model_wrapper.load_bert(path=config.bert_dir)

    # load SAP model
    model = SapBertModel(
        encoder=encoder,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
        pairwise=config.pairwise,
        loss=config.loss,
        use_miner=config.use_miner,
        miner_margin=config.miner_margin,
        type_of_triplets=config.type_of_triplets,
        agg_mode=config.agg_mode,
    )

    if config.use_n_gpu:
        model.encoder = torch.nn.DataParallel(model.encoder)


    def collate_fn_batch_encoding(batch):
        query1, query2, query_id = zip(*batch)
        query_encodings1 = tokenizer.batch_encode_plus(
            list(query1),
            max_length=config.max_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")
        query_encodings2 = tokenizer.batch_encode_plus(
            list(query2),
            max_length=config.max_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")

        query_ids = torch.tensor(list(query_id))
        return query_encodings1, query_encodings2, query_ids

    if config.pairwise:  # 给定一对，positive和negative

        train_set = MetricLearningDataset_pairwise(
            path=config.sap_train_dir,
            tokenizer=tokenizer
        )
        train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=16,
            collate_fn=collate_fn_batch_encoding
        )
    else:
        train_set = MetricLearningDataset(
            path=config.sap_train_dir,
            tokenizer=tokenizer
        )
        # using a sampler
        train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            # shuffle=True,
            sampler=samplers.MPerClassSampler(train_set.query_ids, 2, length_before_new_iter=100000),
            num_workers=16,
        )
    # mixed precision training
    if config.use_amp:
        scaler = GradScaler()
    else:
        scaler = None
    start = time.time()
    global_step = 0
    model.to(device)
    for epoch in range(1, config.num_epochs + 1):

        train_loss = 0
        train_steps = 0

        model.train()

        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):

            model.optimizer.zero_grad()
            batch_x1, batch_x2, batch_y = data
            batch_x_cuda1, batch_x_cuda2 = {}, {}
            for k, v in batch_x1.items():
                batch_x_cuda1[k] = v.to(device)
            for k, v in batch_x2.items():
                batch_x_cuda2[k] = v.to(device)

            batch_y_cuda = batch_y.to(device)

            if config.use_amp:
                with autocast():
                    loss = model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda)
            else:
                loss = model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda)
            if config.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model.optimizer)
                scaler.update()
            else:
                loss.backward()
                model.optimizer.step()

            train_loss += loss.item()
            # wandb.log({"Loss": loss.item()})
            train_steps += 1
            global_step += 1
            # if (i+1) % 10 == 0:
            # LOGGER.info ("epoch: {} loss: {:.3f}".format(i+1,train_loss / (train_steps+1e-9)))
            # LOGGER.info ("epoch: {} loss: {:.3f}".format(i+1, loss.item()))


            # if global_step % args.checkpoint_step == 0:
            #     checkpoint_dir = os.path.join(args.output_dir, "checkpoint_iter_{}".format(str(step_global)))
            #     if not os.path.exists(checkpoint_dir):
            #         os.makedirs(checkpoint_dir)
            #     model_wrapper.save_model(checkpoint_dir)
        train_loss /= (train_steps + 1e-9)


        # save model every epoch
        if config.save_model:
            checkpoint_dir = os.path.join(config.output_dir, "checkpoint_{}".format(epoch))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model_wrapper.save_model(checkpoint_dir)

        # save model last epoch
        if epoch == config.num_epochs:
            model_wrapper.save_model(config.output_dir)

    end = time.time()
    training_time = end - start
    training_hour = int(training_time / 60 / 60)
    training_minute = int(training_time / 60 % 60)
    training_second = int(training_time % 60)
    logger.info("训练花费时间为 {} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))

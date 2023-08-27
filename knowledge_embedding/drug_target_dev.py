# -*- encoding: utf-8 -*-
"""
@File    :   dev.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/17 21:38   
@Description :   这个专用于drug-target的dev任务

"""

import datetime
import logging
import os

import numpy as np
import torch
import wandb
from ipdb import set_trace

from torch.utils.data import DataLoader, TensorDataset, SequentialSampler

from tqdm import tqdm

from src.utils.dataset_utils import KGProcessor, KGBertDataset, InputFeatures

logger = logging.getLogger("main.dev")


def DT_link_predication_dev(config, logger, model, tokenizer, device, train_type='dev'):
    # 数据读取阶段
    processor = KGProcessor()
    drug_entity_list = processor.get_drug_entities(config.data_dir)
    target_entity_list = processor.get_target_entities(config.data_dir)
    train_triples = processor.get_train_triples(config.data_dir)
    dev_triples = processor.get_dev_triples(config.data_dir)
    try:
        test_triples = processor.get_test_triples(config.data_dir)
        all_triples = train_triples + dev_triples + test_triples
    except:
        all_triples = train_triples + dev_triples

    # 构建集合，方便之后的评估
    all_triples_str_set = set()
    for triple in all_triples:
        triple_str = '\t'.join(triple)
        all_triples_str_set.add(triple_str)

    # 这里在验证集上检测时，需要生成negative来测试loss
    dev_examples = processor.get_dev_examples(config.data_dir, type='train')
    dev_dataset = KGBertDataset(config, dev_examples, tokenizer)
    dev_dataloader = DataLoader(dataset=dev_dataset, shuffle=True, num_workers=0,
                                batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    dev_loss = 0.
    dev_acc = 0.
    model.eval()
    for step, batch_data in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader),
                                 desc="计算{}的loss和acc".format(train_type)):

        if config.model_name == 'kgbert':
            input_ids, attention_masks, token_type_ids, label_ids = batch_data
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            token_type_ids = token_type_ids.to(device)
            label_ids = label_ids.to(device)
            with torch.no_grad():
                logits, loss = model(input_ids, token_type_ids, attention_masks, label_ids)
        else:
            input_ids, attention_masks, token_type_ids, label_ids, head_mask, rel_mask, tail_mask = batch_data
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            token_type_ids = token_type_ids.to(device)
            label_ids = label_ids.to(device)

            head_mask = head_mask.to(device)
            rel_mask = rel_mask.to(device)
            tail_mask = tail_mask.to(device)
            with torch.no_grad():
                logits, loss = model(input_ids, token_type_ids, attention_masks, label_ids, head_mask, rel_mask,
                                     tail_mask)

        label_ids = label_ids.cpu().numpy()
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)
        acc = (preds == label_ids).mean()
        dev_loss += loss
        dev_acc += acc
    dev_loss = dev_loss / len(dev_dataloader)
    dev_acc = dev_acc / len(dev_dataloader)

    if not config.metric_verbose:
        return {
            'loss': dev_loss,
            'acc': dev_acc
        }

    # run link prediction
    ranks = []
    ranks_left = []
    ranks_right = []

    hits_left = []
    hits_right = []
    hits = []

    top_ten_hit_count = 0

    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    # 对于每一个test_triple,分别生成其所有的corpus head list和corpus right list来进行评估
    first = 0
    for triple in tqdm(dev_triples, total=len(dev_triples), desc="验证集正在计算metric"):
        first += 1
        head = triple[0].lower()
        relation = triple[1].lower()
        tail = triple[2].lower()
        # print(test_triple, head, relation, tail)
        # 给这里面加一个正确的triple
        head_corrupt_list = [triple]
        for corrupt_ent in drug_entity_list:
            if corrupt_ent != head:
                tmp_triple = [corrupt_ent.lower(), relation, tail]
                tmp_triple_str = '\t'.join(tmp_triple)
                if tmp_triple_str not in all_triples_str_set:
                    # may be slow
                    head_corrupt_list.append(tmp_triple)
        # 这里其实是将corpus list的label设置为1
        # 设置为1的原因其实是用不到这个label,因为之后固定的以label=1对应的值进行...
        head_examples = processor._create_examples(head_corrupt_list, 'dev', config.data_dir)
        head_dataset = KGBertDataset(config, head_examples, tokenizer)
        head_dataloader = DataLoader(dataset=head_dataset, shuffle=False, num_workers=0,
                                     batch_size=config.eval_batch_size,
                                     collate_fn=dev_dataset.collate_fn)

        model.eval()
        preds = []
        all_label_ids = []
        for batch_data in head_dataloader:
            if config.model_name == 'kgbert':
                input_ids, attention_masks, token_type_ids, label_ids = batch_data
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                token_type_ids = token_type_ids.to(device)
                label_ids = label_ids.to(device)
                with torch.no_grad():
                    logits, loss = model(input_ids, token_type_ids, attention_masks, label_ids)
            else:
                input_ids, attention_masks, token_type_ids, label_ids, head_mask, rel_mask, tail_mask = batch_data
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                token_type_ids = token_type_ids.to(device)
                label_ids = label_ids.to(device)

                head_mask = head_mask.to(device)
                rel_mask = rel_mask.to(device)
                tail_mask = tail_mask.to(device)
                with torch.no_grad():
                    logits, loss = model(input_ids, token_type_ids, attention_masks, label_ids, head_mask, rel_mask,
                                         tail_mask)

            batch_logits = logits.detach().cpu().numpy()

            preds.extend(batch_logits)
            all_label_ids.extend(label_ids.cpu().numpy())

        preds = np.array(preds)
        all_label_ids = np.array(all_label_ids)

        # get the dimension corresponding to current label 1
        # print(preds, preds.shape)
        # 这是获得对应于label的输出值
        rel_values = preds[:, all_label_ids[0]]

        rel_values = torch.tensor(rel_values)
        # print(rel_values, rel_values.shape)
        _, argsort1 = torch.sort(rel_values, descending=True)
        # print(max_values)
        # print(argsort1)
        argsort1 = argsort1.cpu().numpy()
        rank1 = np.where(argsort1 == 0)[0][0]
        # print('left: ', rank1)
        ranks.append(rank1 + 1)
        ranks_left.append(rank1 + 1)
        if rank1 < 10:
            top_ten_hit_count += 1
        # 这是对tail进行破坏，

        tail_corrupt_list = [triple]
        for corrupt_ent in target_entity_list:
            if corrupt_ent != tail:
                tmp_triple = [head.lower(), relation.lower(), corrupt_ent.lower()]
                tmp_triple_str = '\t'.join(tmp_triple)
                if tmp_triple_str not in all_triples_str_set:
                    # may be slow
                    tail_corrupt_list.append(tmp_triple)

        preds = []
        tail_examples = processor._create_examples(tail_corrupt_list, 'dev', config.data_dir)
        tail_dataset = KGBertDataset(config, tail_examples, tokenizer)
        tail_dataloader = DataLoader(dataset=tail_dataset, shuffle=False, num_workers=0,
                                     batch_size=config.eval_batch_size,
                                     collate_fn=dev_dataset.collate_fn)
        model.eval()
        all_label_ids = []
        for batch_data in tail_dataloader:
            if config.model_name == 'kgbert':
                input_ids, attention_masks, token_type_ids, label_ids = batch_data
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                token_type_ids = token_type_ids.to(device)
                label_ids = label_ids.to(device)
                with torch.no_grad():
                    logits, loss = model(input_ids, token_type_ids, attention_masks, label_ids)
            else:
                input_ids, attention_masks, token_type_ids, label_ids, head_mask, rel_mask, tail_mask = batch_data
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                token_type_ids = token_type_ids.to(device)
                label_ids = label_ids.to(device)

                head_mask = head_mask.to(device)
                rel_mask = rel_mask.to(device)
                tail_mask = tail_mask.to(device)
                with torch.no_grad():
                    logits, loss = model(input_ids, token_type_ids, attention_masks, label_ids, head_mask, rel_mask,
                                         tail_mask)



            batch_logits = logits.detach().cpu().numpy()

            preds.extend(batch_logits)
            all_label_ids.extend(label_ids.cpu().numpy())

        preds = np.array(preds)
        all_label_ids = np.array(all_label_ids)
        # get the dimension corresponding to current label 1
        rel_values = preds[:, all_label_ids[0]]

        rel_values = torch.tensor(rel_values)
        _, argsort1 = torch.sort(rel_values, descending=True)
        argsort1 = argsort1.cpu().numpy()
        rank2 = np.where(argsort1 == 0)[0][0]
        ranks.append(rank2 + 1)
        ranks_right.append(rank2 + 1)
        # print('right: ', rank2)
        # print('mean rank until now: ', np.mean(ranks))
        if rank2 < 10:
            top_ten_hit_count += 1
        logger.info("hit@10 until now:{:.5f}".format(top_ten_hit_count * 1.0 / len(ranks)))

        # this could be done more elegantly, but here you go
        for hits_level in range(10):
            if rank1 <= hits_level:
                hits[hits_level].append(1.0)
                hits_left[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_left[hits_level].append(0.0)

            if rank2 <= hits_level:
                hits[hits_level].append(1.0)
                hits_right[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_right[hits_level].append(0.0)

    for i in [0, 2, 9]:
        logger.info('Hits left @{0}: {1}'.format(i + 1, np.mean(hits_left[i])))
        logger.info('Hits right @{0}: {1}'.format(i + 1, np.mean(hits_right[i])))
        logger.info('Hits @{0}: {1}'.format(i + 1, np.mean(hits[i])))
    logger.info('Mean rank left: {0}'.format(np.mean(ranks_left)))
    logger.info('Mean rank right: {0}'.format(np.mean(ranks_right)))
    logger.info('Mean rank: {0}'.format(np.mean(ranks)))
    logger.info('Mean reciprocal rank left: {0}'.format(np.mean(1. / np.array(ranks_left))))
    logger.info('Mean reciprocal rank right: {0}'.format(np.mean(1. / np.array(ranks_right))))
    logger.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))
    return {
        'hit@1': np.mean(hits[0]),
        'hit@3': np.mean(hits[2]),
        'hit@10': np.mean(hits[9]),
        'MR': np.mean(ranks),
        'MRR': np.mean(1. / np.array(ranks)),
        'loss': dev_loss,
        'acc': dev_acc,
    }

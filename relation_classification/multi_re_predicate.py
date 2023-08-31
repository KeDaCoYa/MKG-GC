# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :   这个使用已经训练完成的模型来预测无标签的数据
   Author :        kedaxia
   date：          2021/12/03
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/03: 
-------------------------------------------------
"""

import os
import datetime
import pickle
import time
from collections import Counter, defaultdict

from ipdb import set_trace
from tqdm import tqdm

import wandb

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, precision_score, recall_score, \
    classification_report, confusion_matrix
from transformers import BertTokenizer

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset_utils.data_process_utils import read_raw_data
from src.dataset_utils.entity_marker import MTBDataset
from src.dataset_utils.entity_type_marker import NormalDataset, MultiNormalDataset
from src.models.multi_mtb_bert import MultiMtbBertForBC6, MultiMtbBertForBC7
from src.models.multi_entitymarker_model import *

from src.utils.function_utils import correct_datetime, set_seed, save_model, load_model_and_parallel
from src.utils.tips_utils import get_bert_config, get_logger, print_hyperparameters, wandb_log, show_log
from src.utils.train_utils import relation_classification_decode, batch_to_device, choose_model, \
    set_tokenize_special_tag


def predicate(config, ckpt_path=None, logger=None,thresh_hold=0.5):
    """
    thresh_hold:设置关系分类的阈值，默认为0.5
    """

    # 读取所有的相关数据
    examples = read_raw_data(config)

    device = torch.device('cuda') if config.use_gpu else torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
    set_tokenize_special_tag(config, tokenizer)
    # 这个针对sentence-level的关系分类

    if config.data_format == 'single':
        dev_dataset = MultiNormalDataset(examples, config=config, label2id=None, tokenizer=tokenizer, device=device)
    # MTB的方法，这个至少可以解决一些cross-sentence 关系分类
    elif config.data_format == 'cross':
        dev_dataset = MTBDataset(examples, config=config, label2id=None, tokenizer=tokenizer, device=device)
    else:
        raise ValueError

    dev_dataloader = DataLoader(dataset=dev_dataset, shuffle=False, collate_fn=dev_dataset.collate_fn_predicate,
                                num_workers=0, batch_size=config.batch_size)
    # 选择模型

    if config.data_format == 'single':
        model = MultiSingleEntityMarkerForAlldata(config)
    elif config.data_format == 'cross':
        raise NotImplementedError

    # model.bert_model.resize_token_embeddings(len(tokenizer))
    model.bert_model.resize_token_embeddings(len(tokenizer))
    model, device = load_model_and_parallel(model, '0', ckpt_path=ckpt_path, strict=True,
                                            load_type='one2one')
    relation_counter = defaultdict(int)
    model.to(device)

    relation_predicate_res = []

    soft_max = nn.Softmax()
    model.eval()
    start_time = time.time()
    for step, batch_data in tqdm(enumerate(dev_dataloader), desc='正在预测数据{}...'.format(config.dataset_name), total=len(dev_dataloader)):

        if config.data_format == 'single':
            input_ids, token_type_ids, attention_masks, e1_mask, e2_mask, rel_type = batch_data
            labels = None
            with torch.no_grad():
                probability = model(input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask, rel_type)
            predicate_labels = relation_classification_decode(probability)

        elif config.data_format == 'cross':
            input_ids, token_type_ids, attention_masks, e1_mask, e2_mask, rel_type = batch_data
            with torch.no_grad():
                probability = model(input_ids, token_type_ids, attention_masks, None, e1_mask, e2_mask, rel_type)

            predicate_labels = relation_classification_decode(probability)


        else:
            raise ValueError
        probabilities_li = soft_max(probability)
        # 保证他们实体之间存在interaction
        new_predicate_labels = []
        relation_probabilities_li = []
        rel_type = rel_type.cpu().numpy()

        for idx, pred in enumerate(predicate_labels):
            if pred:
                new_predicate_labels.append(rel_type[idx])
                relation_probabilities_li.append(probabilities_li[idx][1].item())
            else:
                new_predicate_labels.append(0)
                relation_probabilities_li.append(probabilities_li[idx][0].item())


        id2label = {
            1: 'PPI',
            2: 'DDI',
            3: 'CPI',
            4: 'GDI',
            5: 'CDI',
        }
        # todo:这里是对多类别的关系进行总结抽取
        for idx in range(len(new_predicate_labels)):

            flag = new_predicate_labels[idx]
            prob = relation_probabilities_li[idx]

            if flag:
                if prob<thresh_hold:
                    continue
                print(len(relation_predicate_res), '/', step * config.batch_size)
                if config.num_labels == 6:
                    relation_counter[id2label[flag]] += 1


                    relation_predicate_res.append({
                        'id': 'r' + str(step * config.batch_size + idx),
                        'abstract_id': examples[step * config.batch_size + idx].abstract_id,
                        'e1_id': examples[step * config.batch_size + idx].ent1_id,
                        'e2_id': examples[step * config.batch_size + idx].ent2_id,
                        'relation_type': id2label[flag],
                        'probability': prob,
                    })


                elif config.num_labels == 2:
                    relation_predicate_res.append({
                        'id': 'r' + str(step * config.batch_size + idx),
                        'abstract_id': examples[step * config.batch_size + idx].abstract_id,
                        'e1_id': examples[step * config.batch_size + idx].ent1_id,
                        'e2_id': examples[step * config.batch_size + idx].ent2_id,
                        'relation_type': 1,
                        'probability': prob,
                    })

    print("花费时间",time.time()-start_time)
    set_trace()

    if config.dataset_name == '1009abstracts':
        path = "./outputs/predicate_outputs/origin_triplets/prob_1009abstracts_relations.pk"
    elif config.dataset_name == '3400abstracts':
        path = "./outputs/predicate_outputs/origin_triplets/prob_3400abstracts_relations.pk"
    else:
        raise ValueError
    with open(path, 'wb') as f:
        pickle.dump(relation_predicate_res, f)
    logger.info("将抽取结果保存到:{}".format(path))
    logger.info('-------预测类别结果----------------')
    logger.info(relation_counter)


if __name__ == '__main__':
    config = get_bert_config()
    # 设置时间
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff

    set_seed(config.seed)

    # 测试wandb

    # project表示这次项目，entity:表示提交人，config为超参数

    logger = get_logger(config)

    logger.info('---------------使用模型预测本次数据---------------------')
    print_hyperparameters(config, logger)
    # ckpt_path = '/root/code/bioner/re/outputs/trained_models/multi/mtb_schema1/model.pt'
    # ckpt_path = '/opt/data/private/luyuwei/code/bioner/re/outputs/save_models/2022-05-26/multi_task_biobert_sing_entity_marker_free_nums7_scheduler0.1_bs32_schema-1_lr1e-05/sing_entity_marker/AllDataset/best_model/model.pt'
    # ckpt_path = '/opt/data/private/luyuwei/code/bioner/re/outputs/save_models/2022-06-04/multi_task_biobert_sing_entity_marker_free_nums9_scheduler0.1_bs32_schema-1_lr1e-05/sing_entity_marker/AllDataset/1/model.pt'
    ckpt_path = '/opt/data/private/luyuwei/code/bioner/re/outputs/save_models/sing_entity_marker_schema_-12/multi/AllDataset/best_model/model.pt'
    config.num_labels = 6
    predicate(config, ckpt_path, logger)

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
from collections import Counter

from ipdb import set_trace
from tqdm import tqdm

import wandb

from sklearn.metrics import precision_recall_fscore_support,accuracy_score,f1_score,precision_score,recall_score,classification_report,confusion_matrix
from transformers import BertTokenizer

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from src.dataset_utils.data_process_utils import read_raw_data
from src.dataset_utils.entity_marker import MTBDataset
from src.dataset_utils.entity_type_marker import NormalDataset

from src.utils.function_utils import correct_datetime, set_seed, save_model, load_model_and_parallel
from src.utils.tips_utils import get_bert_config, get_logger, print_hyperparameters, wandb_log, show_log
from src.utils.train_utils import relation_classification_decode, batch_to_device, choose_model, \
    set_tokenize_special_tag


def predicate(config,ckpt_path=None,logger=None):

    # 读取所有的相关数据
    examples = read_raw_data(config)

    device = torch.device('cuda') if config.use_gpu else torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
    set_tokenize_special_tag(config, tokenizer)
    # 这个针对sentence-level的关系分类
    if config.data_format == 'single':
        dev_dataset = NormalDataset(examples, config=config,label2id=None, tokenizer=tokenizer,device=device)
    # MTB的方法，这个至少可以解决一些cross-sentence 关系分类
    elif config.data_format == 'cross':
        dev_dataset = MTBDataset(examples, config=config,label2id=None, tokenizer=tokenizer,device=device)
    else:
        raise ValueError


    dev_dataloader = DataLoader(dataset=dev_dataset, shuffle=False, collate_fn=dev_dataset.collate_fn_predicate,
                                num_workers=0, batch_size=config.batch_size)
    # 选择模型
    model = choose_model(config)

    model.bert_model.resize_token_embeddings(len(tokenizer))
    # 注意这里load_type
    # 根据训练完成的模型来选择合适的load_type

    model, device = load_model_and_parallel(model, '0', ckpt_path=ckpt_path, strict=True,
                                            load_type='one2one')
    model.to(device)
    all_predicate_tokens = []
    relation_predicate_res = []
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for step, batch_data in tqdm(enumerate(dev_dataloader),desc='正在预测数据...',total=len(dev_dataloader)):
            if config.data_format == 'single':
                input_ids, token_type_ids, attention_masks, e1_mask, e2_mask = batch_data
                labels = None
                logits = model(input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask)

                predicate_token = relation_classification_decode(logits)

            elif config.data_format == 'cross':
                input_ids, token_type_ids, attention_masks, e1_mask, e2_mask = batch_data
                labels = None
                input_ids.to(device)
                token_type_ids.to(device)
                attention_masks.to(device)
                e1_mask.to(device)
                e2_mask.to(device)
                logits = model(input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask)
                predicate_token = relation_classification_decode(logits)

            else:
                raise ValueError

            # 保证他们实体之间存在interaction

            id2label = {
                1:'PPI',
                2:'DDI',
                3:'CPI',
                4:'GDI',
                5:'CDI',
            }
            logger.info("当前数据个数有:{}/{}".format(len(relation_predicate_res),step*config.batch_size))
            for idx in range(config.batch_size):
                try:
                    flag = predicate_token[idx]
                except:
                    print("发生意外....")
                    break

                if flag:

                    if config.num_labels == 6:
                        relation_predicate_res.append({
                            'id': 'r' + str(step * config.batch_size + idx),
                            'abstract_id': examples[step * config.batch_size + idx].abstract_id,
                            'e1_id': examples[step * config.batch_size+idx].ent1_id,
                            'e2_id': examples[step* config.batch_size+idx].ent2_id,
                            'relation_type': id2label[flag],
                        })


                    elif config.num_labels == 2:

                        relation_predicate_res.append({
                            'id': 'r' + str(step * config.batch_size + idx),
                            'abstract_id': examples[step * config.batch_size + idx].abstract_id,
                            'e1_id': examples[step * config.batch_size + idx].ent1_id,
                            'e2_id': examples[step * config.batch_size + idx].ent2_id,
                            'relation_type': 1,
                        })
            # if predicate_token:
            #     logger.info("有效个数为:{}".format(len(relation_predicate_res)))
            #     if config.num_labels == 6:
            #         relation_predicate_res.append({
            #             'id':'r'+step,
            #             'abstract_id':examples[step].abstract_id,
            #             'e1_id':examples[step].ent1_id,
            #             'e2_id':examples[step].ent2_id,
            #             'relation_type':id2label[predicate_token],
            #         })
            #     elif config.num_labels == 2:
            #
            #         relation_predicate_res.append({
            #             'id': 'r' + str(step),
            #             'abstract_id': examples[step].abstract_id,
            #             'e1_id': examples[step].ent1_id,
            #             'e2_id': examples[step].ent2_id,
            #             'relation_type': 1,
            #         })
            # all_predicate_tokens.extend(predicate_token)
        print("花费时间",time.time()-start_time)
        set_trace()
    with open("./outputs/predicate_outputs/{}_re_results.txt".format(config.model_name),'wb') as f:
        pickle.dump(relation_predicate_res,f)

    logger.info('-------预测类别结果----------------')
    print(Counter(all_predicate_tokens))







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
    #ckpt_path = '/root/code/bioner/re/outputs/trained_models/multi/mtb_schema1/model.pt'
    # ckpt_path = '/root/code/bioner/re/trained_models/single/rbert/model.pt'
    ckpt_path = '/opt/data/private/luyuwei/code/bioner/re/outputs/save_models/2022-09-24/single_task_biobert_single_entity_marker_epochs2_free8_scheduler0.1_bs32_schema-12_lr1e-05/single_entity_marker/AllDataset/1/model.pt'
    predicate(config,ckpt_path,logger)
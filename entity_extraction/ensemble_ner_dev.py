# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  这个文件作用主要是拿训练完成的模型直接测试标准数据集
                这里只使用bert-based模型进行集成算法，不使用normal model

   Author :        kedaxia
   date：          2022/01/05
   Copyright:      (c) kedaxia 2022
-------------------------------------------------
   Change Activity:
                   2022/01/05: 
-------------------------------------------------
"""

import logging
import os
import datetime
import time
from collections import defaultdict
import copy


from tqdm import tqdm
from ipdb import set_trace
import torch
import numpy as np
import wandb


from transformers import BertTokenizer, AutoTokenizer

from seqeval.metrics import classification_report,f1_score,accuracy_score,recall_score,precision_score

from src.evaluate.evaluate_crf import entities_evaluate_fpr
from src.evaluate.evaluate_globalpointer import GlobalPointerMetrics
from src.evaluate.evaluate_span import evaluate_span_fpr, evaluate_span_micro
from src.models.bert_crf import EnsembleBertCRF
from src.models.bert_globalpointer import EnsembleBertGlobalPointer
from src.models.bert_mlp import EnsembleBertMLP
from src.models.bert_span import EnsembleBertSpan
from src.ner_predicate import crf_predicate, span_predicate, bert_globalpointer_predicate

from utils.function_utils import choose_model, choose_dataset_and_loader, get_logger, get_config, show_log
from utils.data_process_utils import read_data


from utils.function_utils import load_model_and_parallel


def ensemble_dev(trained_model_path_list,config,device, type_weight,word2id=None,tokenizer=None, metric=None,logger=None,epoch=None,global_step=None,ensemble_method='mix'):
    '''

    :param model:
    :param config:
    :param type_weight:
    :param ckpt_path:
    :param metric:
    :param ensemble_method: ['vote','mix']
    :return:p,r,f1
    '''

    model_name = config.model_name

    if config.model_name == 'bert_mlp':
        ensemble_model = EnsembleBertMLP(trained_model_path_list,config, device,bert_name_list = None)
    elif config.model_name == 'bert_crf':
        ensemble_model = EnsembleBertCRF(trained_model_path_list,config, device,bert_name_list = None)
    elif config.model_name == 'bert_span':
        ensemble_model = EnsembleBertSpan(config, trained_model_path_list, device, bert_name_list=None)
    elif config.model_name == 'bert_globalpointer':
        metric = GlobalPointerMetrics()
        ensemble_model = EnsembleBertGlobalPointer(trained_model_path_list, config, device,bert_name_list = None)

    dev_data, dev_labels = read_data(config.dev_file_path)
    if config.which_model =='bert' and tokenizer is None:
        if config.bert_name == 'biobert':
            tokenizer = BertTokenizer(config.bert_dir)
        elif config.bert_name == 'kebiolm':
            tokenizer = AutoTokenizer.from_pretrained(config.bert_dir)


    dev_dataset, dev_loader = choose_dataset_and_loader(config, dev_data, dev_labels, tokenizer, word2id)


    dev_start_predicate = []
    dev_end_predicate = []
    dev_start_ids = []
    dev_end_ids = []

    # 这是crf需要的label，用于之后的评测...
    dev_predicate = []
    # 这是存放raw_text
    dev_callback_info = []

    dev_orig_token_indexs = []

    batch_sum_dev_f1 = 0.
    batch_sum_dev_p = 0.
    batch_sum_dev_r = 0.


    # 下面两行是为了给每个epoch输出一个reports，方便之后的记录
    true_label_BIOs = []
    predicate_label_BIOs = []


    with torch.no_grad():
        for step, batch_data in tqdm(enumerate(dev_loader)):

            if model_name == 'bert_crf' or model_name == 'bert_bilstm_crf':

                if ensemble_method == 'vote':
                    # 这个就是返回一个预测的实体列表，[(ent_type,start,end,ent_name),....]
                    # 这个评估只能集合评估
                    predicate_entities = ensemble_model.vote_entities(batch_data,device,threshold=0.7)
                    raw_text_list, labels, _, _, _, _, _ = batch_data

                    labels = labels.numpy()
                    true_entities = crf_predicate(labels, config.crf_id2label, raw_text_list)

                    label2id = {"DNA": 0, "protein": 1, "cell_type": 2, "cell_line": 3, "RNA": 4}
                    tmp_dev_f1, tmp_dev_p, tmp_dev_r =  entities_evaluate_fpr(predicate_entities,true_entities,label2id,type_weight,average='micro',verbose=False)
                    batch_sum_dev_f1 += tmp_dev_f1
                    batch_sum_dev_p += tmp_dev_p
                    batch_sum_dev_r += tmp_dev_r

                elif ensemble_method == 'mix': # 将神经网络的输出logits进行加权融合，然后crf进行解码...
                    # 这个返回的就是crf的最终结果，这里默认使用models[0].crf_model进行解码
                    # 这个评估就可以使用seqeval...

                    _, labels, _, _, _, _, _= batch_data

                    tmp_predicate = ensemble_model.predicate(batch_data, device)
                    crf_model = ensemble_model.models[0].crf_model

                    loss_mask = labels.gt(-1)
                    loss_mask = loss_mask.to(device)
                    tmp_predicate = crf_model.decode(emissions=tmp_predicate, mask=loss_mask.byte())

                    true_labels = labels.numpy()

                    predicate_label_BIO = []
                    true_label_BIO = []

                    for i in range(len(tmp_predicate)):
                        predicate_label_BIO.append([config.crf_id2label[x] for x in tmp_predicate[i]])
                        true_label_BIO.append([config.crf_id2label[x] for x in true_labels[i][:len(tmp_predicate[i])]])

                    # 最后的classification report
                    true_label_BIOs.extend(true_label_BIO)
                    predicate_label_BIOs.extend(predicate_label_BIO)

                    # dev计算每个batch下的p,r,f1
                    tmp_dev_p = precision_score(true_label_BIO, predicate_label_BIO)
                    tmp_dev_r = recall_score(true_label_BIO, predicate_label_BIO)
                    tmp_dev_f1 = f1_score(true_label_BIO, predicate_label_BIO)

                    batch_sum_dev_f1 += tmp_dev_f1
                    batch_sum_dev_p += tmp_dev_p
                    batch_sum_dev_r += tmp_dev_r
                else:
                    raise ValueError

            elif model_name == 'bert_mlp':
                ensemble_method = 'mix'
                if ensemble_method == 'vote':
                    # 这个就是返回一个预测的实体列表，[(ent_type,start,end,ent_name),....]
                    # 这个评估只能集合评估

                    predicate_entities = ensemble_model.vote_entities(batch_data, device, threshold=0.7)
                    raw_text_list, labels, _, _, _, _, _ = batch_data

                    labels = labels.numpy()
                    true_entities = crf_predicate(labels, config.crf_id2label, raw_text_list)

                    label2id = {"DNA": 0, "protein": 1, "cell_type": 2, "cell_line": 3, "RNA": 4}
                    tmp_dev_f1, tmp_dev_p, tmp_dev_r = entities_evaluate_fpr(predicate_entities, true_entities,
                                                                             label2id, type_weight, average='micro',
                                                                             verbose=True)

                    batch_sum_dev_f1 += tmp_dev_f1
                    batch_sum_dev_p += tmp_dev_p
                    batch_sum_dev_r += tmp_dev_r

                elif ensemble_method == 'mix':  # 将神经网络的输出logits进行加权融合，然后crf进行解码...
                    # 这个返回的就是crf的最终结果，这里默认使用models[0].crf_model进行解码
                    # 这个评估就可以使用seqeval...

                    raw_text_list, labels, _, _, _, _, _ = batch_data

                    tmp_predicate = ensemble_model.predicate(batch_data, device)

                    output = np.argmax(tmp_predicate.detach().cpu().numpy(), axis=2)
                    output_token = []
                    for i, j in enumerate(output):
                        output_token.append(j[:len(labels[i])])

                    labels = labels.numpy()

                    predicate_label_BIO = []
                    true_label_BIO = []

                    for i in range(len(tmp_predicate)):
                        predicate_label_BIO.append([config.crf_id2label[x] for x in output_token[i][:len(raw_text_list[i])]])

                        true_label_BIO.append([config.crf_id2label[x] for x in labels[i][:len(raw_text_list[i])]])


                    # 最后的classification report

                    true_label_BIOs.extend(true_label_BIO)
                    predicate_label_BIOs.extend(predicate_label_BIO)

                    # dev计算每个batch下的p,r,f1
                    tmp_dev_p = precision_score(true_label_BIO, predicate_label_BIO)
                    tmp_dev_r = recall_score(true_label_BIO, predicate_label_BIO)
                    tmp_dev_f1 = f1_score(true_label_BIO, predicate_label_BIO)

                    batch_sum_dev_f1 += tmp_dev_f1
                    batch_sum_dev_p += tmp_dev_p
                    batch_sum_dev_r += tmp_dev_r

                else:
                    raise ValueError


            elif model_name == 'bert_span':

                if ensemble_method == 'vote':
                    # 这个就是返回一个预测的实体列表，[(ent_type,start,end,ent_name),....]
                    # 这个评估只能集合评估
                    # 这个就是返回一个预测的实体列表，[(ent_type,start,end,ent_name),....]
                    # 这个评估只能集合评估
                    predicate_entities = ensemble_model.vote_entities(batch_data, device, threshold=0.7)
                    raw_text_list, _, _, _, start_ids, end_ids, _ = batch_data

                    start_ids = start_ids.numpy()
                    end_ids = end_ids.numpy()

                    true_entities = span_predicate(start_ids,end_ids,raw_text_list,config.span_id2label)


                    tmp_dev_f1, tmp_dev_p, tmp_dev_r = entities_evaluate_fpr(predicate_entities, true_entities,
                                                                             config.span_label2id, type_weight, average='micro',
                                                                             verbose=True)

                    batch_sum_dev_f1 += tmp_dev_f1
                    batch_sum_dev_p += tmp_dev_p
                    batch_sum_dev_r += tmp_dev_r
                elif ensemble_method == 'mix':  # 将神经网络的输出logits进行加权融合，然后crf进行解码...
                    # 这个返回的就是crf的最终结果，这里默认使用models[0].crf_model进行解码
                    # 这个评估就可以使用seqeval...


                    raw_text_list, _, _, _, start_ids, end_ids, _ = batch_data
                    dev_callback_info.append(raw_text_list)
                    tmp_start_logits, tmp_end_logits = ensemble_model.predicate(batch_data, device)

                    _, span_start_logits = torch.max(tmp_start_logits, dim=-1)
                    _, span_end_logits = torch.max(tmp_end_logits, dim=-1)

                    # dev_start_ids.extend(start_ids.cpu().numpy())
                    # dev_end_ids.extend(end_ids.cpu().numpy())
                    # dev_start_predicate.extend(span_start_logits.cpu().numpy())
                    # dev_end_predicate.extend(span_end_logits.cpu().numpy())

                    tmp_dev_f1, tmp_dev_p, tmp_dev_r = evaluate_span_fpr(span_start_logits.cpu().numpy(), span_end_logits.cpu().numpy(), start_ids.cpu().numpy(),
                                                             end_ids.cpu().numpy(), raw_text_list,
                                                             average=config.evaluate_mode, type_weight=type_weight,
                                                             span_label2id=config.span_label2id, verbose=config.verbose)
                    batch_sum_dev_f1 += tmp_dev_f1
                    batch_sum_dev_p += tmp_dev_p
                    batch_sum_dev_r += tmp_dev_r

                else:
                    raise ValueError



            elif model_name == 'bert_globalpointer':
                ensemble_method = 'mix'
                if ensemble_method == 'vote':

                    predicate_entities = ensemble_model.vote_entities(batch_data, device, threshold=0.7)
                    raw_text_list, batch_true_labels, _, _, _, _, _ = batch_data

                    batch_true_labels = batch_true_labels.numpy()
                    batch_true_labels.dtype = np.float # 由于对正整数解码，所以这里转变一下格式

                    true_entities = bert_globalpointer_predicate(batch_true_labels,config.globalpointer_id2label,raw_text_list)

                    tmp_dev_f1, tmp_dev_p, tmp_dev_r = entities_evaluate_fpr(predicate_entities, true_entities,config.globalpointer_label2id, type_weight,
                                                                             average='micro',
                                                                             verbose=True)
                    print(tmp_dev_f1,tmp_dev_p)
                    batch_sum_dev_f1 += tmp_dev_f1
                    batch_sum_dev_p += tmp_dev_p
                    batch_sum_dev_r += tmp_dev_r
                elif ensemble_method == 'mix':

                    raw_text_list, batch_true_labels, _, _, _, _, _ = batch_data
                    global_pointer_true_labels = batch_true_labels.to(device)

                    globalpointer_predicate = ensemble_model.predicate(batch_data, device)


                    tmp_dev_f1,tmp_dev_p,tmp_dev_r = metric.get_evaluate_fpr(globalpointer_predicate, global_pointer_true_labels,config.globalpointer_label2id,type_weight=type_weight,average=config.evaluate_mode,verbose=True)

                    batch_sum_dev_f1 += tmp_dev_f1
                    batch_sum_dev_p += tmp_dev_p
                    batch_sum_dev_r += tmp_dev_r


            else:
                raise ValueError





    t_total = len(dev_loader)
    if config.decoder_layer == 'span':
        dev_f1, dev_p, dev_r = batch_sum_dev_f1 / t_total, batch_sum_dev_p / t_total, batch_sum_dev_r / t_total
        #dev_f1,dev_p, dev_r = evaluate_span_fpr(dev_start_predicate, dev_end_predicate, dev_start_ids, dev_end_ids, dev_callback_info,average=config.evaluate_mode,type_weight=type_weight,span_label2id=config.span_label2id,verbose=config.verbose)

    elif config.decoder_layer in ['crf','mlp','test']:
        if len(true_label_BIOs)>0:
            report = classification_report(true_label_BIOs, predicate_label_BIOs,digits=4)
            logger.info(report)
        dev_f1, dev_p, dev_r = batch_sum_dev_f1 / t_total, batch_sum_dev_p / t_total, batch_sum_dev_r / t_total

    elif config.decoder_layer == 'globalpointer':
        dev_f1, dev_p, dev_r = batch_sum_dev_f1 / t_total, batch_sum_dev_p / t_total, batch_sum_dev_r / t_total

    else:
        raise ValueError
    show_log(logger, step, len(dev_loader), t_total, epoch, global_step, 0., dev_p, dev_r, dev_f1,0., config.evaluate_mode, type='train', scheme=0)
    return dev_p, dev_r, dev_f1




if __name__ == '__main__':

    config = get_config()

    logger = get_logger(config)

    # 设置时间
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff
    # 设置随机种子

    ensemble_method = 'mix'
    logger.info('开启集成评估方法.................')
    if config.run_type == 'cv5':
        P,R,F1 = 0.,0.,0.
        trained_model_path_list = []
        for i in range(1,6):
            ckpt_path = '/root/code/bioner/ner/outputs/save_models/{}/{}/cv5/cv_{}/best_model/model.pt'.format(
                config.model_name, config.ner_dataset_name, i)
            trained_model_path_list.append(ckpt_path)

        device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')
        dev_p, dev_r, dev_f1 = ensemble_dev(trained_model_path_list,config,device=device, type_weight=None,word2id=None,tokenizer=None, metric=None,logger=logger,epoch=None,global_step=None,ensemble_method=ensemble_method)
        logger.info('集成方法:{}的结果，F1:{},P:{},R:{}'.format(ensemble_method,dev_f1,dev_p,dev_r))
    elif config.run_type == 'cv10':
        P, R, F1 = 0., 0., 0.
        trained_model_path_list = []
        for i in range(1, 11):
            ckpt_path = '/root/code/bioner/ner/outputs/save_models/{}/{}/cv10/cv_{}/best_model/model.pt'.format(
                config.model_name, config.ner_dataset_name, i)
            trained_model_path_list.append(ckpt_path)
        device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')
        ensemble_dev(trained_model_path_list, config, device=device, type_weight=None, word2id=None, tokenizer=None,
                     metric=None, logger=logger, epoch=None, global_step=None, ensemble_method='vote')

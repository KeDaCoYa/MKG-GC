# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  这里是加载已经训练完成的模型，然后输入一个文本，输出存在的实体
                    或者输入一个abstracts，然后输出所有实体，为之后的关系分类提供条件
                    以一句话为单位，也可以输入abstract，full-text...
                    必须设置模型为一次只能读一句话，但是这句话的长度只要不超过max_len就可以
   Author :        kedaxia
   date：          2021/11/08
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/11/08:
-------------------------------------------------
"""

import argparse
import logging
import os
import datetime
import time
import random
import copy
from ipdb import set_trace
import pickle
import json

from tqdm import tqdm

import torch
from transformers import BertTokenizer, AutoTokenizer

from nltk.tokenize import sent_tokenize,wordpunct_tokenize

from src.models.bert_mlp import EnsembleBertMLP
from src.models.bert_span import EnsembleBertSpan
from src.ner_predicate import normal_globalpointer_predicate, crf_predicate, span_predicate
from src.pubmed_util.file_util import read_abstract_text
from utils.function_utils import choose_model, choose_dataset_and_loader, get_config, get_logger, count_parameters
from utils.train_utils import load_model



def read_raw_text():
    '''
    这里读取时，以一个文件一个文件的读取，抽取结果必须得追溯到是从哪个文献得到的
    '''
    abstract_text = 'The FHIT (fragile histidine triad) gene has been recently identified and cloned at chromosome 3p14.2 including FRA3B, the most common fragile site in the human genome. FHIT is suggested to be a candidate tumour suppressor gene in gastrointestinal tract tumours. To elucidate the role of the FHIT gene in gastric cancer, a total of 133 curatively R0-resected gastric carcinomas were investigated for loss of heterozygosity (LOH) at 3p14.2, using four polymorphic microsatellite loci (D3S1300, D3S1313, D3S1481, and D3S1234). LOH of the FHIT gene affecting at least one of the investigated loci was observed in 20 of 123 informative tumours (16.3 per cent). The presence of LOH was correlated neither with major prognostic factors such as pT category, pN category or vascular invasion, nor with histological type or grade of differentiation of the tumours. In addition, there were no differences in the prognosis between patients with gastric carcinomas showing LOH at the FHIT gene and patients with tumours lacking LOH at the FHIT gene. These findings suggest that LOH of the FHIT gene represents an event in the tumourigenesis of only a small subset of gastric carcinomas and does not correlate with tumour progression or prognosis.'
    sents = sent_tokenize(abstract_text)
    sent_li = []
    for sent in sents:
        sent_li.append(wordpunct_tokenize(sent))

    return abstract_text,sent_li



def get_best_trained_model(config,ckpt_path='/root/code/bioner/ner/outputs/save_models/bilstm_globalpointer/jnlpba/best_model/model.pt'):
    '''
    加载已经训练完成的(NER)模型,以及分词器
    :param model:
    :param config:
    :param type_weight:
    :param ckpt_path:
    :param metric:
    :return:p,r,f1
    '''
    word2id, id2word = None, None
    if config.which_model == 'bert':
        model, metric = choose_model(config)  # 这个metric没啥用，只在globalpointer有用
    elif config.which_model == 'normal':
        model, metric, word2id, id2word = choose_model(config)
        # 直接限制normal model为变成batch_len
        config.fixed_batch_length = False
    else:
        raise ValueError("错误，只能选择['bert','normal']")

    model = load_model(model, ckpt_path=ckpt_path)

    device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')
    model.to(device)
    if config.which_model == 'bert':
        tokenizer = BertTokenizer(os.path.join(config.bert_dir, 'vocab.txt'))

        return model,tokenizer,device
    return model,device,word2id

def single_model_predicate(sent_li,config,model,device,word2id=None,tokenizer=None):
    '''
    这是单个模型的预测
    sent_li:是一个abstract或者full text的所有句子，已经分离开来了....

    '''

    model_name = config.model_name

    if config.which_model == 'bert' and tokenizer is None:

        if config.bert_name == 'biobert':
            tokenizer = BertTokenizer(os.path.join(config.bert_dir, 'vocab.txt'))
        elif config.bert_name == 'kebiolm':
            tokenizer = AutoTokenizer.from_pretrained(config.bert_dir)

    dev_dataset, dev_loader = choose_dataset_and_loader(config, sent_li, None, tokenizer, word2id, type_='predicate')

    all_entities = []

    model.eval()
    with torch.no_grad():
        for step, batch_data in enumerate(dev_loader):
            if config.which_model == 'bert':
                if model_name == 'bert_crf' or model_name == 'bert_bilstm_crf' or model_name == 'bert_mlp':

                    raw_text_list, batch_subword_input_ids, batch_subword_token_type_ids, batch_subword_attention_masks, origin_to_subword_indexs, batch_label_mask = batch_data

                    token_ids, attention_masks, token_type_ids = batch_subword_input_ids.to(
                        device), batch_subword_attention_masks.to(device), batch_subword_token_type_ids.to(device)



                    tmp_predicate = model.forward(token_ids, attention_masks=attention_masks,
                                                  token_type_ids=token_type_ids, labels=None,
                                                  input_token_starts=origin_to_subword_indexs)
                    entities = crf_predicate(tmp_predicate, config.crf_id2label, raw_text_list)


                elif model_name == 'bert_span' or model_name == 'interbergruspan':

                    raw_text_list, token_ids, attention_masks, token_type_ids, origin_to_subword_index ,input_true_length= batch_data

                    token_ids, attention_masks, token_type_ids = token_ids.to(device), attention_masks.to(device), token_type_ids.to(device)
                    input_true_length = input_true_length.to(device)
                    tmp_start_logits, tmp_end_logits = model(token_ids, attention_masks=attention_masks,token_type_ids=token_type_ids,input_token_starts=origin_to_subword_index, start_ids=None,end_ids=None,input_true_length=input_true_length)

                    _, span_start_logits = torch.max(tmp_start_logits, dim=-1)
                    _, span_end_logits = torch.max(tmp_end_logits, dim=-1)

                    span_start_logits = span_start_logits.cpu().numpy()
                    span_end_logits = span_end_logits.cpu().numpy()

                    entities = span_predicate(span_start_logits, span_end_logits, raw_text_list, config.span_id2label)



                elif model_name == 'bert_globalpointer':

                    raw_text_list, batch_subword_input_ids, batch_subword_token_type_ids, batch_subword_attention_masks, origin_to_subword_indexs = batch_data

                    token_ids, attention_masks, token_type_ids = batch_subword_input_ids.to(
                        device), batch_subword_attention_masks.to(device), batch_subword_token_type_ids.to(device)


                    globalpointer_predicate = model(token_ids, attention_masks=attention_masks,
                                                    token_type_ids=token_type_ids, labels=None,
                                                    input_token_starts=origin_to_subword_indexs)
                    globalpointer_predicate = globalpointer_predicate.cpu().numpy()
                    entities = normal_globalpointer_predicate(globalpointer_predicate, config.globalpointer_id2label,
                                                   raw_text_list)

                else:
                    raise ValueError
            elif config.which_model == 'normal':
                if config.model_name == 'bilstm_globalpointer' or config.model_name == 'att_bilstm_globalpointer':
                    # 这好像用不到attention

                    raw_text_list, token_ids, attention_masks = batch_data
                    token_ids = token_ids.to(device)

                    globalpointer_predicate = model(token_ids, None)
                    globalpointer_predicate = globalpointer_predicate.cpu().numpy()

                    entities = normal_globalpointer_predicate(globalpointer_predicate, config.globalpointer_id2label,
                                                   raw_text_list)



                elif config.model_name == 'bilstm_crf' or config.model_name == 'att_bilstm_crf':
                    raw_text_list, token_ids, attention_masks = batch_data
                    token_ids = token_ids.to(device)

                    attention_masks = attention_masks.to(device)
                    predicate = model(token_ids, None,
                                      attention_masks)  # scores.shape = (batch_size,max_seq_len,num_class,num_class) 输出标签的种类数量

                    entities = crf_predicate(predicate, config.crf_id2label, raw_text_list)



                elif config.model_name == 'bilstm_cnn_crf' or config.model_name == 'att_bilstm_cnn_crf':
                    # 这里没有fixed_batch_lenght,只有dynamic batch_len
                    raw_text_list, token_ids, char_token_ids, attention_masks = batch_data
                    token_ids = token_ids.to(device)

                    attention_masks = attention_masks.to(device)
                    char_token_ids = char_token_ids.to(device)
                    tmp_predicate = model(token_ids, char_token_ids, None, attention_masks,
                                          pack_unpack=config.lstm_pack_unpack)

                    crf_predicate(tmp_predicate, config.crf_id2label, raw_text_list)
                else:
                    raise ValueError
            else:
                raise ValueError('选择normal,bert....')
            if entities.__len__() != 0:
                #all_entities.append({'raw_text':raw_text_list[0],'entities':entities,'sent_pos':step})
                for ent in entities:
                    ent['pos'] = step

                all_entities.extend(entities)

    return all_entities

def ensemble_model_predicate(ensemble_model,sent_li,config,device,word2id=None,tokenizer=None):
    '''
    这是集成模型的预测
    只针对bert-based model

    '''

    model_name = config.model_name
    config.predicate_flag = True
    if config.which_model == 'bert' and tokenizer is None:

        if config.bert_name == 'biobert':
            tokenizer = BertTokenizer(os.path.join(config.bert_dir, 'vocab.txt'))
        elif config.bert_name == 'kebiolm':
            tokenizer = AutoTokenizer.from_pretrained(config.bert_dir)

    dev_dataset, dev_loader = choose_dataset_and_loader(config, sent_li, None, tokenizer, word2id, type='predicate')
    all_entities = []

    with torch.no_grad():
        for step, batch_data in enumerate(dev_loader):

            if model_name == 'bert_crf' or model_name == 'bert_bilstm_crf' or model_name == 'bert_mlp':

                raw_text_list, batch_subword_input_ids, batch_subword_token_type_ids, batch_subword_attention_masks, origin_to_subword_indexs, batch_label_mask = batch_data

                token_ids, attention_masks, token_type_ids = batch_subword_input_ids.to(
                    device), batch_subword_attention_masks.to(device), batch_subword_token_type_ids.to(device)



                tmp_predicate = ensemble_model(token_ids, attention_masks=attention_masks,
                                              token_type_ids=token_type_ids, labels=None,
                                              input_token_starts=origin_to_subword_indexs)
                entities = crf_predicate(tmp_predicate, config.crf_id2label, raw_text_list)


            elif model_name == 'bert_span':


                entities = ensemble_model.vote_entities(batch_data, device, threshold=0.4)


            elif model_name == 'bert_globalpointer':

                raw_text_list, batch_subword_input_ids, batch_subword_token_type_ids, batch_subword_attention_masks, origin_to_subword_indexs = batch_data

                token_ids, attention_masks, token_type_ids = batch_subword_input_ids.to(
                    device), batch_subword_attention_masks.to(device), batch_subword_token_type_ids.to(device)


                globalpointer_predicate = ensemble_model(token_ids, attention_masks=attention_masks,
                                                token_type_ids=token_type_ids, labels=None,
                                                input_token_starts=origin_to_subword_indexs)
                globalpointer_predicate = globalpointer_predicate.cpu().numpy()
                entities = normal_globalpointer_predicate(globalpointer_predicate, config.globalpointer_id2label,
                                               raw_text_list)

            else:
                raise ValueError

        if entities.__len__() != 0:
            #all_entities.append({'raw_text':raw_text_list[0],'entities':entities,'sent_pos':step})
            for ent in entities:
                ent['pos'] = step

            all_entities.extend(entities)

        return all_entities

def pubmed_abatracts_single_model_predicate_text(config,model,device,word2id=None,tokenizer=None):
    '''
        模型根据输入的数据，输出含有的实体
        inputs：list(str)

    '''

    config.batch_size = 1
    # 获取训练集
    #    logger.info('----------从验证集文件中读取数据-----------')
    abstracts_li = read_abstract_text(abstract_dir = './pubmed_abstracts/cancer_pubmed/abstracts')
    abstract_entities_dict = {}
    # 为每个ent加上ID,方便之后的...
    ent_idx = 0
    start = time.time()
    for abstract in tqdm(abstracts_li):
        #abstract_text = abstract['raw_text']
        ID = abstract['file_id']

        sent_li = abstract['sent_li']
        all_entities = single_model_predicate(sent_li,config,model,device,word2id=word2id,tokenizer=tokenizer)
        new_all_entities = []
        for ent in all_entities:
            ent['id'] = 'e'+str(ent_idx)
            ent_idx += 1
            new_all_entities.append(ent)
        abstract_entities_dict[ID] = {
            'abstract_sentence_li':sent_li,
            'entities':new_all_entities
        }
    print("花费时间", time.time() - start)

    #TODO: 这里暂时关掉
    # with open('outputs/extract_results/normalize_alldataset/normalize_balance_alldataset0.2_entities.json', 'w', encoding='utf-8') as f:
    #     json.dump(abstract_entities_dict,f)

def pubmed_abatracts_ensemble_model_predicate_text(config,device,trained_model_path_list,word2id=None,tokenizer=None):
    '''
        模型根据输入的数据，输出含有的实体
        inputs：list(str)

    '''
    config.batch_size = 1
    abstracts_li = read_abstract_text()
    abstract_entities_dict = {}
    # 选择出合适的ensemble模型
    if config.model_name == 'bert_mlp':
        ensemble_model = EnsembleBertMLP(trained_model_path_list,config, device)
    elif config.model_name == 'bert_crf':
        ensemble_model = EnsembleBertMLP(trained_model_path_list, config, device)
    elif config.model_name == 'bert_span':
        ensemble_model = EnsembleBertSpan(config, trained_model_path_list, device)
    elif config.model_name == 'bert_globalpointer':
        pass
    for abstract in tqdm(abstracts_li):
        #abstract_text = abstract['raw_text']
        ID = abstract['file_id']
        sent_li = abstract['sent_li']

        all_entities = ensemble_model_predicate(ensemble_model,sent_li,config,device,word2id=word2id,tokenizer=tokenizer)
        ent_idx = 0
        new_all_entities = []
        for ent in all_entities:
            ent['id'] = 'e' + str(ent_idx)
            ent_idx += 1
            new_all_entities.append(ent)
        abstract_entities_dict[ID] = {
            'abstract_sentence_li':sent_li,
            'entities':new_all_entities
        }
    with open('./outputs/extract_results/cv5_abstracts_entities_thresh(0.4).json','w',encoding='utf-8') as f:
        json.dump(abstract_entities_dict,f)
def single_main():
    config = get_config()

    logger = get_logger(config)

    # 设置时间
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff
    # 设置随机种子

    logger.info('----------------本次模型运行的参数------------------')
    word2id = None
    tokenize = None
    trained_model_path_list = [
        '/opt/data/private/luyuwei/code/bioner/ner/outputs/save_models/2022-11-30/inter40_biobert_interbergruspan_epochs15_free_nums8_scheduler0.1_bs32_lr1e-05_mx512/interbergruspan/NCBI-disease/1/interbergruspan.pt',

    ]

    ckpt_path = trained_model_path_list[0]

    if config.which_model == 'normal':
        model, device, word2id = get_best_trained_model(config, ckpt_path=ckpt_path)
    elif config.which_model == 'bert':
        model, tokenize, device = get_best_trained_model(config, ckpt_path=ckpt_path)
    model.to(device)
    count_parameters(model)

    set_trace()
    pubmed_abatracts_single_model_predicate_text(config, model, device, word2id=word2id, tokenizer=tokenize)

def ensemble_main():
    config = get_config()

    logger = get_logger(config)

    # 设置时间
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff
    # 设置随机种子

    logger.info('----------------本次模型运行的参数------------------')
    word2id = None
    tokenize = None
    trained_model_path_list = [
        '/root/code/bioner/ner/trained_model/bert_span/cv5/cv_1/best_model/model.pt',
        '/root/code/bioner/ner/trained_model/bert_span/cv5/cv_2/best_model/model.pt',
        '/root/code/bioner/ner/trained_model/bert_span/cv5/cv_3/best_model/model.pt',
        '/root/code/bioner/ner/trained_model/bert_span/cv5/cv_4/best_model/model.pt',
        '/root/code/bioner/ner/trained_model/bert_span/cv5/cv_5/best_model/model.pt',
                               ]
    ckpt_path = trained_model_path_list[0]
    if config.which_model == 'normal':
        model, device, word2id = get_best_trained_model(config, ckpt_path=ckpt_path)
    elif config.which_model == 'bert':
        model, tokenize, device = get_best_trained_model(config, ckpt_path=ckpt_path)


    pubmed_abatracts_ensemble_model_predicate_text(config, device, trained_model_path_list, word2id=None,
                                                   tokenizer=None)


if __name__ == '__main__':
    single_main()
    #ensemble_main()


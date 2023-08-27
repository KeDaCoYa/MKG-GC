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
from collections import defaultdict

from ipdb import set_trace
import pickle
import json

from tqdm import tqdm

import torch
from transformers import BertTokenizer, AutoTokenizer

from nltk.tokenize import sent_tokenize, wordpunct_tokenize

from src.models.bert_mlp import EnsembleBertMLP
from src.models.bert_span import EnsembleBertSpan
from src.ner_predicate import normal_globalpointer_predicate, crf_predicate, span_predicate
from src.pubmed_util.file_util import read_abstract_text
from utils.function_utils import choose_model, choose_dataset_and_loader, get_config, get_logger, \
    get_predicate_dataset_loader, choose_multi_ner_model, count_parameters
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

    return abstract_text, sent_li


def get_best_trained_model(config,ckpt_path):
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
    try:
        tokenizer = BertTokenizer(os.path.join(config.bert_dir, 'vocab.txt'))
    except:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(config.bert_dir, 'vocab.txt'))
    config.vocab_size = len(tokenizer)
    if config.which_model == 'bert':
        model, metric = choose_multi_ner_model(config)  # 这个metric没啥用，只在globalpointer有用
    else:
        raise ValueError("错误，只能选择['bert']")

    device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')
    model.to(device)

    model = load_model(model, ckpt_path=ckpt_path)

    return model, device, tokenizer, word2id


def single_model_predicate(sent_li, config, model, device, word2id=None, tokenizer=None):
    """
    这是单个模型的对输入的sent_li进行预测
    sent_li:是一个abstract或者full text的所有句子，已经分离开来了....

    """

    model_name = config.model_name

    if config.which_model == 'bert' and tokenizer is None:

        if config.bert_name == 'biobert':
            tokenizer = BertTokenizer(os.path.join(config.bert_dir, 'vocab.txt'))
        elif config.bert_name == 'kebiolm':
            tokenizer = AutoTokenizer.from_pretrained(config.bert_dir)

    dev_dataset, dev_loader = get_predicate_dataset_loader(config, sent_li, tokenizer)

    all_entities = []

    all_entities_dict = {}

    model.eval()
    count_ = 0
    for step, batch_data in enumerate(dev_loader):
        entities = []
        if config.which_model == 'bert':

            if model_name in ['binary_bert_span','inter_bert_span','bert_span','inter_binary_bert_gru_mid_span']:
                raw_text_list, token_ids, attention_masks, token_type_ids, origin_to_subword_index, input_true_length, entity_type_ids = batch_data

                token_ids, attention_masks, token_type_ids = token_ids.to(device), attention_masks.to(
                    device), token_type_ids.to(device)
                input_true_length = input_true_length.to(device)
                entity_type_ids = entity_type_ids.to(device)
                with torch.no_grad():
                    tmp_start_logits, tmp_end_logits = model(token_ids, attention_masks=attention_masks,
                                                             token_type_ids=token_type_ids,
                                                             input_token_starts=origin_to_subword_index,
                                                             start_ids=None,
                                                             end_ids=None,
                                                             input_true_length=input_true_length,
                                                             entity_type_ids=entity_type_ids)

                dise_start_logits, chem_start_logits, gene_start_logits, spec_start_logits, celltype_start_logits, cellline_start_logits, dna_start_logits, rna_start_logits = tmp_start_logits
                dise_end_logits, chem_end_logits, gene_end_logits, spec_end_logits, celltype_end_logits, cellline_end_logits, dna_end_logits, rna_end_logits = tmp_end_logits

                _, dise_span_start_logits = torch.max(dise_start_logits, dim=-1)
                _, dise_span_end_logits = torch.max(dise_end_logits, dim=-1)

                _, chem_span_start_logits = torch.max(chem_start_logits, dim=-1)
                _, chem_span_end_logits = torch.max(chem_end_logits, dim=-1)

                _, gene_span_start_logits = torch.max(gene_start_logits, dim=-1)
                _, gene_span_end_logits = torch.max(gene_end_logits, dim=-1)

                _, spec_span_start_logits = torch.max(spec_start_logits, dim=-1)
                _, spec_span_end_logits = torch.max(spec_end_logits, dim=-1)

                _, celltype_span_start_logits = torch.max(celltype_start_logits, dim=-1)
                _, celltype_span_end_logits = torch.max(celltype_start_logits, dim=-1)

                _, cellline_span_start_logits = torch.max(cellline_start_logits, dim=-1)
                _, cellline_span_end_logits = torch.max(cellline_end_logits, dim=-1)

                _, dna_span_start_logits = torch.max(dna_start_logits, dim=-1)
                _, dna_span_end_logits = torch.max(dna_end_logits, dim=-1)

                _, rna_span_start_logits = torch.max(rna_start_logits, dim=-1)
                _, rna_span_end_logits = torch.max(rna_end_logits, dim=-1)

                dise_span_start_logits = dise_span_start_logits.cpu().numpy()
                dise_span_end_logits = dise_span_end_logits.cpu().numpy()

                chem_span_start_logits = chem_span_start_logits.cpu().numpy()
                chem_span_end_logits = chem_span_end_logits.cpu().numpy()

                gene_span_start_logits = gene_span_start_logits.cpu().numpy()
                gene_span_end_logits = gene_span_end_logits.cpu().numpy()

                spec_span_start_logits = spec_span_start_logits.cpu().numpy()
                spec_span_end_logits = spec_span_end_logits.cpu().numpy()

                celltype_span_start_logits = celltype_span_start_logits.cpu().numpy()
                celltype_span_end_logits = celltype_span_end_logits.cpu().numpy()

                cellline_span_start_logits = cellline_span_start_logits.cpu().numpy()
                cellline_span_end_logits = cellline_span_end_logits.cpu().numpy()

                dna_span_start_logits = dna_span_start_logits.cpu().numpy()
                dna_span_end_logits = dna_span_end_logits.cpu().numpy()

                rna_span_start_logits = rna_span_start_logits.cpu().numpy()
                rna_span_end_logits = rna_span_end_logits.cpu().numpy()

                dise_entities = span_predicate(dise_span_start_logits, dise_span_end_logits, raw_text_list,
                                               config.span_id2label)
                chem_entities = span_predicate(chem_span_start_logits, chem_span_end_logits, raw_text_list,
                                               config.span_id2label)
                gene_entities = span_predicate(gene_span_start_logits, gene_span_end_logits, raw_text_list,
                                               config.span_id2label)
                spec_entities = span_predicate(spec_span_start_logits, spec_span_end_logits, raw_text_list,
                                               config.span_id2label)
                celltype_entities = span_predicate(celltype_span_start_logits, celltype_span_end_logits, raw_text_list,
                                                   config.span_id2label)
                cellline_entities = span_predicate(cellline_span_start_logits, cellline_span_end_logits, raw_text_list,
                                                   config.span_id2label)
                dna_entities = span_predicate(dna_span_start_logits, dna_span_end_logits, raw_text_list,
                                              config.span_id2label)
                rna_entities = span_predicate(rna_span_start_logits, rna_span_end_logits, raw_text_list,
                                              config.span_id2label)
                for ent in dise_entities:
                    if ent['entity_type'] != 'Disease':
                        continue
                    else:
                        entities.append(ent)
                for ent in chem_entities:
                    if ent['entity_type'] != 'Chemical/Drug':
                        continue
                    else:
                        entities.append(ent)
                for ent in gene_entities:
                    if ent['entity_type'] != 'Gene/Protein':
                        continue
                    else:
                        entities.append(ent)
                for ent in spec_entities:
                    if ent['entity_type'] != 'Species':
                        continue
                    else:
                        entities.append(ent)
                for ent in celltype_entities:
                    if ent['entity_type'] != 'cell_type':
                        continue
                    else:
                        entities.append(ent)

                for ent in cellline_entities:
                    if ent['entity_type'] != 'cell_line':
                        continue
                    else:
                        entities.append(ent)

                for ent in dna_entities:
                    if ent['entity_type'] != 'DNA':
                        continue
                    else:
                        entities.append(ent)
                for ent in rna_entities:
                    if ent['entity_type'] != 'RNA':
                        continue
                    else:
                        entities.append(ent)

        else:
            raise ValueError('选择normal,bert....')

        if entities.__len__() != 0:
            # all_entities.append({'raw_text':raw_text_list[0],'entities':entities,'sent_pos':step})
            for ent in entities:
                ent['pos'] = step
            count_ += len(entities)
            all_entities.extend(entities)

    return all_entities


def post_preprocess_entities(entities_li):
    """
    这是以一个abstract的为单位进行,对抽取到的实体进行检查
    相当于是一个投票法进行
     {
        'entity_type': span_id2label[s_type],
        'start_idx': str(i),
        'end_idx': str(i+j),
        'entity_name': " ".join(raw_text_li[id][i:i+j+1])
    }
    """
    entities_dict = defaultdict(lambda: defaultdict(int))
    for ent in entities_li:
        ent_name = ent['entity_name']
        ent_type = ent['entity_type']
        entities_dict[ent_name][ent_type] += 1
    new_entities_li = []
    count = 0

    for ent in entities_li:
        ent_name = ent['entity_name']
        ent_types_li_dict = entities_dict[ent_name]

        if len(ent_types_li_dict) > 1:
            more_ent_type = sorted(ent_types_li_dict.items(), key=lambda x: x[1], reverse=True)[0]
            if more_ent_type[1] > 1:
                ent['entity_type'] = more_ent_type[0]
                count += 1
        new_entities_li.append(ent)


    return new_entities_li


def pubmed_abatracts_single_model_predicate_text(logger,config, model, device=None, word2id=None, tokenizer=None):
    """
        模型根据输入的数据，输出含有的实体
        inputs：list(str)

    """

    config.batch_size = 1

    # 这是1009篇abstracts
    if config.ner_dataset_name == '1009abstracts':
        abstract_dir = './pubmed_abstracts/cancer_pubmed/abstracts'
        type_ = 'json'
        output_path = './outputs/extract_results/1009abstracts_entities.json'
    elif config.ner_dataset_name == '3400abstracts':
        # 这是3400篇abstracts
        abstract_dir = './pubmed_abstracts/cancer_pubmed/output_key_word_abstracts'
        type_ = 'text'
        output_path = './outputs/extract_results/3400abstracts_entities.json'
    else:
        raise ValueError


    abstracts_li = read_abstract_text(abstract_dir, type_)
    abstract_entities_dict = {}
    # 为每个ent加上ID,方便之后的...
    ent_idx = 0
    entities_count = 0
    entity_type_counter = defaultdict(int)
    start_time = time.time()
    for abstract in tqdm(abstracts_li, total=len(abstracts_li), desc='正在预测{}....'.format(config.ner_dataset_name)):
        # abstract_text = abstract['raw_text']
        ID = abstract['file_id']

        sent_li = abstract['sent_li']


        all_entities = single_model_predicate(sent_li, config, model, device, word2id=word2id, tokenizer=tokenizer)

        entities_count += len(all_entities)
        print("已抽取实体个数:{}".format(entities_count))
        all_entities = post_preprocess_entities(all_entities)
        new_all_entities = []
        for ent in all_entities:
            ent['id'] = 'e' + str(ent_idx)
            ent_idx += 1
            new_all_entities.append(ent)
            entity_type_counter[ent['entity_type']] += 1
        abstract_entities_dict[ID] = {
            'abstract_sentence_li': sent_li,
            'entities': new_all_entities
        }
    print("花费时间", time.time() - start_time)


    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(abstract_entities_dict, f)
    logger.info("抽取的实体数据保存到:{}".format(os.path.abspath(output_path)))


def single_main():
    config = get_config()

    logger = get_logger(config)

    # 设置时间
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff
    # 设置随机种子

    logger.info('----------------本次模型运行的参数------------------')

    # ckpt_path = '/opt/data/private/luyuwei/code/bioner/ner/outputs/save_models/2022-06-16/MT_biobert_binary_bert_span_epochs15_free8_scheduler0.1_bs64_lr1e-05/binary_bert_span/multi_all_dataset_v1_lite/best_model/binary_bert_span.pt'
    # ckpt_path = '/opt/data/private/luyuwei/code/bioner/ner/trained_model/multi_all_dataset_large/multi_bert_span/bert_span.pt'
    ckpt_path = '/opt/data/private/luyuwei/code/bioner/ner/outputs/save_models/2022-11-30/MT_biobert_inter_binary_bert_gru_mid_span_epochs15_free8_scheduler0.1_bs32_lr1e-05/inter_binary_bert_gru_mid_span/multi_all_dataset_plus/13/inter_binary_bert_gru_mid_span.pt'

    model, device, tokenizer, word2id = get_best_trained_model(config, ckpt_path=ckpt_path)

    pubmed_abatracts_single_model_predicate_text(logger,config, model, device=device, word2id=word2id, tokenizer=tokenizer)


if __name__ == '__main__':
    single_main()
    # ensemble_main()

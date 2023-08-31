# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :   这个是模型的预测，输入一个mention，直接输出所有结果
   Author :        kedaxia
   date：          2022/01/18
   Copyright:      (c) kedaxia 2022
-------------------------------------------------
   Change Activity:
                   2022/01/18: 
-------------------------------------------------
"""
import datetime
import json
import os
import pickle
import time
from collections import defaultdict

import torch
from ipdb import set_trace
from tqdm import tqdm

from src.models.biosyn import BioSyn
from utils.dataset_utils import DictionaryDataset
from utils.function_utils import get_config, get_logger, set_seed
from utils.predicate_utils import load_cache_dictionary, return_dictionary_url
from utils.preprocess_utils import TextPreprocess


def cache_or_load_dictionary(biosyn: BioSyn, model_name_or_path, dictionary_path):
    dictionary_name = os.path.splitext(os.path.basename(dictionary_path))[0]

    cached_dictionary_path = os.path.join(
        './tmp',
        f"cached_{model_name_or_path.split('/')[-1]}_{dictionary_name}.pk"
    )

    # If exist, load the cached dictionary
    if os.path.exists(cached_dictionary_path):
        with open(cached_dictionary_path, 'rb') as fin:
            cached_dictionary = pickle.load(fin)
        print("Loaded dictionary from cached file {}".format(cached_dictionary_path))

        dictionary, dict_sparse_embeds, dict_dense_embeds = (
            cached_dictionary['dictionary'],
            cached_dictionary['dict_sparse_embeds'],
            cached_dictionary['dict_dense_embeds'],
        )

    else:

        logger.info("开始对字典:{}进行初次初始化".format(dictionary_path))
        dictionary = DictionaryDataset(dictionary_path=dictionary_path).data
        dictionary_names = dictionary[:, 0]
        dict_sparse_embeds = biosyn.get_sparse_representation(mentions=dictionary_names, verbose=True)
        dict_dense_embeds = biosyn.get_dense_representation(mentions=dictionary_names, verbose=True)

        cached_dictionary = {
            'dictionary': dictionary,
            'dict_sparse_embeds': dict_sparse_embeds,
            'dict_dense_embeds': dict_dense_embeds
        }

        if not os.path.exists('./tmp'):
            os.mkdir('./tmp')

        with open(cached_dictionary_path, 'wb') as fin:
            pickle.dump(cached_dictionary, fin,protocol=4)
        print("Saving dictionary into cached file {}".format(cached_dictionary_path))

    return dictionary, dict_sparse_embeds, dict_dense_embeds



def abstract_biosyn_predicate(config, logger, model_name_or_path,file_path=None):
    """
    这是批量进行实体标准化
    但其实依然是对单个进行标准化
    :param config:
    :param logger:
    :param model_name_or_path:
    :return:
    """
    # load biosyn model
    device = torch.device('cuda') if config.use_gpu else torch.device('cpu')
    biosyn = BioSyn(config, device)
    # 加载已经训练的模型,dense_encoder,sparse_encoder
    logger.info("开始加载模型:{}".format(model_name_or_path))
    biosyn.load_model(model_name_or_path=model_name_or_path)
    logger.info('加载完成....')
    # 加载字典

    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    abstract_ids = list(all_data.keys())

    normalize_entities = {}

    for entity_type_ in ['Gene/Protein','cell_line','cell_type','DNA','RNA', 'Disease',  'Chemical/Drug', 'Species']:
        if entity_type_ == 'Disease':
            config.dictionary_path = './dictionary/mesh/disease_dictionary.txt'
        elif entity_type_ == 'Chemical/Drug':
            config.dictionary_path = './dictionary/mesh/chemical_and_drug_dictionary.txt'
        elif entity_type_ in['Gene/Protein','DNA','RNA'] :
            config.dictionary_path = './dictionary/gene/entre_gene_dictionary.txt'
        elif entity_type_ == 'cell_type':
            config.dictionary_path = './dictionary/cell/cell_type_dictionary.txt'
        elif entity_type_ == 'cell_line':
            config.dictionary_path = './dictionary/cell/cell_line_dictionary.txt'
        elif entity_type_ == 'Species':
            config.dictionary_path = './dictionary/Taxonomy/species_dictionary.txt'
        logger.info("开始对{}进行标准化".format(entity_type_))
        dictionary, dict_sparse_embeds, dict_dense_embeds = cache_or_load_dictionary(biosyn, model_name_or_path,
                                                                                     config.dictionary_path)

        for id in tqdm(abstract_ids,desc="正在对abstracts的{}进行标准化".format(entity_type_)):

            abstract_data = all_data[id]
            abstract_sentence_li = abstract_data['abstract_sentence_li']
            entities = abstract_data['entities']
            for idx,ent in enumerate(entities):
                ent_name = ent['entity_name']
                ent_type = ent['entity_type']
                if ent_type == entity_type_:
                    ent_id = ent['id']
                    ent_synonyms = biosyn_model_predicate(ent_name, biosyn, dictionary, dict_sparse_embeds, dict_dense_embeds)
                    all_data[id]['entities'][idx]['synonyms'] = ent_synonyms


def batch_biosyn_predicate_v2(config, logger, model_name_or_path):
    """
    这是批量进行实体标准化
    单个进行标准化
    :param config:
    :param logger:
    :param model_name_or_path:
    :return:
    """
    # load biosyn model
    device = torch.device('cuda') if config.use_gpu else torch.device('cpu')
    biosyn = BioSyn(config, device)
    # 加载已经训练的模型,dense_encoder,sparse_encoder
    logger.info("开始加载模型:{}".format(model_name_or_path))
    biosyn.load_model(model_name_or_path=model_name_or_path)
    logger.info('加载完成....')
    # 加载字典

    # 读取文件
    # with open("./dataset/1009abstracts/1009abstracts_entities_dict.json",'rb') as f:
    #     entities_dict = pickle.load(f)

    with open('./dataset/1009abstracts/1009abstracts_entities_dict.json', 'r', encoding='utf-8') as f:
        entities_dict = json.load(f)


    logger.info("开始加载disease信息.....")
    dise_dict, dise_sparse_embds, dise_dense_embds = load_cache_dictionary(config.disease_cache_path)
    logger.info("开始加载chem_drug词典信息.....")
    chem_drug_dict, chem_drug_sparse_embds, chem_drug_dense_embds = load_cache_dictionary(
        config.chemical_drug_dictionary_path)
    logger.info("开始加载gene词典信息.....")
    gene_dict, gene_sparse_embds, gene_dense_embds = load_cache_dictionary(
        config.gene_protein_cache_path)
    logger.info("开始加载cell_type词典信息.....")
    cell_type_dict, cell_type_sparse_embds, cell_type_dense_embds = load_cache_dictionary(
        config.cell_type_cache_path)
    logger.info("开始加载chem_line词典信息.....")
    cell_line_dict, cell_line_sparse_embds, cell_line_dense_embds = load_cache_dictionary(
        config.cell_line_cache_path)
    logger.info("开始加载species词典信息.....")
    species_dict, species_sparse_embds, species_dense_embds = load_cache_dictionary(config.species_cache_path)
    normalize_entities = {}
    new_entities = []

    for idx, ent_id in tqdm(enumerate(entities_dict),total=len(entities_dict)):
        ent = entities_dict[ent_id]
        counter = defaultdict(int)
        ent_name = ent['entity_name']
        ent_type = ent['entity_type']

        if ent_type == 'Disease':
            synonyms = biosyn_model_predicate(ent_name, biosyn, dise_dict, dise_sparse_embds,dise_dense_embds)
        elif ent_type == "Chemical/Drug":
            synonyms = biosyn_model_predicate(ent_name, biosyn, chem_drug_dict, chem_drug_sparse_embds, chem_drug_dense_embds)
        elif ent_type == 'Gene/Protein' or ent_type == 'DNA' or ent_type == 'RNA':
            synonyms = biosyn_model_predicate(ent_name, biosyn, gene_dict,gene_sparse_embds, gene_dense_embds)
        elif ent_type == 'cell_line':
            synonyms = biosyn_model_predicate(ent_name, biosyn, cell_line_dict,cell_line_sparse_embds, cell_line_dense_embds)
        elif ent_type == 'cell_type':
            synonyms = biosyn_model_predicate(ent_name, biosyn, cell_type_dict, cell_type_sparse_embds, cell_type_dense_embds)
        elif ent_type == 'Species':
            synonyms = biosyn_model_predicate(ent_name, biosyn, species_dict, species_sparse_embds,species_dense_embds)
        else:
            print(ent_type)
            raise NotImplementedError
        for syn in synonyms:
            counter[syn['id']] += 1

        sort_ent = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        if sort_ent[0][1] > 1:
            most_prob_id = sort_ent[0][0]
            most_norm_name = ''
            for s in synonyms:
                if s['id'] == most_prob_id:
                    most_norm_name = s['name']
                    break
        else:
            most_prob_id = synonyms[0]['id']
            most_norm_name = synonyms[0]['name']

        ent['norm_id'] = most_prob_id
        ent['url'] = return_dictionary_url(most_prob_id, ent_type)
        ent['norm_name'] = most_norm_name
        new_entities.append(ent)
        normalize_entities[ent_id] = ent

    with open("./dataset/abstract_results/normalize_entities_dict.pk",'wb') as f:
        pickle.dump(normalize_entities,f)



def single_model_abstract_batch_biosyn_predicate(config, logger, model_name_or_path):
    """
   单任务模型对某个类型实体进行批量标准化
    :param config:
    :param logger:
    :param model_name_or_path:
    :return:
    """
    # load biosyn model
    device = torch.device('cuda') if config.use_gpu else torch.device('cpu')
    biosyn = BioSyn(config, device)
    # 加载已经训练的模型,dense_encoder,sparse_encoder
    logger.info("开始加载模型:{}".format(model_name_or_path))
    biosyn.load_model(model_name_or_path=model_name_or_path)
    logger.info('加载完成....')
    # 加载字典

    # 读取文件
    # with open("./dataset/1009abstracts/1009abstracts_entities_dict.json",'rb') as f:
    #     entities_dict = pickle.load(f)

    with open('./dataset/1009abstracts/1009abstracts_entities_dict.json', 'r', encoding='utf-8') as f:
        entities_dict = json.load(f)


    logger.info("开始加载disease信息.....")
    dise_dict, dise_sparse_embds, dise_dense_embds = load_cache_dictionary(config.disease_cache_path)
    # logger.info("开始加载chem_drug词典信息.....")
    # chem_drug_dict, chem_drug_sparse_embds, chem_drug_dense_embds = load_cache_dictionary(
    #     config.chemical_drug_dictionary_path)
    # logger.info("开始加载gene词典信息.....")
    # gene_dict, gene_sparse_embds, gene_dense_embds = load_cache_dictionary(
    #     config.gene_protein_cache_path)
    # logger.info("开始加载cell_type词典信息.....")
    # cell_type_dict, cell_type_sparse_embds, cell_type_dense_embds = load_cache_dictionary(
    #     config.cell_type_cache_path)
    # logger.info("开始加载chem_line词典信息.....")
    # cell_line_dict, cell_line_sparse_embds, cell_line_dense_embds = load_cache_dictionary(
    #     config.cell_line_cache_path)
    # logger.info("开始加载species词典信息.....")
    # species_dict, species_sparse_embds, species_dense_embds = load_cache_dictionary(config.species_cache_path)
    normalize_entities = {}
    new_entities = []
    set_trace()
    for idx, ent_id in tqdm(enumerate(entities_dict),total=len(entities_dict)):
        ent = entities_dict[ent_id]
        counter = defaultdict(int)
        ent_name = ent['entity_name']
        ent_type = ent['entity_type']

        if ent_type == 'Disease':
            synonyms = biosyn_model_predicate(ent_name, biosyn, dise_dict, dise_sparse_embds,dise_dense_embds)
        # elif ent_type == "Chemical/Drug":
        #     synonyms = biosyn_model_predicate(ent_name, biosyn, chem_drug_dict, chem_drug_sparse_embds, chem_drug_dense_embds)
        # elif ent_type == 'Gene/Protein' or ent_type == 'DNA' or ent_type == 'RNA':
        #     synonyms = biosyn_model_predicate(ent_name, biosyn, gene_dict,gene_sparse_embds, gene_dense_embds)
        # elif ent_type == 'cell_line':
        #     synonyms = biosyn_model_predicate(ent_name, biosyn, cell_line_dict,cell_line_sparse_embds, cell_line_dense_embds)
        # elif ent_type == 'cell_type':
        #     synonyms = biosyn_model_predicate(ent_name, biosyn, cell_type_dict, cell_type_sparse_embds, cell_type_dense_embds)
        # elif ent_type == 'Species':
        #     synonyms = biosyn_model_predicate(ent_name, biosyn, species_dict, species_sparse_embds,species_dense_embds)
        else:
            print(ent_type)
            continue
        for syn in synonyms:
            counter[syn['id']] += 1

        sort_ent = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        if sort_ent[0][1] > 1:
            most_prob_id = sort_ent[0][0]
            most_norm_name = ''
            for s in synonyms:
                if s['id'] == most_prob_id:
                    most_norm_name = s['name']
                    break
        else:
            most_prob_id = synonyms[0]['id']
            most_norm_name = synonyms[0]['name']

        ent['norm_id'] = most_prob_id
        ent['url'] = return_dictionary_url(most_prob_id, ent_type)
        ent['norm_name'] = most_norm_name
        new_entities.append(ent)
        normalize_entities[ent_id] = ent

    with open("./dataset/abstract_results/normalize_entities_dict.pk",'wb') as f:
        pickle.dump(normalize_entities,f)


def biosyn_model_predicate(input_mention,biosyn,dictionary, dict_sparse_embeds, dict_dense_embeds):
    # preprocess 输入的 mention
    mention = TextPreprocess().run(input_mention)

    # embed mention
    mention_sparse_embeds = biosyn.get_sparse_representation(mentions=[mention])
    mention_dense_embeds = biosyn.get_dense_representation(mentions=[mention])

    output = {
        'mention': input_mention,
    }

    if config.verbose:
        output = {
            'mention': input_mention,
            'mention_sparse_embeds': mention_sparse_embeds.squeeze(0),
            'mention_dense_embeds': mention_dense_embeds.squeeze(0)
        }

    if config.dictionary_path == None:
        logger.info('insert the dictionary path')
        return



    # 计算得到sparse score和dense score
    sparse_score_matrix = biosyn.get_score_matrix(
        query_embeds=mention_sparse_embeds,
        dict_embeds=dict_sparse_embeds
    )
    dense_score_matrix = biosyn.get_score_matrix(
        query_embeds=mention_dense_embeds,
        dict_embeds=dict_dense_embeds
    )

    sparse_weight = biosyn.get_sparse_weight().item()

    hybrid_score_matrix = sparse_weight * sparse_score_matrix + dense_score_matrix
    # 获得topk个最相似的单词
    hybrid_candidate_idxs = biosyn.retrieve_candidate(
        score_matrix=hybrid_score_matrix,
        topk=5
    )

    # 只能从字典中获得具体的名称
    predictions = dictionary[hybrid_candidate_idxs].squeeze(0)
    output['predictions'] = []

    for prediction in predictions:
        predicted_name = prediction[0]
        predicted_id = prediction[1]
        output['predictions'].append({
            'name': predicted_name,
            'id': predicted_id
        })

    return output['predictions']

def biosyn_predicate(config, logger, input_mention, model_name_or_path):
    """
    这是使用biosyn进行标准化
    对单个实体名称进行标准化
    :param config:
    :param logger:
    :param input_mention:
    :param model_name_or_path:
    :return:
    """
    # load biosyn model
    device = torch.device('cuda') if config.use_gpu else torch.device('cpu')
    biosyn = BioSyn(config, device)
    # 加载已经训练的模型,dense_encoder,sparse_encoder
    biosyn.load_model(model_name_or_path=model_name_or_path)
    # preprocess 输入的 mention
    start_time = time.time()
    mention = TextPreprocess().run(input_mention)

    # embed mention
    mention_sparse_embeds = biosyn.get_sparse_representation(mentions=[mention])
    mention_dense_embeds = biosyn.get_dense_representation(mentions=[mention])

    output = {
        'mention': input_mention,
    }

    if config.verbose:
        output = {
            'mention': input_mention,
            'mention_sparse_embeds': mention_sparse_embeds.squeeze(0),
            'mention_dense_embeds': mention_dense_embeds.squeeze(0)
        }

    if config.dictionary_path == None:
        logger.info('insert the dictionary path')
        return

    # cache or load dictionary
    config.dictionary_path = './dictionary/mesh/disease_dictionary.txt'

    dictionary, dict_sparse_embeds, dict_dense_embeds = cache_or_load_dictionary(biosyn, model_name_or_path,
                                                                                 config.dictionary_path)

    # 计算得到sparse score和dense score
    sparse_score_matrix = biosyn.get_score_matrix(
        query_embeds=mention_sparse_embeds,
        dict_embeds=dict_sparse_embeds
    )
    dense_score_matrix = biosyn.get_score_matrix(
        query_embeds=mention_dense_embeds,
        dict_embeds=dict_dense_embeds
    )

    sparse_weight = biosyn.get_sparse_weight().item()

    hybrid_score_matrix = sparse_weight * sparse_score_matrix + dense_score_matrix
    hybrid_candidate_idxs = biosyn.retrieve_candidate(
        score_matrix=hybrid_score_matrix,
        topk=5
    )

    # get predictions from dictionary
    predictions = dictionary[hybrid_candidate_idxs].squeeze(0)
    output['predictions'] = []

    for prediction in predictions:
        predicted_name = prediction[0]
        predicted_id = prediction[1]
        output['predictions'].append({
            'name': predicted_name,
            'id': predicted_id
        })

    print(output['predictions'])
    print("花费时间",time.time()-start_time)
    set_trace()

if __name__ == '__main__':
    config = get_config()
    logger = get_logger(config)

    # 设置时间
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff
    # 设置随机种子
    set_seed(config.seed)

    # 使用biosyn进行预测
    biosyn_ckpt_path = '../embedding/SapBERT-from-PubMedBERT-fulltext'
    file_path = './dataset/multi_task_rtx_2080_8types.json'

    single_model_abstract_batch_biosyn_predicate(config,logger, biosyn_ckpt_path)

    #biosyn_predicate(config, logger, 'Gastric Cancer', model_name_or_path=biosyn_ckpt_path)

    # 使用sapbert进行预测
    #spabert_ckpt_path = '/opt/data/private/luyuwei/code/bioner/embedding/SapBERT-from-PubMedBERT-fulltext'
    #biosyn_predicate(config, logger, 'cancer', model_name_or_path=spabert_ckpt_path)
    #

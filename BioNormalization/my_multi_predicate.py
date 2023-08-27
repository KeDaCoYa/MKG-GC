# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :   这是五任务的实体标准化模型:Disease,Cell type,cell line ,chemical ,gene
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
from src.models.my_multi_biosyn import MyMultiBioSynModel
from utils.dataset_utils import DictionaryDataset
from utils.function_utils import get_config, get_logger, set_seed
from utils.predicate_utils import load_cache_dictionary, return_dictionary_url
from utils.preprocess_utils import TextPreprocess


def cache_or_load_dictionary(biosyn, dictionary_path,type_=0):
    dictionary_name = os.path.splitext(os.path.basename(dictionary_path))[0]

    cached_dictionary_path = os.path.join(
        './tmp',
        f"cached_{dictionary_name}.pk"
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

        if type_ == 0:
            dict_sparse_embeds = biosyn.get_disease_sparse_representation(mentions=dictionary_names, verbose=True)
        elif type_ == 1:
            dict_sparse_embeds = biosyn.get_chemical_drug_sparse_representation(mentions=dictionary_names, verbose=True)
        elif type_ == 2:
            dict_sparse_embeds = biosyn.get_gene_protein_sparse_representation(mentions=dictionary_names, verbose=True)
        elif type_ == 3:
            dict_sparse_embeds = biosyn.get_cell_type_sparse_representation(mentions=dictionary_names, verbose=True)
        elif type_ == 4:
            dict_sparse_embeds = biosyn.get_cell_line_sparse_representation(mentions=dictionary_names, verbose=True)
        elif type_ == 5:
            dict_sparse_embeds = biosyn.get_sparse_representation(mentions=dictionary_names, verbose=True)
        else:
            raise ValueError

        dict_dense_embeds = biosyn.get_dense_representation(mentions=dictionary_names, verbose=True,type_=type_)

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



def biosyn_model_batch_predicate(input_mentions, biosyn, dictionary, dict_sparse_embeds, dict_dense_embeds,type_=0):
    """
    这个是批量进行预测，是single的升级版本
    :param input_mentions:
    :param biosyn:
    :param dictionary:
    :param dict_sparse_embeds:
    :param dict_dense_embeds:
    :param type_:
    :return:
    """
    # preprocess 输入的 mention
    mentions = []
    for ment in input_mentions:
        mentions.append(TextPreprocess().run(ment))

    if type_ == 0:
        mentions_sparse_embeds = biosyn.get_disease_sparse_representation(mentions=mentions)
        sparse_weight = biosyn.disease_sparse_weight.item()
    elif type_ == 1:
        mentions_sparse_embeds = biosyn.get_chemical_drug_sparse_representation(mentions=mentions)
        sparse_weight = biosyn.chemical_drug_sparse_weight.item()
    elif type_ == 2:
        mentions_sparse_embeds = biosyn.get_gene_protein_sparse_representation(mentions=mentions)
        sparse_weight = biosyn.gene_sparse_weight.item()
    elif type_ == 3:
        mentions_sparse_embeds = biosyn.get_cell_type_sparse_representation(mentions=mentions)
        sparse_weight = biosyn.cell_type_sparse_weight.item()
    elif type_ == 4:
        mentions_sparse_embeds = biosyn.get_cell_line_sparse_representation(mentions=mentions)
        sparse_weight = biosyn.cell_line_sparse_weight.item()
    elif type_ == 5:
        mentions_sparse_embeds = biosyn.get_sparse_representation(mentions=mentions)
        sparse_weight = biosyn.get_sparse_weight().item()
    else:
        raise ValueError
    mentions_dense_embeds = biosyn.get_dense_representation(mentions=mentions, type_=type_)


    # 计算得到sparse score和dense score
    sparse_score_matrix = biosyn.get_score_matrix(
        query_embeds=mentions_sparse_embeds,
        dict_embeds=dict_sparse_embeds
    )
    dense_score_matrix = biosyn.get_score_matrix(
        query_embeds=mentions_dense_embeds,
        dict_embeds=dict_dense_embeds
    )



    hybrid_score_matrix = sparse_weight * sparse_score_matrix + dense_score_matrix
    # 获得topk个最相似的单词
    hybrid_candidate_idxs = biosyn.retrieve_candidate(
        score_matrix=hybrid_score_matrix,
        topk=5
    )

    # 只能从字典中获得具体的名称

    entity_predictions = dictionary[hybrid_candidate_idxs]
    output = []
    for predictions in entity_predictions:
        tmp = []
        for prediction in predictions:
            predicted_name = prediction[0]
            predicted_id = prediction[1]
            tmp.append({
                'name': predicted_name,
                'id': predicted_id
            })
        output.append(tmp)

    return output

def normalize_by_single(config, logger, model_name_or_path,file_path):
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
    biosyn = MyMultiBioSynModel(config, device)
    species_model = BioSyn(config,device,initial_sparse_weight=1.)

    species_model.load_model(config.bert_dir)

    # 加载已经训练的模型,dense_encoder,sparse_encoder
    logger.info("开始加载模型:{}".format(model_name_or_path))
    # all_sparse_weights = (13.532,14.085,17.761,16.979,18.67)
    # biosyn.load_model(model_name_or_path=model_name_or_path,all_sparse_weights=all_sparse_weights)
    biosyn.load_model(model_name_or_path=model_name_or_path)

    logger.info('加载模型完成....')


    # 读取文件
    with open(file_path,'r',encoding='utf-8') as f:
        entities_dict = json.load(f)

    logger.info("开始加载species词典信息.....")
    species_dict, species_sparse_embds, species_dense_embds = cache_or_load_dictionary(species_model,config.species_dictionary_path,type_=5)
    logger.info("开始加载disease信息.....")
    dise_dict, dise_sparse_embds, dise_dense_embds = cache_or_load_dictionary(biosyn,config.disease_dictionary_path,type_=0)
    logger.info("开始加载chem_drug词典信息.....")
    chem_drug_dict, chem_drug_sparse_embds, chem_drug_dense_embds = cache_or_load_dictionary(biosyn,config.chemical_drug_dictionary_path,type_=1)
    logger.info("开始加载gene词典信息.....")
    gene_dict, gene_sparse_embds, gene_dense_embds = cache_or_load_dictionary(biosyn,config.gene_protein_dictionary_path,type_=2)
    logger.info("开始加载cell_type词典信息.....")
    cell_type_dict, cell_type_sparse_embds, cell_type_dense_embds = cache_or_load_dictionary(biosyn,config.cell_type_dictionary_path,type_=3)
    logger.info("开始加载cell_line词典信息.....")
    cell_line_dict, cell_line_sparse_embds, cell_line_dense_embds = cache_or_load_dictionary(biosyn,config.cell_line_dictionary_path,type_=4)


    normalize_entities = {}
    new_entities = []

    for idx, ent_id in tqdm(enumerate(entities_dict),total=len(entities_dict)):
        ent = entities_dict[ent_id]

        counter = defaultdict(int)

        ent_name = ent['entity_name']
        ent_type = ent['entity_type']

        if ent_type == 'Disease':
            synonyms = biosyn_model_single_predicate(ent_name, biosyn, dise_dict, dise_sparse_embds,dise_dense_embds,type_=0)

        elif ent_type == "Chemical/Drug":
            synonyms = biosyn_model_single_predicate(ent_name, biosyn, chem_drug_dict, chem_drug_sparse_embds, chem_drug_dense_embds,type_=1)

        elif ent_type == 'Gene/Protein' or ent_type == 'DNA' or ent_type == 'RNA':
            synonyms = biosyn_model_single_predicate(ent_name, biosyn, gene_dict,gene_sparse_embds, gene_dense_embds,type_=2)

        elif ent_type == 'cell_line':
            synonyms = biosyn_model_single_predicate(ent_name, biosyn, cell_line_dict,cell_line_sparse_embds, cell_line_dense_embds,type_=3)

        elif ent_type == 'cell_type':
            synonyms = biosyn_model_single_predicate(ent_name, biosyn, cell_type_dict, cell_type_sparse_embds, cell_type_dense_embds,type_=4)

        elif ent_type == 'Species':

            synonyms = biosyn_model_single_predicate(ent_name, species_model, species_dict, species_sparse_embds,species_dense_embds,type_=5)
        else:

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

    if config.dataset_name == '1009abstracts':
        output_path = './dataset/1009abstracts/1009abstracts_normalize_entities_dict.json'
    elif config.dataset_name == '3400abstracts':
        output_path = './dataset/3400abstracts/3400abstracts_normalize_entities_dict.json'
    else:
        raise ValueError
    with open(output_path,'w') as f:
        json.dump(normalize_entities,f)


def biosyn_model_single_predicate(input_mention,biosyn:MyMultiBioSynModel,dictionary, dict_sparse_embeds, dict_dense_embeds,type_=0):
    """
    这个只能对一个input mention进行predicate
    :param input_mention:
    :param biosyn:
    :param dictionary:
    :param dict_sparse_embeds:
    :param dict_dense_embeds:
    :param type_:
    :return:
    """
    # preprocess 输入的 mention
    mention = TextPreprocess().run(input_mention)

    # embed mention

    if type_ == 0:
        mention_sparse_embeds = biosyn.get_disease_sparse_representation(mentions=[mention])
        sparse_weight = biosyn.disease_sparse_weight.item()
    elif type_ == 1:
        mention_sparse_embeds = biosyn.get_chemical_drug_sparse_representation(mentions=[mention])
        sparse_weight = biosyn.chemical_drug_sparse_weight.item()
    elif type_ == 2:
        mention_sparse_embeds = biosyn.get_gene_protein_sparse_representation(mentions=[mention])
        sparse_weight = biosyn.gene_sparse_weight.item()
    elif type_ == 3:
        mention_sparse_embeds = biosyn.get_cell_type_sparse_representation(mentions=[mention])
        sparse_weight = biosyn.cell_type_sparse_weight.item()
    elif type_ == 4:
        mention_sparse_embeds = biosyn.get_cell_line_sparse_representation(mentions=[mention])
        sparse_weight = biosyn.cell_line_sparse_weight.item()
    elif type_ == 5:

        mention_sparse_embeds = biosyn.get_sparse_representation(mentions=[mention])
        sparse_weight = biosyn.get_sparse_weight().item()
    else:
        raise ValueError
    mention_dense_embeds = biosyn.get_dense_representation(mentions=[mention],type_=type_)


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
    单个进行标准化
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

    dictionary, dict_sparse_embeds, dict_dense_embeds = cache_or_load_dictionary(biosyn, model_name_or_path,config.dictionary_path)

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


def normalize_by_batch(config, logger, model_name_or_path, file_path):
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
    biosyn = MyMultiBioSynModel(config, device)

    species_model = BioSyn(config, device, initial_sparse_weight=1.)
    species_model.load_model(config.bert_dir)

    # 加载已经训练的模型,dense_encoder,sparse_encoder
    logger.info("开始加载模型:{}".format(model_name_or_path))
    # all_sparse_weights = (13.532,14.085,17.761,16.979,18.67)
    # biosyn.load_model(model_name_or_path=model_name_or_path,all_sparse_weights=all_sparse_weights)
    biosyn.load_model(model_name_or_path=model_name_or_path)

    logger.info('加载模型完成....')

    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        entities_dict = json.load(f)

    logger.info("开始加载species词典信息.....")
    species_dict, species_sparse_embds, species_dense_embds = cache_or_load_dictionary(species_model,
                                                                                       config.species_dictionary_path,
                                                                                       type_=5)

    logger.info("开始加载disease词典信息.....")
    dise_dict, dise_sparse_embds, dise_dense_embds = cache_or_load_dictionary(biosyn, config.disease_dictionary_path,
                                                                              type_=0)
    logger.info("开始加载chem_drug词典信息.....")
    chem_drug_dict, chem_drug_sparse_embds, chem_drug_dense_embds = cache_or_load_dictionary(biosyn,
                                                                                             config.chemical_drug_dictionary_path,
                                                                                             type_=1)
    logger.info("开始加载gene——protein词典信息.....")
    gene_dict, gene_sparse_embds, gene_dense_embds = cache_or_load_dictionary(biosyn,
                                                                              config.gene_protein_dictionary_path,
                                                                              type_=2)
    logger.info("开始加载cell_type词典信息.....")
    cell_type_dict, cell_type_sparse_embds, cell_type_dense_embds = cache_or_load_dictionary(biosyn,config.cell_type_dictionary_path,type_=3)
    logger.info("开始加载cell_line词典信息.....")
    cell_line_dict, cell_line_sparse_embds, cell_line_dense_embds = cache_or_load_dictionary(biosyn,config.cell_line_dictionary_path,type_=4)

    normalize_entities = {}
    new_entities = []

    # 这是存储每个ent的ent type,方便之后的标准化
    ent_type2id={
        'Disease':0,
        'Chemical/Drug':1,
        'Gene/Protein':2,
        'DNA':6,
        'RNA':7,
        'cell_line':3,
        'cell_type':4,
        'Species':5,
    }

    ent_type_dict_li = defaultdict(list)

    for idx, ent_id in tqdm(enumerate(entities_dict)):
        ent = entities_dict[ent_id]
        ent_type = ent['entity_type']
        ent_type_dict_li[ent_type].append(ent)


    start_time = time.time()
    for ent_type in ent_type2id:
        entities_li = ent_type_dict_li[ent_type]

        for i in tqdm(range(0, len(entities_li), config.batch_size),desc="正在对{}进行预测".format(ent_type)):
            entity_names = [x['entity_name'] for x in
                            entities_li[i:i+config.batch_size]]

            if ent_type == 'Disease':
                synonyms = biosyn_model_batch_predicate(entity_names, biosyn, dise_dict,
                                                             dise_sparse_embds,
                                                             dise_dense_embds, type_=0)

            elif ent_type == 'Gene/Protein' or ent_type == 'DNA' or ent_type == 'RNA':

                synonyms = biosyn_model_batch_predicate(entity_names, biosyn, gene_dict,
                                                             gene_sparse_embds, gene_dense_embds, type_=2)
            elif ent_type == 'Chemical/Drug':
                synonyms = biosyn_model_batch_predicate(entity_names, biosyn, chem_drug_dict,
                                                             chem_drug_sparse_embds, chem_drug_dense_embds,
                                                             type_=1)
            elif ent_type == 'cell_line':
                synonyms = biosyn_model_batch_predicate(entity_names, biosyn, cell_line_dict,
                                                       cell_line_sparse_embds, cell_line_dense_embds,type_=4)
            elif ent_type == 'cell_type':

                synonyms = biosyn_model_batch_predicate(entity_names, biosyn, cell_type_dict,
                                                       cell_type_sparse_embds, cell_type_dense_embds,type_=3)

            elif ent_type == 'Species':

                synonyms = biosyn_model_batch_predicate(entity_names, species_model, species_dict,
                                                       species_sparse_embds, species_dense_embds,type_=5)
            else:
                raise ValueError('---')
            for idx, ent in enumerate(entities_li[i:i+config.batch_size]):
                counter = defaultdict(int)
                for syn in synonyms[idx]:
                    counter[syn['id']] += 1

                sort_ent = sorted(counter.items(), key=lambda x: x[1], reverse=True)
                if sort_ent[0][1] > 1:
                    most_prob_id = sort_ent[0][0]
                    most_norm_name = ''
                    for s in synonyms[idx]:
                        if s['id'] == most_prob_id:
                            most_norm_name = s['name']
                            break
                else:
                    most_prob_id = synonyms[idx][0]['id']
                    most_norm_name = synonyms[idx][0]['name']

                ent['norm_id'] = most_prob_id
                ent['norm_name'] = most_norm_name
                new_entities.append(ent)

    print("花费时间", time.time() - start_time)



    normalize_entities = {}

    for ent in new_entities:
        ent_id = ent['id']

        normalize_entities[ent_id] = ent



    if config.dataset_name == '1009abstracts':
        output_path = './dataset/1009abstracts/1009abstracts_normalize_entities_dict.json'
    elif config.dataset_name == '3400abstracts':
        output_path = './dataset/3400abstracts/3400abstracts_normalize_entities_dict.json'
    else:
        raise ValueError
    logger.info("结果保存到:{}".format(output_path))
    with open(output_path, 'w') as f:
        json.dump(normalize_entities, f)


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
    biosyn_ckpt_path = '/root/code/bioner/embedding/SapBERT-from-PubMedBERT-fulltext'
    ckpt_path = '/opt/data/private/luyuwei/code/bioner/BioNormalization/outputs/save_models/2022-06-20/Five_task_biosyn_bs24_free8_lr0.0001_1_encoder_type_bert_maxlen25/biosyn/multi_task_five_dataset/best_model'
    if config.dataset_name == '1009abstracts':
        file_path = './dataset/1009abstracts/1009abstracts_entities_dict.json'
    elif config.dataset_name == '3400abstracts':
        file_path = './dataset/3400abstracts/3400abstracts_entities_dict.json'
    else:
        raise ValueError
    # normalize_by_single(config,logger, ckpt_path,file_path)
    start = time.time()
    normalize_by_batch(config,logger, ckpt_path,file_path)
    logger.info("实体标准化花费时间:{}".format(time.time()-start))


    #biosyn_predicate(config, logger, 'Gastric Cancer', model_name_or_path=biosyn_ckpt_path)

    # 使用sapbert进行预测
    # spabert_ckpt_path = '/root/code/bioner/knowledgenormalization/BioNormalization/outputs/save_models/checkpoint_1'
    # sapbert_predicate(config, logger, 'cancer', ckpt_path=spabert_ckpt_path)
    #

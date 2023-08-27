# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  一句话一句话的预测,也就是用于之后的abstract预测
   Author :        kedaxia
   date：          2021/11/20
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/11/20: 
-------------------------------------------------
"""
from transformers import BertModel, BertTokenizer
from src.bert_models import BertGlobalPointer
import json
import torch
import numpy as np
from tqdm import  tqdm
import logging

from ipdb import set_trace
from config import BertConfig,NormalConfig
from src.normal_models import BiLSTM_GlobalPointer, BiLSTM_CRF,Att_BiLSTM_CRF
from utils.data_process_utils import read_data, build_vocab

import time

from utils.function_utils import BIO_decode_json

logger = logging.getLogger()

def bert_globalpointer_predicate(config:BertConfig,best_model_ckpt_path='/root/code/biobert_nlp/ner/outputs/bert_globalpointer/jnlpba/best_model/model.pt'):
    '''
    这个专门用于评测bert+globalpointer_predicate的结果
    从读取测试集，然后对测试集的每一句话进行预测
    :param best_model_ckpt_path:
    :param task_type:
    :param word2id:
    :return:
    '''
    # 这个路径是训练集BC5CDR-disease的模型

    globalpointer_label2id = config.globalpointer_label2id
    globalpointer_id2label = config.globalpointer_id2label

    tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
    # 这是预训练的BERT模型

    model = BertGlobalPointer(config, config.num_gp_class, 64)
    device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')
    model.to(device)
    logger.info('-----加载与预训练的模型-----')
    model.load_state_dict(torch.load(best_model_ckpt_path, map_location='cuda:{}'.format(config.gpu_id)))
    all_ = []
    model.eval()
    test_data,test_labels = read_data('./original_dataset/jnlpba/dev.txt')

    with torch.no_grad():

        for idx,text in tqdm(enumerate(test_data)):
            '''
                这个text按照word进行切分
            '''
            entities = []

            encoder_txt = tokenizer.encode_plus(text, max_length=config.max_len,truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).cuda()
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).cuda()
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).cuda()
            scores = model(input_ids, attention_mask, token_type_ids)[0].data.cpu().numpy()
            # 这里在decode的时候将开头和末尾给去掉，因为这是[CLS],[SEP]
            scores[:, [0, -1]] -= np.inf
            scores[:, :, [0, -1]] -= np.inf
            # 这里相当于直接将预测结果进行解码，
            for l, start, end in zip(*np.where(scores > 0)):
                entities.append({"start_idx":start, "end_idx":end, "type": globalpointer_id2label[l],'entity':" ".join(text[start:end+1])})

            res = BIO_decode_json(text,test_labels[idx])


            all_.append({"text": text, "entities": entities})
    json.dump(
        all_,
        open('./outputs/BC5CDR-disease_test.json', 'w'),
        indent=4,
        ensure_ascii=False
    )

def span_predicate(config:NormalConfig,best_model_ckpt_path=' ./output/bilstm_crf/BC5CDR-disease/11/model.pt',**kwargs):
    '''
    一般只有bert采用这个span，使用bilstm，这个span没啥意思
    :param config:
    :param best_model_ckpt_path:
    :param kwargs:
    :return:
    '''


def normal_model_crf_predicate(config:NormalConfig,best_model_ckpt_path='/root/code/biobert_nlp/ner/output/bilstm_crf/BC5CDR-disease/8/model.pt',**kwargs):
    '''
    这个是最后是crf的进行decode
    :param config:
    :param best_model_ckpt_path:
    :return:
    '''
    crf_label2id = config.crf_label2id
    crf_id2label = config.crf_id2label
    if config.task_type == 'bilstm_crf':
        model = BiLSTM_CRF(config)
    elif config.task_type == 'att_bilstm_crf':
        model = Att_BiLSTM_CRF(config)

    device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')
    model.to(device)
    print('-----从{}加载预训练的模型-----'.format(best_model_ckpt_path))
    model.load_state_dict(torch.load(best_model_ckpt_path, map_location='cuda:{}'.format(config.gpu_id)))
    all_ = []
    model.eval()
    test_data, test_labels = read_data(config.test_file_path)
    with torch.no_grad():

        for text in tqdm(test_data[:100]):

            if config.task_type in ['bilstm_crf','att_bilstm_crf']:
                token_ids = [[word2id.get(word,word2id.get('unk')) for word in text]]
                token_ids = torch.tensor(token_ids).to(device)
                crf_decode_token = model(token_ids,None)[0][0]

            else: #bert_crf类型
                pass

            pass


            entities.append({
                'start_idx': str(start),
                'end_idx': str(end),
                'type': globalpointer_id2label[l],
                'entity': " ".join(text[start:end + 1])
            })
            all_.append({"text": text, "entities": entities})
    json.dump(
        all_,
        open('./predicate/crf_BC5CDR-disease_test.json', 'w',encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )


def normal_globalpointer_predicate(config:NormalConfig,word2id,best_model_ckpt_path):
    '''
    这个是对BiLSTM_globalpointer或者att_bilstm_globalpointer的评估
    :param config:
    :param best_model_ckpt_path:
    :return:
    '''
    globalpointer_label2id = config.globalpointer_label2id
    globalpointer_id2label = config.globalpointer_id2label
    if config.task_type == 'bilstm_globalpointer':
        model = BiLSTM_GlobalPointer(config)
    elif config.task_type == 'att_bilstm_globalpointer':
        pass
    device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')
    model.to(device)
    logger.info('-----加载与预训练的模型-----')
    model.load_state_dict(torch.load(best_model_ckpt_path, map_location='cuda:{}'.format(config.gpu_id)))
    all_ = []
    model.eval()
    test_data, test_labels = read_data(config.test_file_path)
    with torch.no_grad():

        for text in tqdm(test_data[:100]):



            token_ids = [[word2id.get(word,word2id.get('unk')) for word in text]]
            token_ids = torch.tensor(token_ids).to(device)

            scores  = model(token_ids)[0].cpu().numpy()
            # 这里在decode的时候将开头和末尾给去掉，因为这是[CLS],[SEP]
            scores[:, [0, -1]] -= np.inf
            scores[:, :, [0, -1]] -= np.inf
            # 这里相当于直接将预测结果进行解码，
            entities = []
            for l, start, end in zip(*np.where(scores > 0)):

                entities.append({
                    'start_idx':str(start),
                    'end_idx':str(end),
                    'type':globalpointer_id2label[l],
                    'entity':" ".join(text[start:end+1])
                })


            all_.append({"text": text, "entities": entities})
    json.dump(
        all_,
        open('./predicate/BC5CDR-disease_test.json', 'w',encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )


if __name__ == '__main__':

    config = BertConfig(ner_type='jnlpba', task_type='globalpointer')
    bert_globalpointer_predicate(config)
    config = NormalConfig(ner_type='BC5CDR-disease', task_type='bilstm_crf')
    word2id = build_vocab(None, config.vocab_path)
    id2word = {j: i for i, j in word2id.items()}
    config.vocab_size = len(word2id)
    #normal_globalpointer_predicate(config,word2id=word2id,best_model_ckpt_path='/root/code/biobert_nlp/ner/output/bilstm_globalpointer/BC5CDR-disease/14/model.pt')
    #crf_predicate(config,word2id=word2id)




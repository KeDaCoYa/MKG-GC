# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2022/01/18
   Copyright:      (c) kedaxia 2022
-------------------------------------------------
   Change Activity: 这个是针对我自己构建的数据集进行的训练，共五个任务
                   2022/01/18: 
-------------------------------------------------
"""
import logging
import os
import pickle

from ipdb import set_trace
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from huggingface_hub import hf_hub_url, cached_download

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer, default_data_collator, BertLayer

from config import MyBertConfig
from src.data_loader import NamesDataset
from src.models.gau import GAU
from utils.train_utils import save_model, load_model_and_parallel

logger = logging.getLogger('main.sparse_encoder')


class MultiEncoder(nn.Module):
    def __init__(self, config:MyBertConfig):
        super().__init__()
        self.base_bert_encoder = AutoModel.from_pretrained(config.bert_dir)


        self.dropout = nn.Dropout(0.2)
        # chem encoder
        if config.encoder_type == 'bert':
            encoder_class = BertLayer
        elif config.encoder_type == 'gau':
            encoder_class = GAU
        else:
            raise ValueError

        self.disease_encoder = nn.ModuleList([encoder_class(config) for _ in range(config.task_encoder_nums)])
        self.drug_chemical_encoder = nn.ModuleList([encoder_class(config) for _ in range(config.task_encoder_nums)])
        self.gene_protein_encoder = nn.ModuleList([encoder_class(config) for _ in range(config.task_encoder_nums)])
        self.cell_type_encoder = nn.ModuleList([encoder_class(config) for _ in range(config.task_encoder_nums)])
        self.cell_line_encoder = nn.ModuleList([encoder_class(config) for _ in range(config.task_encoder_nums)])
        # self.species_encoder = nn.ModuleList([encoder_class(config) for _ in range(5)])
        # print(self.disease_encoder)
        # print(self.drug_chemical_encoder)
        # print(self.gene_protein_encoder)
        # print(self.cell_type_encoder)
        # print(self.cell_line_encoder)
        self.bert_model = AutoModel.from_pretrained(config.bert_dir)
        if config.freeze_bert:
            logger.info("对{}进行冻结".format(config.freeze_layers))
            self.freeze_parameter(config.freeze_layers)

    def freeze_parameter(self, freeze_layers):
        '''
        对指定的layers进行冻结参数
        :param layers: 格式为['layer.10','layer.11','bert.pooler','out.']
        :return:
        '''

        for name, param in self.bert_model.named_parameters():

            for ele in freeze_layers:
                if ele in name:
                    param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, type_=0):
        """

        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :param type_:
            0:disease
            1:chemical-drug
            2:cell_type
            2:cell_line
            2:cell_line
        :return:
        """
        base_output = \
            self.bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]

        if type_ == 0:
            for i in range(len(self.disease_encoder)):
                base_output = self.disease_encoder[i](base_output)[0]

        elif type_ == 1:
            for i in range(len(self.drug_chemical_encoder)):
                base_output = self.drug_chemical_encoder[i](base_output)[0]

        elif type_ == 2:
            for i in range(len(self.gene_protein_encoder)):
                base_output = self.gene_protein_encoder[i](base_output)[0]
        elif type_ == 3:
            for i in range(len(self.cell_type_encoder)):
                base_output = self.cell_type_encoder[i](base_output)[0]
        elif type_ == 4:
            for i in range(len(self.cell_line_encoder)):
                base_output = self.cell_line_encoder[i](base_output)[0]
        elif type_ == 5:
            for i in range(len(self.species_encoder)):
                base_output = self.species_encoder[i](base_output)[0]
        return (base_output,)


class MyMultiBioSynModel:
    def __init__(self, config, device, initial_sparse_weight=None):
        """
        这个模型就是统筹兼顾sparse encoder和dense encoder
        这个也是关键模型
        :param config:
        """

        self.config = config
        self.device = device
        self.max_len = config.max_len
        self.dense_encoder = None
        self.tokenizer = None

        self.sparse_encoder = None

        self.disease_sparse_weight = None
        self.chemical_drug_sparse_weight = None
        self.gene_sparse_weight = None
        self.cell_type_sparse_weight = None
        self.cell_line_sparse_weight = None

        if initial_sparse_weight != None:
            self.sparse_weight = self.init_sparse_weight(initial_sparse_weight)


    def init_sparse_weight(self, initial_sparse_weight):
        """
        初始化sparse weight
        ----------
        initial_sparse_weight : float
            initial sparse weight
        """

        self.disease_sparse_weight = nn.Parameter(torch.empty(1).to(self.device))
        self.chemical_drug_sparse_weight = nn.Parameter(torch.empty(1).to(self.device))
        self.gene_sparse_weight = nn.Parameter(torch.empty(1).to(self.device))
        self.cell_type_sparse_weight = nn.Parameter(torch.empty(1).to(self.device))
        self.cell_line_sparse_weight = nn.Parameter(torch.empty(1).to(self.device))

        self.disease_sparse_weight.data.fill_(initial_sparse_weight) # init sparse_weight
        self.chemical_drug_sparse_weight.data.fill_(initial_sparse_weight) # init sparse_weight
        self.gene_sparse_weight.data.fill_(initial_sparse_weight) # init sparse_weight
        self.cell_type_sparse_weight.data.fill_(initial_sparse_weight) # init sparse_weight
        self.cell_line_sparse_weight.data.fill_(initial_sparse_weight) # init sparse_weight


        return self.disease_sparse_weight,self.chemical_drug_sparse_weight,self.gene_sparse_weight,self.cell_type_sparse_weight,self.cell_line_sparse_weight


    def init_disease_sparse_encoder(self, corpus):
        """
        初始化使用tf-idf计算sparse representation
        :param corpus:
        :return:
        """
        self.disease_sparse_encoder = SparseEncoder(self.device).fit(corpus)

        return self.disease_sparse_encoder

    def init_gene_protein_sparse_encoder(self, corpus):
        """
        初始化使用tf-idf计算sparse representation
        :param corpus:
        :return:
        """
        self.gene_protein_sparse_encoder = SparseEncoder(self.device).fit(corpus)

        return self.gene_protein_sparse_encoder

    def init_chemical_drug_sparse_encoder(self, corpus):
        """
        初始化使用tf-idf计算sparse representation
        :param corpus:
        :return:
        """
        self.chemical_drug_sparse_encoder = SparseEncoder(self.device).fit(corpus)

        return self.chemical_drug_sparse_encoder

    def init_cell_type_sparse_encoder(self, corpus):
        """
        初始化使用tf-idf计算sparse representation
        :param corpus:
        :return:
        """
        self.cell_type_sparse_encoder = SparseEncoder(self.device).fit(corpus)

        return self.cell_type_sparse_encoder

    def init_cell_line_sparse_encoder(self, corpus):
        """
        初始化使用tf-idf计算sparse representation
        :param corpus:
        :return:
        """
        self.cell_line_sparse_encoder = SparseEncoder(self.device).fit(corpus)

        return self.cell_line_sparse_encoder

    def init_species_sparse_encoder(self, corpus):
        '''
        初始化使用tf-idf计算sparse representation
        :param corpus:
        :return:
        '''
        self.species_sparse_encoder = SparseEncoder(self.device).fit(corpus)

        return self.species_sparse_encoder

    def load_dense_encoder(self,bert_dir,ckpt_path=None):
        """
        这也就是为了可以自由加载Transformer的model
        """

        self.dense_encoder = MultiEncoder(self.config)
        if ckpt_path is not None:
            load_model_and_parallel(self.dense_encoder, '0', ckpt_path=ckpt_path+'/model.pt', load_type='one2one')
            logger.info("从{}加载模型....".format(ckpt_path))
        self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)
        self.dense_encoder = self.dense_encoder.to(self.device)
        return self.dense_encoder, self.tokenizer

    def load_disease_sparse_encoder(self, sparse_model_path):
        """
        加载已有的sparse model(其实就是weight)
        :param sparse_model_path:
        :return:
        """
        sparse_encoder_path = os.path.join(sparse_model_path, 'disease_sparse_encoder.pk')
        # check file exists
        if not os.path.isfile(sparse_encoder_path):
            raise ValueError("不存在disease sparse encoder.pk")

        self.disease_sparse_encoder = SparseEncoder(self.device).load_encoder(path=sparse_encoder_path)

        return self.disease_sparse_encoder

    def load_chemical_drug_sparse_encoder(self, sparse_model_path):

        sparse_encoder_path = os.path.join(sparse_model_path, 'chemical_drug_sparse_encoder.pk')
        # check file exists
        if not os.path.isfile(sparse_encoder_path):
            raise ValueError("chemical drug sparse encoder.pk")

        self.chemical_drug_sparse_encoder = SparseEncoder(self.device).load_encoder(path=sparse_encoder_path)

        return self.chemical_drug_sparse_encoder

    def load_gene_protein_sparse_encoder(self, sparse_model_path):
        """
        加载已有的sparse model(其实就是weight)
        :param sparse_model_path:
        :return:
        """
        sparse_encoder_path = os.path.join(sparse_model_path, 'gene_protein_sparse_encoder.pk')
        if not os.path.isfile(sparse_encoder_path):
            raise ValueError("chemical drug sparse encoder.pk")
        self.gene_protein_sparse_encoder = SparseEncoder(self.device).load_encoder(path=sparse_encoder_path)

        return self.gene_protein_sparse_encoder

    def load_cell_type_sparse_encoder(self, sparse_model_path):

        sparse_encoder_path = os.path.join(sparse_model_path, 'cell_type_sparse_encoder.pk')
        # check file exists
        if not os.path.isfile(sparse_encoder_path):
            raise ValueError("chemical drug sparse encoder.pk")
        self.cell_type_parse_encoder = SparseEncoder(self.device).load_encoder(path=sparse_encoder_path)
        return self.cell_type_parse_encoder

    def load_cell_line_sparse_encoder(self, sparse_model_path):
        """
        加载已有的sparse model(其实就是weight)
        :param sparse_model_path:
        :return:
        """
        sparse_encoder_path = os.path.join(sparse_model_path, 'cell_line_sparse_encoder.pk')
        # check file exists
        if not os.path.isfile(sparse_encoder_path):
            raise ValueError("cell line sparse encoder.pk")
        self.cell_line_sparse_encoder = SparseEncoder(self.device).load_encoder(path=sparse_encoder_path)
        return self.cell_line_sparse_encoder

    def load_species_sparse_encoder(self, sparse_model_path):

        sparse_encoder_path = os.path.join(sparse_model_path, 'species_sparse_encoder.pk')
        # check file exists
        if not os.path.isfile(sparse_encoder_path):
            raise ValueError("species sparse encoder.pk")

        self.species_sparse_encoder = SparseEncoder(self.device).load_encoder(path=sparse_encoder_path)

        return self.species_sparse_encoder

    def get_dense_encoder(self):

        return self.dense_encoder

    def get_dense_tokenizer(self):

        return self.tokenizer

    def get_chemical_drug_sparse_encoder(self):

        return self.chemical_drug_sparse_encoder

    def get_disease_sparse_encoder(self):

        return self.disease_sparse_encoder

    def get_gene_protein_sparse_encoder(self):

        return self.gene_protein_sparse_encoder

    def get_cell_type_sparse_encoder(self):

        return self.cell_type_sparse_encoder

    def get_cell_line_sparse_encoder(self):

        return self.cell_line_sparse_encoder

    def get_species_sparse_encoder(self):

        return self.species_sparse_encoder

    def get_sparse_weight(self):

        return self.disease_sparse_weight,self.chemical_drug_sparse_weight,self.gene_sparse_weight,self.cell_type_sparse_weight,self.cell_line_sparse_weight

    def get_disease_sparse_representation(self, mentions, verbose=False):
        """
        将数据集中的mention，使用sparse encoder进行编码，得到sparse representations
        :param mentions:
        :param verbose:
        :return:
        """
        batch_size = 1024
        sparse_embeds = []

        for start in tqdm(range(0, len(mentions), batch_size), disable=not verbose,
                          desc='get disease sparse embedding...'):
            end = min(start + batch_size, len(mentions))
            batch = mentions[start:end]
            # 调用tf-ids的transform，得到

            batch_sparse_embeds = self.disease_sparse_encoder(batch)

            batch_sparse_embeds = batch_sparse_embeds.cpu().numpy()
            sparse_embeds.append(batch_sparse_embeds)
        sparse_embeds = np.concatenate(sparse_embeds, axis=0)

        return sparse_embeds

    def get_chemical_drug_sparse_representation(self, mentions, verbose=False):
        '''
        将数据集中的mention，使用sparse encoder进行编码，得到sparse representations
        :param mentions:
        :param verbose:
        :return:
        '''
        batch_size = 1024
        sparse_embeds = []

        for start in tqdm(range(0, len(mentions), batch_size), disable=not verbose,
                          desc='get chemical_drug sparse embedding'):
            end = min(start + batch_size, len(mentions))
            batch = mentions[start:end]
            # 调用tf-ids的transform，得到
            batch_sparse_embeds = self.chemical_drug_sparse_encoder(batch)
            batch_sparse_embeds = batch_sparse_embeds.cpu().numpy()
            sparse_embeds.append(batch_sparse_embeds)
        sparse_embeds = np.concatenate(sparse_embeds, axis=0)

        return sparse_embeds

    def get_gene_protein_sparse_representation(self, mentions, verbose=False):
        '''
        将数据集中的mention，使用sparse encoder进行编码，得到sparse representations
        :param mentions:
        :param verbose:
        :return:
        '''
        batch_size = 1024
        sparse_embeds = []

        for start in tqdm(range(0, len(mentions), batch_size), disable=not verbose,
                          desc='get gene_protein sparse embedding'):
            end = min(start + batch_size, len(mentions))
            batch = mentions[start:end]
            # 调用tf-ids的transform，得到
            batch_sparse_embeds = self.gene_protein_sparse_encoder(batch)
            batch_sparse_embeds = batch_sparse_embeds.cpu().numpy()
            sparse_embeds.append(batch_sparse_embeds)
        sparse_embeds = np.concatenate(sparse_embeds, axis=0)

        return sparse_embeds

    def get_cell_type_sparse_representation(self, mentions, verbose=False):
        '''
        将数据集中的mention，使用sparse encoder进行编码，得到sparse representations
        :param mentions:
        :param verbose:
        :return:
        '''
        batch_size = 1024
        sparse_embeds = []

        for start in tqdm(range(0, len(mentions), batch_size), disable=not verbose,
                          desc='get cell_tyoe sparse embedding'):
            end = min(start + batch_size, len(mentions))
            batch = mentions[start:end]
            # 调用tf-ids的transform，得到
            batch_sparse_embeds = self.gene_protein_sparse_encoder(batch)
            batch_sparse_embeds = batch_sparse_embeds.cpu().numpy()
            sparse_embeds.append(batch_sparse_embeds)
        sparse_embeds = np.concatenate(sparse_embeds, axis=0)

        return sparse_embeds

    def get_cell_line_sparse_representation(self, mentions, verbose=False):
        '''
        将数据集中的mention，使用sparse encoder进行编码，得到sparse representations
        :param mentions:
        :param verbose:
        :return:
        '''
        batch_size = 1024
        sparse_embeds = []

        for start in tqdm(range(0, len(mentions), batch_size), disable=not verbose,
                          desc='get cell_line sparse embedding'):
            end = min(start + batch_size, len(mentions))
            batch = mentions[start:end]
            # 调用tf-ids的transform，得到
            batch_sparse_embeds = self.gene_protein_sparse_encoder(batch)
            batch_sparse_embeds = batch_sparse_embeds.cpu().numpy()
            sparse_embeds.append(batch_sparse_embeds)
        sparse_embeds = np.concatenate(sparse_embeds, axis=0)

        return sparse_embeds

    def get_species_sparse_representation(self, mentions, verbose=False):
        '''
        将数据集中的mention，使用sparse encoder进行编码，得到sparse representations
        :param mentions:
        :param verbose:
        :return:
        '''
        batch_size = 1024
        sparse_embeds = []

        for start in tqdm(range(0, len(mentions), batch_size), disable=not verbose,
                          desc='get species sparse embedding'):
            end = min(start + batch_size, len(mentions))
            batch = mentions[start:end]
            # 调用tf-ids的transform，得到
            batch_sparse_embeds = self.gene_protein_sparse_encoder(batch)
            batch_sparse_embeds = batch_sparse_embeds.cpu().numpy()
            sparse_embeds.append(batch_sparse_embeds)
        sparse_embeds = np.concatenate(sparse_embeds, axis=0)

        return sparse_embeds

    def get_dense_representation(self, mentions, verbose=False, type_=0):
        '''
        这是使用BERT等预训练模型对mentions进行encode
        :param mentions:
        :param verbose:
        :return:
        '''
        # 这时候注意要开启eval，不使用dropout....
        self.dense_encoder.eval()  # prevent dropout
        batch_size = 1024

        if isinstance(mentions, np.ndarray):
            mentions = mentions.tolist()
        # 添加了这些参数之后，竟然相当于encode_plus，可能是参数return_tensor
        # name_encoding.shape = (1587,25)
        name_encodings = self.tokenizer(mentions, padding="max_length", max_length=self.max_len, truncation=True,
                                        return_tensors="pt")

        name_encodings = name_encodings.to(self.device)

        name_dataset = NamesDataset(name_encodings)
        # default_data_collator这个是非常简单的一个预处理函数
        name_dataloader = DataLoader(name_dataset, shuffle=False, collate_fn=default_data_collator,
                                     batch_size=batch_size)

        # 这里还设置不要进行梯度记录，这里就相当于只是利用模型进行计算，不进行更新
        dense_embeds = []
        if type_ == 0:
            desc = 'get disease dense representation'
        elif type_ == 1:
            desc = 'get chemical_drug dense representation'
        elif type_ == 2:
            desc = 'get gene protein dense representation'
        elif type_ == 3:
            desc = 'get cell_type dense representation'
        elif type_ == 4:
            desc = 'get cell_line dense representation'
        elif type_ == 5:
            desc = 'get species dense representation'
        else:
            raise ValueError("type_应该为0,1,2,3,4,5,不能是{}".format(type_))
        with torch.no_grad():
            for batch in tqdm(name_dataloader, disable=not verbose, desc=desc):
                outputs = self.dense_encoder(**batch, type_=type_)
                # 但是不知道为啥不选择outputs[1]作为[CLS]的output...
                # [CLS] representations,shape=(batch_size,hidden_size)=(1024,768)
                batch_dense_embeds = outputs[0][:,0].cpu().detach().numpy()
                dense_embeds.append(batch_dense_embeds)
        dense_embeds = np.concatenate(dense_embeds, axis=0)

        return dense_embeds

    def get_score_matrix(self, query_embeds, dict_embeds):
        """
        返回相似分数矩阵，通过点击
        Parameters
        ----------
        query_embeds :shape = (len(dataset),dim)
        dict_embeds : (len(dict),dim)

        Returns
        -------
        score_matrix : np.array
            2d numpy array of scores
        """
        score_matrix = np.matmul(query_embeds, dict_embeds.T)
        # shape = (len(dataset),len(dict))
        return score_matrix

    def retrieve_candidate(self, score_matrix, topk):
        """
        对相似矩阵检索，得到对于每个mention，得到与其最相似的topk个字典中的数值
        Parameters
        ----------
        score_matrix : 相似分数矩阵 np.array shape=(len(train_dataset),len(dict))
            2d numpy array of scores
        topk : int
            The number of candidates
        Returns
        -------
        topk_idxs : np.array
            2d numpy array of scores [# of query , # of dict] =(1578,71924),就是输入数据和字典中各个实体的相似分数
        """

        def indexing_2d(arr, cols):
            '''
            这是二维取值，根据cols的index来取值
            arr:
            '''
            rows = np.repeat(np.arange(0, cols.shape[0])[:, np.newaxis], cols.shape[1], axis=1)
            # 这个rows结果为
            #       [[   0,    0,    0, ...,    0,    0,    0],
            #        [   1,    1,    1, ...,    1,    1,    1],
            #        [   2,    2,    2, ...,    2,    2,    2],
            #        ...,
            #        [1584, 1584, 1584, ..., 1584, 1584, 1584],
            #        [1585, 1585, 1585, ..., 1585, 1585, 1585],
            #        [1586, 1586, 1586, ..., 1586, 1586, 1586]]

            return arr[rows, cols]

        # 这里是使用argpartion,通过对倒数第20个进行划分，得到最大的20个index
        # 也就是每个单词对应词典中最相似的topk个name，这个时候并没有排序

        topk_idxs = np.argpartition(score_matrix, -topk)[:, -topk:]  # shape=(1587,20)

        # 根据上面得到的index到score_matrix取值
        topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
        # 对index排序
        topk_argidxs = np.argsort(-topk_score_matrix)
        topk_idxs = indexing_2d(topk_idxs, topk_argidxs)

        return topk_idxs

    def save_model(self, path,config,epoch=0,mode='normal'):
        """
        保存所有的模型
            1. dense encoder(BERT等模型)
            2. sparse weight
        :param path:
        :return:
        """
        # save dense encoder,这是transformers自带的方法进行保存模型，

        save_model(config, self.dense_encoder, epoch=epoch,mode=mode)
        # self.dense_encoder.save_pretrained(path)
        # self.tokenizer.save_pretrained(path)

        # save sparse encoder
        disease_sparse_encoder_path = os.path.join(path, 'disease_sparse_encoder.pk')
        self.disease_sparse_encoder.save_encoder(path=disease_sparse_encoder_path)

        chemical_drug_sparse_encoder_path = os.path.join(path, 'chemical_drug_sparse_encoder.pk')
        self.chemical_drug_sparse_encoder.save_encoder(path=chemical_drug_sparse_encoder_path)

        gene_protein_encoder_path = os.path.join(path, 'gene_protein_sparse_encoder.pk')
        self.gene_protein_sparse_encoder.save_encoder(path=gene_protein_encoder_path)

        cell_type_sparse_encoder_path = os.path.join(path, 'cell_type_sparse_encoder.pk')
        self.cell_type_sparse_encoder.save_encoder(path=cell_type_sparse_encoder_path)

        cell_line_sparse_encoder_path = os.path.join(path, 'cell_line_sparse_encoder.pk')
        self.cell_line_sparse_encoder.save_encoder(path=cell_line_sparse_encoder_path)

        # species_sparse_encoder_path = os.path.join(path, 'species_sparse_encoder.pk')
        # self.species_sparse_encoder.save_encoder(path=species_sparse_encoder_path)

        disease_sparse_weight_file = os.path.join(path, 'disease_sparse_weight.pt')
        torch.save(self.disease_sparse_weight, disease_sparse_weight_file)
        logging.info("Sparse weight saved in {}".format(disease_sparse_weight_file))

        chemical_drug_sparse_weight_file = os.path.join(path, 'chemical_drug_sparse_weight.pt')
        torch.save(self.chemical_drug_sparse_weight, chemical_drug_sparse_weight_file)
        logging.info("Sparse weight saved in {}".format(chemical_drug_sparse_weight_file))

        gene_sparse_weight_file = os.path.join(path, 'gene_sparse_weight.pt')
        torch.save(self.gene_sparse_weight, gene_sparse_weight_file)
        logging.info("Sparse weight saved in {}".format(gene_sparse_weight_file))

        cell_type_sparse_weight_file = os.path.join(path, 'cell_type_sparse_weight.pt')
        torch.save(self.cell_type_sparse_weight, cell_type_sparse_weight_file)
        logging.info("Sparse weight saved in {}".format(cell_type_sparse_weight_file))

        cell_line_sparse_weight_file = os.path.join(path, 'cell_line_sparse_weight.pt')
        torch.save(self.cell_line_sparse_weight, cell_line_sparse_weight_file)
        logging.info("Sparse weight saved in {}".format(cell_line_sparse_weight_file))

    def load_model(self, model_name_or_path,all_sparse_weights=None):
        """
        这是加载所有的模型
        """

        self.load_dense_encoder(self.config.bert_dir,model_name_or_path)
        self.load_disease_sparse_encoder(model_name_or_path)
        self.load_chemical_drug_sparse_encoder(model_name_or_path)
        self.load_gene_protein_sparse_encoder(model_name_or_path)
        self.load_cell_type_sparse_encoder(model_name_or_path)
        self.load_cell_line_sparse_encoder(model_name_or_path)
        # self.load_species_sparse_encoder(model_name_or_path)
        self.load_all_sparse_weight(model_name_or_path,all_sparse_weights)



    def load_sparse_weight(self, model_name_or_path):
        sparse_weight_path = os.path.join(model_name_or_path, 'sparse_weight.pt')
        # check file exists
        if not os.path.isfile(sparse_weight_path):
            # download from huggingface hub and cache it
            sparse_weight_url = hf_hub_url(model_name_or_path, filename="sparse_weight.pt")
            sparse_weight_path = cached_download(sparse_weight_url)

        self.sparse_weight = torch.load(sparse_weight_path)

        return self.sparse_weight
    def load_all_sparse_weight(self, model_name_or_path,all_weights=None):
        """
        加载所有的sparse weight
        :param model_name_or_path:
        :return:
        """
        disease_sparse_weight_path = os.path.join(model_name_or_path, 'disease_sparse_weight.pt')
        chemical_drug_sparse_weight_path = os.path.join(model_name_or_path, 'chemical_drug_sparse_weight.pt')
        gene_sparse_weight_path = os.path.join(model_name_or_path, 'gene_sparse_weight.pt')
        cell_type_sparse_weight_path = os.path.join(model_name_or_path, 'cell_type_sparse_weight.pt')
        cell_line_sparse_weight_path = os.path.join(model_name_or_path, 'cell_line_sparse_weight.pt')
        # check file exists
        if all_weights is not None:
            self.disease_sparse_weight = nn.Parameter(torch.tensor(all_weights[0]).to(self.device))
            self.chemical_drug_sparse_weight = nn.Parameter(torch.tensor(all_weights[1]).to(self.device))
            self.gene_sparse_weight = nn.Parameter(torch.tensor(all_weights[2]).to(self.device))
            self.cell_type_sparse_weight = nn.Parameter(torch.tensor(all_weights[3]).to(self.device))
            self.cell_line_sparse_weight = nn.Parameter(torch.tensor(all_weights[4]).to(self.device))


        else:

            self.disease_sparse_weight = torch.load(disease_sparse_weight_path)
            self.chemical_drug_sparse_weight = torch.load(chemical_drug_sparse_weight_path)
            self.gene_sparse_weight = torch.load(gene_sparse_weight_path)
            self.cell_type_sparse_weight = torch.load(cell_type_sparse_weight_path)
            self.cell_line_sparse_weight = torch.load(cell_line_sparse_weight_path)




class SparseEncoder(object):
    '''
        这是Sparse Encoder,
        这就是Tf-idf计算得到spare representation，使用uni，bi-gram...
        直接使用sklearn进行计算，然后转变为tensor类别
    '''

    def __init__(self, device):
        self.encoder = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
        self.device = device

    def fit(self, train_corpus):
        self.encoder.fit(train_corpus)
        return self

    def transform(self, mentions):
        '''
        将mentions使用tf-idf进行转变
        :param mentions:
        :return:
        '''

        vec = self.encoder.transform(mentions).toarray()
        vec = torch.FloatTensor(vec)  # return torch float tensor
        vec = vec.to(self.device)
        return vec

    def __call__(self, mentions):
        return self.transform(mentions)

    def vocab(self):
        return self.encoder.vocabulary_

    def save_encoder(self, path):
        with open(path, 'wb') as fout:
            pickle.dump(self.encoder, fout)
            logging.info("Sparse encoder saved in {}".format(path))

    def load_encoder(self, path):
        with open(path, 'rb') as fin:
            self.encoder = pickle.load(fin)
            logging.info("Sparse encoder loaded from {}".format(path))

        return self


class MultiRerankNet(nn.Module):
    """
        这个是训练模型，用于对模型的训练...
    """

    def __init__(self, config, dense_encoder, sparse_weight, device):

        super(MultiRerankNet, self).__init__()
        self.dense_encoder = dense_encoder
        self.config = config

        self.device = device

        self.criterion = self.marginal_nll

        self.disease_sparse_weight, self.chemical_drug_sparse_weight, self.gene_sparse_weight, self.cell_type_sparse_weight, self.cell_line_sparse_weight = sparse_weight

        self.disease_sparse_weight.requires_grad = True
        self.chemical_drug_sparse_weight.requires_grad = True
        self.gene_sparse_weight.requires_grad = True
        self.cell_type_sparse_weight.requires_grad = True
        self.cell_line_sparse_weight.requires_grad = True



    def forward(self, x,type_=0):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        # 这里只预先得到sparse的similiarity scores、
        # dense的similarity scores则是动态得到...
        query_token, candidate_tokens, candidate_s_scores = x
        batch_size, topk, max_length = candidate_tokens['input_ids'].shape

        candidate_s_scores = candidate_s_scores.to(self.device)

        query_token['input_ids'] = query_token['input_ids'].to(self.device)
        query_token['token_type_ids'] = query_token['token_type_ids'].to(self.device)
        query_token['attention_mask'] = query_token['attention_mask'].to(self.device)

        candidate_tokens['input_ids'] = candidate_tokens['input_ids'].to(self.device)
        candidate_tokens['token_type_ids'] = candidate_tokens['token_type_ids'].to(self.device)
        candidate_tokens['attention_mask'] = candidate_tokens['attention_mask'].to(self.device)

        # dense embed for query and candidates
        query_embed = self.dense_encoder(
            input_ids=query_token['input_ids'].squeeze(1),
            token_type_ids=query_token['token_type_ids'].squeeze(1),
            attention_mask=query_token['attention_mask'].squeeze(1)
        )
        query_embed = query_embed[0][:, 0].unsqueeze(1)  # query : [batch_size, 1, hidden]

        candidate_embeds = self.dense_encoder(
            input_ids=candidate_tokens['input_ids'].reshape(-1, max_length),
            token_type_ids=candidate_tokens['token_type_ids'].reshape(-1, max_length),
            attention_mask=candidate_tokens['attention_mask'].reshape(-1, max_length)
        )
        candidate_embeds = candidate_embeds[0][:, 0].reshape(batch_size, topk, -1)  # [batch_size, topk, hidden]

        # 得到dense similiarity scores,shape=(batch_size,topk)
        candidate_d_score = torch.bmm(query_embed, candidate_embeds.permute(0, 2, 1)).squeeze(1)


        if type_ == 0:
            score = self.disease_sparse_weight * candidate_s_scores + candidate_d_score
        elif type_ == 1:
            score = self.chemical_drug_sparse_weight * candidate_s_scores + candidate_d_score
        elif type_ == 2:

            score = self.gene_sparse_weight * candidate_s_scores + candidate_d_score
        elif type_ == 3:
            score = self.cell_type_sparse_weight * candidate_s_scores + candidate_d_score
        elif type_ == 4:
            score = self.cell_line_sparse_weight * candidate_s_scores + candidate_d_score
        else:
            raise ValueError

        return score

    def reshape_candidates_for_encoder(self, candidates):
        """
        reshape candidates for encoder input shape
        [batch_size, topk, max_length] => [batch_size*topk, max_length]
        """
        _, _, max_length = candidates.shape
        candidates = candidates.contiguous().view(-1, max_length)
        return candidates

    def get_loss(self, outputs, targets):
        targets = targets.to(self.device)
        loss = self.criterion(outputs, targets)
        return loss

    def get_embeddings(self, mentions, batch_size=1024):
        """
        Compute all embeddings from mention tokens.
        """
        embedding_table = []
        with torch.no_grad():
            for start in tqdm(range(0, len(mentions), batch_size)):
                end = min(start + batch_size, len(mentions))
                batch = mentions[start:end]
                batch_embedding = self.vectorizer(batch)
                batch_embedding = batch_embedding.cpu()
                embedding_table.append(batch_embedding)
        embedding_table = torch.cat(embedding_table, dim=0)
        return embedding_table

    def marginal_nll(self, score, target):
        """
        sum all scores among positive samples
        损失函数计算
        """
        predict = F.softmax(score, dim=-1)
        loss = predict * target  # 只记录target=1的预测分数
        loss = loss.sum(dim=-1)  # sum all positive scores
        loss = loss[loss > 0]  # filter sets with at least one positives
        # 将loss值给限定在[1e-9,1]范围，
        loss = torch.clamp(loss, min=1e-9, max=1)  # for numerical stability
        loss = -torch.log(loss)  # for negative log likelihood
        if len(loss) == 0:
            loss = loss.sum()  # will return zero loss
        else:
            loss = loss.mean()
        return loss

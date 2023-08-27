# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2022/01/13
   Copyright:      (c) kedaxia 2022
-------------------------------------------------
   Change Activity:
                   2022/01/13: 
-------------------------------------------------
"""
from transformers.configuration_bert import BertConfig

class MyBertConfig(BertConfig):
    
    def __init__(self,**kwargs):
        """
        这里Config必须得基于BertConfig，不然一些地方用不了....
        :param kwargs:
        """
        super(MyBertConfig, self).__init__(**kwargs)

        self.model_name = kwargs['model_name']
        self.task_name = kwargs['task_name']
        self.bert_name = kwargs['bert_name']
        self.bert_dir = kwargs['bert_dir']
        self.dataset_name = kwargs['dataset_name']


        self.learning_rate = kwargs['learning_rate']
        self.use_scheduler = kwargs['use_scheduler']
        self.warmup_proportion = kwargs['warmup_proportion']  # 学习率调整
        self.batch_size = kwargs['batch_size']
        self.max_len = kwargs['max_len']  # 这里一般设置为25
        self.seed = kwargs['seed']
        self.num_epochs = kwargs['num_epochs']
        self.gradient_accumulation_steps = kwargs['gradient_accumulation_steps']

        self.max_grad_norm = kwargs['max_grad_norm']
        self.weight_decay = kwargs['weight_decay']
        self.dropout_prob = kwargs['dropout_prob']




        # 微调最后四层
        self.freeze_bert = kwargs['freeze_bert']
        self.freeze_layers = ['layer.0', 'layer.1.', 'layer.2', 'layer.3']
        self.freeze_layer_nums = len(self.freeze_layers)
        self.use_gpu = kwargs['use_gpu']
        self.use_n_gpu = kwargs['use_n_gpu']
        self.use_fp16 = kwargs['use_fp16']
        self.gpu_id = kwargs['gpu_id']
        self.use_amp = kwargs['use_amp']
        self.save_model = kwargs['save_model']
        self.save_predictions = kwargs['save_predictions']

        self.logfile_name = kwargs['logfile_name']
        self.logs_dir = './outputs/logs/{}/'.format(self.model_name)
        self.tensorboard_dir = './outputs/tensorboard/{}/'.format(self.model_name)

        self.output_dir = './outputs/save_models/{}/{}'.format(self.model_name,self.task_name)

        self.use_wandb = kwargs['use_wandb']
        self.use_metric_summary_writer = kwargs['use_metric_summary_writer']
        self.use_parameter_summary_writer = kwargs['use_parameter_summary_writer']

        self.debug = kwargs['debug']
        self.encoder_type = kwargs['encoder_type']
        self.task_encoder_nums = kwargs['task_encoder_nums'] # 表示多任务的对应encoder个数


        # 数据集的路径
        if self.dataset_name == 'cell_line_dataset':
            self.dictionary_path = 'dataset/cell_line_dataset/cell_line_dictionary.txt'
            self.train_path = './dataset/cell_line_dataset/train.txt'
            self.dev_path = './dataset/cell_line_dataset/dev.txt'

        elif self.dataset_name == 'cell_type_dataset':
            self.dictionary_path = './dataset/cell_type_dataset/cell_type_dictionary.txt'
            self.train_path = './dataset/cell_type_dataset/train.txt'
            self.dev_path = './dataset/cell_type_dataset/dev.txt'
        elif self.dataset_name == 'mesh_chemical_drug':
            self.dictionary_path = 'dataset/mesh_chemical_drug/chemical_and_drug_dictionary.txt'
            self.train_path = './dataset/mesh_chemical_drug/train.txt'
            self.dev_path = './dataset/mesh_chemical_drug/dev.txt'
        elif self.dataset_name == 'mesh_disease':
            self.dictionary_path = 'dataset/mesh_disease/disease_dictionary.txt'
            self.train_path = './dataset/mesh_disease/train.txt'
            self.dev_path = './dataset/mesh_disease/dev.txt'
        elif self.dataset_name == 'gene_protein':
            self.train_path = './dataset/cell_type_dataset/train.txt'
            self.dev_path = './dataset/gene_protein/dev.txt'
            self.dictionary_path = 'dataset/gene_protein/entre_gene_dictionary.txt'
        elif self.dataset_name == 'BC2GM':
            self.train_path = './dataset/BC2GM/train.txt'
            self.dev_path = './dataset/BC2GM/dev.txt'
            self.dictionary_path = './dataset/BC2GM/entre_gene_dictionary.txt'
        elif self.dataset_name == 'species':
            self.train_path = './dataset/species/train.txt'
            self.dev_path = './dataset/species/dev.txt'
            self.dictionary_path = './dataset/species/species_dictionary.txt'
        else:
            self.train_dictionary_path = './dataset/{}/train_dictionary.txt'.format(self.dataset_name)
            self.dev_dictionary_path = './dataset/{}/dev_dictionary.txt'.format(self.dataset_name)
            self.test_dictionary_path = './dataset/{}/test_dictionary.txt'.format(self.dataset_name)
            self.train_dir = './dataset/{}/processed_traindev'.format(self.dataset_name)
            self.dev_dir = './dataset/{}/processed_dev'.format(self.dataset_name)
            self.test_dir = './dataset/{}/processed_test'.format(self.dataset_name)

            # 这个针对predicate的时候的path
            self.dictionary_path = './dataset/{}/train_dictionary.txt'.format(self.dataset_name)

        # 专属于BioNormalization的超参数
        self.dense_ratio = kwargs['dense_ratio']
        self.topk = kwargs['topk']

        # SapBERT的超参数
        self.type_of_triplets = kwargs['type_of_triplets']
        self.miner_margin = kwargs['miner_margin']
        self.agg_mode = kwargs['agg_mode']
        self.loss = kwargs['loss']
        self.use_miner = kwargs['use_miner']
        self.pairwise = kwargs['pairwise']

        # 可以替换成其他的各种文件，但是格式嘚一致
        self.sap_train_dir = './corpus/UMLS/umls2021AB_en_uncased_no_dup_pairwise_pair_th50.txt'.format(self.dataset_name)

        self.disease_dictionary_path = "./dataset/mesh_disease/disease_dictionary.txt"
        self.disease_cache_path = "./tmp/cached_disease_dictionary.pk"

        self.chemical_drug_dictionary_path = "./dataset/mesh_chemical_drug/chemical_and_drug_dictionary.txt"
        self.chemical_drug_cache_path = "./tmp/cache_chemical_drug_dictionary.pk"

        self.cell_type_dictionary_path = "./dataset/cell_type_dataset/cell_type_dictionary.txt"
        self.cell_type_cache_path = "./tmp/cache_cell_type_dictionary.pk"

        self.cell_line_dictionary_path = "./dataset/cell_line_dataset/cell_line_dictionary.txt"
        self.cell_line_cache_path = "./tmp/cache_cell_line_dictionary.pk"

        self.gene_protein_dictionary_path = "./dataset/gene_protein/entre_gene_dictionary.txt"
        self.gene_protein_cache_path = "./tmp/cache_entre_gene_dictionary.pk"

        self.species_dictionary_path = "./dataset/species/species_dictionary.txt"
        self.species_cache_path = "./tmp/cache_species_dictionary.pk"



        # 大量文件路径，为了多任务的训练
        self.bc5cdr_disease_train_path = './dataset/bc5cdr-disease/train.txt'
        self.bc5cdr_disease_dev_path = './dataset/bc5cdr-disease/dev.txt'
        self.bc5cdr_disease_test_path = './dataset/bc5cdr-disease/test.txt'

        self.bc5cdr_disease_dictionary_path = './dataset/bc5cdr-disease/bc5cdr-disease_dictionary.txt'
        self.bc5cdr_disease_train_dictionary_path = './dataset/bc5cdr-disease/train_dictionary.txt'
        self.bc5cdr_disease_dev_dictionary_path = './dataset/bc5cdr-disease/dev_dictionary.txt'
        self.bc5cdr_disease_test_dictionary_path = './dataset/bc5cdr-disease/test_dictionary.txt'

        self.bc5cdr_chemical_train_path = './dataset/bc5cdr-chemical/train.txt'
        self.bc5cdr_chemical_dev_path = './dataset/bc5cdr-chemical/dev.txt'
        self.bc5cdr_chemical_test_path = './dataset/bc5cdr-chemical/test.txt'

        self.bc5cdr_chemical_dictionary_path = './dataset/bc5cdr-disease/bc5cdr-chemical_dictionary.txt'
        self.bc5cdr_chemical_train_dictionary_path = './dataset/bc5cdr-chemical/train_dictionary.txt'
        self.bc5cdr_chemical_dev_dictionary_path = './dataset/bc5cdr-chemical/dev_dictionary.txt'
        self.bc5cdr_chemical_test_dictionary_path = './dataset/bc5cdr-chemical/test_dictionary.txt'

        self.ncbi_disease_train_path = "./dataset/ncbi-disease/train.txt"
        self.ncbi_disease_dev_path = "./dataset/ncbi-disease/dev.txt"
        self.ncbi_disease_test_path = "./dataset/ncbi-disease/test.txt"

        self.ncbi_disease_dictionary_path = './dataset/ncbi-disease/bc5cdr-chemical_dictionary.txt'
        self.ncbi_disease_train_dictionary_path = './dataset/ncbi-disease/train_dictionary.txt'
        self.ncbi_disease_dev_dictionary_path = './dataset/ncbi-disease/dev_dictionary.txt'
        self.ncbi_disease_test_dictionary_path = './dataset/ncbi-disease/test_dictionary.txt'

        self.bc2gm_train_path = "./dataset/bc2gm/train.txt"
        self.bc2gm_test_path = "./dataset/bc2gm/test.txt"

        self.bc2gm_dictionary_path = './dataset/bc2gm/entre_gene_dictionary.txt'
        self.bc2gm_train_dictionary_path = './dataset/bc2gm/entre_gene_dictionary.txt'
        self.bc2gm_dev_dictionary_path = './dataset/bc2gm/entre_gene_dictionary.txt'
        self.bc2gm_test_dictionary_path = './dataset/bc2gm/entre_gene_dictionary.txt'


        self.cell_type_train_path = "./dataset/cell_type_dataset/train.txt"
        self.cell_type_dev_path = "./dataset/cell_type_dataset/dev.txt"

        self.cell_line_train_path = "./dataset/cell_line_dataset/train.txt"
        self.cell_line_dev_path = "./dataset/cell_line_dataset/dev.txt"

        self.gene_protein_train_path = "./dataset/gene_protein/train.txt"
        self.gene_protein_dev_path = "./dataset/gene_protein/dev.txt"

        self.chemical_drug_train_path = "./dataset/mesh_chemical_drug/train.txt"
        self.chemical_drug_dev_path = "./dataset/mesh_chemical_drug/dev.txt"

        self.mesh_disease_train_path = "./dataset/mesh_disease/train.txt"
        self.mesh_disease_dev_path = "./dataset/mesh_disease/dev.txt"

        self.species_train_path = "./dataset/species/train.txt"
        self.species_dev_path = "./dataset/species/dev.txt"






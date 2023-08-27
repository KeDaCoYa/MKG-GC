# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  这是所有模型的配置文件
   Author :        kedaxia
   date：          2021/11/08
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/11/08: 今天是个好日期
-------------------------------------------------
"""
import json

from ipdb import set_trace
from transformers import BertConfig,PretrainedConfig



class BaseConfig:
    def __init__(self, **kwargs):
        '''

        :param gpu_ids:
        :param model_name:
        :param ner_type:
        :param model_name: 这个自由写名字，没啥区别
        '''
        # 这个是选择数据集name

        self.ner_dataset_name = kwargs['ner_dataset_name']
        self.which_model = kwargs['which_model']
        self.run_type = kwargs['run_type']
        self.inter_scheme = kwargs['inter_scheme']
        # 这个选择使用哪个路径下的数据集
        self.wandb_notes = kwargs['wandb_notes']

        if self.ner_dataset_name in ['jnlpba','multi_jnlpba','origin_jnlpba']:
            self.num_entity_class = 5
        elif self.ner_dataset_name == 'AllDataset':
            self.num_entity_class = 7
        elif self.ner_dataset_name in ['Balance_AllDataset','Normalize_AllDataset','Normalize_Balance_AllDataset']:
            self.num_entity_class = 6
        elif self.ner_dataset_name in ['multi_alldataset','multi_all_dataset_v1_lite','multi_all_dataset_large','eight_type_alldataset','1009abstracts','3400abstracts'] or 'CV' in self.ner_dataset_name:
            self.num_entity_class = 8
        elif self.ner_dataset_name in ['BC6ChemProt','BC7DrugProt','multi_BC6','multi_BC7','multi_BC5','BC5CDR']:
            self.num_entity_class = 2
        elif self.ner_dataset_name in ['DDI2013']:
            self.num_entity_class = 4
        else:
            self.num_entity_class = 1
        self.use_n_gpu = kwargs['use_n_gpu']  # 这里只在bert模型下使用，其他模型一般不需要...
        self.model_name = kwargs['model_name']
        self.task_name = kwargs['task_name']
        self.decoder_layer = kwargs['decoder_layer']
        # BERT model的 可选择的参数：[bert_span,bert_mlp,bert_bilstm_crf,globalpointer]
        # Normal model的可选择参数： [bilstm_crf,att_bilstm_crf,bilstm_globalpointer,att_bilstm_globalpointer]

        self.seed = kwargs['seed']
        self.entity_type = kwargs['entity_type']

        self.over_fitting_rate = kwargs['over_fitting_rate']  # 这个用于表示训练集的f1和验证集f1之间的差距，如果为1表示不会限制
        self.over_fitting_epoch = kwargs['over_fitting_epoch']
        self.early_stop = kwargs['early_stop']
        self.use_scheduler = kwargs['use_scheduler']

        self.use_gpu = kwargs['use_gpu']
        self.gpu_id = kwargs['gpu_id']

        self.warmup_proportion = kwargs['warmup_proportion']  # 学习率调整
        self.weight_decay = 0.01
        self.ema_decay = 0.999

        self.gradient_accumulation_steps = kwargs['gradient_accumulation_steps']
        self.max_grad_norm = kwargs['max_grad_norm']

        # 这个参数如果为True则
        self.fixed_batch_length = kwargs['fixed_batch_length']  # 这个参数控制batch的长度是否固定

        self.logfile_name = kwargs['logfile_name']
        self.logs_dir = './outputs/logs/{}/{}/'.format(self.model_name, self.ner_dataset_name)
        self.tensorboard_dir = './outputs/tensorboard/{}/{}/tensorboard/'.format(self.model_name, self.ner_dataset_name)
        if self.run_type == 'normal':
            self.output_dir = './outputs/save_models/{}_{}/{}/'.format(kwargs["bert_name"],self.model_name, self.ner_dataset_name)
        elif self.run_type == 'cv5':
            self.output_dir = './outputs/save_models/cv5/{}/{}_{}/{}/'.format(1,kwargs["bert_name"],self.model_name, self.ner_dataset_name)
        # 最后模型对文本预测的结果存放点
        self.predicate_dir = './outputs/predicate_outputs/{}_{}/{}/'.format(kwargs["bert_name"],self.model_name, self.ner_dataset_name)

        # NERdata，original_dataset
        self.data_dir = './NERdata/{}'.format(self.ner_dataset_name)
        if self.run_type == 'normal':
            self.train_file_path = './NERdata/{}/train.txt'.format(self.ner_dataset_name)
            self.dev_file_path = './NERdata/{}/dev.txt'.format(self.ner_dataset_name)
            self.test_file_path = './NERdata/{}/test.txt'.format(self.ner_dataset_name)
        else:
            pass

        # 更换数据集，只需要修改的部分
        if 'crf' in self.model_name or 'mlp' in self.model_name or 'test' in self.model_name:
            self.crf_label2id_path = './NERdata/{}/crf_label2id.json'.format(self.ner_dataset_name)
            self.crf_label2id = json.load(open(self.crf_label2id_path, 'r', encoding='utf-8'))
            self.crf_id2label_path = './NERdata/{}/crf_id2label.json'.format(self.ner_dataset_name)
            self.crf_id2label = json.load(open(self.crf_id2label_path, 'r', encoding='utf-8'))
            self.crf_id2label = {int(key): value for key, value in self.crf_id2label.items()}
            # 这个用于crf的classes，这个就是label2id的个数
            self.num_crf_class = len(self.crf_label2id)
        elif 'globalpointer' in self.model_name:
            self.globalpointer_label2id_path = './NERdata/{}/globalpointer_label2id.json'.format(self.ner_dataset_name)
            self.globalpointer_label2id = json.load(open(self.globalpointer_label2id_path, 'r', encoding='utf-8'))
            self.globalpointer_id2label_path = './NERdata/{}/globalpointer_id2label.json'.format(self.ner_dataset_name)
            self.globalpointer_id2label = json.load(open(self.globalpointer_id2label_path, 'r', encoding='utf-8'))
            self.globalpointer_id2label = {int(key): value for key, value in self.globalpointer_id2label.items()}
            # 这个用于globalpointer的classes，这个就是实体类别的个数
            self.num_gp_class = self.num_entity_class  # 这是对于globalpointer模型,就是实体类别个数
        elif 'span' in self.model_name:
            self.span_label2id_path = './NERdata/{}/span_label2id.json'.format(self.ner_dataset_name)
            self.span_label2id = json.load(open(self.span_label2id_path, 'r', encoding='utf-8'))
            self.span_id2label_path = './NERdata/{}/span_id2label.json'.format(self.ner_dataset_name)
            self.span_id2label = json.load(open(self.span_id2label_path, 'r', encoding='utf-8'))
            self.span_id2label = {int(key): value for key, value in self.span_id2label.items()}
            # span指针的classses
            # 这个是表示实体类别的个数，但是要包括非实体O，例如实体类别有疾病、化学、蛋白质，那么类别个数为3+1=4
            self.num_span_class = self.num_entity_class + 1

        # 对数据集进行排序
        self.use_sort = kwargs['use_sort']
        self.use_fp16 = kwargs['use_fp16']
        # 评价方式，micro或者macro
        self.evaluate_mode = kwargs['evaluate_mode']
        self.num_epochs = kwargs['num_epochs']

        self.use_ema = kwargs['use_ema']

        self.verbose = kwargs['verbose']

        self.use_wandb = kwargs['use_wandb']

        self.metric_summary_writer = kwargs['metric_summary_writer']
        self.parameter_summary_writer = kwargs['parameter_summary_writer']
        self.print_step = kwargs['print_step']
        self.save_model = kwargs['save_model']

        self.predicate_flag = False  # 在训练和验证的时候都是False，只有面对无标签的测试集才会打开
        self.debug = kwargs['debug']





class FLASHConfig(PretrainedConfig):
    model_type = "flash_quad"

    def __init__(
        self,
        vocab_size=31090,
        hidden_size=768,
        num_hidden_layers=24,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        pad_token_id=0,
        expansion_factor=2,
        s=128,
        norm_type="scalenorm",
        gradient_checkpointing=False,
        dropout=0.0,
        hidden_act="swish",
        classifier_dropout=0.1,
        **kwargs
    ):
        """

        :param vocab_size:
        :param hidden_size:
        :param num_hidden_layers: base model=24,因为base bert=12,所以这里12*2
        :param max_position_embeddings:
        :param type_vocab_size:token_type_embedding就是2
        :param initializer_range:
        :param layer_norm_eps:
        :param pad_token_id:
        :param expansion_factor:
        :param s: 这个是GAU的，一般为128
        :param norm_type: layernorm或者普通的ScaleNorm
        :param gradient_checkpointing:
        :param dropout:
        :param hidden_act: 激活函数
        :param classifier_dropout:
        :param kwargs:
        """
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.expansion_factor = expansion_factor
        self.s = s
        self.norm_type = norm_type
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.gradient_checkpointing = gradient_checkpointing
        self.classifier_dropout = classifier_dropout


class MyBertConfig(BaseConfig,BertConfig,FLASHConfig):
    def __init__(self, **kwargs):
        '''
        开始使用argparse控制config的参数

        :param model_name:
        :param ner_type:
        '''
        super(MyBertConfig, self).__init__(**kwargs)
        super(BertConfig, self).__init__(**kwargs)
        super(FLASHConfig, self).__init__(**kwargs)

        self.bert_dir = kwargs['bert_dir']
        self.bert_name = kwargs['bert_name']

        self.batch_size = kwargs['batch_size']
        self.span_loss_type = kwargs['span_loss_type']

        self.max_len = kwargs['max_len']  # BC5CDR-disease的文本最长为121


        self.other_lr = kwargs['other_lr']

        self.adam_epsilon = 1e-8

        self.dropout_prob = kwargs['dropout_prob']  # 设置bert的dropout
        # self.freeze_layers = ['layer.1.', 'layer.3','layer.4', 'layer.5', 'layer.7','layer.8','layer.9']
        # self.freeze_layers = ['layer.2','layer.3','layer.4', 'layer.5', 'layer.6','layer.7','layer.8','layer.9','layer.10']

        # 这是之前的层
        # self.freeze_layers = ['layer.1.', 'layer.3', 'layer.4', 'layer.5', 'layer.7', 'layer.9', 'layer.10']

        # 微调最后四层
        self.freeze_bert = kwargs['freeze_bert']

        if self.bert_name in['flash_quad']:
            self.freeze_layers = ['layer.0', 'layer.1.', 'layer.2', 'layer.3', 'layer.4', 'layer.5', 'layer.6', 'layer.7']
        elif self.bert_name in ['biobert','scibert','bert','wwm_bert']:
            self.freeze_layers = ['layer.0', 'layer.1.', 'layer.2', 'layer.3']
        elif self.bert_name in ['kebiolm']:
            self.freeze_layers = ['layer.0', 'layer.1.', 'layer.2', 'layer.3']

        elif self.bert_name == 'flash':
            self.freeze_layers = ['layers.0', 'layers.1.', 'layers.2', 'layers.3', 'layers.4', 'layers.5', 'layers.6','layers.7', 'layers.8', 'layers.9', 'layers.10','layers.11']

        self.freeze_layer_nums = len(self.freeze_layers)


        self.subword_weight_mode = kwargs['subword_weight_mode']


class KebioConfig(BertConfig):
    """Configuration for `KebioModel`."""

    def __init__(self, vocab_size, num_entities, **kwargs):
        super(KebioConfig, self).__init__()
        self.vocab_size = vocab_size
        self.num_entities = num_entities

        self.max_mentions = 50
        self.max_candidate_entities = 100
        self.hidden_size = 768
        self.use_n_gpu = kwargs['use_n_gpu']
        self.which_model = kwargs['which_model']

        self.entity_size = 100  # 这个应该是entity embedding dim
        self.num_hidden_layers = 12
        self.num_context_layers = 8
        self.num_attention_heads = 12
        self.intermediate_size = 3072
        self.hidden_act = "gelu"
        self.model_type = "bert"
        self.pad_token_id = 0
        self.layer_norm_eps = 1e-12
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512

        # 这个是token_type_embedding的token_type个数，很神奇...
        self.type_vocab_size = 2
        self.initializer_range = 0.02
        self.gradient_accumulation_steps = kwargs['gradient_accumulation_steps']

        self.bert_dir = kwargs['bert_dir']
        self.bert_name = kwargs['bert_name']
        self.run_type = kwargs['run_type']

        # 这个是选择数据集name
        self.ner_dataset_name = kwargs['ner_dataset_name']
        # 这个选择使用哪个路径下的数据集
        self.task_name = kwargs['task_name']  # original_dataset,NERdata
        self.model_name = kwargs['model_name']
        self.decoder_layer = kwargs['decoder_layer']
        # BERT model的 可选择的参数：[bert_span,bert_mlp,bert_bilstm_crf,globalpointer]
        # Normal model的可选择参数： [bilstm_crf,att_bilstm_crf,bilstm_globalpointer,att_bilstm_globalpointer]

        self.batch_size = kwargs['batch_size']
        self.gpu_id = kwargs['gpu_id']
        self.span_loss_type = kwargs['span_loss_type']

        self.warmup_proportion = kwargs['warmup_proportion']  # 学习率调整
        self.weight_decay = 0.01
        self.other_lr = kwargs['other_lr']
        self.max_len = kwargs['max_len']  # BC5CDR-disease的文本最长为121
        self.entity_type = kwargs['entity_type']  # BC5CDR-disease的文本最长为121
        self.learning_rate = kwargs['learning_rate']
        self.adam_epsilon = 1e-8
        self.dropout_prob = kwargs['dropout_prob']  # 设置bert的dropout
        self.max_grad_norm = kwargs['max_grad_norm']

        self.freeze_bert = kwargs['freeze_bert']
        self.freeze_layers = ['layer.0', 'layer.1.', 'layer.2', 'layer.3', 'layer.4', 'layer.5', 'layer.6', 'layer.7']

        self.ema_decay = 0.999

        self.subword_weight_mode = kwargs['subword_weight_mode']

        if self.ner_dataset_name in ['jnlpba','multi_jnlpba','origin_jnlpba']:
            self.num_entity_class = 5
        elif self.ner_dataset_name == 'AllDataset':
            self.num_entity_class = 7
        elif self.ner_dataset_name in ['Balance_AllDataset', 'Normalize_AllDataset', 'Normalize_Balance_AllDataset']:
            self.num_entity_class = 6
        elif self.ner_dataset_name in ['multi_alldataset','multi_all_dataset_v1_lite','multi_all_dataset_large']:
            self.num_entity_class = 8
        elif self.ner_dataset_name in ['BC6ChemProt', 'BC7DrugProt']:
            self.num_entity_class = 2
        else:
            self.num_entity_class = 1

        self.model_name = kwargs['model_name']
        self.seed = kwargs['seed']

        self.over_fitting_rate = kwargs['over_fitting_rate']  # 这个用于表示训练集的f1和验证集f1之间的差距，如果为1表示不会限制
        self.over_fitting_epoch = kwargs['over_fitting_epoch']
        self.early_stop = kwargs['early_stop']

        self.use_gpu = kwargs['use_gpu']
        self.use_fp16 = kwargs['use_fp16']
        self.use_scheduler = kwargs['use_scheduler']

        self.fixed_batch_length = kwargs['fixed_batch_length']  # 这个参数控制batch的长度是否固定

        self.logfile_name = kwargs['logfile_name']
        self.logs_dir = './outputs/logs/{}/{}/'.format(self.model_name, self.ner_dataset_name)
        self.tensorboard_dir = './outputs/tensorboard/{}/{}/tensorboard/'.format(self.model_name, self.ner_dataset_name)
        self.output_dir = './outputs/save_models/{}/{}/'.format(self.model_name, self.ner_dataset_name)
        # 最后模型对文本预测的结果存放点
        self.predicate_dir = './outputs/predicate_outputs/{}/{}/'.format(self.model_name, self.ner_dataset_name)

        # NERdata，original_dataset
        self.train_file_path = './NERdata/{}/train.txt'.format(self.ner_dataset_name)
        self.dev_file_path = './NERdata/{}/dev.txt'.format(self.ner_dataset_name)
        self.test_file_path = './NERdata/{}/test.txt'.format(self.ner_dataset_name)
        self.data_dir = './NERdata/{}'.format(self.ner_dataset_name)
        # 更换数据集，只需要修改的部分
        if 'crf' in self.model_name or 'mlp' in self.model_name or 'test' in self.model_name:
            self.crf_label2id_path = './NERdata/{}/crf_label2id.json'.format(self.ner_dataset_name)
            self.crf_label2id = json.load(open(self.crf_label2id_path, 'r', encoding='utf-8'))
            self.crf_id2label_path = './NERdata/{}/crf_id2label.json'.format(self.ner_dataset_name)
            self.crf_id2label = json.load(open(self.crf_id2label_path, 'r', encoding='utf-8'))
            self.crf_id2label = {int(key): value for key, value in self.crf_id2label.items()}
            # 这个用于crf的classes，这个就是label2id的个数
            self.num_crf_class = len(self.crf_label2id)
        elif 'globalpointer' in self.model_name:
            self.globalpointer_label2id_path = './NERdata/{}/globalpointer_label2id.json'.format(self.ner_dataset_name)
            self.globalpointer_label2id = json.load(open(self.globalpointer_label2id_path, 'r', encoding='utf-8'))
            self.globalpointer_id2label_path = './NERdata/{}/globalpointer_id2label.json'.format(self.ner_dataset_name)
            self.globalpointer_id2label = json.load(open(self.globalpointer_id2label_path, 'r', encoding='utf-8'))
            self.globalpointer_id2label = {int(key): value for key, value in self.globalpointer_id2label.items()}
            # 这个用于globalpointer的classes，这个就是实体类别的个数
            self.num_gp_class = self.num_entity_class  # 这是对于globalpointer模型,就是实体类别个数
        elif 'span' in self.model_name:
            self.span_label2id_path = './NERdata/{}/span_label2id.json'.format(self.ner_dataset_name)
            self.span_label2id = json.load(open(self.span_label2id_path, 'r', encoding='utf-8'))
            self.span_id2label_path = './NERdata/{}/span_id2label.json'.format(self.ner_dataset_name)
            self.span_id2label = json.load(open(self.span_id2label_path, 'r', encoding='utf-8'))
            self.span_id2label = {int(key): value for key, value in self.span_id2label.items()}
            # span指针的classses
            # 这个是表示实体类别的个数，但是要包括非实体O，例如实体类别有疾病、化学、蛋白质，那么类别个数为3+1=4
            self.num_span_class = self.num_entity_class + 1

        # 对数据集进行排序
        self.use_sort = kwargs['use_sort']
        self.use_fp16 = kwargs['use_fp16']
        # 评价方式，micro或者macro
        self.evaluate_mode = kwargs['evaluate_mode']
        self.use_ema = kwargs['use_ema']
        self.verbose = kwargs['verbose']

        self.num_epochs = kwargs['num_epochs']

        self.metric_summary_writer = kwargs['metric_summary_writer']
        self.parameter_summary_writer = kwargs['parameter_summary_writer']

        self.print_step = kwargs['print_step']
        self.save_model = kwargs['save_model']
        self.use_wandb = kwargs['use_wandb']

        self.predicate_flag = False  # 在训练和验证的时候都是False，只有面对无标签的测试集才会打开
        self.debug = kwargs['debug']

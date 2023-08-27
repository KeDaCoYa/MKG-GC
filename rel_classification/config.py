# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2021/12/02
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/02: 
-------------------------------------------------
"""
from ipdb import set_trace
from transformers import BertConfig, PretrainedConfig


class BaseConfig:
    def __init__(self, **kwargs):
        """

        :param gpu_ids:
        :param task_type:
        :param ner_type:
        :param model_name: 这个自由写名字，没啥区别
        """
        # 这个是选择数据集name
        self.dataset_type = kwargs['dataset_type']  # general_domain_dataset,original_dataset,REdata
        # 这个选择使用哪个路径下的数据集
        self.dataset_name = kwargs['dataset_name']
        self.model_name = kwargs['model_name']

        self.seed = kwargs['seed']

        self.other_lr = kwargs['other_lr']

        self.over_fitting_rate = kwargs['over_fitting_rate']  # 这个用于表示训练集的f1和验证集f1之间的差距，如果为1表示不会限制
        self.over_fitting_epoch = kwargs['over_fitting_epoch']
        self.use_parameter_summary_writer = kwargs['use_parameter_summary_writer']
        self.use_metric_summary_writer = kwargs['use_metric_summary_writer']
        self.use_gpu = kwargs['use_gpu']
        self.use_n_gpu = kwargs['use_n_gpu']
        self.gpu_id = kwargs['gpu_id']


        self.num_labels = kwargs['num_labels']

        self.fixed_batch_length = kwargs['fixed_batch_length']  # 这个参数控制batch的长度是否固定

        self.logfile_name = kwargs['logfile_name']
        self.logs_dir = './outputs/logs/{}/{}/{}/'.format(self.model_name, self.dataset_name,kwargs['scheme'])
        self.tensorboard_dir = './outputs/tensorboard/{}/{}/'.format(self.model_name, self.dataset_name)
        
        # 最后模型对文本预测的结果存放点
        self.predicate_dir = './outputs/predicate_outputs/{}/{}/'.format(self.model_name, self.dataset_name)
        
        # original_dataset
        self.train_file_path = './{}/{}/mid_dataset/train/sentences.txt'.format(self.dataset_type, self.dataset_name)
        self.train_labels_path = './{}/{}/mid_dataset/train/labels.txt'.format(self.dataset_type, self.dataset_name)
        self.dev_file_path = './{}/{}/mid_dataset/dev/sentences.txt'.format(self.dataset_type, self.dataset_name)
        self.dev_labels_path = './{}/{}/mid_dataset/dev/labels.txt'.format(self.dataset_type, self.dataset_name)

        self.class_type = kwargs['class_type']
        if self.class_type == 'multi':
            self.train_mtb_path = './{}/{}/multi_mid_dataset/train/mtb_train.txt'.format(self.dataset_type, self.dataset_name)
            self.train_normal_path = './{}/{}/multi_mid_dataset/train/normal_train.txt'.format(self.dataset_type,self.dataset_name)
            self.dev_mtb_path = './{}/{}/multi_mid_dataset/dev/mtb_dev.txt'.format(self.dataset_type, self.dataset_name)
            self.dev_normal_path = './{}/{}/multi_mid_dataset/dev/normal_dev.txt'.format(self.dataset_type, self.dataset_name)
            self.relation_labels = './{}/{}/multi_mid_dataset/labels.txt'.format(self.dataset_type, self.dataset_name)
            self.output_dir = './outputs/save_models/{}_schema_{}/multi/{}/'.format(self.model_name, kwargs['scheme'],
                                                                                            self.dataset_name)
            self.test_mtb_path = './{}/{}/multi_mid_dataset/test/mtb_test.txt'.format(self.dataset_type, self.dataset_name)
            self.test_normal_path = './{}/{}/multi_mid_dataset/test/normal_test.txt'.format(self.dataset_type,
                                                                                      self.dataset_name)


        elif self.class_type == 'single':
            self.train_mtb_path = './{}/{}/single_mid_dataset/train/mtb_train.txt'.format(self.dataset_type,
                                                                                         self.dataset_name)
            self.train_normal_path = './{}/{}/single_mid_dataset/train/normal_train.txt'.format(self.dataset_type,
                                                                                               self.dataset_name)
            self.dev_mtb_path = './{}/{}/single_mid_dataset/dev/mtb_dev.txt'.format(self.dataset_type, self.dataset_name)
            self.dev_normal_path = './{}/{}/single_mid_dataset/dev/normal_dev.txt'.format(self.dataset_type,
                                                                                         self.dataset_name)
            self.test_mtb_path = './{}/{}/single_mid_dataset/test/mtb_test.txt'.format(self.dataset_type,
                                                                                    self.dataset_name)
            self.test_normal_path = './{}/{}/single_mid_dataset/test/normal_test.txt'.format(self.dataset_type,
                                                                                          self.dataset_name)
            self.relation_labels = './{}/{}/single_mid_dataset/labels.txt'.format(self.dataset_type, self.dataset_name)
            self.output_dir = './outputs/save_models/{}_schema_{}/single/{}/'.format(self.model_name, kwargs['scheme'],
                                                                              self.dataset_name)
        else:
            self.train_mtb_path = './{}/{}/mid_dataset/train/mtb_train.txt'.format(self.dataset_type,
                                                                                         self.dataset_name)
            self.train_normal_path = './{}/{}/mid_dataset/train/normal_train.txt'.format(self.dataset_type,
                                                                                               self.dataset_name)
            self.dev_mtb_path = './{}/{}/mid_dataset/dev/mtb_dev.txt'.format(self.dataset_type, self.dataset_name)
            self.dev_normal_path = './{}/{}/mid_dataset/dev/normal_dev.txt'.format(self.dataset_type,self.dataset_name)

            self.test_mtb_path = './{}/{}/mid_dataset/test/mtb_test.txt'.format(self.dataset_type, self.dataset_name)
            self.test_normal_path = './{}/{}/mid_dataset/test/normal_test.txt'.format(self.dataset_type,self.dataset_name)

            self.relation_labels = './{}/{}/mid_dataset/labels.txt'.format(self.dataset_type, self.dataset_name)
            self.output_dir = './outputs/save_models/{}_schema_{}/{}/'.format(self.model_name, kwargs['scheme'],
                                                                              self.dataset_name)
        
        # 对数据集进行排序
        self.use_sort = kwargs['use_sort']
        # 评价方式，micro或者macro
        self.evaluate_mode = kwargs['evaluate_mode']
        self.use_ema = kwargs['use_ema']
        self.use_scheduler = kwargs['use_scheduler']
        self.train_verbose = kwargs['train_verbose']

        self.num_epochs = kwargs['num_epochs']

        self.print_step = kwargs['print_step']
        self.use_wandb = kwargs['use_wandb']
        self.debug = kwargs['debug']

        self.save_model = kwargs['save_model']


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
        hidden_act="silu",
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
        :param gpu_ids:
        :param task_type:
        :param ner_type:
        '''
        super(MyBertConfig, self).__init__(**kwargs)
        super(BaseConfig,self).__init__(**kwargs)
        super(BertConfig,self).__init__(**kwargs)
        super(FLASHConfig,self).__init__(**kwargs)

        self.bert_name = kwargs['bert_name']
        self.run_type = kwargs['run_type']
        self.bert_dir = kwargs['bert_dir']

        self.batch_size = kwargs['batch_size']

        self.warmup_proportion = kwargs['warmup_proportion']  # 学习率调整的step
        self.weight_decay = 0.01

        self.max_len = kwargs['max_len']
        self.bert_lr = kwargs['bert_lr']
        self.gradient_accumulation_steps = kwargs['gradient_accumulation_steps']
        self.use_fp16 = kwargs['use_fp16']

        self.adam_epsilon = 1e-8
        self.dropout_prob = kwargs['dropout_prob']  # 设置bert的dropout


        self.freeze_bert = kwargs['freeze_bert']
        if self.bert_name in ['biobert', 'wwm_bert', 'bert', 'scibert']:
            self.freeze_layers = ['layer.0', 'layer.1.', 'layer.2', 'layer.3']
        elif self.bert_name in ['flash', 'flash_quad']:
            self.freeze_layers = ['layer.0', 'layer.1.', 'layer.2', 'layer.3', 'layer.4','layer.5', 'layer.6', 'layer.7']
        else:
            raise ValueError



        self.freeze_layer_nums = len(self.freeze_layers)

        self.ema_decay = 0.999
        self.subword_weight_mode = kwargs['subword_weight_mode']
        # 实体的

        self.scheme = kwargs['scheme']
        self.data_format = kwargs['data_format']

        self.ent1_start_tag = '[s1]'
        self.ent1_end_tag = '[e1]'
        self.ent2_start_tag = '[s2]'
        self.ent2_end_tag = '[e2]'
        self.special_tags = [self.ent1_start_tag, self.ent1_end_tag, self.ent2_start_tag, self.ent2_end_tag]
        self.total_special_toks = 3


class KebioConfig(BertConfig):
  """Configuration for `KebioModel`."""

  def __init__(self,
               vocab_size,
               num_entities,
               max_mentions=15,
               max_candidate_entities=100,
               hidden_size=768,
               entity_size=50,
               num_hidden_layers=12,
               num_context_layers=8,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02, **kwargs):
    super(KebioConfig, self).__init__(vocab_size=vocab_size,
                                      hidden_size=hidden_size,
                                      num_hidden_layers=num_hidden_layers,
                                      num_attention_heads=num_attention_heads,
                                      intermediate_size=intermediate_size,
                                      hidden_act=hidden_act,
                                      hidden_dropout_prob=hidden_dropout_prob,
                                      attention_probs_dropout_prob=attention_probs_dropout_prob,
                                      max_position_embeddings=max_position_embeddings,
                                      type_vocab_size=type_vocab_size,
                                      initializer_range=initializer_range, **kwargs)
    self.num_context_layers = num_context_layers
    self.entity_size = entity_size
    self.num_entities = num_entities
    self.max_mentions = max_mentions
    self.max_candidate_entities = max_candidate_entities

class MyKebioConfig(BaseConfig,BertConfig):
  """Configuration for `KebioModel`."""

  def __init__(self,
               vocab_size=28895,
               num_entities=477039,
               max_mentions=50,
               max_candidate_entities=100,
               hidden_size=768,
               entity_size=100,
               num_hidden_layers=12,
               num_context_layers=8,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=2,
               initializer_range=0.02, **kwargs):
    BaseConfig.__init__(self, **kwargs)
    BertConfig.__init__(self,vocab_size=vocab_size,
                                      hidden_size=hidden_size,
                                      num_hidden_layers=num_hidden_layers,
                                      num_attention_heads=num_attention_heads,
                                      intermediate_size=intermediate_size,
                                      hidden_act=hidden_act,
                                      hidden_dropout_prob=hidden_dropout_prob,
                                      attention_probs_dropout_prob=attention_probs_dropout_prob,
                                      max_position_embeddings=max_position_embeddings,
                                      type_vocab_size=type_vocab_size,
                                      initializer_range=initializer_range, **kwargs)


    self.num_context_layers = num_context_layers
    self.entity_size = entity_size
    self.num_entities = num_entities
    self.max_mentions = max_mentions
    self.max_candidate_entities = max_candidate_entities

    self.class_name = 'KeBioLM Config'

    self.bert_name = kwargs['bert_name']
    self.run_type = kwargs['run_type']
    self.bert_dir = kwargs['bert_dir']

    self.batch_size = kwargs['batch_size']

    self.warmup_proportion = kwargs['warmup_proportion'] # 学习率调整的step
    self.weight_decay = 0.01

    self.max_len = kwargs['max_len']  # BC5CDR-disease的文本最长为121
    self.bert_lr = kwargs['bert_lr']

    self.adam_epsilon = 1e-8
    self.dropout_prob = kwargs['dropout_prob']  # 设置bert的dropout

    self.freeze_bert = kwargs['freeze_bert']
    self.freeze_layers = ['layer.0', 'layer.1.', 'layer.2', 'layer.3', 'layer.4', 'layer.5', 'layer.6']
    self.freeze_layer_nums = len(self.freeze_layers)

    self.ema_decay = 0.999
    self.subword_weight_mode = kwargs['subword_weight_mode']
    # 实体的

    self.scheme = kwargs['scheme']
    self.data_format = kwargs['data_format']

    self.ent1_start_tag = '[s1]'
    self.ent1_end_tag = '[e1]'
    self.ent2_start_tag = '[s2]'
    self.ent2_end_tag = '[e2]'
    self.special_tags = [self.ent1_start_tag, self.ent1_end_tag, self.ent2_start_tag, self.ent2_end_tag]
    self.total_special_toks = 3

class NormalConfig(BaseConfig):
    def __init__(self, **kwargs):
        super(NormalConfig, self).__init__(**kwargs)

        self.attention_mechanism = kwargs['attention_mechanism']  # ['mha','normal','sa']
        self.use_pretrained_embedding = kwargs['use_pretrained_embedding']

        self.batch_size = kwargs['batch_size']

        # 这个注意也要修改
        self.max_len = kwargs['max_len']

        self.lstm_pack_unpack = kwargs['lstm_pack_unpack']

        # 针对OOV问题的PAD和UNK,表示会使用的坐标(最然我觉得这两个差别应该不大)
        self.PAD = 0
        self.UNK = 1

        # word embedding dim根据预训练的结果
        self.word2vec_embedding_path = '../embedding/bionlp_embed/word2vec_optimized_extrinsic/word2vec_optimized_extrinsic'
        self.fast_embedding_path = '../embedding/bionlp_embed/fasttext_optimized_extrinsic/fasttext_optimized_extrinsic'
        self.embedding_type = kwargs['embedding_type']  # [fasttext,word2vec]

        # word2idx中的个数，在使用时再赋值
        self.vocab_size = 0

        self.other_lr = kwargs['other_lr']

        self.ema_decay = 0.999

        # 网络架构参数
        # 这是embedding的dim，但是一般由pretrained embedding来决定
        self.word_embedding_dim = 50
        self.freeze_embeddings = False  # 是否对pretrained embedding进行微调
        self.char_embedding_dim = 25  # character embedding一般是自己学习，并不需要pretrained character embedding
        self.lstm_hidden_dim = 128

        self.char_size = 100  # 表示有多少个字符
        self.dropout_prob = kwargs['dropout_prob']
        self.num_bilstm_layers = kwargs['num_bilstm_layers']  # 表示bilstm的层数

        # PCNN参数设置
        self.filter_num = 128  # 卷积核的个数
        self.position_embedding_dim = 10
        self.pos_dis_limit = 50
        self.filters = [2, 3, 4, 5]
        self.weight_decay = 1e-5

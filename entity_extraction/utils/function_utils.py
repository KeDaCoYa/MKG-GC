# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2021/11/08
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/11/08: 
-------------------------------------------------
"""
import argparse
import os
import random
import datetime
import logging
import time
from collections import defaultdict

import numpy as np
import torch
from ipdb import set_trace
from prettytable import PrettyTable
from torch.utils.data import DataLoader

from src.dataset_util.base_dataset import NERProcessor
from src.dataset_util.bert_crf_dataset import BertCRFDataset_dynamic, BertCRFDataset_test
from src.dataset_util.bert_globalpointer_dataset import BertGlobalPointerDataset
from src.dataset_util.bert_span_dataset import BertSpanDataset_dynamic

from src.evaluate.evaluate_globalpointer import GlobalPointerMetrics

from src.models.bert_bilstm_crf import Bert_BiLSTM_CRF
from src.models.bert_bilstm_span import InterBertBiLSTMSpan
from src.models.bert_crf import BertCRF
from src.models.bert_globalpointer import BertGlobalPointer
from src.models.bert_mlp import BertMLP, BertTest
from src.models.bert_span import Bert_Span
from src.models.inter_bert_span import InterBertSpan
from src.models.multi_binary_inter_gru_span import MultiBinaryInterGRUSpanForEight

from src.models.multi_binary_span import MultiBinarySpanForBinary, MultiBinarySpanForEight, MultiBinarySpanForFour, \
    MultiBinarySpanForFive
from src.models.multi_span import MultiSpanForBinary, MultiSpanForFour, MultiSpanForFive, MultiSpanForEight

from utils.data_process_utils import load_pretrained_word2vec, load_pretrained_fasttext, read_data

logger = logging.getLogger("main.function_utils")


def get_logger(config):
    logger = logging.getLogger('main')
    logging.Formatter.converter = correct_datetime

    logger.setLevel(level=logging.INFO)

    if not os.path.exists(config.logs_dir):
        os.makedirs(config.logs_dir)

    now = datetime.datetime.now() + datetime.timedelta(hours=8)
    year, month, day, hour, minute, secondas = now.year, now.month, now.day, now.hour, now.minute, now.second
    handler = logging.FileHandler(os.path.join(config.logs_dir,
                                               '{} {}_{}_{} {}:{}:{}.txt'.format(config.logfile_name, year, month, day,
                                                                                 hour, minute, secondas)))

    handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def get_config():
    from config import MyBertConfig, KebioConfig

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False, help='是否开启debug模式，debug模式使用少量数据运行')

    parser.add_argument('--which_model', type=str, default='bert', choices=['bert', 'normal'],
                        help='选择哪个模型，normal或者bert')

    parser.add_argument('--run_type', type=str, default='normal', choices=['cv5', 'cv10', 'normal'],
                        help='选择模型的数据训练方式，是交叉验证，还是最普通的模式')

    parser.add_argument('--ner_dataset_name', type=str, default='jnlpba', help='数据集是哪个NER dataset')
    parser.add_argument('--inter_scheme', type=int, default=1, help='交互模式')
    parser.add_argument('--entity_type', type=str, default='single', help='表示实体类别的个数，要么是单类别，要么是多类别',
                        choices=['single', 'multiple'])
    parser.add_argument('--decoder_layer', type=str, default='crf',
                        help='NER模型的decode layer,目前是globalpointer,mlp,test,crf,span')

    parser.add_argument('--bert_name', type=str, default='biobert', help='这是使用的哪个预训练模型')
    parser.add_argument('--bert_dir', type=str, default='', help='预训练模型的路径')

    parser.add_argument('--model_name', type=str, default='', help='正式的模型名称，非常标准的名称')
    parser.add_argument('--task_name', type=str, default='五名字', help='给这次模型取个名字')

    parser.add_argument('--gpu_id', type=str, default='1', help='选择哪块gpu使用，0,1,2...')
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--use_n_gpu', type=bool, default=False, help='是否使用多个GPU同时训练...')
    parser.add_argument('--use_fp16', type=bool, default=False)
    parser.add_argument('--wandb_notes', type=str, default='')

    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=256, help='最大长度')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--fixed_batch_length', type=bool, default=False, help='动态batch或者根据batch修改')
    parser.add_argument('--dropout_prob', type=float, default=0.1, help='BERT使用的dropout')
    parser.add_argument('--other_lr', type=float, default=1e-4, help='BERT之外的网络学习率')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪...')

    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='warm_lienar的学习率调整期')

    parser.add_argument('--freeze_bert', type=bool, default=False, help='是否冻结bert的部分层数')

    parser.add_argument('--use_scheduler', type=bool, default=False, help='是否使用学习率调整期')
    parser.add_argument('--subword_weight_mode', type=str, default='first',
                        help='选择第一个subword作为token representation；或者是平均值', choices=['first', 'avg'])

    parser.add_argument('--span_loss_type', type=str, default='ls_ce', help='为bert_span选择损失函数')
    parser.add_argument('--use_ema', type=bool, default=False, help='是否使用EMA')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='梯度累加次数...')

    parser.add_argument('--evaluate_mode', type=str, default='micro', choices=['micro', 'macro', 'all', 'weight'],
                        help='对数据集的评估方式')

    parser.add_argument('--over_fitting_rate', type=float, default=0.15, help='验证集和训练集的f1差别在多大的时候停止')
    parser.add_argument('--over_fitting_epoch', type=int, default=5, help='表示有几个epoch没有超过最大f1则停止')
    parser.add_argument('--early_stop', type=bool, default=False, help='采用早停机制,防止过拟合')

    parser.add_argument('--metric_summary_writer', type=bool, default=False, help='是否使用SummaryWriter记录参数')
    parser.add_argument('--parameter_summary_writer', type=bool, default=False, help='是否使用SummaryWriter记录参数')

    parser.add_argument('--logfile_name', type=str, default='', help='给logfile起个名字')
    parser.add_argument('--save_model', type=bool, default=False, help='是否保存最佳模型....')
    parser.add_argument('--print_step', type=int, default=1, help='打印频次')
    parser.add_argument('--verbose', type=bool, default=False, help='是否在训练过程中每个batch显示各种值')
    parser.add_argument('--use_wandb', type=bool, default=False, help='是否使用wandb来记录训练结果')

    parser.add_argument('--use_sort', type=bool, default=False, help='对数据集按照顺序进行排序')  # 这个参数属实没用

    # ----------------下面是Normal Model专属的参数--------------------

    parser.add_argument('--embedding_type', type=str, default='word2vec')
    parser.add_argument('--use_pretrained_embedding', type=bool, default=True)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    parser.add_argument('--attention_mechanism', type=str, default='sa')

    parser.add_argument('--lstm_pack_unpack', type=bool, default=False, help='对bilstm的前向传播是否使用此方式')
    parser.add_argument('--num_bilstm_layers', type=int, default=2, help='bilstm的层数')

    args = parser.parse_args()

    ner_dataset_name = args.ner_dataset_name
    debug = args.debug
    inter_scheme = args.inter_scheme
    run_type = args.run_type
    which_model = args.which_model
    entity_type = args.entity_type
    bert_name = args.bert_name
    bert_dir = args.bert_dir
    task_name = args.task_name
    model_name = args.model_name
    seed = args.seed

    logfile_name = args.logfile_name
    save_model = args.save_model
    wandb_notes = args.wandb_notes

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    use_sort = args.use_sort
    max_len = args.max_len

    gpu_id = args.gpu_id
    use_gpu = args.use_gpu
    use_fp16 = args.use_fp16
    use_n_gpu = args.use_n_gpu

    dropout_prob = args.dropout_prob
    gradient_accumulation_steps = args.gradient_accumulation_steps
    max_grad_norm = args.max_grad_norm
    warmup_proportion = args.warmup_proportion
    use_ema = args.use_ema
    use_scheduler = args.use_scheduler

    over_fitting_rate = args.over_fitting_rate
    over_fitting_epoch = args.over_fitting_epoch
    early_stop = args.early_stop
    fixed_batch_length = args.fixed_batch_length

    metric_summary_writer = args.metric_summary_writer
    parameter_summary_writer = args.parameter_summary_writer
    print_step = args.print_step
    evaluate_mode = args.evaluate_mode

    decoder_layer = args.decoder_layer


    learning_rate = args.learning_rate


    verbose = args.verbose
    use_wandb = args.use_wandb

    # ---------针对bert model--------------
    freeze_bert = args.freeze_bert
    span_loss_type = args.span_loss_type
    subword_weight_mode = args.subword_weight_mode
    other_lr = args.other_lr

    ner_dataset_name = ner_dataset_name.strip()
    task_name = task_name.strip()
    bert_dir = bert_dir.strip()

    subword_weight_mode = subword_weight_mode.strip()
    span_loss_type = span_loss_type.strip()
    bert_name = bert_name.strip()
    decoder_layer = decoder_layer.strip()


    if bert_name in ['scibert', 'biobert', 'flash', 'flash_quad', 'wwm_bert','bert']:
        config = MyBertConfig(model_name=model_name, gpu_id=gpu_id, task_name=task_name,
                              ner_dataset_name=ner_dataset_name, num_epochs=num_epochs, batch_size=batch_size,
                              use_fp16=use_fp16, which_model=which_model, decoder_layer=decoder_layer,
                              use_sort=use_sort, early_stop=early_stop, run_type=run_type,
                              gradient_accumulation_steps=gradient_accumulation_steps,inter_scheme=inter_scheme,
                              evaluate_mode=evaluate_mode, use_ema=use_ema, over_fitting_rate=over_fitting_rate,
                              save_model=save_model, debug=debug, learning_rate=learning_rate,
                              seed=seed, other_lr=other_lr, dropout_prob=dropout_prob, logfile_name=logfile_name,
                              fixed_batch_length=fixed_batch_length, over_fitting_epoch=over_fitting_epoch,
                              max_grad_norm=max_grad_norm, metric_summary_writer=metric_summary_writer,
                              verbose=verbose, parameter_summary_writer=parameter_summary_writer, max_len=max_len,
                              use_wandb=use_wandb, use_scheduler=use_scheduler, warmup_proportion=warmup_proportion,
                              subword_weight_mode=subword_weight_mode, print_step=print_step, use_n_gpu=use_n_gpu,
                              span_loss_type=span_loss_type, bert_dir=bert_dir, bert_name=bert_name,
                              use_gpu=use_gpu,wandb_notes=wandb_notes,
                              freeze_bert=freeze_bert, entity_type=entity_type)
    elif bert_name == 'kebiolm':
        config = KebioConfig(model_name=model_name, gpu_id=gpu_id, task_name=task_name, run_type=run_type,
                             ner_dataset_name=ner_dataset_name, use_fp16=use_fp16, decoder_layer=decoder_layer,
                             num_epochs=num_epochs, batch_size=batch_size, use_sort=use_sort,
                             evaluate_mode=evaluate_mode, use_ema=use_ema, over_fitting_rate=over_fitting_rate,
                             seed=seed, early_stop=early_stop, which_model=which_model,
                             gradient_accumulation_steps=gradient_accumulation_steps,
                             parameter_summary_writer=parameter_summary_writer, learning_rate=learning_rate,
                             other_lr=other_lr, dropout_prob=dropout_prob, logfile_name=logfile_name,
                             save_model=save_model, metric_summary_writer=metric_summary_writer,
                             fixed_batch_length=fixed_batch_length, over_fitting_epoch=over_fitting_epoch,
                             verbose=verbose, max_len=max_len, use_wandb=use_wandb,
                             subword_weight_mode=subword_weight_mode, print_step=print_step,
                             max_grad_norm=max_grad_norm, debug=debug, use_scheduler=use_scheduler,
                             warmup_proportion=warmup_proportion,
                             span_loss_type=span_loss_type, entity_type=entity_type, use_n_gpu=use_n_gpu,
                             bert_dir=bert_dir, bert_name=bert_name, vocab_size=28895, num_entities=477039,
                             use_gpu=use_gpu, freeze_bert=freeze_bert)
    else:
        raise ValueError('-----------')

    return config


def print_hyperparameters(config):
    hyper_parameters = vars(config)
    logger.info('-------此次任务的名称为:{}---------'.format(config.task_name))
    for key, value in hyper_parameters.items():
        if key == 'ner_dataset_name':
            logger.info('NER数据集：{}'.format(value))
        elif key == 'bert_dir':
            logger.info('预训练模型：{}'.format(value))
        elif key == 'use_pretrained_embedding':
            if value:
                logger.info('预训练的词嵌入{}'.format(config.embedding_type))
        elif key == 'model_name':
            logger.info('模型名称:{}'.format(value))
        elif key == 'evaluate_mode':
            logger.info('评价标准:{}'.format(value))
        elif key == 'seed':
            logger.info('随机种子:{}'.format(value))
        elif key == 'batch_size':
            logger.info('batch_size:{}'.format(value))
        elif key == 'logs_dir':
            logger.info('日志保存路径:{}'.format(value))
        elif key == 'tensorboard_dir':
            logger.info('tensorboard的存储文件在:{}'.format(value))
        elif key == 'output_dir':
            logger.info('模型的保存地址:{}'.format(value))
        elif key == 'num_bilstm_layers':
            logger.info('BiLSTM的层数为{}'.format(value))
        elif key == 'use_fp16':
            if value:
                logger.info('使用fp16加速模型训练...')
        elif key == 'lstm_pack_unpack':
            if value:
                logger.info('这里BilSTM的计算采用pad_pack方式')
        elif key == 'use_gpu':
            if value:
                logger.info('显卡使用的:{}'.format(torch.cuda.get_device_name(int(config.gpu_id))))
                logger.info('显卡使用的:{}'.format((config.gpu_id)))
        elif key == 'fixed_batch_length':
            if value:
                logger.info("固定batch data的最大长度为：{}".format(config.max_len))
            else:
                logger.info('动态batch data长度，并不固定')
        elif key == 'lr':

            logger.info('BERT的学习率:{}'.format(value))
            logger.info('其他网络的学习率:{}'.format(config.other_lr))

        elif key == 'attention_mechanism' and 'att' in config.model_name:
            if value == 'sa':
                logger.info('注意力机制：Self-Attention')
            if value == 'mha':
                logger.info('注意力机制：Multi-Head Attention')
        elif key == 'use_ema':
            if value:
                logger.info('使用滑动加权平均模型')

        elif key == 'freeze_bert':
            if value:
                logger.info('冻结BERT层:{}'.format(config.freeze_layers))
        # else:
        #     logger.info("{}:{}".format(key,value))


# 初始参数设置



def choose_multi_ner_model(config, type='train'):
    """
    根据输入的条件，选择合适的模型
    这个模型只能是bert span
    """
    model_name = config.model_name
    metric = None

    if 'binary' in model_name:
        if config.ner_dataset_name in ['multi_BC6', 'multi_BC7', 'multi_BC5']:
            model = MultiBinarySpanForBinary(config)

        elif config.ner_dataset_name in['multi_dataset','multi_all_dataset_v1_lite','multi_all_dataset_large','1009abstracts','3400abstracts']:
            if model_name == 'inter_binary_bert_gru_mid_span':
                model = MultiBinaryInterGRUSpanForEight(config)
            else:
                model = MultiBinarySpanForEight(config)
        elif config.ner_dataset_name in ['multi_DDI2013']:
            model = MultiBinarySpanForFour(config)
        elif config.ner_dataset_name in ['multi_jnlpba']:
            model = MultiBinarySpanForFive(config)
    else:
        if model_name == 'bert_span' and config.ner_dataset_name in['multi_dataset','multi_all_dataset_v1_lite','multi_all_dataset_large','1009abstracts','3400abstracts'] or 'CV' in config.ner_dataset_name:

            model = MultiSpanForEight(config)
        elif model_name == 'bert_span' and config.ner_dataset_name == 'multi_jnlpba':
            model = MultiSpanForFive(config)
        elif model_name == 'bert_span' and config.ner_dataset_name in ['multi_BC6','multi_BC7','multi_BC5']:
            model = MultiSpanForBinary(config)
        elif model_name == 'bert_span' and config.ner_dataset_name in ['multi_DDI2013']:
            model = MultiSpanForFour(config)


    return model, metric


def choose_model(config, type='train'):
    """
    根据输入的条件，选择合适的模型
    """
    model_name = config.model_name
    metric = None


    if model_name == 'bert_crf':
        model = BertCRF(config)
    elif model_name == 'bert_mlp':
        model = BertMLP(config)
    elif model_name == 'bert_span':
        # 这里的num_tags=2，不能是1
        model = Bert_Span(config)
    elif model_name == 'inter_bert_span':
        model = InterBertSpan(config)
    elif model_name == 'bert_bilstm_crf':
        model = Bert_BiLSTM_CRF(config)
    elif model_name == 'bert_globalpointer':
        metric = GlobalPointerMetrics()
        model = BertGlobalPointer(config)
    elif model_name == 'bert_test':
        model = BertTest(config)
    elif model_name == 'interbertbilstmspan':
        model = InterBertBiLSTMSpan(config)
    elif model_name == 'interbergruspan':
        model = InterBertSpan(config)
    else:
        raise ValueError



    return model, metric


def get_predicate_dataset_loader(config, train_data, tokenizer):
    """
        这个是专用于多任务学习的，将entity type设为1
    """
    processor = NERProcessor(0)
    train_examples = processor.get_examples(train_data, None)
    train_dataset = BertSpanDataset_dynamic(config, train_examples, tokenizer)
    train_loader = DataLoader(dataset=train_dataset, shuffle=False, num_workers=0,
                              batch_size=config.batch_size,
                              collate_fn=train_dataset.multi_collate_fn_predicate)
    return train_dataset, train_loader


def choose_multi_dataset_and_loader(config, tokenizer=None, type_='train'):

    if config.ner_dataset_name == 'multi_all_dataset_v1_lite':
        entity_type_dict = {
            "NCBI-disease": 1,
            "BC4CHEMD": 2,
            "BC2GM": 3,
            "linnaeus": 4,
            "jnlpba-cell_line": 5,
            "jnlpba-DNA": 6,
            "jnlpba-RNA": 7,
            "jnlpba-cell_type": 8,
        }
    elif config.ner_dataset_name == 'multi_jnlpba':
        entity_type_dict = {
            "jnlpba-cell_line": 4,
            "jnlpba-protein": 2,
            "jnlpba-DNA": 1,
            "jnlpba-RNA": 5,
            "jnlpba-cell_type": 3,
        }
    elif config.ner_dataset_name in ['multi_BC6','multi_BC7']:
        entity_type_dict = {
            "chem": 1,
            "gene": 2,
        }
    elif config.ner_dataset_name in ['multi_DDI2013']:
        entity_type_dict = {
            "drug": 1,
            "drug_n": 2,
            "group": 3,
            "brand": 4,
        }
    elif config.ner_dataset_name in ['multi_BC5']:
        entity_type_dict = {
            "disease": 1,
            "chem": 2,
        }
    elif config.ner_dataset_name == 'multi_all_dataset_large' or 'CV' in config.ner_dataset_name:
        """
        使用全部的数据集
        """
        entity_type_dict = {
            "BC5CDR-disease": 1,
            "NCBI-disease": 1,
            "BC4CHEMD": 2,
            "BC5CDR-chem": 2,
            "BC6ChemProt-chem": 2,
            "BC7DrugProt-chem": 2,
            "BC2GM": 3,
            "HPRD-50": 3,
            "IEPA": 3,
            "fsu-prge": 3,
            "BC6ChemProt-gene": 3,
            "BC7DrugProt-gene": 3,
            "linnaeus": 4,
            "s800": 4,
            "CLL": 5,
            "cellus": 5,
            "jnlpba-cell_line": 5,
            "jnlpba-DNA": 6,
            "jnlpba-RNA": 7,
            "jnlpba-cell_type": 8,
        }
    elif config.ner_dataset_name == 'multi_all_dataset_plus':
        """
        使用全部的数据集
        """
        entity_type_dict = {
            "BC5CDR-disease": 1,
            "NCBI-disease": 1,
            "BC4CHEMD": 2,
            "BC5CDR-chem": 2,
            "BC2GM": 3,
            "jnlpba-protein": 3,
            "linnaeus": 4,
            "s800": 4,
            "jnlpba-cell_line": 5,
            "jnlpba-DNA": 6,
            "jnlpba-RNA": 7,
            "jnlpba-cell_type": 8,
        }
    all_examples = []
    for file in os.listdir(os.path.join(config.data_dir, type_)):

        file_name = file.split('/')[-1].split('.')[0]
        if not file.endswith('.txt') or file_name not in entity_type_dict:
            continue
        logger.info("使用数据集:{}".format(file))
        file_name = file.split('/')[-1].split('.')[0]

        processor = NERProcessor(entity_type_dict[file_name])
        train_data, train_labels = read_data(os.path.join(config.data_dir, type_, file))
        train_examples = processor.get_examples(train_data, train_labels)
        all_examples.extend(train_examples)

    # 现在就对所有数据进行随机打乱
    random.shuffle(all_examples)
    if config.debug:
        all_examples = all_examples[:10 * config.batch_size]
    train_dataset = BertSpanDataset_dynamic(config, all_examples, tokenizer)

    if type_ == 'predicate':
        train_loader = DataLoader(dataset=train_dataset, shuffle=False, num_workers=0,
                                  batch_size=config.batch_size,
                                  collate_fn=train_dataset.multi_collate_fn_tokenize)
    else:
        train_loader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=0,
                                  batch_size=config.batch_size,
                                  collate_fn=train_dataset.multi_collate_fn_tokenize)

    return train_dataset, train_loader


def get_normalize_all_dataset(config, tokenizer=None, type='train'):
    """
        专用于多任务学习NER
        当
    """
    entity_type_dict = {
        "BC5CDR-disease": 1,
        "NCBI-disease": 1,
        "scai": 1,
        "BC4CHEMD": 2,
        "BC5CDR-chem": 2,
        "BC6ChemProt-chem": 2,
        "BC7DrugProt-chem": 2,
        "BC2GM": 3,
        "HPRD-50": 3,
        "IEPA": 3,
        "jnlpba-protein": 3,
        "fsu-prge": 3,
        "BC6ChemProt-gene": 3,
        "BC7DrugProt-gene": 3,
        "linnaeus": 4,
        "s800": 4,
        "CLL": 5,
        "cellus": 5,
        "jnlpba-cell_line": 5,
        "jnlpba-DNA": 6,
        "jnlpba-RNA": 7,
        "jnlpba-cell_type": 8,
    }
    all_examples = []
    for file in os.listdir(os.path.join(config.data_dir, type)):

        file_name = file.split('/')[-1].split('.')[0]
        if not file.endswith('.txt') or file_name not in entity_type_dict:
            continue
        logger.info("使用数据集:{}".format(file))
        file_name = file.split('/')[-1].split('.')[0]

        processor = NERProcessor(entity_type_dict[file_name])
        train_data, train_labels = read_data(os.path.join(config.data_dir, type, file))
        train_examples = processor.get_examples(train_data, train_labels)
        all_examples.extend(train_examples)

    # 现在就对所有数据进行随机打乱
    random.shuffle(all_examples)
    if config.debug and type == 'train':
        all_examples = all_examples[:5 * config.batch_size]
    train_dataset = BertSpanDataset_dynamic(config, all_examples, tokenizer)
    if type == 'predicate':
        train_loader = DataLoader(dataset=train_dataset, shuffle=False, num_workers=0,
                                  batch_size=config.batch_size,
                                  collate_fn=train_dataset.multi_collate_fn_tokenize)
    else:
        train_loader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=0,
                                  batch_size=config.batch_size,
                                  collate_fn=train_dataset.multi_collate_fn_tokenize)

    return train_dataset, train_loader


def choose_dataset_and_loader(config, train_data, train_labels, tokenizer=None, word2id=None, type_='train'):
    """
    根据输入，选择dataset和loader
    """
    # todo:这里变量名需要修改
    processor = NERProcessor()
    train_examples = processor.get_examples(train_data, train_labels)
    model_name = config.model_name
    # 只有模型在训练的时候才shuffle
    if type_ == 'train':
        shuffle = True
    else:
        shuffle = False


    if model_name == 'bert_crf' or model_name == 'bert_mlp' or model_name == 'bert_bilstm_crf':

        train_dataset = BertCRFDataset_dynamic(config, train_examples, tokenizer)
        if type_ == 'predicate':
            train_loader = DataLoader(dataset=train_dataset, shuffle=shuffle, num_workers=0,
                                      batch_size=config.batch_size,
                                      collate_fn=train_dataset.collate_fn_predicate)
        else:
            train_loader = DataLoader(dataset=train_dataset, shuffle=shuffle, num_workers=0,
                                      batch_size=config.batch_size,
                                      collate_fn=train_dataset.collate_fn)

    elif model_name in ['bert_span','inter_bert_span','interbertbilstmspan','interbergruspan']:
        train_dataset = BertSpanDataset_dynamic(config, train_examples, tokenizer)
        if type_ == 'predicate':
            train_loader = DataLoader(dataset=train_dataset, shuffle=shuffle, num_workers=0,
                                      batch_size=config.batch_size,
                                      collate_fn=train_dataset.collate_fn_predicate)
        else:
            train_loader = DataLoader(dataset=train_dataset, shuffle=shuffle, num_workers=0,
                                      batch_size=config.batch_size,
                                      collate_fn=train_dataset.collate_fn_tokenize)
    elif model_name == 'bert_globalpointer':
        train_dataset = BertGlobalPointerDataset(data=train_examples, tokenizer=tokenizer, config=config)
        if type_ == 'predicate':
            train_loader = DataLoader(dataset=train_dataset, shuffle=shuffle, num_workers=0,
                                      batch_size=config.batch_size, collate_fn=train_dataset.collate_predicate)
        else:
            train_loader = DataLoader(dataset=train_dataset, shuffle=shuffle, num_workers=0,
                                      batch_size=config.batch_size,
                                      collate_fn=train_dataset.collate_tokenize)
    elif model_name == 'bert_test':
        train_dataset = BertCRFDataset_test(config, train_examples, tokenizer)
        if type_ == 'predicate':
            train_loader = DataLoader(dataset=train_dataset, num_workers=0,
                                      batch_size=config.batch_size, shuffle=shuffle,
                                      collate_fn=train_dataset.collate_fn_predicate)
        else:
            train_loader = DataLoader(dataset=train_dataset, shuffle=shuffle, num_workers=0,
                                      batch_size=config.batch_size,
                                      collate_fn=train_dataset.collate_fn)
    else:
        raise ValueError

    return train_dataset, train_loader

def get_wandb_name(config):
    if config.freeze_bert:
        if config.use_scheduler:
            wandb_name = f'{config.bert_name}_{config.model_name}_epochs{config.num_epochs}_free_nums{config.freeze_layer_nums}_scheduler{config.warmup_proportion}_bs{config.batch_size}_lr{config.learning_rate}_mx{config.max_len}'
        else:
            wandb_name = f'{config.bert_name}_{config.model_name}_epochs{config.num_epochs}_free_nums{config.freeze_layer_nums}_bs{config.batch_size}_lr{config.learning_rate}_mx{config.max_len}'

    else:
        if config.use_scheduler:
            wandb_name = f'{config.bert_name}_{config.model_name}_epochs{config.num_epochs}_no_free_scheduler{config.warmup_proportion}_bs{config.batch_size}_lr{config.learning_rate}_mx{config.max_len}'
        else:
            wandb_name = f'{config.bert_name}_{config.model_name}_epochs{config.num_epochs}_no_free_bs{config.batch_size}_lr{config.learning_rate}_mx{config.max_len}'

    if 'inter' in config.model_name:
        wandb_name = 'inter{}_'.format(config.inter_scheme) + wandb_name
    if 'mlp' in config.model_name or 'crf' in config.model_name:
        wandb_name = 'mid'+wandb_name
    if 'CV' in config.ner_dataset_name:
        wandb_name = config.logfile_name + '_' + wandb_name
        project_name = "实体抽取-eight_type_alldataset"
    else:
        project_name = "实体抽取-{}".format(config.ner_dataset_name)
    return wandb_name,project_name
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def correct_datetime(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def count_parameters(model):
    '''
    统计参数量，统计需要梯度更新的参数量，并且参数不共享
    :param model:
    :return:
    '''
    requires_grad_nums = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameter_nums = sum(p.numel() for p in model.parameters())
    return requires_grad_nums,parameter_nums



def gentl_print(gentl_li, average='micro'):
    tb = PrettyTable()
    tb.field_names = ['实体类别', '{}-f1'.format(average), '{}-precision'.format(average), '{}-recall'.format(average),
                      'support']
    for a, f1, p, r, support in gentl_li:
        tb.add_row([a, round(f1, 5), round(p, 5), round(r, 5), support])

    logger.info(tb)


def gentl_print_confusion_matrix(gentl_li):
    """
    这个只是针对TP,TN,FP,FN的
    """
    tb = PrettyTable()
    tb.field_names = ['实体类别', 'TP', 'FP', 'FN', 'support']
    for a, TP, FP, FN, support in gentl_li:
        tb.add_row([a, TP, FP, FN, support])

    logger.info(tb)


def final_print_score(train_res_li, dev_res_li):
    for i in range(len(train_res_li)):
        logger.info('epoch{} 训练集 f1:{:.5f},p:{:.5f},r:{:.5f}'.format(i + 1, train_res_li[i]['f1'],
                                                                     train_res_li[i]['p'],
                                                                     train_res_li[i]['r']))
        logger.info('        验证集 f1:{:.5f},p:{:.5f},r:{:.5f}'.format(dev_res_li[i]['f1'],
                                                                     dev_res_li[i]['p'],
                                                                     dev_res_li[i]['r']))


def get_type_weights(labels):
    '''
    计算每种实体类别对应的权重，方便之后的评估
    :param labels:如果是crf，则只需要根据BIO进行统计
    :return:
    '''

    type_weights = defaultdict(int)
    count = 0
    for label in labels:
        for word in label:
            if word != 'O':
                BIO_format, entity_type = word.split('-')
                if BIO_format == 'B':
                    type_weights[entity_type] += 1
                    count += 1
    for key, value in type_weights.items():
        type_weights[key] = value / count
    return type_weights


def argsort(seq):
    '''
    这里seq传入的是一个列表，其值为列表的长度，但是都是负数
    '''
    # 这里也可以修改为如下代码
    res = sorted(range(len(seq)), key=lambda x: seq[x])
    return sorted(range(len(seq)), key=seq.__getitem__)


def argsort_sequences_by_lens(list_in):
    '''
    这个函数是对list_in按照序列长度进行排序,这个list_in的值就是序列的长度
    sort_indices为从大到小，对应句子index的list
    '''
    data_num = len(list_in)
    # 这里得到的就是从大到小的indice
    sort_indices = argsort([-len(item) for item in list_in])
    reverse_sort_indices = [-1 for _ in range(data_num)]
    for i in range(data_num):
        reverse_sort_indices[sort_indices[i]] = i
    # 这个reverse_sort_indices就是sort_indices的排序，正好相反
    return sort_indices, reverse_sort_indices


def BIO_decode_json(text, pred_token):
    '''
    将BIO的结果进行decode,得到json结果
    这个函数也可以看作是BIO数据和json数据的转换
    :param text:  这是一句话
    :param pred_token: 这是text这句话对应的BIO标签 ['O', 'B-cell_type', 'I-cell_type', 'O']
    :return:
    '''

    entities = []
    start_index = 0
    actual_len = len(text)
    while start_index < actual_len:
        cur_label = pred_token[start_index].split('-')
        if len(cur_label) == 2:
            # 这个entity type和entity type id是全局的数据
            BIO_format, entity_type = cur_label

        else:
            BIO_format = cur_label[0]
            entity_type = 'One'

        if start_index + 1 < actual_len:
            next_label = pred_token[start_index + 1].split('-')
            if len(next_label) == 2:
                BIO_, _ = next_label
            elif len(next_label) == 1:
                BIO_ = next_label[0]

        if BIO_format == 'B' and start_index + 1 < actual_len and BIO_ == 'O':  # 实体是一个单词
            entities.append({
                'start_idx': start_index,
                'end_idx': start_index,
                'type': entity_type,
                'entity': text[start_index]

            })
            start_index += 1
        elif BIO_format == 'B' and start_index + 1 >= actual_len:  # 最后只有一个实体，并且只有一个单词，到达了最后
            entities.append({
                'start_idx': start_index,
                'end_idx': start_index,
                'type': entity_type,
                'entity': text[start_index]
            })
            break
        elif BIO_format == 'B':
            j = start_index + 1
            while j < actual_len:
                j_label = pred_token[j].split('-')
                if len(j_label) == 2:
                    BIO_, _ = j_label
                elif len(j_label) == 1:
                    BIO_ = j_label[0]

                if BIO_ == 'I':
                    j += 1
                else:
                    entities.append({
                        'start_idx': start_index,
                        'end_idx': j - 1,
                        'type': entity_type,
                        'entity': " ".join(text[start_index:j])
                    })

                    break
            if j >= actual_len:
                j_label = pred_token[j - 1].split('-')
                if len(j_label) == 2:
                    BIO_, _ = j_label
                elif len(j_label) == 1:
                    BIO_ = j_label[0]

                if BIO_ == 'I':
                    entities.append({
                        'start_idx': start_index,
                        'end_idx': j - 1,
                        'type': entity_type,
                        'entity': " ".join(text[start_index:j])
                    })

            start_index = j
        else:
            start_index += 1
    return entities


def load_model_and_parallel(model, gpu_ids, ckpt_path=None, strict=True, load_type='one2one'):
    """
    加载模型 & 放置到 GPU 中（单卡 / 多卡）
    主要是多卡模型训练和保存的问题...
    load_type:表示加载模型的类别，one2one,one2many,many2one,many2many
    """
    gpu_ids = gpu_ids.split(',')
    device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])

    if load_type == 'one2one':
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint)


    elif load_type == 'one2many':
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint)
            gpu_ids = [int(x) for x in gpu_ids]
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        else:
            # 如果是在一个卡上训练，多个卡上加载，那么
            gpu_ids = [int(gpu_ids[0])]
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    elif load_type == 'many2one':
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)

            gpu_ids = [int(gpu_ids[0])]
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
            model.load_state_dict(checkpoint)
        else:
            gpu_ids = [int(x) for x in gpu_ids]

            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            gpu_ids = [int(x) for x in gpu_ids]

            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
            # set_trace()
            model.load_state_dict(checkpoint)
        else:
            gpu_ids = [int(x) for x in gpu_ids]
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.to(device)

    return model, device


def save_model(config, model, epoch=100, mode='best_model'):
    '''
        无论是多卡保存还是单卡保存，所有的保存都是一样的
    '''
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    if config.use_n_gpu and torch.cuda.device_count() > 1:
        model = model.module
    if mode == 'best_model':
        output_dir = os.path.join(config.output_dir, 'best_model')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        logger.info('-----将模型保存到 {}------'.format(output_dir))

        torch.save(model.state_dict(), os.path.join(output_dir, '{}.pt'.format(config.model_name)))
    else:

        output_dir = os.path.join(config.output_dir, str(epoch))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        logger.info('-----将模型保存到 {}------'.format(output_dir))

        torch.save(model.state_dict(), os.path.join(output_dir, '{}.pt'.format(config.model_name)))


def save_metric_writer(metric_writer, loss, global_step, p, r, f1, type='Training'):
    """
    这个主要是记录模型的performance
    """
    metric_writer.add_scalar("{}/loss".format(type), loss, global_step)
    metric_writer.add_scalar("{}/precision".format(type), p, global_step)
    metric_writer.add_scalar("{}/recall".format(type), r, global_step)
    metric_writer.add_scalar("{}/f1".format(type), f1, global_step)


def save_parameter_writer(model_writer, model, step):
    '''
    这个是记录模型的参数、梯度等信息
    '''
    for name, param in model.named_parameters():
        model_writer.add_histogram('model_param_' + name, param.clone().cpu().data.numpy(), step)
        if param.grad is not None:
            model_writer.add_histogram('model_grad_' + name, param.grad.clone().cpu().numpy(), step)


def show_log(logger, idx, len_dataloader, t_total, epoch, global_step, loss, p, r, f1, acc, evaluate_mode, type='train',
             scheme=0):
    if scheme == 0:
        if type == 'train':
            logger.info(
                '训练集训练中...:  Epoch {} | Step:{}/{}|{}/{}'.format(epoch, idx, len_dataloader, global_step, t_total))
        else:
            logger.info(
                '验证集评估中...:  Epoch {}'.format(epoch))

    else:
        if type == 'train':
            logger.info('********Epoch {} [训练集完成]********'.format(epoch))
        elif type == 'dev':
            logger.info('********Epoch {} [验证集完成]********'.format(epoch))
        else:
            logger.info('********Epoch {} [测试集完成]********'.format(epoch))

    logger.info('---------------{}--------------'.format(evaluate_mode))
    logger.info('Loss:{:.5f}'.format(loss))
    logger.info('Accuracy:{:.5f}'.format(acc))

    logger.info('Precision:{:.5f}'.format(p))
    logger.info('Recall:{:.5f}'.format(r))
    logger.info('F1:{:.5f}'.format(f1))


def wandb_log(wandb, epoch, global_step, f1, p, r, acc, loss, type,evaluate_mode='micro', **kwargs):
    f1_key = '{}_{}_f1'.format(type,evaluate_mode)
    p_key = '{}_{}_p'.format(type,evaluate_mode)
    r_key = '{}_{}_r'.format(type,evaluate_mode)
    acc_key = '{}_acc'.format(type)
    loss_type = '{}_loss'.format(type)
    if type == 'train':
        lr_key = '{}_lr'.format(type)
        wandb.log(
            {"train-epoch": epoch, f1_key: f1, p_key: p, r_key: r, loss_type: loss, acc_key: acc, lr_key: kwargs['lr']},
            step=global_step)
    else:
        wandb.log(
            {"{}-epoch".format(type): epoch, f1_key: f1, p_key: p, r_key: r, loss_type: loss},
            step=global_step)



def list_find(li, value):
    try:
        if isinstance(li, torch.Tensor):
            li = li.cpu().numpy().tolist()
        return li.index(value)
    except:
        return len(li)

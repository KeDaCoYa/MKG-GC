# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2021/11/25
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/11/25: 
-------------------------------------------------
"""

import argparse
import logging
import os
import datetime
import time
from ipdb import set_trace
import torch
import random
from tqdm import tqdm
import copy
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from seqeval.metrics import classification_report,f1_score,accuracy_score,recall_score,precision_score

from config import BertConfig
from src.dataset_util.base_dataset import NERProcessor
from src.dataset_util.bert_crf_dataset import BertCRFDataset, BertCRFDataset_dynamic
from src.dataset_util.bert_globalpointer_dataset import BertGlobalPointerDataset
from src.dataset_util.bert_span_dataset import BertSpanDataset, BertSpanDataset_dynamic
from src.decode.decode_globalpointer import GlobalPointerMetrics
from src.decode.decode_span import evaluate_span_macro, evaluate_span_micro
from src.decode.decode_crf import bert_evaluate_crf, bert_evaluate_crf_tokenize

from src.models.bert_bilstm_crf import Bert_BiLSTM_CRF
from src.models.bert_crf import BertCRF
from src.models.bert_globalpointer import BertGlobalPointer
from src.models.bert_mlp import BertMLP
from src.models.bert_span import Bert_Span


from utils.function_utils import correct_datetime, set_seed, print_hyperparameters, gentl_print, get_type_weights, \
    argsort_sequences_by_lens
from utils.data_process_utils import read_data, convert_example_to_span_features, convert_example_to_crf_features, \
    sort_by_lengths

from utils.loss_utils import multilabel_categorical_crossentropy
from utils.train_utils import build_optimizer, build_optimizer_and_scheduler
from utils.function_utils import save_model,load_model
from utils.trick_utils import EMA

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id',type=str,default='1',help='选择哪块gpu使用，0,1,2...')
parser.add_argument('--ner_type',type=str,default='jnlpba',help='选择的NER任务是')
parser.add_argument('--dataset_name',type=str,default='NERdata',help='选择的NER任务是')
parser.add_argument('--task_type',type=str,default='bilstm_globalpointer',help='选择使用的模型是')
parser.add_argument('--model_name',type=str,default='Luge Model',help='给model一个名字')
parser.add_argument('--num_epochs',type=int,default=100,help='给model一个名字')
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--seed',type=int,default=1234)  #[sa,mha,normal]
parser.add_argument('--use_sort',type=bool,default=True,help='对数据集按照顺序进行排序')
parser.add_argument('--evaluate_mode',type=str,default='micro',help='对数据集按照顺序进行排序')
parser.add_argument('--fixed_batch_length',type=bool,default=False,help='动态batch或者根据batch修改')
parser.add_argument('--use_ema',type=bool,default=False,help='是否使用EMA')
parser.add_argument('--over_fitting_rate',type=float,default=0.5,help='验证集和训练集的f1差别在多大的时候停止')
parser.add_argument('--dropout_prob',type=float,default=0.1,help='BERT使用的dropout')
parser.add_argument('--other_lr',type=float,default=1e-4,help='BERT之外的网络学习率')
parser.add_argument('--logfile_name',type=str,default='',help='给logfile起个名字')
parser.add_argument('--over_fitting_epoch',type=int,default=5,help='表示有几个epoch没有超过最大f1则停止')
parser.add_argument('--train_verbose',type=bool,default=True,help='是否在训练过程中每个batch显示各种值')
parser.add_argument('--summary_writer',type=bool,default=False,help='是否使用SummaryWriter记录参数')
parser.add_argument('--max_len',type=int,default=360,help='最大长度')
parser.add_argument('--subword_weight_mode',type=str,default='first',help='选择第一个subword作为token representation；或者是平均值',choices=['first','avg'])
parser.add_argument('--print_step',type=int,default=1,help='打印频次')

args = parser.parse_args()
gpu_id = args.gpu_id
ner_type = args.ner_type
task_type = args.task_type
model_name = args.model_name
dataset_name = args.dataset_name
num_epochs = args.num_epochs
batch_size = args.batch_size
use_sort = args.use_sort
evaluate_mode = args.evaluate_mode
use_ema = args.use_ema
other_lr = args.other_lr
seed = args.seed
over_fitting_rate = args.over_fitting_rate
dropout_prob = args.dropout_prob
logfile_name = args.logfile_name
over_fitting_epoch = args.over_fitting_epoch
fixed_batch_length =  args.fixed_batch_length
train_verbose = args.train_verbose
summary_writer = args.summary_writer
max_len = args.max_len
subword_weight_mode = args.subword_weight_mode
print_step = args.print_step

ner_type = ner_type.strip()
task_type = task_type.strip()
dataset_name = dataset_name.strip()
subword_weight_mode = subword_weight_mode.strip()


# 选择bert+模型
if task_type in ['crf','span','mlp','bilstm_crf','globalpointer']:
    model_name = 'bert_{}'.format(task_type)
else:
    model_name = task_type

config = BertConfig(model_name=model_name,gpu_ids=gpu_id,task_type=task_type,ner_type=ner_type,dataset_name=dataset_name,num_epochs=num_epochs,batch_size=batch_size,use_sort=use_sort,
                    evaluate_mode=evaluate_mode,use_ema=use_ema,over_fitting_rate=over_fitting_rate,seed=seed,other_lr=other_lr,dropout_prob=dropout_prob,logfile_name=logfile_name,
                    fixed_batch_length=fixed_batch_length,over_fitting_epoch=over_fitting_epoch,train_verbose=train_verbose,summary_writer=summary_writer,max_len=max_len,
                    subword_weight_mode=subword_weight_mode,print_step=print_step)


logger = logging.getLogger('main')
logging.Formatter.converter = correct_datetime

logger.setLevel(level=logging.INFO)

if not os.path.exists(config.logs_dir):
    os.makedirs(config.logs_dir)

now = datetime.datetime.now()+datetime.timedelta(hours=8)
year,month,day,hour,minute,secondas = now.year,now.month,now.day,now.hour,now.minute,now.second
handler = logging.FileHandler(os.path.join(config.logs_dir,'{} {}_{}_{} {}:{}:{}.txt'.format(logfile_name,year,month,day,hour,minute,secondas)))

handler.setLevel(level=logging.INFO)
formatter =  logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(level=logging.INFO)
console.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console)


# 设置时间
now = datetime.datetime.now()
diff = datetime.timedelta(hours=8)
now = now+diff

set_seed(config.seed)

logger.info('----------------本次模型运行的参数------------------')
print_hyperparameters(config)


def span_decode(start_logits,end_logits,raw_text):
    '''
    Bert解码的时候注意<CLS>,<SEP>
    这是macro的计算方式
    :param start_logits:
    :param end_logits:
    :param start_ids:
    :param end_ids:
    :param mask:
    :param raw_text:
    :return:
    '''
    lens_li = [len(i) for i in raw_text] #使用raw_text作为具体长度


    sum_ = 0
    A, B, C = 1e-10, 1e-10, 1e-10

    for id,length in enumerate(lens_li):
        # 这里是一个集合，共由四部分组成(entity,entity_type,start_idx,end_idx)
        predicate_set = set()

        tmp_start_logits = start_logits[id][1:length+1]
        tmp_end_logits = end_logits[id][1:length+1]

        tmp_raw_text = raw_text[id]

        for i,s_type in enumerate(tmp_start_logits):
            if s_type == 0:
                continue
            for j,e_type in enumerate(tmp_end_logits[i:]):
                if s_type == e_type:
                    predicate_set.add((" ".join(tmp_raw_text[i:i+j+1]), s_type, i, i))
                    break



def dev(model, config, type_weight,ckpt_path='/home/kedaxia/code/bioner/ner/outputs/save_models/bert_span/jnlpba/18/model.pt', metric=None):
    '''

    :param model:
    :param config:
    :param type_weight:
    :param ckpt_path:
    :param metric:
    :return:p,r,f1
    '''

    task_type = config.task_type
    if model is None:
        logger.info('----------加载预训练的模型-----------')
        if task_type == 'crf':
            model = BertCRF(config)
        elif task_type == 'mlp':
            model = BertMLP(config)
        elif task_type == 'span':
            model = Bert_Span(config)
        elif task_type == 'bilstm_crf':
            model = Bert_BiLSTM_CRF(config)
        elif task_type == 'globalpointer':
            pass
        model = load_model(model, ckpt_path=ckpt_path)

    device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')

    model.to(device)

    # 获取训练集
    #    logger.info('----------从验证集文件中读取数据-----------')
    dev_data, dev_labels = read_data(config.dev_file_path)
    processor = NERProcessor()

    dev_examples = processor.get_examples(dev_data, dev_labels)

    # 加载预训练模型的分词器
    tokenizer = BertTokenizer(os.path.join(config.bert_dir, 'vocab.txt'))
    # logger.info('----------预处理验证集数据数据-----------')
    if config.fixed_batch_length:
        if task_type == 'crf' or task_type == 'bilstm_crf':
            dev_features, dev_callback_info = convert_example_to_crf_features(dev_examples, config.crf_label2id, tokenizer,
                                                                              config.max_len)

            dev_dataset = BertCRFDataset(dev_features)
            dev_loader = DataLoader(dataset=dev_dataset, shuffle=False, num_workers=0, batch_size=config.batch_size)
        elif task_type == 'mlp':
            dev_callback_info = None
        elif task_type == 'span':
            dev_features, dev_callback_info = convert_example_to_span_features(dev_examples, tokenizer, config.max_len)

            dev_dataset = BertSpanDataset(dev_features)
            # 这里的shuffle为False，因为dev的时候是不需要的
            dev_loader = DataLoader(dataset=dev_dataset, shuffle=False, num_workers=0, batch_size=config.batch_size)

    else:
        if task_type == 'crf' or task_type == 'bilstm_crf' or task_type == 'mlp':

            dev_dataset = BertCRFDataset_dynamic(config, dev_examples, tokenizer)
            dev_loader = DataLoader(dataset=dev_dataset, shuffle=True, num_workers=0, batch_size=config.batch_size,
                                    collate_fn=dev_dataset.collate_fn)
        elif task_type == 'span':
            dev_dataset = BertSpanDataset_dynamic(config, dev_examples, tokenizer)
            dev_loader = DataLoader(dataset=dev_dataset, shuffle=True, num_workers=0, batch_size=config.batch_size,
                                    collate_fn=dev_dataset.collate_fn)

        elif task_type == 'globalpointer':
            dev_dataset = BertGlobalPointerDataset(data=dev_examples, tokenizer=tokenizer,config=config)
            dev_loader = DataLoader(dataset=dev_dataset, shuffle=True, num_workers=0, batch_size=config.batch_size,
                                    collate_fn=dev_dataset.collate_tokenize)

    # writer = SummaryWriter(os.path.join(config.tensorboard_dir, "dev {}-{} {}-{}-{} global_step{}".format(now.month,now.day,now.hour,now.minute,now.second,global_step)))


    model.eval()
    # 这是span需要的label，用于之后的evaluate...
    dev_start_predicate = []
    dev_end_predicate = []
    dev_start_ids = []
    dev_end_ids = []

    # 这是crf需要的label，用于之后的评测...
    dev_labels = []
    dev_predicate = []

    dev_callback_info = []

    dev_orig_token_indexs = []

    batch_sum_f1 = 0.
    batch_sum_p = 0.
    batch_sum_r = 0.
    with torch.no_grad():
        for step, batch_data in enumerate(dev_loader):
            if config.fixed_batch_length:
                token_ids = batch_data['token_ids'].to(device)
                attention_masks = batch_data['attention_masks'].to(device)
                token_type_ids = batch_data['token_type_ids'].to(device)
                if task_type == 'crf' or task_type == 'bilstm_crf':
                    labels = batch_data['labels'].to(device)
                    res = model(token_ids, attention_masks, token_type_ids, labels)
                    dev_predicate.extend(res)

                elif task_type == 'mlp':
                   pass

                elif task_type == 'span':
                        start_ids = batch_data['start_ids'].to(device)
                        end_ids = batch_data['end_ids'].to(device)

                        start_logits, end_logits = model(token_ids, attention_masks, token_type_ids, start_ids, end_ids)
                        _, start_ind = torch.max(start_logits, dim=-1)
                        _, end_ind = torch.max(end_logits, dim=-1)
                        start_ind = start_ind.cpu().numpy()
                        end_ind = end_ind.cpu().numpy()
                        dev_start_predicate.extend(start_ind)
                        dev_end_predicate.extend(end_ind)
            else:

                if task_type == 'crf' or task_type == 'bilstm_crf' or task_type == 'mlp':
                    # raw_text_list, token_ids, attention_masks, token_type_ids, labels = batch_data
                    # token_ids, attention_masks, token_type_ids, labels = token_ids.to(device), attention_masks.to(
                    #     device), token_type_ids.to(device), labels.to(device)
                    # dev_callback_info.extend(raw_text_list)
                    # res = model(token_ids, attention_masks, token_type_ids, labels)
                    # labels = labels.cpu().numpy()
                    # dev_labels.extend(labels)
                    # dev_predicate.extend(res)
                    # --------上面是普通模式------------
                    raw_text_list, batch_true_labels, batch_subword_input_ids, batch_subword_token_type_ids, batch_subword_attention_masks, origin_to_subword_indexs, batch_label_mask = batch_data


                    token_ids, attention_masks, token_type_ids = batch_subword_input_ids.to(
                        device), batch_subword_attention_masks.to(device), batch_subword_token_type_ids.to(device)

                    true_labels = batch_true_labels.to(device)



                    dev_callback_info.extend(raw_text_list)
                    dev_orig_token_indexs.extend(origin_to_subword_indexs)
                    if config.subword_weight_mode == 'first':
                        tmp_predicate = model.forward(token_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=true_labels,input_token_starts=origin_to_subword_indexs,)
                    elif config.subword_weight_mode == 'avg':
                        tmp_predicate = model.weight_forward(token_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=true_labels,input_token_starts=origin_to_subword_indexs)

                    true_labels = true_labels.cpu().numpy()

                    predicate_label_BIO = []
                    true_label_BIO = []

                    for i in range(len(tmp_predicate)):
                        predicate_label_BIO.append([config.crf_id2label[x] for x in tmp_predicate[i]])
                        true_label_BIO.append([config.crf_id2label[x] for x in true_labels[i][:len(tmp_predicate[i])]])


                    p = precision_score(true_label_BIO, predicate_label_BIO)
                    r = recall_score(true_label_BIO, predicate_label_BIO)
                    f1 = f1_score(true_label_BIO, predicate_label_BIO)
                    batch_sum_f1 += f1
                    batch_sum_p += p
                    batch_sum_r += r


                elif task_type == 'span':
                    raw_text_list, token_ids, attention_masks, token_type_ids, start_ids, end_ids = batch_data
                    dev_callback_info.extend(raw_text_list)
                    token_ids, attention_masks, token_type_ids, start_ids, end_ids = token_ids.to(
                        device), attention_masks.to(device), token_type_ids.to(device), start_ids.to(device), end_ids.to(
                        device)
                    tmp_start_logits, tmp_end_logits = model(token_ids, attention_masks, token_type_ids, start_ids, end_ids)

                    _, tmp_start_logits = torch.max(tmp_start_logits, dim=-1)
                    _, tmp_end_logits = torch.max(tmp_end_logits, dim=-1)
                    tmp_start_logits = tmp_start_logits.cpu().numpy()
                    tmp_end_logits = tmp_end_logits.cpu().numpy()

                    dev_start_ids.extend(start_ids)
                    dev_end_ids.extend(end_ids)

                    dev_start_predicate.extend(tmp_start_logits)
                    dev_end_predicate.extend(tmp_end_logits)
                elif task_type == 'globalpointer':

                    # raw_text_list, token_ids, attention_masks, token_type_ids, labels = batch_data
                    # dev_callback_info.extend(raw_text_list)
                    # token_ids, attention_masks, token_type_ids, labels = token_ids.to(device), attention_masks.to(
                    #     device), token_type_ids.to(device), labels.to(device)
                    #
                    # logits = model(token_ids, attention_masks, token_type_ids,labels)
                    #
                    # dev_predicate.extend(logits)
                    # dev_labels.extend(labels)
                    # ------------上面是普通模式------------------
                    raw_text_list, batch_true_labels, batch_subword_input_ids, batch_subword_token_type_ids, batch_subword_attention_masks, origin_to_subword_indexs, batch_label_mask = batch_data

                    token_ids, attention_masks, token_type_ids = batch_subword_input_ids.to(
                        device), batch_subword_attention_masks.to(device), batch_subword_token_type_ids.to(device)

                    global_pointer_true_labels = batch_true_labels.to(device)

                    dev_callback_info.extend(raw_text_list)


                    globalpointer_predicate = model(token_ids, attention_masks=attention_masks,
                                                                           token_type_ids=token_type_ids,
                                                                           labels=global_pointer_true_labels,
                                                                           input_token_starts=origin_to_subword_indexs)

                    f1, p, r = metric.get_evaluate_fpr_micro(globalpointer_predicate, global_pointer_true_labels, type_weight,
                                                             config.globalpointer_label2id)
                    batch_sum_f1 += f1
                    batch_sum_p += p
                    batch_sum_r += r
    t_total = len(dev_loader)
    if task_type == 'span':
        if config.evaluate_mode == 'micro':
            f1,p, r = evaluate_span_micro(dev_start_predicate, dev_end_predicate, dev_start_ids, dev_end_ids, dev_callback_info,type_weight=type_weight,span_label2id=config.span_label2id)
        else:
            f1, p, r = evaluate_span_macro(dev_start_predicate, dev_end_predicate, dev_start_ids, dev_end_ids,
                                           dev_callback_info)

    elif task_type == 'crf' or task_type == 'bilstm_crf' or task_type == 'mlp':

        # 这里需要预先住那边形式为[B-]
        # predicate_label_BIO = []
        # true_label_BIO = []
        #
        # for i in range(len(dev_predicate)):
        #     predicate_label_BIO.append([config.crf_id2label[x] for x in dev_predicate[i]])
        #     true_label_BIO.append([config.crf_id2label[x] for x in dev_labels[i][:len(dev_predicate[i])]])
        #
        # #acc = accuracy_score(true_label_BIO, predicate_label_BIO)
        # p = precision_score(true_label_BIO, predicate_label_BIO)
        # r = recall_score(true_label_BIO, predicate_label_BIO)
        # f1 = f1_score(true_label_BIO, predicate_label_BIO)
        #f1,p, r = bert_evaluate_crf(dev_predicate, dev_labels, dev_callback_info,crf_id2label=config.crf_id2label,type_weight=type_weight,mode=config.evaluate_mode,verbose=True)


        f1, p, r = batch_sum_f1 / t_total, batch_sum_p / t_total, batch_sum_r / t_total

    elif task_type == 'globalpointer':
        if config.evaluate_mode == 'micro':

            f1, p, r = batch_sum_f1/t_total,batch_sum_p/t_total,batch_sum_r/t_total
        else:
            raise ValueError
    else:
        raise ValueError
    return p, r, f1


if __name__ == '__main__':

    train_data, train_labels = read_data(config.train_file_path)
    type_weight = get_type_weights(train_labels)

    dev(None,config,type_weight)
# -*- encoding: utf-8 -*-
"""
@File    :   multi_ner_main.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/3/27 15:27   
@Description :   这个是多任务学习的NER

"""


import datetime
import os

from ipdb import set_trace
import torch
import numpy as np
import wandb
from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer

from src.evaluate.evaluate_span import evaluate_span_fpr, error_analysis
from utils.function_utils import correct_datetime, set_seed, print_hyperparameters, gentl_print, get_type_weights, \
    get_logger, save_metric_writer, save_parameter_writer, choose_model, choose_dataset_and_loader, final_print_score, \
    get_config, list_find, choose_multi_dataset_and_loader, choose_multi_ner_model, load_model_and_parallel

from utils.function_utils import show_log, wandb_log


def dev(model, config,device, type_weight=None,ckpt_path=None,word2id=None,tokenizer=None, metric=None,logger=None,epoch=None,global_step=None,type_='dev',wandb=None):


    dev_dataset, dev_loader = choose_multi_dataset_and_loader(config,tokenizer, type_=type_)



    dev_loss = 0.
    dev_start_ids = []
    dev_end_ids = []
    dev_start_logits = []
    dev_end_logits = []
    dev_callback_info = []

    t_total = len(dev_loader)
    model.eval()

    for step, batch_data in tqdm(enumerate(dev_loader),total=t_total,desc="{}:正在评估中....".format(config.ner_dataset_name)):


        if config.model_name in  ['binary_bert_span','inter_binary_bert_span','bert_span','inter_bert_span','inter_binary_bert_bilstm_span','inter_binary_bert_bilstm_double_mid_span','inter_binary_bert_linear_mid_span','inter_binary_bert_gru_mid_span']:

            raw_text_list, token_ids, attention_masks, token_type_ids, start_ids, end_ids, origin_to_subword_index,input_true_length,entity_type_ids = batch_data
            token_ids, attention_masks, token_type_ids, start_ids, end_ids = token_ids.to(device), attention_masks.to(device), token_type_ids.to(device), start_ids.to(device), end_ids.to(device)
            entity_type_ids = entity_type_ids.to(device)
            input_true_length = input_true_length.to(device)
            origin_to_subword_index = origin_to_subword_index.to(device)

            dev_callback_info.extend(raw_text_list)
            with torch.no_grad():
                tmp_loss, tmp_start_logits, tmp_end_logits = model(token_ids, attention_masks=attention_masks,
                                                           token_type_ids=token_type_ids,
                                                           start_ids=start_ids,
                                                           end_ids=end_ids,
                                                           input_token_starts=origin_to_subword_index,
                                                           input_true_length = input_true_length,
                                                           entity_type_ids = entity_type_ids)


            _, span_start_logits = torch.max(tmp_start_logits, dim=-1)
            _, span_end_logits = torch.max(tmp_end_logits, dim=-1)

            span_start_logits = span_start_logits.cpu().numpy()
            span_end_logits = span_end_logits.cpu().numpy()
            start_ids = start_ids.cpu().numpy()
            end_ids = end_ids.cpu().numpy()

            if 'binary' in config.model_name:
                # 这是将结构进行转变
                new_start_logits = []
                new_end_logits = []
                for bs in range(len(span_start_logits)):
                    tmp_start = []
                    tmp_end = []
                    for idx in range(len(span_start_logits[0])):
                        if span_start_logits[bs][idx] == 1:
                            tmp_start.append(entity_type_ids[bs][0].item())
                        else:
                            tmp_start.append(0)

                        if span_end_logits[bs][idx] == 1:
                            tmp_end.append(entity_type_ids[bs][0].item())
                        else:
                            tmp_end.append(0)
                    new_start_logits.append(tmp_start)
                    new_end_logits.append(tmp_end)
                span_start_logits = new_start_logits
                span_end_logits = new_end_logits
            dev_start_ids.extend(start_ids)
            dev_end_ids.extend(end_ids)
            dev_start_logits.extend(span_start_logits)
            dev_end_logits.extend(span_end_logits)
            # set_trace()
            # dev_f1, dev_p, dev_r = evaluate_span_fpr(span_start_logits, span_end_logits,
            #                                          start_ids, end_ids, dev_callback_info,
            #                                          config.span_label2id, None,
            #                                          average=config.evaluate_mode,
            #                                          verbose=True)

            dev_loss += tmp_loss.mean()

    # error_analysis(dev_start_logits, dev_end_logits,dev_start_ids, dev_end_ids, dev_callback_info,config.span_label2id)
    wandb_dict = None
    if config.use_wandb:
        wandb_dict = {}
        wandb_dict['global_step'] = global_step
        wandb_dict['epoch'] = epoch
        wandb_dict['type_'] = 'dev'
    dev_f1, dev_p, dev_r = evaluate_span_fpr(dev_start_logits, dev_end_logits,
                                                         dev_start_ids, dev_end_ids, dev_callback_info,
                                                         config.span_label2id, None,
                                                         average=config.evaluate_mode,
                                                         verbose=True,wandb_dict=wandb_dict,wandb=wandb)
    # dev_f1 = dev_f1/t_total
    # dev_r = dev_r/t_total
    # dev_p = dev_p/t_total
    dev_loss = dev_loss/t_total
    show_log(logger, step, len(dev_loader), t_total, epoch, global_step, dev_loss, dev_p, dev_r, dev_f1,0., config.evaluate_mode, type='dev', scheme=0)

    return dev_p, dev_r, dev_f1, dev_loss



if __name__ == '__main__':

    config = get_config()

    logger = get_logger(config)

    # 设置时间
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff
    # 设置随机种子
    set_seed(config.seed)
    tokenizer = None
    if config.which_model == 'bert':
        if config.bert_name in ['scibert', 'biobert', 'flash', 'flash_quad', 'wwm_bert', 'bert']:
            tokenizer = BertTokenizer(os.path.join(config.bert_dir, 'vocab.txt'))
        elif config.bert_name == 'kebiolm':
            tokenizer = AutoTokenizer.from_pretrained(config.bert_dir)
        else:
            raise ValueError

    model, metric = choose_multi_ner_model(config)  # 这个metric没啥用，只在globalpointer有用
    # ckpt_path = '/root/code/bioner/ner/outputs/save_models/biobert_bert_span/multi_BC5/best_model/bert_span.pt'

    ckpt_path = '/opt/data/private/luyuwei/code/bioner/ner/outputs/save_models/2022-06-16/MT_biobert_binary_bert_span_epochs15_free8_scheduler0.1_bs64_lr1e-05/binary_bert_span/multi_all_dataset_v1_lite/11/binary_bert_span.pt'


    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint)
    # model, device = load_model_and_parallel(model, '0,1', ckpt_path=None, load_type='one2one')
    device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')
    model.to(device)

    dev(model, config, device, type_weight=None, ckpt_path=ckpt_path, word2id=None, tokenizer=tokenizer, metric=None, logger=logger,
        epoch=0, global_step=0, type_='dev')


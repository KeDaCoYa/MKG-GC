
import logging
import os
import datetime
import time
from collections import defaultdict
import copy

from tqdm import tqdm
from ipdb import set_trace
import torch
import numpy as np
import wandb


from transformers import BertTokenizer, AutoTokenizer

from seqeval.metrics import classification_report,f1_score,accuracy_score,recall_score,precision_score


from src.evaluate.evaluate_span import evaluate_span_fpr, evaluate_span_micro, transfer_span_to_BIO

from utils.function_utils import choose_model, choose_dataset_and_loader, get_logger, get_config, show_log, list_find
from utils.data_process_utils import read_data


from utils.function_utils import load_model_and_parallel


def dev(model, config,device, type_weight,ckpt_path='/root/code/biobert_task/ner/output/best_model/model.pt',word2id=None,tokenizer=None, metric=None,logger=None,epoch=None,global_step=None,type_='dev'):
    """

    :param model:
    :param config:
    :param type_weight:
    :param ckpt_path:
    :param metric:
    :return:p,r,f1
    """

    model_name = config.model_name

    if model is None:
        logger.info('----------加载训练完成的模型-----------')
        if config.which_model == 'bert':
            model, metric = choose_model(config)  # 这个metric没啥用，只在globalpointer有用
        elif config.which_model == 'normal':
            model, metric, word2id, id2word = choose_model(config, type='train')
            config.fixed_batch_length = False

        model, device = load_model_and_parallel(model, '0,1', ckpt_path=ckpt_path, load_type='one2one')

    model.to(device)

    # 获取训练集
    #    logger.info('----------从验证集文件中读取数据-----------')
    if type_ == 'dev':
        dev_data, dev_labels = read_data(config.dev_file_path)
    elif type_ == 'test':
        dev_data, dev_labels = read_data(config.test_file_path)
    if config.debug:
        dev_data = dev_data[:config.batch_size]
        dev_labels = dev_labels[:config.batch_size]

    if config.which_model =='bert' and tokenizer is None:
        # todo:这里需要完善，添加限制
        if config.bert_name == 'biobert' or config.bert_name == 'wwm_bert':
            tokenizer = BertTokenizer(os.path.join(config.bert_dir, 'vocab.txt'))
        elif config.bert_name == 'kebiolm':
            tokenizer = AutoTokenizer.from_pretrained(config.bert_dir)


    dev_dataset, dev_loader = choose_dataset_and_loader(config, dev_data, dev_labels, tokenizer, word2id,type_='test')


    dev_start_predicate = []
    dev_end_predicate = []
    dev_start_ids = []
    dev_end_ids = []

    # 这是crf需要的label，用于之后的评测...
    dev_predicate = []
    # 这是存放raw_text
    dev_callback_info = []

    dev_orig_token_indexs = []

    batch_sum_dev_f1 = 0.
    batch_sum_dev_p = 0.
    batch_sum_dev_r = 0.

    batch_sum_macro_f1 = 0.
    batch_sum_macro_p = 0.
    batch_sum_macro_r = 0.


    # 下面两行是为了给每个epoch输出一个reports，方便之后的记录
    true_label_BIOs = []
    predicate_label_BIOs = []
    dev_loss = 0.
    model.eval()
    with torch.no_grad():
        for step, batch_data in tqdm(enumerate(dev_loader),total=len(dev_loader)):
            if config.which_model == 'bert':
                if model_name == 'bert_crf' or model_name == 'bert_bilstm_crf' or model_name == 'bert_mlp':

                    raw_text_list, batch_true_labels, batch_subword_input_ids, batch_subword_token_type_ids, batch_subword_attention_masks, origin_to_subword_indexs, batch_label_mask = batch_data

                    token_ids, attention_masks, token_type_ids = batch_subword_input_ids.to(
                        device), batch_subword_attention_masks.to(device), batch_subword_token_type_ids.to(device)

                    true_labels = batch_true_labels.to(device)

                    dev_callback_info.extend(raw_text_list)
                    dev_orig_token_indexs.extend(origin_to_subword_indexs)

                    loss, tmp_predicate = model(token_ids,
                                                attention_masks=attention_masks,
                                                token_type_ids=token_type_ids,
                                                labels=true_labels,
                                                input_token_starts=origin_to_subword_indexs)
                    # elif config.subword_weight_mode == 'avg':
                    #     tmp_predicate = model.weight_forward(token_ids, attention_masks=attention_masks, token_type_ids=token_type_ids, labels=true_labels,input_token_starts=origin_to_subword_indexs)

                    if config.decoder_layer == 'crf':
                        tmp_predicate = [
                            tmp_predicate[i][:list_find(origin_to_subword_indexs[i], 0)].cpu().numpy().tolist() for i
                            in
                            range(token_ids.shape[0])]
                    else:
                        tmp_predicate = np.argmax(tmp_predicate.detach().cpu().numpy(), axis=2)

                        tmp_predicate = [
                            tmp_predicate[i][:list_find(origin_to_subword_indexs[i], 0)].tolist()
                            for i in range(token_ids.shape[0])]

                    true_labels = true_labels.cpu().numpy()

                    predicate_label_BIO = []
                    true_label_BIO = []

                    for i in range(len(tmp_predicate)):
                        predicate_label_BIO.append([config.crf_id2label[x] for x in tmp_predicate[i]])
                        true_label_BIO.append([config.crf_id2label[x] for x in true_labels[i][:len(tmp_predicate[i])]])

                    # 最后的classification report
                    true_label_BIOs.extend(true_label_BIO)
                    predicate_label_BIOs.extend(predicate_label_BIO)

                    # dev计算每个batch下的p,r,f1
                    tmp_dev_p = precision_score(true_label_BIO, predicate_label_BIO)
                    tmp_dev_r = recall_score(true_label_BIO, predicate_label_BIO)
                    tmp_dev_f1 = f1_score(true_label_BIO, predicate_label_BIO)

                    batch_sum_dev_f1 += tmp_dev_f1
                    batch_sum_dev_p += tmp_dev_p
                    batch_sum_dev_r += tmp_dev_r


                    # 这是专用于多分类，用macro进行计算
                    if config.ner_dataset_name in ['jnlpba','multi_jnlpba']:
                        tmp_macro_dev_p = precision_score(true_label_BIO, predicate_label_BIO,average='macro')
                        tmp_macro_dev_r = recall_score(true_label_BIO, predicate_label_BIO,average='macro')
                        tmp_macro_dev_f1 = f1_score(true_label_BIO, predicate_label_BIO,average='macro')

                        batch_sum_macro_f1 += tmp_macro_dev_f1
                        batch_sum_macro_p += tmp_macro_dev_p
                        batch_sum_macro_r += tmp_macro_dev_r



                elif model_name == 'bert_span':


                    raw_text_list, token_ids, attention_masks, token_type_ids, start_ids, end_ids, origin_to_subword_index, input_true_length = batch_data
                    token_ids, attention_masks, token_type_ids, start_ids, end_ids = token_ids.to(
                        device), attention_masks.to(device), token_type_ids.to(device), start_ids.to(
                        device), end_ids.to(device)
                    input_true_length = input_true_length.to(device)
                    origin_to_subword_index = origin_to_subword_index.to(device)
                    dev_callback_info.extend(raw_text_list)
                    # 注意这里start_logits返回的结果已经不再带有[CLS]
                    loss,tmp_start_logits, tmp_end_logits = model(token_ids, attention_masks=attention_masks,token_type_ids=token_type_ids,input_token_starts=origin_to_subword_index,start_ids=start_ids,end_ids=end_ids,input_true_length=input_true_length)

                    _, span_start_logits = torch.max(tmp_start_logits, dim=-1)
                    _, span_end_logits = torch.max(tmp_end_logits, dim=-1)

                    dev_start_ids.extend(start_ids.cpu().numpy())
                    dev_end_ids.extend(end_ids.cpu().numpy())
                    dev_start_predicate.extend(span_start_logits.cpu().numpy())
                    dev_end_predicate.extend(span_end_logits.cpu().numpy())


                elif model_name == 'bert_globalpointer':


                    raw_text_list, batch_true_labels, batch_subword_input_ids, batch_subword_token_type_ids, batch_subword_attention_masks, origin_to_subword_indexs, batch_label_mask,input_true_length = batch_data

                    token_ids, attention_masks, token_type_ids = batch_subword_input_ids.to(
                        device), batch_subword_attention_masks.to(device), batch_subword_token_type_ids.to(device)

                    global_pointer_true_labels = batch_true_labels.to(device)
                    input_true_length = input_true_length.to(device)

                    dev_callback_info.extend(raw_text_list)

                    loss,globalpointer_predicate = model(token_ids, attention_masks=attention_masks,
                                                                     token_type_ids=token_type_ids,
                                                                     labels=global_pointer_true_labels,
                                                                     input_token_starts=origin_to_subword_indexs,
                                                                    input_true_length=input_true_length)

                    tmp_dev_f1,tmp_dev_p,tmp_dev_r = metric.get_evaluate_fpr(globalpointer_predicate, global_pointer_true_labels,config.globalpointer_label2id,type_weight=type_weight,average=config.evaluate_mode)

                    batch_sum_dev_f1 += tmp_dev_f1
                    batch_sum_dev_p += tmp_dev_p
                    batch_sum_dev_r += tmp_dev_r

                    if config.ner_dataset_name in ['jnlpba','multi_jnlpba']:
                        tmp_micro_dev_f1, tmp_micro_dev_p, tmp_micro_dev_r = metric.get_evaluate_fpr(globalpointer_predicate,
                                                                                   global_pointer_true_labels,
                                                                                   config.globalpointer_label2id,
                                                                                   type_weight=type_weight,
                                                                                   average='macro')

                        batch_sum_macro_f1 += tmp_micro_dev_f1
                        batch_sum_macro_p += tmp_micro_dev_p
                        batch_sum_macro_r += tmp_micro_dev_r

                elif model_name == 'bert_test':

                    input_ids, token_type_ids, attention_masks, subword_labels, true_labels = batch_data
                    input_ids = input_ids.to(device)
                    token_type_ids = token_type_ids.to(device)
                    attention_masks = attention_masks.to(device)
                    subword_labels = subword_labels.to(device)

                    loss, logits = model(input_ids, attention_masks=attention_masks, token_type_ids=token_type_ids,labels=subword_labels)

                    output = np.argmax(logits.detach().cpu().numpy(), axis=2)
                    subword_labels = subword_labels.detach().cpu().numpy()
                    tmp_predicate = []
                    batch_size, seq_len = logits.shape[:2]
                    for i in range(batch_size):
                        tmp_token = []
                        for j in range(seq_len):

                            if subword_labels[i][j] != -1:
                                tmp_token.append(output[i][j])
                        tmp_predicate.append(tmp_token)

                    predicate_label_BIO = []
                    true_label_BIO = []

                    for i in range(len(tmp_predicate)):
                        predicate_label_BIO.append([config.crf_id2label[x] for x in tmp_predicate[i]])
                        true_label_BIO.append([config.crf_id2label[x] for x in true_labels[i][:len(tmp_predicate[i])]])

                    true_label_BIOs.extend(true_label_BIO)
                    predicate_label_BIOs.extend(predicate_label_BIO)

                    tmp_dev_p = precision_score(true_label_BIO, predicate_label_BIO)
                    tmp_dev_r = recall_score(true_label_BIO, predicate_label_BIO)
                    tmp_dev_f1 = f1_score(true_label_BIO, predicate_label_BIO)

                    batch_sum_dev_f1 += tmp_dev_f1
                    batch_sum_dev_p += tmp_dev_p
                    batch_sum_dev_r += tmp_dev_r
                else:
                    raise ValueError

            else:
                raise ValueError('选择normal,bert....')

            loss = loss.mean()
            dev_loss += loss.item()
    dev_loss = dev_loss/len(dev_loader)

    t_total = len(dev_loader)
    if config.decoder_layer == 'span':

        BIO = transfer_span_to_BIO(dev_start_predicate, dev_end_predicate, dev_callback_info)

        dev_f1,dev_p, dev_r = evaluate_span_fpr(dev_start_predicate, dev_end_predicate, dev_start_ids, dev_end_ids, dev_callback_info,average=config.evaluate_mode,type_weight=type_weight,span_label2id=config.span_label2id,verbose=config.verbose)
        if config.ner_dataset_name in ['jnlpba', 'multi_jnlpba']:
            dev_macro_f1, dev_macro_p, dev_macro_r = evaluate_span_fpr(dev_start_predicate, dev_end_predicate, dev_start_ids, dev_end_ids,
                                                     dev_callback_info, average='macro',
                                                     type_weight=type_weight, span_label2id=config.span_label2id,
                                                     verbose=config.verbose)
    elif config.decoder_layer in ['crf','mlp','test']:
        try:
            report = classification_report(true_label_BIOs, predicate_label_BIOs,digits=4)
            dev_f1, dev_p, dev_r = batch_sum_dev_f1 / t_total, batch_sum_dev_p / t_total, batch_sum_dev_r / t_total
            if config.ner_dataset_name in ['jnlpba', 'multi_jnlpba']:
                dev_macro_f1, dev_macro_p, dev_macro_r = batch_sum_macro_f1 / t_total, batch_sum_macro_p / t_total, batch_sum_macro_r / t_total
            logger.info(report)
        except:# 这是因为label中没有实体，所以会报错
            dev_f1,dev_p,dev_r = 0.,0.,0.



    elif config.decoder_layer == 'globalpointer':
        dev_f1, dev_p, dev_r = batch_sum_dev_f1 / t_total, batch_sum_dev_p / t_total, batch_sum_dev_r / t_total
        if config.ner_dataset_name in ['jnlpba', 'multi_jnlpba']:
            dev_macro_f1, dev_macro_p, dev_macro_r = batch_sum_macro_f1 / t_total, batch_sum_macro_p / t_total, batch_sum_macro_r / t_total

    else:
        raise ValueError
    show_log(logger, step, len(dev_loader), t_total, epoch, global_step, loss, dev_p, dev_r, dev_f1,0., config.evaluate_mode, type='train', scheme=0)
    if config.ner_dataset_name in ['jnlpba', 'multi_jnlpba']:

        f1_key = '{}_macro_f1'.format(type_)
        p_key = '{}_macro_p'.format(type_)
        r_key = '{}_macro_r'.format(type_)

        wandb.log(
            {"train-epoch": epoch, f1_key: dev_macro_f1, p_key: dev_macro_p, r_key: dev_macro_r},
            step=global_step)

    return dev_p, dev_r, dev_f1,dev_loss




if __name__ == '__main__':

    config = get_config()

    logger = get_logger(config)

    # 设置时间
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff
    # 设置随机种子

    if config.use_wandb:
        wandb.init(project="实体抽取dev-{}".format(config.ner_dataset_name), entity="kedaxia", config=vars(config))
    if config.run_type == 'normal':
        ckpt_path = '/root/code/bioner/ner/outputs/save_models/bert_span/jnlpba/2/bert_span.pt'

        dev(model=None, config=config, device=None, type_weight=None, ckpt_path=ckpt_path,word2id=None, tokenizer=None, metric=None, logger=logger, epoch=0, global_step=0,type_='test')
    elif config.run_type == 'cv5':
        P,R,F1 = 0.,0.,0.
        for i in range(1,6):
            ckpt_path = '/root/code/bioner/ner/outputs/save_models/{}/{}/cv5/cv_{}/best_model/model.pt'.format(config.model_name,config.ner_dataset_name,i)
            config.dev_file_path  = '/root/code/bioner/ner/NERdata/{}/cv5/{}/test.txt'.format(config.ner_dataset_name,i)
            logger.info('数据集:{}'.format(config.dev_file_path))
            logger.info('模型:{}'.format(ckpt_path))
            dev_p, dev_r, dev_f1,dev_loss = dev(model=None, config=config, device=None, type_weight=None, ckpt_path=ckpt_path, word2id=None,
                tokenizer=None, metric=None, logger=logger, epoch=0, global_step=0)
            P += dev_p
            R += dev_r
            F1 += dev_f1
        F1 = F1/5
        P = P/5
        R = R/5
        logger.info('测试集结果:F1:{},P:{},R:{}'.format(F1,P,R))

    elif config.run_type == 'cv10':
        for i in range(1, 11):
            ckpt_path = '/root/code/bioner/ner/outputs/save_models/{}/{}/cv10/cv_{}/best_model/model.pt'.format(config.model_name, config.ner_dataset_name, i)
            dev(model=None, config=config, device=None, type_weight=None, ckpt_path=ckpt_path, word2id=None,
                tokenizer=None, metric=None, logger=logger, epoch=0, global_step=0)



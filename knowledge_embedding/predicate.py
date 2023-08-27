# -*- encoding: utf-8 -*-
"""
@File    :   train.py
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/17 20:25
@Description :   这是三元组的预测，判断这个三元组是否是一个正常的结果，其实就是直接一个二分类

"""
import collections
import datetime
import os
import logging

import numpy as np
import torch
import wandb
from ipdb import set_trace

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BertTokenizer

from dev import link_predication_dev
from src.models.kg_bert import KGBertModel
from src.models.kgbert_convkb import KGBertConvKBModel
from src.utils.dataset_utils import KGProcessor, KGBertDataset
from src.utils.function_utils import get_config, get_logger, set_seed, save_parameter_writer, wandb_log, show_log, \
    print_hyperparameters, dir_exists, file_exists
from src.utils.metric_utils import safe_ranking
from src.utils.train_utils import load_model_and_parallel, build_optimizer, build_bert_optimizer_and_scheduler


logger = logging.getLogger('main.predicate')


def link_predication_train(config, logger):
    # 数据读取阶段
    processor = KGProcessor()

    label_list = processor.get_labels()
    num_labels = len(label_list)

    entity_list = processor.get_entities(config.data_dir)

    train_examples = processor.get_train_examples(config.data_dir)

    if config.debug:  # 开启debug，则试验一个batch的数据
        train_examples = train_examples[:config.batch_size * 5]



    if config.bert_name == 'biobert' or config.bert_name == 'wwm_bert' or config.bert_name == 'flash_quad' or config.bert_name == 'flash':
        tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
    else:
        raise ValueError("bert_name的值应该为['biobert','wwm_bert','flash_quad','flash']")

    train_dataset = KGBertDataset(config, train_examples, tokenizer)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=0,
                              batch_size=config.batch_size,
                              collate_fn=train_dataset.collate_fn)

    if config.model_name == 'kgbert':
        model = KGBertModel(config, 2)
    else:
        model = KGBertConvKBModel(config, 2)

    #  加载模型是否是多GPU或者
    if config.use_n_gpu and torch.cuda.device_count() > 1:
        model, device = load_model_and_parallel(model, '0,1', ckpt_path=None, load_type='many2one')
    else:
        # model, device = load_model_and_parallel(model, '0,1', ckpt_path=None, load_type='one2one')
        device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')
        model.to(device)
    if config.use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    if config.metric_summary_writer:
        if not os.path.exists(config.tensorboard_dir):
            os.makedirs(config.tensorboard_dir)
        metric_writer = SummaryWriter(os.path.join(config.tensorboard_dir,
                                                   "metric_{} {}-{} {}-{}-{}".format(config.model_name, now.month,
                                                                                     now.day,
                                                                                     now.hour, now.minute,
                                                                                     now.second)))
    if config.parameter_summary_writer:
        if not os.path.exists(config.tensorboard_dir):
            os.makedirs(config.tensorboard_dir)
        parameter_writer = SummaryWriter(
            os.path.join(config.tensorboard_dir, "parameter_{} {}-{} {}-{}-{}".format(config.model_name, now.month,
                                                                                      now.day,
                                                                                      now.hour, now.minute,
                                                                                      now.second)))
    t_total = config.num_epochs * len(train_loader)
    best_epoch = 0
    best_model = None

    if config.use_scheduler:
        optimizer, scheduler = build_bert_optimizer_and_scheduler(config, model, t_toal=t_total)
        logger.info('学习率调整器:{}'.format(scheduler))
    else:
        optimizer = build_optimizer(config, model)

    logger.info('优化器:{}'.format(optimizer))
    global_step = 0


    for epo_idx,epoch in enumerate(range(config.num_epochs)):
        model.train()
        epoch_acc = 0.
        epoch_loss = 0.
        for step, batch_data in enumerate(train_loader):
            if config.model_name == 'kgbert':
                input_ids,attention_masks,token_type_ids = batch_data
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                token_type_ids = token_type_ids.to(device)


                logits, loss = model(input_ids,token_type_ids,attention_masks,label_ids=None)
            else:
                input_ids, attention_masks, token_type_ids, label_ids,head_mask,rel_mask,tail_mask = batch_data
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                token_type_ids = token_type_ids.to(device)


                head_mask = head_mask.to(device)
                rel_mask = rel_mask.to(device)
                tail_mask = tail_mask.to(device)

                logits = model(input_ids, token_type_ids, attention_masks, None,head_mask,rel_mask,tail_mask)



            label_ids = label_ids.cpu().numpy()
            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)

            acc = (preds == label_ids).mean()
            epoch_loss += loss
            epoch_acc += acc

            global_step += 1



def star_predict(config, raw_examples, dataset_list, model, device,verbose=True):
    """
    :param:dataset_list:就是train_dataset,dev_dataset,test_dataset的三合一
    : raw_examples 这个是需要evaluate的三元组，一般为测试集
    """
    logger.info("***** Running Prediction*****")
    model.eval()
    # get the last one (i.e., test) to make use if its useful functions and data
    standard_dataset = dataset_list[-1] # 这个就是test dataset

    # 这是将训练集、验证集、测试集都进行收集吗，构建为ent set,build graph
    ents = set()
    g_subj2objs = collections.defaultdict(lambda: collections.defaultdict(set)) #global_subj2objs
    g_obj2subjs = collections.defaultdict(lambda: collections.defaultdict(set))
    for _ds in dataset_list:

        for _raw_ex in _ds.raw_examples:
            _head, _rel, _tail = _raw_ex
            ents.add(_head)
            ents.add(_tail)
            g_subj2objs[_head][_rel].add(_tail)
            g_obj2subjs[_tail][_rel].add(_head)
    ent_list = list(sorted(ents))
    rel_list = list(sorted(standard_dataset.rel_list))

    # ========= get all embeddings ==========
    logger.info("获取当前数据集的所有embeddings")


    save_path = os.path.join(config.output_dir, "saved_emb_mat.np")

    if dir_exists(config.output_dir) and file_exists(save_path):
        logger.info("已存在所有的embedding，进行加载")
        emb_mat = torch.load(save_path)
    else:
        logger.info("重新获取所有的embedding")
        # 这里是根据所有的ent和rel进行遍历，相当于得到所有可能的组合情况
        input_ids_list, mask_ids_list, segment_ids_list = [], [], []
        for _ent in tqdm(ent_list):
            for _idx_r, _rel in enumerate(rel_list):

                head_ids, rel_ids, tail_ids = standard_dataset.convert_raw_example_to_features(
                    [_ent, _rel, _ent], method="4")

                head_ids, rel_ids, tail_ids = head_ids[1:-1], rel_ids[1:-1], tail_ids[1:-1]
                # truncate
                max_ent_len = standard_dataset.max_seq_length - 3 - len(rel_ids)
                head_ids = head_ids[:max_ent_len]
                tail_ids = tail_ids[:max_ent_len]

                src_input_ids = [standard_dataset._cls_id] + head_ids + [standard_dataset._sep_id] + rel_ids + [
                    standard_dataset._sep_id]
                src_mask_ids = [1] * len(src_input_ids)
                src_segment_ids = [0] * (len(head_ids) + 2) + [1] * (len(rel_ids) + 1)

                if _idx_r == 0:
                    # 这里就是为了获取tail的对应，因为tial只需一个
                    # 注意这里是先放入tail，然后才是head_rel
                    tgt_input_ids = [standard_dataset._cls_id] + tail_ids + [standard_dataset._sep_id]
                    tgt_mask_ids = [1] * len(tgt_input_ids)
                    tgt_segment_ids = [0] * (len(tail_ids) + 2)
                    input_ids_list.append(tgt_input_ids)
                    mask_ids_list.append(tgt_mask_ids)
                    segment_ids_list.append(tgt_segment_ids)

                input_ids_list.append(src_input_ids)
                mask_ids_list.append(src_mask_ids)
                segment_ids_list.append(src_segment_ids)

        # # padding
        try:
            max_len = max(len(_e) for _e in input_ids_list)
        except:
            set_trace()
        assert max_len <= standard_dataset.max_seq_length
        input_ids_list = [_e + [standard_dataset._pad_id] * (max_len - len(_e)) for _e in input_ids_list]
        mask_ids_list = [_e + [0] * (max_len - len(_e)) for _e in mask_ids_list]
        segment_ids_list = [_e + [0] * (max_len - len(_e)) for _e in segment_ids_list]
        # # dataset
        enc_dataset = TensorDataset(
            torch.tensor(input_ids_list, dtype=torch.long),
            torch.tensor(mask_ids_list, dtype=torch.long),
            torch.tensor(segment_ids_list, dtype=torch.long),
        )
        enc_dataloader = DataLoader(
            enc_dataset, sampler=SequentialSampler(enc_dataset), batch_size=config.eval_batch_size)
        print("\t get all emb via model")
        # 将上面得到的所有id，然后经过bert，得到所有的embeddings
        embs_list = []
        for batch in tqdm(enc_dataloader, desc="entity embedding", disable=(not verbose)):
            batch = tuple(t.to(device) for t in batch)
            _input_ids, _mask_ids, _segment_ids = batch
            with torch.no_grad():
                embs = model.encoder(_input_ids, attention_mask=_mask_ids, token_type_ids=_segment_ids)
                embs = embs.detach().cpu()
                embs_list.append(embs)

        # 这是得到了所有的embeddings，shape=(6345,768)
        emb_mat = torch.cat(embs_list, dim=0).contiguous()
        assert emb_mat.shape[0] == len(input_ids_list)
        # save emb_mat
        if dir_exists(config.output_dir):
            torch.save(emb_mat, save_path)

    # # assign to ent
    assert len(ent_list) *(1+len(rel_list)) == emb_mat.shape[0]

    ent_rel2emb = collections.defaultdict(dict)
    ent2emb = dict()

    # 这里将得到的embedding映射到字典中，
    ptr_row = 0
    for _ent in ent_list:
        for _idx_r, _rel in enumerate(rel_list):
            if _idx_r == 0:
                ent2emb[_ent] = emb_mat[ptr_row]
                ptr_row += 1
            ent_rel2emb[_ent][_rel] = emb_mat[ptr_row]
            ptr_row += 1
    # ========= run link prediction ==========

    # * begin to get hit
    ranks_left, ranks_right, ranks = [], [], []
    hits_left, hits_right, hits = [], [], []
    top_ten_hit_count = 0
    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    for _idx_ex, _triplet in enumerate(tqdm(raw_examples, desc="开始正式的预测....")):
        _head, _rel, _tail = _triplet

        head_ent_list = []
        tail_ent_list = []

        # head corrupt
        _pos_head_ents = g_obj2subjs[_tail][_rel]
        _neg_head_ents = ents - _pos_head_ents # 这是得到当前(_,rel,tail)中head位置的错误ent
        # -------------------------------------------
        # _tail_s = set()
        # _tail_s.add(_tail)
        # _neg_head_ents = _neg_head_ents - _tail_s
        # -------------------------------------------
        head_ent_list.append(_head)  # positive example
        head_ent_list.extend(_neg_head_ents)  # negative examples
        tail_ent_list.extend([_tail] * (1 + len(_neg_head_ents)))

        # 这个参数是一个划分，表示是head corrupt，另一边是tail corrupt
        split_idx = len(head_ent_list)

        # tail corrupt
        _pos_tail_ents = g_subj2objs[_head][_rel]
        _neg_tail_ents = ents - _pos_tail_ents
        # -------------------------------------------
        # _head_s = set()
        # _head_s.add(_head)
        # _neg_tail_ents = _neg_tail_ents - _head_s
        # -------------------------------------------
        head_ent_list.extend([_head] * (1 + len(_neg_tail_ents)))
        tail_ent_list.append(_tail)  # positive example
        tail_ent_list.extend(_neg_tail_ents)  # negative examples

        triplet_list = list([_h, _rel, _t] for _h, _t in zip(head_ent_list, tail_ent_list))

        # build dataset
        rep_src_list = [ent_rel2emb[_h][_rel] for _h, _rel, _ in triplet_list]
        rep_tgt_list = [ent2emb[_t] for _, _rel, _t in triplet_list]
        all_rep_src = torch.stack(rep_src_list, dim=0).to(device)
        all_rep_tgt = torch.stack(rep_tgt_list, dim=0).to(device)

        local_scores_list = []
        sim_batch_size = config.eval_batch_size

        # 这里选择不同的交互方式

        if config.cls_method == "dis":
            for _idx_r in range(0, all_rep_src.shape[0], sim_batch_size):
                _rep_src, _rep_tgt = all_rep_src[_idx_r: _idx_r + sim_batch_size], all_rep_tgt[
                                                                                   _idx_r: _idx_r + sim_batch_size]
                with torch.no_grad():
                    distances = model.distance_metric_fn(_rep_src, _rep_tgt)
                    local_scores = - distances
                    local_scores = local_scores.detach().cpu().numpy()
                local_scores_list.append(local_scores)
        elif config.cls_method == "cls":
            for _idx_r in range(0, all_rep_src.shape[0], sim_batch_size):
                _rep_src, _rep_tgt = all_rep_src[_idx_r: _idx_r + sim_batch_size], all_rep_tgt[
                                                                                   _idx_r: _idx_r + sim_batch_size]
                with torch.no_grad():
                    logits = model.classifier(_rep_src, _rep_tgt)
                    logits = torch.softmax(logits, dim=-1)
                    local_scores = logits.detach().cpu().numpy()[:, 1]
                local_scores_list.append(local_scores)
        scores = np.concatenate(local_scores_list, axis=0)

        # left
        left_scores = scores[:split_idx]
        left_rank = safe_ranking(left_scores)
        ranks_left.append(left_rank)
        ranks.append(left_rank)

        # right
        right_scores = scores[split_idx:]
        right_rank = safe_ranking(right_scores)
        ranks_right.append(right_rank)
        ranks.append(right_rank)

        # log
        top_ten_hit_count += (int(left_rank <= 10) + int(right_rank <= 10))
        if (_idx_ex + 1) % 10 == 0:
            logger.info("hit@10 until now: {}".format(top_ten_hit_count * 1.0 / len(ranks)))
            logger.info('mean rank until now: {}'.format(np.mean(ranks)))

        # hits
        for hits_level in range(10):
            if left_rank <= hits_level + 1:

                hits[hits_level].append(1.0)
                hits_left[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_left[hits_level].append(0.0)

            if right_rank <= hits_level + 1:
                hits[hits_level].append(1.0)
                hits_right[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_right[hits_level].append(0.0)
    if verbose:
        for i in [0, 2, 9]:
            logger.info('Hits left @{0}: {1}'.format(i + 1, np.mean(hits_left[i])))
            logger.info('Hits right @{0}: {1}'.format(i + 1, np.mean(hits_right[i])))
            logger.info('Hits @{0}: {1}'.format(i + 1, np.mean(hits[i])))
        logger.info('Mean rank left: {0}'.format(np.mean(ranks_left)))
        logger.info('Mean rank right: {0}'.format(np.mean(ranks_right)))
        logger.info('Mean rank: {0}'.format(np.mean(ranks)))
        logger.info('Mean reciprocal rank left: {0}'.format(np.mean(1. / np.array(ranks_left))))
        logger.info('Mean reciprocal rank right: {0}'.format(np.mean(1. / np.array(ranks_right))))
        logger.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

        tuple_ranks = [[int(_l), int(_r)] for _l, _r in zip(ranks_left, ranks_right)]
        metric_res =  {
            'hit@1': np.mean(hits[0]),
            'hit@3': np.mean(hits[2]),
            'hit@10': np.mean(hits[9]),
            'MR': np.mean(ranks),
            'MRR': np.mean(1. / np.array(ranks)),
        }
        return tuple_ranks,metric_res

if __name__ == '__main__':
    config = get_config()
    logger = get_logger(config)
    if config.use_wandb:
        wandb_name = f'{config.bert_name}_{config.model_name}_bs{config.batch_size}_mxl_{config.max_len}'
        wandb.init(project="知识嵌入-{}".format(config.dataset_name), entity="kedaxia", config=vars(config),
                   name=wandb_name)
    # 设置时间
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff
    # 设置随机种子
    set_seed(config.seed)
    print_hyperparameters(config)
    link_predication_train(config, logger)

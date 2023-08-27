# -*- encoding: utf-8 -*-
"""
@File    :   lpbert_dataset.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/5/6 21:45   
@Description :   这是生成LPBERT的预训练所需要的数据

"""
import json
import logging
import os
import pickle
import random
import multiprocessing
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ipdb import set_trace
from transformers import AutoTokenizer, BertTokenizer


class MemExamples:
    def __init__(self):
        pass


class LpBERTDataset(Dataset):
    def __init__(self, max_len, triplets, cui2concepts, cui2desc, tokenizer):
        """
        将数据集转变为MLM和NSP任务所需要的格式
        :param triplets：这里data是原始的三元组
        :param cui2concepts：这是cui对应的所有同义词
        :param cui2desc：这是给cui的详细解释
        """

        self.triplets = triplets
        self.cui2desc = cui2desc
        self.cui2concepts = cui2concepts
        self.nums = len(triplets)
        self.max_len = max_len  # 这是每句话的最大长度
        self.tokenizer = tokenizer

    def __len__(self):
        return self.nums
    def collate_fn(self,data):
        return data
    def __getitem__(self, index):

        """
        每取一个数据，处理一个数据,这个时候需要保证data是一个两列的数据....
        这里将数据转变为MLM和NSP所需要的格式....
        :param index:
        :return:
        """
        triple = self.triplets[index].strip().split('\t')

        head_cui, rel, tail_cui = triple
        rel = rel.replace("_", " ")
        head_desc = ""
        if head_cui in self.cui2desc:
            head_desc = self.cui2desc[head_cui]
        tail_desc = ""
        if tail_cui in self.cui2desc:
            tail_desc = self.cui2desc[tail_cui]
        try:
            head_ent = random.choice(self.cui2concepts[head_cui])
            tail_ent = random.choice(self.cui2concepts[tail_cui])
        except:
            return None

        rel_tokens = self.tokenizer.tokenize(rel)
        rel_token_ids = self.tokenizer.convert_tokens_to_ids(rel_tokens)

        # rel mrm + mlm
        rel_tokens_ids, rel_mrm_label, head_ent_desc_ids, head_ent_desc_label, tail_ent_desc_ids, tail_ent_desc_label = self.rel_mlm(
            head_ent, head_desc, tail_ent, tail_desc, rel)
        rel_mrm_input_ids = [self.tokenizer.convert_tokens_to_ids('[CLS]')] + head_ent_desc_ids + [
            self.tokenizer.convert_tokens_to_ids('[SEP]')] + rel_tokens_ids + [
                                self.tokenizer.convert_tokens_to_ids('[SEP]')] + tail_ent_desc_ids + [
                                self.tokenizer.convert_tokens_to_ids('[SEP]')]
        rel_mrm_attention_mask = [1] * len(rel_mrm_input_ids)
        rel_mrm_token_type_ids = [0] * (2 + len(head_ent_desc_ids))
        rel_mrm_token_type_ids += [1] * (1 + len(rel_token_ids))
        rel_mrm_token_type_ids += [2] * (1 + len(tail_ent_desc_ids))
        rel_mrm_label_ids = [-100] + head_ent_desc_label + [-100] + rel_mrm_label + [-100] + tail_ent_desc_label + [
            -100]
        assert len(rel_mrm_input_ids) == len(rel_mrm_attention_mask) == len(rel_mrm_token_type_ids) == len(
            rel_mrm_label_ids), "mrm 数据生成不一致"

        if len(rel_mrm_input_ids) > self.max_len:
            # todo:需要从desc进行下手裁剪
            # 直接跳过这个数据
            return
        else:
            pad_len = self.max_len - len(rel_mrm_label_ids)
            rel_mrm_input_ids += [0] * pad_len
            rel_mrm_attention_mask += [0] * pad_len
            rel_mrm_token_type_ids += [0] * pad_len
            rel_mrm_label_ids += [-100] * pad_len

        head_tokens_ids, head_mem_label = self.head_entity_mlm(head_ent)

        # 这是head mem任务+mlm任务
        head_desc_token_ids = self.tokenizer.encode(head_desc)[1:-1]
        head_mem_input_ids = [self.tokenizer.convert_tokens_to_ids('[CLS]')] + head_tokens_ids + head_desc_token_ids + [
            self.tokenizer.convert_tokens_to_ids('[SEP]')] + rel_token_ids + [
                                 self.tokenizer.convert_tokens_to_ids('[SEP]')] + tail_ent_desc_ids + [
                                 self.tokenizer.convert_tokens_to_ids('[SEP]')]
        head_mem_attention_mask = [1] * len(head_mem_input_ids)
        head_mem_token_type_ids = [0] * (2 + len(head_tokens_ids) + len(head_desc_token_ids))
        head_mem_token_type_ids += [1] * (1 + len(rel_token_ids))
        head_mem_token_type_ids += [2] * (1 + len(tail_ent_desc_ids))
        head_mem_label_ids = [-100] + head_mem_label + [-100] * len(head_desc_token_ids) + [-100] + [-100] * len(
            rel_token_ids) + [-100] + tail_ent_desc_label + [-100]
        # todo: 最长长度进行限制
        assert len(head_mem_input_ids) == len(head_mem_attention_mask) == len(head_mem_attention_mask) == len(
            head_mem_label_ids)

        if len(head_mem_input_ids) > self.max_len:
            # todo:需要从desc进行下手裁剪
            pass
        else:
            pad_len = self.max_len - len(head_mem_input_ids)
            head_mem_input_ids += [0] * pad_len
            head_mem_attention_mask += [0] * pad_len
            head_mem_token_type_ids += [0] * pad_len
            head_mem_label_ids += [-100] * pad_len

        # tail mem + mlm
        tail_tokens_ids, tail_mem_label = self.tail_entity_mlm(tail_ent)
        tail_desc_token_ids = self.tokenizer.encode(tail_desc)[1:-1]
        tail_mem_input_ids = [self.tokenizer.convert_tokens_to_ids('[CLS]')] + head_ent_desc_ids + [
            self.tokenizer.convert_tokens_to_ids('[SEP]')] + rel_token_ids + [self.tokenizer.convert_tokens_to_ids(
            '[SEP]')] + tail_tokens_ids + tail_desc_token_ids + [self.tokenizer.convert_tokens_to_ids('[SEP]')]
        tail_mem_attention_mask = [1] * len(tail_mem_input_ids)
        tail_mem_token_type_ids = [0] * (2 + len(head_ent_desc_ids))
        tail_mem_token_type_ids += [1] * (1 + len(rel_token_ids))
        tail_mem_token_type_ids += [2] * (1 + len(tail_tokens_ids) + len(tail_desc_token_ids))
        tail_mem_label_ids = [-100] + head_ent_desc_label + [-100] + [-100] * len(rel_token_ids) + [
            -100] + tail_mem_label + [-100] * len(tail_desc_token_ids) + [-100]

        assert len(tail_mem_input_ids) == len(tail_mem_attention_mask) == len(tail_mem_token_type_ids) == len(
            tail_mem_label_ids), "tail mem 数据生成错误"

        if len(tail_mem_input_ids) > self.max_len:
            # todo:需要从desc进行下手裁剪
            pass
        else:
            pad_len = self.max_len - len(tail_mem_input_ids)
            tail_mem_input_ids += [0] * pad_len
            tail_mem_attention_mask += [0] * pad_len
            tail_mem_token_type_ids += [0] * pad_len
            tail_mem_label_ids += [-100] * pad_len

        rel_mrm_input_ids = torch.tensor(rel_mrm_input_ids).long()
        rel_mrm_attention_mask = torch.tensor(rel_mrm_attention_mask).bool()
        rel_mrm_token_type_ids = torch.tensor(rel_mrm_token_type_ids).long()
        rel_mrm_label_ids = torch.tensor(rel_mrm_label_ids).long()
        mrm_res = {
            'input_ids': rel_mrm_input_ids,
            'attention_mask': rel_mrm_attention_mask,
            'token_type_ids': rel_mrm_token_type_ids,
            'label_ids': rel_mrm_label_ids
        }
        head_mem_input_ids = torch.tensor(head_mem_input_ids).long()
        head_mem_attention_mask = torch.tensor(head_mem_attention_mask).bool()
        head_mem_token_type_ids = torch.tensor(head_mem_token_type_ids).long()
        head_mem_label_ids = torch.tensor(head_mem_label_ids).long()
        head_mem_res = {
            'input_ids': head_mem_input_ids,
            'attention_mask': head_mem_attention_mask,
            'token_type_ids': head_mem_token_type_ids,
            'label_ids': head_mem_label_ids
        }
        tail_mem_input_ids = torch.tensor(tail_mem_input_ids).long()
        tail_mem_attention_mask = torch.tensor(tail_mem_attention_mask).bool()
        tail_mem_token_type_ids = torch.tensor(tail_mem_token_type_ids).long()
        tail_mem_label_ids = torch.tensor(tail_mem_label_ids).long()
        tail_mem_res = {
            'input_ids': tail_mem_input_ids,
            'attention_mask': tail_mem_attention_mask,
            'token_type_ids': tail_mem_token_type_ids,
            'label_ids': tail_mem_label_ids
        }
        return head_mem_res, tail_mem_res, mrm_res

    def head_entity_mlm(self, head_ent):
        """

        这是对head entity的随机mask
        对head ent进行完全mask，对tail ent和tail desc进行mlm任务
        :param head_ent
        :param head_desc
        :return:
        """

        # head_token_ids = self.tokenizer.encode(head_ent)[1:-1]
        # head_desc_ids = self.tokenizer.encode(head_desc)[1:-1]
        # rel_ids = self.tokenizer.encode(rel)[1:-1]

        # head_labels =deepcopy(head_token_ids)

        head_tokens = self.tokenizer.tokenize(head_ent)
        head_tokens_ids = self.tokenizer.convert_tokens_to_ids(head_tokens)
        # 对head ent进行完全mask
        head_mem_label = deepcopy(head_tokens_ids)
        head_tokens_ids = [self.tokenizer.convert_tokens_to_ids('[MASK]')] * len(head_tokens)

        return head_tokens_ids, head_mem_label

    def tail_entity_mlm(self, tail_ent):
        """

        这是对head entity的随机mask
        对head ent进行完全mask，对tail ent和tail desc进行mlm任务
        :param head_ent
        :param head_desc
        :return:
        """

        tail_tokens = self.tokenizer.tokenize(tail_ent)
        tail_tokens_ids = self.tokenizer.convert_tokens_to_ids(tail_tokens)
        # 对tao; ent进行完全mask
        tail_mem_label = deepcopy(tail_tokens_ids)
        tail_tokens_ids = [self.tokenizer.convert_tokens_to_ids('[MASK]')] * len(tail_tokens)

        return tail_tokens_ids, tail_mem_label

    def rel_mlm(self, head_ent, head_desc, tail_ent, tail_desc, rel):

        rel_tokens = self.tokenizer.tokenize(rel)
        rel_tokens_ids = self.tokenizer.convert_tokens_to_ids(rel_tokens)
        # rel 进行完全mask
        rel_mrm_label = deepcopy(rel_tokens_ids)
        rel_tokens_ids = [self.tokenizer.convert_tokens_to_ids('[MASK]')] * len(rel_tokens_ids)

        head_ent_desc = head_ent + " " + head_desc
        tokens = self.tokenizer.tokenize(head_ent_desc)
        head_ent_desc_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        head_ent_mlm_label = [-100] * len(head_ent_desc_ids)

        for i, token in enumerate(tokens):
            if i == 0 or i == len(tokens) - 1:
                continue
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                # 80% randomly change token to mask token
                if prob < 0.9:
                    # 这是whole word mask的中间subword
                    if len(tokens[i]) > 1 and tokens[i][0] == '#':
                        self.label_mlm(head_ent_desc_ids, head_ent_mlm_label, tokens, i, special_token='[MASK]')
                    else:
                        if i + 1 < len(tokens) and len(tokens[i + 1]) > 1 and tokens[i + 1][0] == '#':
                            # 这个情况是，一个word是由多个subword组成，这里正好选到了word的第一个subword，因此需要对整个mask
                            self.label_mlm(head_ent_desc_ids, head_ent_mlm_label, tokens, i + 1, special_token='[MASK]')
                        else:
                            head_ent_desc_ids[i] = self.tokenizer.convert_tokens_to_ids('[MASK]')
                            head_ent_mlm_label[i] = self.tokenizer.convert_tokens_to_ids(token)

            else:  # 85%的概率不修改
                pass
        # 对tail_ent+tail_desc进行MLM
        tail_ent_desc = tail_ent + " " + tail_desc
        tokens = self.tokenizer.tokenize(tail_ent_desc)
        tail_ent_desc_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        tail_mlm_label = [-100] * len(tail_ent_desc_ids)

        for i, token in enumerate(tokens):
            if i == 0 or i == len(tokens) - 1:
                continue
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                # 80% randomly change token to mask token
                if prob < 0.9:
                    # 这是whole word mask的中间subword
                    if len(tokens[i]) > 1 and tokens[i][0] == '#':
                        self.label_mlm(tail_ent_desc_ids, tail_mlm_label, tokens, i, special_token='[MASK]')
                    else:
                        if i + 1 < len(tokens) and len(tokens[i + 1]) > 1 and tokens[i + 1][0] == '#':
                            # 这个情况是，一个word是由多个subword组成，这里正好选到了word的第一个subword，因此需要对整个mask
                            self.label_mlm(tail_ent_desc_ids, tail_mlm_label, tokens, i + 1, special_token='[MASK]')
                        else:
                            tail_ent_desc_ids[i] = self.tokenizer.convert_tokens_to_ids('[MASK]')
                            tail_mlm_label[i] = self.tokenizer.convert_tokens_to_ids(token)

            else:  # 85%的概率不修改
                pass
        return rel_tokens_ids, rel_mrm_label, head_ent_desc_ids, head_ent_mlm_label, tail_ent_desc_ids, tail_mlm_label

    def label_mlm(self, input_ids, mlm_label, tokens, cur_pos, special_token='[MASK]'):
        """
        这是开始以cur_pos为中心，向前后进行试探，然后对whole word进行mask
        :param mlm_label:
        :param tokens:
        :param cur_pos:
        :return:
        """
        # 从当前到后 [cur_pos:]
        index_ = cur_pos
        lens = len(mlm_label)
        if special_token == '[MASK]':
            while index_ < lens:  # 这是从当前向后查找
                if len(tokens[index_]) > 1 and tokens[index_][0] == '#':
                    input_ids[index_] = self.tokenizer.convert_tokens_to_ids(special_token)
                    mlm_label[index_] = self.tokenizer.convert_tokens_to_ids(tokens[index_])
                else:
                    break
                index_ += 1
            index_ = cur_pos - 1

            while index_ >= 0:  # 这是从当前向前查找
                if len(tokens[index_]) > 1 and tokens[index_][0] == '#':

                    input_ids[index_] = self.tokenizer.convert_tokens_to_ids(special_token)
                    mlm_label[index_] = self.tokenizer.convert_tokens_to_ids(tokens[index_])

                else:
                    input_ids[index_] = self.tokenizer.convert_tokens_to_ids(special_token)
                    mlm_label[index_] = self.tokenizer.convert_tokens_to_ids(tokens[index_])
                    break
                index_ -= 1
            # 注意这里，在向前查找的时候，第一个单词是没有#，这个时候别忘了处理


        else:
            raise NotImplementedError


def generate_lp_dataset(tokenizer, max_len=512):
    '''
    将tokenize之后的数据convert_to_ids
    :param tokenizer:
    :param file_path:
    :param target_file_path:
    :param max_len:
    :return:
    '''


    with open("../../umls/umls_triplets.txt", 'r', encoding='utf-8') as f:
        triplets = f.readlines()
    print("读取三元组完成")
    with open("../../umls/cui2desc.json", 'r', encoding='utf-8') as f:
        cui2desc = json.load(f)
    print("读取cui2desc完成")
    with open("../../umls/cui2concept.json", 'r', encoding='utf-8') as f:
        cui2concepts = json.load(f)
    print("读取cui2concept完成")

    train_dataset = LpBERTDataset(max_len, triplets, cui2concepts, cui2desc, tokenizer)
    train_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=1,collate_fn=train_dataset.collate_fn)
    new_line = []
    file_idx = 1
    for step, batch_data in tqdm(enumerate(train_loader), total=len(train_dataset)):

        if batch_data[0] is None:
            continue

        head_mem_res, tail_mem_res, mrm_res = batch_data[0]

        for key in ['input_ids','attention_mask','token_type_ids','label_ids']:
            head_mem_res[key] = head_mem_res[key].squeeze().numpy().tolist()
            tail_mem_res[key] = tail_mem_res[key].squeeze().numpy().tolist()
            mrm_res[key] = mrm_res[key].squeeze().numpy().tolist()


        new_line.append((head_mem_res, tail_mem_res, mrm_res))
        if len(new_line) >150000:
            file_path = '../../Lpcorpus/{}_150000.pk'.format(file_idx)
            print("当前保存到文件", file_path)
            with open(file_path, 'wb') as f:
                pickle.dump(new_line, f)
            new_line = []
            file_idx += 1

    # print("文件存储到:{}".format(target_file_path))


if __name__ == '__main__':
    import time

    max_len = 512

    tokenizer_file_path = '../../../embedding/scibert_scivocab_uncased'
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_file_path)
    except:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_file_path)
    generate_lp_dataset(tokenizer, max_len=max_len)
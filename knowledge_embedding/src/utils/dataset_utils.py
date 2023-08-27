# -*- encoding: utf-8 -*-
"""
@File    :   dataset_utils.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/17 20:09   
@Description :   这个是用于KGBERT的dataset_utils


"""
import csv
import os
import sys
import random

import torch
from ipdb import set_trace
from torch.utils.data import Dataset
from tqdm import tqdm

from config import MyBertConfig


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
        """
        这个就是存储三元组作为三个text
        :param guid: 这是数据的id,没啥用
        :param text_a: 原始的head entity
        :param text_b:
        :param text_c:
        :param label: 这个就是表示triple的label，1或者0，test dataset可以不设置
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """
    Base class for data converters for sequence classification data sets.
    """

    def get_train_examples(self, data_dir):
        """
        返回训练集，a list of InputExample
        :param data_dir:
        :return:
        """
        raise not NotImplementedError

    def get_dev_examples(self, data_dir):
        """
        返回验证集，a list of InputExample
        :param data_dir:
        :return:
        """
        raise not NotImplementedError

    def get_test_examples(self, data_dir):
        """
        返回测试集，a list of InputExample
        :param data_dir:
        :return:
        """
        raise not NotImplementedError

    def read_csv_file(self, input_file, quotechar=None):
        """
        读取原始Linkpredication文件
        一行就是(head,rel,tail)
        :param input_file:
        :param quotechar:
        :return: [(h1,r1,t1),(h2,r2,t2),...]
        """
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in tqdm(reader,desc="读取:{}文件".format(input_file)):
                lines.append(line)
        return lines


class KGProcessor(DataProcessor):
    """
        对于知识图谱数据集的详细处理类
    """

    def __init__(self,negative_ratio=5,debug=False):
        """

        :param negative_ratio:  这个是negative sample的比例，表示一个positive会有negative_ratio个negative samples
        """

        self.negative_ratio = negative_ratio
        self.debug = debug

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self.read_csv_file(os.path.join(data_dir, "train.tsv"))
        if self.debug:
            lines = lines[:100]
        return self._create_examples(lines, "train", data_dir)

    def get_dev_examples(self, data_dir,type='dev'):
        """See base class."""
        lines = self.read_csv_file(os.path.join(data_dir, "dev.tsv"))
        if self.debug:
            lines = lines[:100]
        return self._create_examples(lines, type, data_dir)

    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self.read_csv_file(os.path.join(data_dir, "test.tsv"))
        if self.debug:
            lines = lines[:100]
        return self._create_examples(lines, "test", data_dir)

    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""

        with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

    def get_labels(self):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        return [0, 1]

    def get_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        with open(os.path.join(data_dir, "entities.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities
    def get_drug_entities(self,data_dir):
        """
        这个是专用于 drug-target interaction，获取所有的drug entity
        :param data_dir:
        :return:
        """
        with open(os.path.join(data_dir, "drug_entities.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities
    def get_target_entities(self,data_dir):
        """
        这个是专用于 drug-target interaction，获取所有的target entity
        之后破坏三元组
        :param data_dir:
        :return:
        """
        with open(os.path.join(data_dir, "target_entities.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities
    def get_train_triples(self, data_dir):
        """
            获取原始的数据集三元组
            也就是train.tsv,....
        """
        return self.read_csv_file(os.path.join(data_dir, "train.tsv"))

    def get_dev_triples(self, data_dir):

        return self.read_csv_file(os.path.join(data_dir, "dev.tsv"))

    def get_test_triples(self, data_dir):

        return self.read_csv_file(os.path.join(data_dir, "test.tsv"))

    def _create_examples(self, lines, set_type, data_dir):
        """
        Creates examples for the training and dev sets.
        lines:就是读取数据集的原始结果，一行都是三个
        """

        # entity to text
        ent2text = {}

        # 这是读取每个实体的文本形式
        # 这是因为FB15K-237,WinRR等数据集的实体都是一些代号啥的，
        # 这里就相当于把真正的实体名称弄来，这也是为了encode....
        with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    end = temp[1]  # .find(',')
                    ent2text[temp[0]] = temp[1]  # [:end]
        # 如果数据集是FK15,那么使用更长的文本进行描述,不适用短文本
        if data_dir.find("FB15") != -1:
            with open(os.path.join(data_dir, "entity2textlong.txt"), 'r') as f:
                ent_lines = f.readlines()
                for line in ent_lines:
                    temp = line.strip().split('\t')
                    # first_sent_end_position = temp[1].find(".")
                    ent2text[temp[0]] = temp[1]  # [:first_sent_end_position + 1]

        entities = list(ent2text.keys())

        rel2text = {}
        # 这个就是将relation使用文本进行代替
        # FK15，WN等数据的关系都是代码表示，这里进行替换一下
        # 对于umls则只是真正的名称，没有使用代号
        with open(os.path.join(data_dir, "relation2text.txt"), 'r') as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split('\t')
                rel2text[temp[0]] = temp[1]

        # 这个是去除数据集的重复， {'embryonic_structure\tpart_of\thuman', 'family_group\texhibits\tbehavior', 'anatomical_structure\tisa\tentity'}
        # 方便之后的negative sample，通过集合来判断是否重合

        lines_str_set = set(['\t'.join(line) for line in lines])
        examples = []

        for (i, line) in enumerate(lines):
            # 这里则相当于获得head,tail,relation的对应text

            head_ent_text = ent2text[line[0]]
            tail_ent_text = ent2text[line[2]]
            relation_text = rel2text[line[1]]


            if set_type == "dev" or set_type == "test":

                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text

                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=1))

            elif set_type == "train":
                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=1))

                rnd = random.random()
                guid = "%s-%s" % (set_type + "_corrupt", i)
                # 这个就是生成negative examples
                # 将head或者tail进行随机替换，然后生成negative
                # 然后生成5个negative

                if rnd <= 0.5:
                    # corrupting head
                    for _ in range(self.negative_ratio):
                        while True:
                            tmp_ent_list = set(entities)
                            tmp_ent_list.remove(line[0])
                            tmp_ent_list = list(tmp_ent_list)
                            tmp_head = random.choice(tmp_ent_list)
                            tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                            if tmp_triple_str not in lines_str_set:
                                break
                        tmp_head_text = ent2text[tmp_head]
                        examples.append(
                            InputExample(guid=guid, text_a=tmp_head_text, text_b=text_b, text_c=text_c, label=0))
                else:
                    # corrupting tail
                    for _ in range(self.negative_ratio):
                        while True:
                            tmp_ent_list = set(entities)
                            tmp_ent_list.remove(line[2])
                            tmp_ent_list = list(tmp_ent_list)
                            tmp_tail = random.choice(tmp_ent_list)
                            tmp_triple_str = line[0] + '\t' + line[1] + '\t' + tmp_tail
                            if tmp_triple_str not in lines_str_set:
                                break
                        tmp_tail_text = ent2text[tmp_tail]
                        examples.append(
                            InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=tmp_tail_text, label=0))
        return examples


class KGBertDataset(Dataset):
    def __init__(self, config: MyBertConfig, data, tokenizer):
        '''
        这个数据处理适合模型：Bert_CRF,BERT_BiLSTM_CRF,Bert_MLP
        :param data:InputExamples
        '''
        super(KGBertDataset, self).__init__()
        self.nums = len(data)

        # 这里的data就是InputExamples，格式为
        # ipdb> train_examples[0].text
        # ['Identification', 'of', 'APC2', ',', 'a', 'homologue', 'of', 'the', 'adenomatous', 'polyposis', 'coli', 'tumour', 'suppressor', '.']
        # train_examples[0].labels
        # ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'O', 'O']
        self.data = data
        self.max_len = config.max_len  # 这是每句话的最大长度
        self.nums = len(data)
        self.config = config
        self.tokenizer = tokenizer
        self.is_train = True

    def __len__(self):
        return self.nums

    def collate_fn(self, examples: InputExample):
        """
        这个函数用于DataLoader，一次处理一个batch的数据Input example
        :return:
        """

        batch_input_ids = []
        batch_attention_masks = []
        batch_token_type_ids = []
        batch_label_ids = []

        # 分别获得三元组head,rel,tail所在的位置mask
        batch_head_mask = []
        batch_rel_mask = []
        batch_tail_mask = []


        for (idx, example) in enumerate(examples):


            tokens_a = self.tokenizer.tokenize(example.text_a)

            tokens_b = None
            tokens_c = None

            if example.text_b and example.text_c:
                tokens_b = self.tokenizer.tokenize(example.text_b)
                tokens_c = self.tokenizer.tokenize(example.text_c)
                # Modifies `tokens_a`, `tokens_b` and `tokens_c`in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP], [SEP] with "- 4"
                # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
                _truncate_seq_triple(tokens_a, tokens_b, tokens_c, self.max_len - 4)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > self.max_len - 2:
                    tokens_a = tokens_a[:(self.max_len - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            # (c) for sequence triples:
            #  tokens: [CLS] Steve Jobs [SEP] founded [SEP] Apple Inc .[SEP]
            #  type_ids: 0 0 0 0 1 1 0 0 0 0
            head_mask = [0]+[1]*len(tokens_a)+[0]
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            token_type_ids = [0] * len(tokens)

            if tokens_b:
                rel_mask = [0]*len(head_mask)+[1]*len(tokens_b)+[0]
                tokens += tokens_b + ["[SEP]"]
                token_type_ids += [1] * (len(tokens_b) + 1)
            if tokens_c:
                tail_mask = [0]*len(rel_mask)+[1]*len(tokens_c)+[0]
                tokens += tokens_c + ["[SEP]"]
                token_type_ids += [0] * (len(tokens_c) + 1)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_masks = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (self.max_len - len(input_ids))
            input_ids += padding
            attention_masks += padding
            token_type_ids += padding

            # 对三个mask进行补全
            head_mask += [0]*(self.max_len-len(head_mask))
            rel_mask += [0]*(self.max_len-len(rel_mask))
            tail_mask += [0]*(self.max_len-len(tail_mask))


            assert len(input_ids) == self.max_len
            assert len(attention_masks) == self.max_len
            assert len(token_type_ids) == self.max_len

            label_id = example.label

            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_masks)
            batch_token_type_ids.append(token_type_ids)
            batch_label_ids.append(label_id)

            batch_head_mask.append(head_mask)
            batch_rel_mask.append(rel_mask)
            batch_tail_mask.append(tail_mask)


        batch_input_ids = torch.tensor(batch_input_ids).long()
        batch_attention_masks = torch.tensor(batch_attention_masks).long()
        batch_token_type_ids = torch.tensor(batch_token_type_ids).long()
        batch_label_ids = torch.tensor(batch_label_ids).long()

        batch_head_mask = torch.tensor(batch_head_mask).float()
        batch_rel_mask = torch.tensor(batch_rel_mask).float()
        batch_tail_mask = torch.tensor(batch_tail_mask).float()

        if self.config.model_name == 'kgbert':
            return batch_input_ids,batch_attention_masks,batch_token_type_ids,batch_label_ids
        else:
            return batch_input_ids, batch_attention_masks, batch_token_type_ids, batch_label_ids,batch_head_mask,batch_rel_mask,batch_tail_mask

    def __getitem__(self, index):
        return self.data[index]

def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence triple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
            tokens_b.pop()
        elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
            tokens_c.pop()
        else:
            tokens_c.pop()
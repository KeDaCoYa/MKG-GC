# -*- encoding: utf-8 -*-
"""
@File    :   dti_dataset_utils.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/5/1 22:15   
@Description :   None 

"""

import os
import json
import pickle
from src.utils.dataset_utils import DataProcessor

class DTIInputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, head_ent, rel=None, tail_ent=None, label=None):
        """
        这个就是存储三元组作为三个text
        :param guid: 这是数据的id,没啥用
        :param text_a: 原始的head entity
        :param text_b:
        :param text_c:
        :param label: 这个就是表示triple的label，1或者0，test dataset可以不设置
        """

        self.head_ent = head_ent
        self.rel = rel
        self.tail_ent = tail_ent
        self.label = label


class DTIProcessor(DataProcessor):
    """
        这个处理之后的药物发现任务
    """


    def get_train_examples(self, data_dir):
        """See base class."""
        file_path = os.path.join(data_dir, "train.pk")
        train_examples = pickle.load(file_path)

        return self._create_examples(train_examples, "train", data_dir)

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
    def get_vocab2id(self,data_dir):
        file_path = os.path.join(data_dir,'vocab2id.json')
        with open(file_path,'r',encoding='utf-8') as f:
            vocab2id = json.load(f)
        id2vocab = {}
        for v,id_ in vocab2id.items():
            id2vocab[id_] = v
        return vocab2id,id2vocab
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


    def _create_examples(self, lines,vocab2id):
        """
        将三元组转变为id形式
        :param lines:
        :param type_:
        :param data_dir:
        :return:
        """
        examples = []
        for line in lines:
            head_ent,rel,tai_ent,label = line
            head_id = vocab2id[head_ent.lower()]
            rel_id = 0
            tail_id = vocab2id[tai_ent.lower()]
            label = int(label)
            examples.append(DTIInputExample(head_id,rel_id,tail_id,label))
        return examples





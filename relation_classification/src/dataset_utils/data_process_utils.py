# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  数据读取
   Author :        kedaxia
   date：          2021/12/02
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/02: 
-------------------------------------------------
"""
import random

from ipdb import set_trace

import numpy as np

from gensim.models import Word2Vec, FastText


class InputExamples(object):
    def __init__(self, text, label, ent1_type, ent2_type, ent1_name, ent2_name, ent1_id=None, ent2_id=None,
                 abstract_id=None, rel_type=None):
        '''
        针对sentence-level的关系分类任务....
        :param text_a:
        :param text_b:
        :param label:
        :param ent1_type:
        :param ent2_type:
        '''
        self.text = text
        self.label = label
        self.ent1_type = ent1_type
        self.ent2_type = ent2_type
        self.ent1_name = ent1_name
        self.ent2_name = ent2_name
        self.ent1_id = ent1_id
        self.ent2_id = ent2_id
        self.abstract_id = abstract_id
        self.rel_type = rel_type


class MTBExamples(object):
    def __init__(self, text_a, text_b, label, ent1_type, ent2_type, ent1_name=None, ent2_name=None, ent1_id=None,
                 ent2_id=None, abstract_id=None, rel_type=None):
        '''
        MTB的cross-sentence 关系分类任务
        :param text_a:
        :param text_b:
        :param label:
        :param ent1_type:
        :param ent2_type:
        :param ent1_name:
        :param ent2_name:
        '''
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.ent1_type = ent1_type
        self.ent2_type = ent2_type
        self.ent1_name = ent1_name
        self.ent2_name = ent2_name
        self.ent1_id = ent1_id
        self.ent2_id = ent2_id
        self.abstract_id = abstract_id
        self.rel_type = rel_type


def load_pretrained_fasttext(fastText_embedding_path):
    '''
    加载预训练的fastText
    :param fastText_embedding_path:
    :return:fasttext,word2id,id2word
    '''
    fasttext = FastText.load(fastText_embedding_path)

    id2word = {i + 1: j for i, j in enumerate(fasttext.wv.index2word)}  # 共1056283个单词，也就是这些embedding
    word2id = {j: i for i, j in id2word.items()}
    fasttext = fasttext.wv.syn0
    word_hidden_dim = fasttext.shape[1]
    # 这是为了unk
    fasttext = np.concatenate([np.zeros((1, word_hidden_dim)), np.zeros((1, word_hidden_dim)), fasttext])
    return fasttext, word2id, id2word


def load_pretrained_word2vec(word2vec_embedding_path):
    '''
    加载预训练的fastText
    :param word2vec_embedding_path:
    :return:word2vec, word2id, id2word
    '''
    word2vec = Word2Vec.load(word2vec_embedding_path)

    # 空出0和1，0是pad，1是unknow
    id2word = {i + 2: j for i, j in enumerate(word2vec.wv.index2word)}  # 共1056283个单词，也就是这些embedding
    word2id = {j: i for i, j in id2word.items()}
    word2vec = word2vec.wv.syn0
    word_hidden_dim = word2vec.shape[1]
    # 这是为了pad和unk
    word2id['unk'] = 1
    word2id['pad'] = 0
    id2word[0] = 'pad'
    id2word[1] = 'unk'
    word2vec = np.concatenate([np.zeros((1, word_hidden_dim)), np.zeros((1, word_hidden_dim)), word2vec])

    # word2vec = np.concatenate([[copy.deepcopy(word2vec[0])], word2vec])

    return word2vec, word2id, id2word


def get_relative_pos_feature(x, limit):
    """
       :param x = idx - entity_idx
       这个方法就是不管len(sentence)多长，都限制到这个位置范围之内

       x的范围就是[-len(sentence),len(sentence)] 转换到都是正值范围
       -limit ~ limit => 0 ~ limit * 2+2
       将范围转换一下，为啥
   """
    if x < -limit:
        return 0
    elif x >= -limit and x <= limit:
        return x + limit + 1
    else:
        return limit * 2 + 2


def get_label2id(label_file):
    f = open(label_file, 'r')
    t = f.readlines()
    f.close()
    label2id = {}
    id2label = {}
    for i, label in enumerate(t):
        label = label.strip()
        label2id[label] = i
        id2label[i] = label

    return label2id, id2label


def read_semeval2010(sentences_file, labels_file):
    '''
        关系分类任务，一般是读取两个文件，sentence.txt labels.txt
        这里就是读取数据
        :param file_path:
        :return:
            sents:列表，每一个为元组(start_idx,end_idx,entity_name)
            labels:对应的关系类别
        '''
    sents = list()
    # Replace each token by its index if it is in vocab, else use index of unk_word
    with open(sentences_file, 'r') as f:
        for i, line in enumerate(f):
            # 这里分离出实体对和句子
            e1, e2, sent = line.strip().split('\t')
            words = sent.split(' ')  # 将句子划分为一个一个单词
            sents.append((e1, e2, words))

    # Replace each label by its index
    f = open(labels_file, 'r')
    labels = f.readlines()
    f.close()
    labels = [label.strip() for label in labels]

    return sents, labels


def read_file(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    t = f.readlines()
    f.close()
    res = [x.strip() for x in t]
    return res


def read_raw_data(config):
    '''
    这里根据不同的数据集，需要读取不同格式的数据集，但是最后输出会保持一致，一个是sentence，另一个是label
    :param config:
    :param type:
    :return:
    '''

    if config.data_format == 'single':  # 格式为<CLS>sentence a<sep>sentence b <sep>
        examples = process_raw_normal_data(config.dev_normal_path)

    elif config.data_format == 'cross':
        examples = process_raw_mtb_data(config.dev_mtb_path)
    else:
        raise ValueError("data_format错误")
    return examples


def read_data(config, type_='train'):
    '''
    这里根据不同的数据集，需要读取不同格式的数据集，但是最后输出会保持一致，一个是sentence，另一个是label
    :param config:
    :param type:
    :return:
    '''

    if config.dataset_name == 'semeval2010':
        if type_ == 'train':
            return read_semeval2010(config.train_file_path, config.train_labels_path)
        else:
            return read_semeval2010(config.dev_file_path, config.dev_labels_path)
    else:
        if config.data_format == 'single':  # 格式为<CLS>sentence a<sep>sentence b <sep>
            if type_ == 'train':
                examples = process_normal_data(config.train_normal_path, config.dataset_name)
            elif type_ == 'dev':
                examples = process_normal_data(config.dev_normal_path, config.dataset_name)
            elif type_ == 'test':
                examples = process_normal_data(config.test_normal_path, config.dataset_name)
        elif config.data_format == 'cross':
            if type_ == 'train':
                examples = process_mtb_data(config.train_mtb_path, config.dataset_name)
            elif type_ == 'dev':
                examples = process_mtb_data(config.dev_mtb_path, config.dataset_name)
            elif type_ == 'test':
                examples = process_mtb_data(config.test_mtb_path, config.dataset_name)
        elif config.data_format == 'inter':
            if type_ == 'train':
                examples = process_mtb_data(config.train_mtb_path, config.dataset_name)
            elif type_ == 'dev':
                examples = process_mtb_data(config.dev_mtb_path, config.dataset_name)
            elif type_ == 'test':
                examples = process_mtb_data(config.test_mtb_path, config.dataset_name)
        else:
            raise ValueError("data_format value error， please choise ['single','cross']")
        return examples


def process_mtb_data(file_path, dataset_name):
    '''

    :param file_path:
    :return:
    '''
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    res = []

    if dataset_name in ['DDI2013', 'LLL', 'HPRD-50', 'IEPA', 'AIMed','BioInfer']:  # 针对二分类数据
        for idx, line in enumerate(lines):
            line = line.strip()
            line = line.split('\t')

            sent1, sent2, ent1_name, ent2_name, ent1_type, ent2_type, label = line
            example = MTBExamples(sent1, sent2, label, ent1_type, ent2_type, ent1_name, ent2_name)
            res.append(example)
    elif dataset_name in ['BC6ChemProt', 'BC7DrugProt']:
        for idx, line in enumerate(lines[1:]):

            line = line.strip()  # 去除换行符
            line = line.split('\t')
            if dataset_name == 'BC6ChemProt':
                sent1, sent2, ent1_type, ent2_type, ent1_name, ent2_name, label, _, _, _ = line
                label2rel={
                    'CPR:1':1,
                    'CPR:2':2,
                    'CPR:3':3,
                    'CPR:4':4,
                    'CPR:5':5,
                    'CPR:6':6,
                    'CPR:7':7,
                    'CPR:8':8,
                    'CPR:9':9,
                    'CPR:10':10,
                }
            else:
                sent1, sent2, ent1_type, ent2_type, ent1_name, ent2_name, label, _, _ = line
                label2rel = {
                    'INHIBITOR': 1,
                    'PART-OF': 2,
                    'SUBSTRATE': 3,
                    'ACTIVATOR': 4,
                    'INDIRECT-DOWNREGULATOR': 5,
                    'ANTAGONIST': 6,
                    'INDIRECT-UPREGULATOR': 7,
                    'AGONIST': 8,
                    'DIRECT-REGULATOR': 9,
                    'PRODUCT-OF': 10,
                    'AGONIST-ACTIVATOR': 11,
                    'AGONIST-INHIBITOR': 12,
                    'SUBSTRATE_PRODUCT-OF': 130,

                }


            example = MTBExamples(sent1, sent2, label, ent1_type, ent2_type, ent1_name, ent2_name,rel_type=label2rel[label])
            res.append(example)
    elif dataset_name in ['BC5CDR', 'two_BC6', 'two_BC7']:
        for idx, line in enumerate(lines):
            line = line.strip()  # 去除换行符
            line = line.split('\t')
            sent1, sent2, ent1_type, ent2_type, ent1_name, ent2_name, label, _ = line
            example = MTBExamples(sent1, sent2, label, ent1_type, ent2_type, ent1_name, ent2_name)
            res.append(example)
    elif dataset_name == 'AllDataset':
        for idx, line in enumerate(lines[1:]):
            line = line[:-1]  # 去除换行符
            line = line.split('\t')

            sent1, sent2, ent1_name, ent2_name, ent1_type, ent2_type, label, _ = line
            label2rel = {
                '0': 1,
                '1': 2,
                '2': 3,
                '3': 4,
                '4': 5,
                '5': 6,
            }
            example = MTBExamples(sent1, sent2, label, ent1_type, ent2_type, ent1_name, ent2_name,rel_type=label2rel[label])
            res.append(example)
    else:
        raise ValueError("选择正确的数据集名称")
    return res


def process_raw_mtb_data(file_path):
    '''

    :param file_path:
    :return:
    '''
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    res = []
    for idx, line in enumerate(lines[1:]):
        line = line[:-1]  # 去除换行符
        line = line.split('\t')
        abstract_id, sent1, sent2, ent1_name, ent2_name, ent1_type, ent2_type, ent1_id, ent2_id, distance = line
        example = MTBExamples(sent1, sent2, None, ent1_type, ent2_type, ent1_name, ent2_name, ent1_id, ent2_id)
        res.append(example)
    return res


def process_raw_normal_data(file_path):
    """
    这是处理predicate所需要的raw dataset
    """
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    res = []
    for idx, line in enumerate(lines[1:]):
        line = line.strip()
        line = line.split('\t')

        abstract_id, sent, ent1_name, ent2_name, ent1_type, ent2_type, ent1_id, ent2_id, distance = line

        if (ent1_type,ent2_type) in [("Gene/Protein","Gene/Protein"),('DNA','Gene/Protein'),('Gene/Protein','DNA'),('RNA','Gene/Protein'),('Gene/Protein','RNA'),('RNA','RNA'),('DNA','DNA')]:
            rel_type = 1
        elif (ent1_type,ent2_type) in [("Chemical/Drug","Chemical/Drug")]:
            rel_type = 2

        elif (ent1_type, ent2_type) in [('Gene/Protein','Chemical/Drug'),("Chemical/Drug","Gene/Protein")]:
            rel_type = 3

        elif (ent1_type, ent2_type) in [("Gene/Protein","Disease"),("Disease","Gene/Protein")]:
            rel_type = 4
        elif (ent1_type, ent2_type) in [("Chemical/Drug","Disease"),("Disease","Chemical/Drug")]:
            rel_type = 5
        else:
            raise ValueError

        example = InputExamples(sent, None, ent1_type, ent2_type, ent1_name, ent2_name, ent1_id, ent2_id,
                                abstract_id=abstract_id,rel_type=rel_type)
        res.append(example)

    return res


def process_normal_data(file_path, dataset_name):
    """
    这是处理标准数据集，数据格式为normal格式
    :param file_path:
    :return:
    """

    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    res = []
    if dataset_name in ['2018n2c2_track2']:
        for line in lines:
            line = line.strip()
            rel_type, text_a, text_b, ent1_type, ent2_type, ent1_id, ent_id, _ = line.split('\t')
            example = MTBExamples(text_a, text_b, rel_type, ent1_type, ent2_type)
            res.append(example)
    elif dataset_name in ['euadr', 'GAD', 'DDI2013', 'LLL', 'HPRD-50', 'IEPA', 'AIMed','BioInfer','PPI','CPI','GDI']:  # 针对二分类数据
        for idx, line in enumerate(lines):

            line = line.strip()
            line = line.split('\t')
            if dataset_name in ['euadr', 'GAD','GDI']:
                line = line[1:]
            if dataset_name == 'CPI':
                sent, ent1_type, ent2_type, ent1_name, ent2_name, label, _ = line
            else:
                sent, ent1_name, ent2_name, ent1_type, ent2_type, label = line


            example = InputExamples(sent, label, ent1_type, ent2_type, ent1_name, ent2_name, rel_type=0)
            res.append(example)

    elif dataset_name in ['BC6ChemProt', 'BC7DrugProt']:
        for idx, line in enumerate(lines[1:]):

            line = line.strip()
            line = line.split('\t')

            if dataset_name == 'BC6ChemProt':
                sent, ent1_type, ent2_type, ent1_name, ent2_name, label, _, _, _ = line
                label2rel={
                    'CPR:1':1,
                    'CPR:2':2,
                    'CPR:3':3,
                    'CPR:4':4,
                    'CPR:5':5,
                    'CPR:6':6,
                    'CPR:7':7,
                    'CPR:8':8,
                    'CPR:9':9,
                    'CPR:10':10,
                }
            else:
                sent, ent1_type, ent2_type, ent1_name, ent2_name, label, _, _ = line

                label2rel = {
                    'INHIBITOR': 1,
                    'PART-OF': 2,
                    'SUBSTRATE': 3,
                    'ACTIVATOR': 4,
                    'INDIRECT-DOWNREGULATOR': 5,
                    'ANTAGONIST': 6,
                    'INDIRECT-UPREGULATOR': 7,
                    'AGONIST': 8,
                    'DIRECT-REGULATOR': 9,
                    'PRODUCT-OF': 10,
                    'AGONIST-ACTIVATOR': 11,
                    'AGONIST-INHIBITOR': 12,
                    'SUBSTRATE_PRODUCT-OF': 130,

                }

            example = InputExamples(sent, label, ent1_type, ent2_type, ent1_name, ent2_name, rel_type=label2rel[label])
            res.append(example)
    elif dataset_name in ['BC5CDR', 'two_BC6', 'two_BC7','CDI']:
        for idx, line in enumerate(lines):
            line = line.strip()
            line = line.split('\t')
            sent, ent1_type, ent2_type, ent1_name, ent2_name, label, _ = line

            example = InputExamples(sent, label, ent1_type, ent2_type, ent1_name, ent2_name,rel_type=0)
            res.append(example)
    elif dataset_name == 'AllDataset' or 'CV' in dataset_name:
        for idx, line in enumerate(lines[1:]):
            line = line.strip()
            line = line.split('\t')
            sent, ent1_name, ent2_name, ent1_type, ent2_type, label, _ = line

            if ent1_name == ent2_name:
                continue
            # if (ent1_type,ent2_type) in [("Gene/Protein","Gene/Protein"),('DNA','Gene/Protein'),('Gene/Protein','DNA'),('RNA','Gene/Protein'),('Gene/Protein','RNA'),('RNA','RNA'),('DNA','DNA')]:
            #     rel_type = 1
            # elif (ent1_type,ent2_type) in [("Chemical/Drug","Chemical/Drug")]:
            #     rel_type = 2
            #
            # elif (ent1_type, ent2_type) in [('Gene/Protein','Chemical/Drug'),("Chemical/Drug","Gene/Protein")]:
            #     rel_type = 3
            # elif (ent1_type, ent2_type) in [("Gene/Protein","Disease"),("Disease","Gene/Protein")]:
            #     rel_type = 4
            # elif (ent1_type, ent2_type) in [("Chemical/Drug","Disease"),("Disease","Chemical/Drug")]:
            #     rel_type = 5
            if (ent1_type,ent2_type) in [("protein","protein")]:
                rel_type = 1
            elif (ent1_type,ent2_type) in [("drug","drug")]:
                rel_type = 2

            elif (ent1_type, ent2_type) in [('CHEMICAL','protein'),("protein","CHEMICAL")]:
                rel_type = 3
            elif (ent1_type, ent2_type) in [("GENE","DISEASE"),("DISEASE","GENE")]:
                rel_type = 4
            elif (ent1_type, ent2_type) in [("Chemical","Disease"),("Disease","Chemical")]:
                rel_type = 5
            else:
                print(ent1_type,ent2_type)
                raise ValueError


            example = InputExamples(sent, label, ent1_type, ent2_type, ent1_name, ent2_name,rel_type=rel_type)
            res.append(example)


    else:
        raise ValueError("选择正确的数据集名称")
    return res


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    '''
    这里对数据进行pad，不同的batch里面使用不同的长度
    这个方法从多个方面考虑pad，写的很高级
    这个方法一般写不出来，阿西吧


    Numpy函数，将序列padding到同一长度
    按照一个batch的最大长度进行padding
    :param inputs:(batch_size,None),每个序列的长度不一样
    :param seq_dim: 表示对哪些维度进行pad，默认为1，只有当对label进行pad的时候，seq_dim=3,因为labels.shape=(batch_size,entity_type,seq_len,seq_len)
        因为一般都是对(batch_size,seq_len)进行pad，，，
    :param length: 这个是设置补充之后的长度，一般为None，根据batch的实际长度进行pad
    :param value:
    :param mode:
    :return:
    '''
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)  # length=np.array([max_batch_length])
    elif not hasattr(length, '__getitem__'):  # 如果这个length的类别不是列表....,就进行转变
        length = [length]
    # logger.info('这个batch下面的最长长度为{}'.format(length[0]))

    slices = [np.s_[:length[i]] for i in
              range(seq_dims)]  # 获得针对针对不同维度的slice，对于seq_dims=0,slice=[None:max_len:None],max_len是seq_dims的最大值
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]  # 有多少个维数，就需要多少个(0,0),一般是一个

    outputs = []
    for x in inputs:
        # X为一个列表
        # 这里就是截取长度
        x = x[slices]
        for i in range(seq_dims):  # 对不同的维度逐步进行扩充
            if mode == 'post':
                # np.shape(x)[i]是获得当前的实际长度
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)


if __name__ == '__main__':
    sentences_file = './general_domain_dataset/semeval2008/mid_dataset/train/sentences.txt'
    labels_file = './general_domain_dataset/semeval2008/mid_dataset/train/labels.txt'
    sents, labels = read_data(sentences_file, labels_file)
    label2id, id2label = get_label2id('./general_domain_dataset/semeval2008/mid_dataset/labels.txt')

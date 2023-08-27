# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description : 这里定义数据的转换,这里的代码基本放弃使用
   Author :        kedaxia
   date：          2021/11/08
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/11/08:
-------------------------------------------------
"""


import os
import pickle
import copy
from ipdb import set_trace
from tqdm import tqdm
import numpy as np

from gensim.models import Word2Vec,FastText

from src.dataset_util.base_dataset import NormalCRFFeature, CRFFeature, SpanFeature


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
    fasttext = np.concatenate([np.zeros((1, word_hidden_dim)),np.zeros((1, word_hidden_dim)), fasttext])
    return fasttext,word2id,id2word


def load_pretrained_word2vec(word2vec_embedding_path):
    '''
    加载预训练的fastText
    :param word2vec_embedding_path:
    :return:word2vec, word2id, id2word
    '''
    word2vec = Word2Vec.load(word2vec_embedding_path)

    id2word = {i + 1: j for i, j in enumerate(word2vec.wv.index2word)}  # 共1056283个单词，也就是这些embedding
    word2id = {j: i for i, j in id2word.items()}
    word2vec = word2vec.wv.syn0
    word_hidden_dim = word2vec.shape[1]
    # 这是为了pad和unk
    word2vec = np.concatenate([np.zeros((1, word_hidden_dim)),np.zeros((1, word_hidden_dim)), word2vec])

   # word2vec = np.concatenate([[copy.deepcopy(word2vec[0])], word2vec])

    return word2vec, word2id, id2word



def build_map(lists):

    maps = {'pad':0,'unk':1}
    
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps

def build_vocab(word_lists,vocab_path):
    """
        根据训练集数据构建字典
        word_lists(按照word进行划分)，是readlines

    
    """
    
    #建立标签和index的map  {'B-NAME': 0, 'E-NAME': 1, 'O': 2, 'B-CONT': 3, 'M-CONT': 4}

    if os.path.exists(vocab_path):
        word2id = pickle.load(open(vocab_path,"rb"))
    else:
        # 建立单词和index之间的映射'高': 0,'勇': 1,'：': 2,'男': 3,'，': 4,'中': 5,}，不含有重复值
        word2id = build_map(word_lists)
        pickle.dump(word2id,open(vocab_path,"wb"))
    
    return  word2id
    



def sort_by_lengths(word_lists, label_ids):
    '''
    将所有的数据按照长度进行排序，自上到下
    :param word_lists:
    :param tag_lists:
    :return:
    '''
    pairs = list(zip(word_lists, label_ids))

    #将原来的索引顺序保存下来
    indices = sorted(range(len(pairs)),key=lambda k: len(pairs[k][0]),reverse=True)
    pairs = [pairs[i] for i in indices]
    # pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

    word_lists, tag_lists = list(zip(*pairs))

    return word_lists, tag_lists, indices


def read_data(file_path):
    data = []
    labels = []
    tmp_data = []
    tmp_label = []
    f = open(file_path, 'r', encoding='utf-8')
    t = f.readlines()
    f.close()
    for words in t:
        if words == '\n':
            assert len(tmp_data) == len(tmp_label), 'error'
            if tmp_data and tmp_label:
                data.append(tmp_data)
                labels.append(tmp_label)
            tmp_data = []
            tmp_label = []
        else:
            words_ = words.strip()
            try:
                word, label = words_.split()
            except:
                continue
            tmp_data.append(word)
            tmp_label.append(label)

    return data,labels



def vectorization(X,y,word2id,tag2id):
    '''
    进行向量化
    :param X:
    :param y:
    :param word2id:
    :param tag2id:
    :return:
    '''
    #获取每个batch中序列的最高长度，一最高长度为基准，若低于此长度就用PAD进行补充
    #每个batch的序列最高长度是都不一样的
    max_seq_len = len(X[0])
    vector_x =[]
    vector_y = []
    for line in X:
        tmp_x = []
        for ind in range(max_seq_len):
            if ind < len(line):
                #根据word2id字典中的值进行转换
                tmp_x.append(word2id.get(line[ind],word2id.get('unk')))
            else:
                tmp_x.append(word2id.get('pad'))
        vector_x.append(tmp_x)
    for line in y:
        tmp_y = []
        for ind in range(max_seq_len):
            if ind < len(line):
                tmp_y.append(tag2id.get(line[ind],tag2id.get('O')))
            else:
                #对补位补充”O“
                tmp_y.append(tag2id.get('O'))
        vector_y.append(tmp_y)
    return vector_x,vector_y


def normal_convert_example_to_crf_features(examples, word2id, label2id, max_len):
    '''
        将所有的InputExample转变为Input Feature
        这是将所有的数据转变为固定长度
    '''
    features = []
    callback_info = []
    for example in tqdm(examples):
        raw_text_li = example.text
        callback_info.append(raw_text_li)
        labels = example.labels

        # 开始对token进行向量化
        token_ids = [word2id.get(word, word2id.get('unk')) for word in raw_text_li]
        label_ids = [label2id.get(c) for c in labels]
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
            label_ids = label_ids[:max_len]
        else:
            pad_len = max_len - len(label_ids)
            # 全部补齐为 O
            label_ids = label_ids + [0] * pad_len
            token_ids = token_ids + [0] * pad_len

        feature = NormalCRFFeature(token_ids=token_ids, labels=label_ids)
        features.append(feature)
    return features,callback_info

def normal_convert_example_to_globalpointer_features(examples, word2id, label2id, max_len):
    '''
        将所有的InputExample转变为Input Feature,针对globalpointer的数据
        这是将所有的数据转变为固定长度
        只需要对label进行修改即可
    '''
    features = []
    callback_info = []
    for example in tqdm(examples):
        raw_text_li = example.text
        callback_info.append(raw_text_li)
        true_labels = example.labels

        # 开始对token进行向量化
        token_ids = [word2id.get(word, word2id.get('unk')) for word in raw_text_li]
        labels = np.zeros((1,max_len,max_len))
        start_index = 0
        # 获取正确的set
        actual_len = len(true_labels)
        while start_index < actual_len:

            if true_labels[start_index] == 'B' and start_index + 1 < actual_len and true_labels[
                start_index + 1] == 'O':  # 实体是一个单词
                labels[0, start_index, start_index] = 1
                start_index += 1
            elif true_labels[start_index] == 'B' and start_index + 1 >= actual_len:  # 最后只有一个实体，并且只有一个单词，到达了最后
                labels[0, start_index, start_index] = 1
                break
            elif true_labels[start_index] == 'B':
                j = start_index + 1
                while j < actual_len:
                    if true_labels[j] == 'I':
                        j += 1
                    else:
                        labels[0, start_index, j - 1] = 1

                        break
                if j >= actual_len:
                    if true_labels[j - 1] == 'I':
                        labels[0, start_index, j - 1] = 1

                start_index = j
            else:
                start_index += 1

        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]

        else:
            pad_len = max_len - len(true_labels)
            # 全部补齐为 O

            token_ids = token_ids + [0] * pad_len

        feature = NormalCRFFeature(token_ids=token_ids, labels=labels)
        features.append(feature)
    return features,callback_info

def convert_example_to_crf_features(examples, label2id, tokenizer, max_len,shuffle=False):
    '''

        将所有的InputExample(全部数据)转变为Input Feature
    :param examples:
    :param label2id:
    :param tokenizer:
    :param max_len:
    :param shuffle: 默认为False，在这里进行shuffle，而不是使用DataLoader，这是因为这里shuffle才能用于对训练集的p,r,f1
    :return:
    '''

    sequence_ids = list(range(len(examples)))
    if shuffle:
        np.random.shuffle(sequence_ids)

    features = []
    callback_info = []
    for ids in sequence_ids:
        example = examples[ids]
        raw_text_li = example.text
        if len(raw_text_li) > max_len - 2:
            continue
        callback_info.append(raw_text_li)
        labels = example.labels

        label_ids = [label2id.get(c) for c in labels]


        label_ids = [0] + label_ids + [0]
        pad_len = max_len - len(label_ids)
        label_ids = label_ids + [0] * (pad_len)
        # 这里自动在开头添加[cls],[sep]
        res = tokenizer.encode_plus(text=raw_text_li, max_length=max_len, pad_to_max_length=True, truncation=True,
                                    is_pretokenized=False, return_token_type_ids=True, return_attention_mask=True)


        feature = CRFFeature(token_ids=res['input_ids'], attention_masks=res['attention_mask'], labels=label_ids,
                             token_type_ids=res['token_type_ids'])
        features.append(feature)
    return features,callback_info


def convert_example_to_span_features(examples, tokenizer, max_len):
    '''
        将所有的InputExample转变为Input Feature
    '''
    features = []
    callback_info = []
    for example in tqdm(examples):

        raw_text_li = example.text

        callback_info.append(raw_text_li)
        if len(raw_text_li) > max_len - 2:
            continue
        start_ids = [0] * len(raw_text_li)
        end_ids = [0] * len(raw_text_li)

        labels = example.labels

        # 开始根据BIO的标注模式转变为ＳＰＡＮ
        start_index = 0

        while start_index < len(raw_text_li):

            if labels[start_index] == 'B':
                start_ids[start_index] = 1
                start_index += 1
                while start_index < len(raw_text_li) and labels[start_index] == 'I':
                    start_index += 1
                if start_index == len(raw_text_li) or labels[start_index] == 'O':
                    end_ids[start_index - 1] = 1
            else:
                start_index += 1

        start_ids = [0] + start_ids + [0]
        end_ids = [0] + end_ids + [0]
        pad_len = max_len - len(start_ids)

        start_ids = start_ids + [0] * pad_len
        end_ids = end_ids + [0] * pad_len

        # 这里自动在开头添加[cls],[sep]
        res = tokenizer.encode_plus(text=raw_text_li, max_length=max_len, pad_to_max_length=True, truncation=True,
                                    is_pretokenized=False, return_token_type_ids=True, return_attention_mask=True)

        feature = SpanFeature(token_ids=res['input_ids'], attention_masks=res['attention_mask'], start_ids=start_ids,
                              end_ids=end_ids, token_type_ids=res['token_type_ids'])

        features.append(feature)
    return features,callback_info



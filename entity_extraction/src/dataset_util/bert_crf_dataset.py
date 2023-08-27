# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  只要是decoder是CRF、MLP都是用这个进行dataset...
   Author :        kedaxia
   date：          2021/11/25
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/11/25: 
-------------------------------------------------
"""
from ipdb import set_trace

import torch
from torch.utils.data import Dataset
import logging
import numpy as np
from config import MyBertConfig
from src.dataset_util.base_dataset import sequence_padding, tokenize_text, tokenize_text_predicate

logger = logging.getLogger('main.crf_dataset')


class BertCRFDataset(Dataset):
    def __init__(self,features):
        '''
            在这里将所有数据转变为tensor，可以直接使用
        :param data:
        '''
        super(BertCRFDataset,self).__init__()
        self.nums = len(features)
        # 因为这里一次只取一个，所以一个tensor就行了...
        self.tokens_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks).long() for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]

        self.labels = [torch.tensor(example.labels).long() for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {
            'token_ids':self.tokens_ids[index],
            'attention_masks':self.attention_masks[index],
            'token_type_ids':self.token_type_ids[index],
            'labels':self.labels[index],

        }
        return data



class BertCRFDataset_dynamic(Dataset):
    def __init__(self,config:MyBertConfig,data,tokenizer):
        '''
        这个数据处理适合模型：Bert_CRF,BERT_BiLSTM_CRF,Bert_MLP
        :param data:
        '''
        super(BertCRFDataset_dynamic,self).__init__()
        self.nums = len(data)

        # 这里的data就是InputExamples，格式为
        #ipdb> train_examples[0].text
        # ['Identification', 'of', 'APC2', ',', 'a', 'homologue', 'of', 'the', 'adenomatous', 'polyposis', 'coli', 'tumour', 'suppressor', '.']
        # train_examples[0].labels
        # ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'O', 'O']
        self.data = data
        self.num_tags = config.num_crf_class
        self.max_len = config.max_len #这是每句话的最大长度
        self.nums = len(data)
        self.config = config
        self.tokenizer = tokenizer
        self.label2id = config.crf_label2id



    def __len__(self):
        return self.nums

    def normal_encoder(self, item):
        '''
        这是对一个数据的encode
        一般中文使用此方式进行encode
        :param item:
        :return:
        '''

        raw_text = item.text

        encoder_res = self.tokenizer.encode_plus(raw_text, truncation=True, max_length=self.max_len)
        input_ids = encoder_res['input_ids']
        token_type_ids = encoder_res['token_type_ids']
        attention_mask = encoder_res['attention_mask']
        return raw_text, input_ids, token_type_ids, attention_mask


    def tokenize_encoder(self,item):
        '''
        这是对一个数据(InputExample)的encoder
        :param item:
            raw_text: 这个就是未分词之前的word list
            true_labels:
        :return:
        '''

        raw_text = item.text
        true_labels = item.labels

        true_labels = [self.label2id[label] for label in true_labels]

        subword_tokens,origin_to_subword_index = tokenize_text(self.tokenizer,item,self.max_len)

        ## 这里需要截断，因为因为subword可能太长，超过了config.max_len,需要截断，因此true label和raw_text都会进行截断
        true_labels = true_labels[:len(origin_to_subword_index)]
        raw_text = raw_text[:len(origin_to_subword_index)]

        encoder_res = self.tokenizer.encode_plus(subword_tokens,truncation=True,max_length=self.max_len)
        subword_input_ids = encoder_res['input_ids']
        subword_token_type_ids = encoder_res['token_type_ids']
        subword_attention_mask = encoder_res['attention_mask']
        origin_to_subword_index = [x+1 for x in origin_to_subword_index]

        return raw_text,true_labels,subword_input_ids,subword_token_type_ids,subword_attention_mask,origin_to_subword_index

    def tokenize_encoder_predicate(self, item):
        '''
        这是对一个数据(InputExample)的encoder
        :param item:
            raw_text: 这个就是未分词之前的word list
            true_labels:
        :return:
        '''

        raw_text = item.text


        subword_tokens, origin_to_subword_index = tokenize_text_predicate(self.tokenizer, item, self.max_len)


        raw_text = raw_text[:len(origin_to_subword_index)]

        encoder_res = self.tokenizer.encode_plus(subword_tokens, truncation=True, max_length=self.max_len)
        subword_input_ids = encoder_res['input_ids']
        subword_token_type_ids = encoder_res['token_type_ids']
        subword_attention_mask = encoder_res['attention_mask']
        origin_to_subword_index = [x + 1 for x in origin_to_subword_index]

        return raw_text, subword_input_ids, subword_token_type_ids, subword_attention_mask, origin_to_subword_index
    def collate_fn(self, examples):
        '''
        这个函数用于DataLoader，一次处理一个batch的数据Input example

        :param features: 这个就是一个InputExample,有两个属性
            1. raw_text:['IL-13', ',', 'in', 'the', 'peripheral', 'blood', ';', '(', '2', ')', 'the']
            2. labels:['B-protein', 'O', 'O', 'B-cell_type', 'O']

        :return:
        '''

        raw_text_list = []
        batch_subword_input_ids = [] # 这是针对token-level，这里数据一般
        batch_subword_attention_masks = []

        batch_subword_token_type_ids = []
        batch_true_labels = []
        # 这个是对subword的一个mask，只保留subword的首词为1，其余为0
        batch_label_mask = []
        origin_to_subword_indexs = []

        # subword之后的值
        batch_subword_max_len = 0
        for item in examples: #一个item都是一个InputExample，在这里进行tokenize

            # 这里开始进行数据转化
            raw_text,true_labels,subword_input_ids,subword_token_type_ids,subword_attention_mask,origin_to_subword_index = self.tokenize_encoder(item)

            # 预留special token，在crf中这两个special token非常重要，相当于<start>,<end>,不能少
            tmp_len = sum(subword_attention_mask)-2
            if tmp_len>batch_subword_max_len:
                batch_subword_max_len = tmp_len

            raw_text_list.append(raw_text)
            batch_subword_input_ids.append(subword_input_ids)
            batch_subword_attention_masks.append(subword_attention_mask)
            batch_subword_token_type_ids.append(subword_token_type_ids)
            origin_to_subword_indexs.append(origin_to_subword_index)

            batch_true_labels.append(true_labels)

        for i,index in enumerate(origin_to_subword_indexs):
            label_mask = np.zeros(batch_subword_max_len+2,dtype=np.int)
            for j in index:
                label_mask[j] = 1

            batch_label_mask.append(label_mask)


        # 这里batch_input_ids,batch_segment_ids,batch_attention_mask都没有进行pad，这里需要进行pad，将长度进行统一补齐...
        if self.config.fixed_batch_length:
            pad_length = self.max_len
        else:
            pad_length = min(batch_subword_max_len, self.max_len)

        # 也需要对origin_to_subword_index进行检查
        new_origin_to_subword_indexs = []
        for i, subword_index in enumerate(origin_to_subword_indexs):
            new_index = []
            for ele in subword_index:
                if ele >= pad_length:
                    break
                else:
                    new_index.append(ele)
            new_origin_to_subword_indexs.append(new_index)
        # input_true_length = [len(x) for x in new_origin_to_subword_indexs]
        # input_true_length = torch.tensor(input_true_length).long()

        batch_origin_to_subword_indexs = torch.tensor(
            sequence_padding(new_origin_to_subword_indexs, length=pad_length)).long()

        # 这里是自动pad，按照每个batch的最大长度进行pad
        batch_subword_input_ids = torch.tensor(sequence_padding(batch_subword_input_ids,length=pad_length)).long()
        batch_subword_token_type_ids = torch.tensor(sequence_padding(batch_subword_token_type_ids,length=pad_length)).long()
        batch_subword_attention_masks = torch.tensor(sequence_padding(batch_subword_attention_masks,length=pad_length)).float()
        # 这里对labels也进行切分你
        # lens = max([len(x) for x in new_origin_to_subword_indexs])
        # set_trace()
        batch_true_labels = torch.tensor(sequence_padding(batch_true_labels,value=-1,length=pad_length)).long()
        batch_label_mask = torch.tensor(sequence_padding(batch_label_mask,length=pad_length)).long()

        return raw_text_list, batch_true_labels,batch_subword_input_ids,batch_subword_token_type_ids,batch_subword_attention_masks,batch_origin_to_subword_indexs,batch_label_mask
    def collate_fn_predicate(self, examples):
        '''
        这个函数用于DataLoader，一次处理一个batch的数据Input example

        :param features: 这个就是一个InputExample,有两个属性
            1. raw_text:['IL-13', ',', 'in', 'the', 'peripheral', 'blood', ';', '(', '2', ')', 'the']
            2. labels:['B-protein', 'O', 'O', 'B-cell_type', 'O']

        :return:
        '''

        raw_text_list = []
        batch_subword_input_ids = [] # 这是针对token-level，这里数据一般
        batch_subword_attention_masks = []

        batch_subword_token_type_ids = []

        # 这个是对subword的一个mask，只保留subword的首词为1，其余为0
        batch_label_mask = []
        origin_to_subword_indexs = []

        # subword之后的值

        batch_subword_max_len = 0
        for item in examples: #一个item都是一个InputExample，在这里进行tokenize

            # 这里开始进行数据转化

            raw_text,subword_input_ids,subword_token_type_ids,subword_attention_mask,origin_to_subword_index = self.tokenize_encoder_predicate(item)

            # 预留special token，在crf中这两个special token非常重要，相当于<start>,<end>,不能少
            tmp_len = sum(subword_attention_mask)-2
            if tmp_len>batch_subword_max_len:
                batch_subword_max_len = tmp_len

            raw_text_list.append(raw_text)
            batch_subword_input_ids.append(subword_input_ids)
            batch_subword_attention_masks.append(subword_attention_mask)
            batch_subword_token_type_ids.append(subword_token_type_ids)
            origin_to_subword_indexs.append(origin_to_subword_index)



        for i,index in enumerate(origin_to_subword_indexs):
            label_mask = np.zeros(batch_subword_max_len+2,dtype=np.int)
            for j in index:
                label_mask[j] = 1

            batch_label_mask.append(label_mask)


        # 这里batch_input_ids,batch_segment_ids,batch_attention_mask都没有进行pad，这里需要进行pad，将长度进行统一补齐...
        if self.config.fixed_batch_length:
            pad_length = self.max_len
        else:
            pad_length = min(batch_subword_max_len, self.max_len)

        # 也需要对origin_to_subword_index进行检查
        new_origin_to_subword_indexs = []
        for i, subword_index in enumerate(origin_to_subword_indexs):
            new_index = []
            for ele in subword_index:
                if ele >= pad_length:
                    break
                else:
                    new_index.append(ele)
            new_origin_to_subword_indexs.append(new_index)

        # 这里是自动pad，按照每个batch的最大长度进行pad
        batch_subword_input_ids = torch.tensor(sequence_padding(batch_subword_input_ids)).long()
        batch_subword_token_type_ids = torch.tensor(sequence_padding(batch_subword_token_type_ids)).long()
        batch_subword_attention_masks = torch.tensor(sequence_padding(batch_subword_attention_masks)).float()


        batch_label_mask = torch.tensor(sequence_padding(batch_label_mask)).long()



        return raw_text_list,batch_subword_input_ids,batch_subword_token_type_ids,batch_subword_attention_masks,new_origin_to_subword_indexs,batch_label_mask
    def __getitem__(self, index):

        return self.data[index]



class BertCRFDataset_test(Dataset):
    def __init__(self,config:MyBertConfig,data,tokenizer):
        '''
        这个数据处理适合模型：Bert_CRF,BERT_BiLSTM_CRF,Bert_MLP
        :param data:
        '''
        super(BertCRFDataset_test,self).__init__()
        self.nums = len(data)

        # 这里的data就是InputExamples，格式为
        #ipdb> train_examples[0].text
        # ['Identification', 'of', 'APC2', ',', 'a', 'homologue', 'of', 'the', 'adenomatous', 'polyposis', 'coli', 'tumour', 'suppressor', '.']
        # train_examples[0].labels
        # ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'O', 'O']
        self.data = data
        self.num_tags = config.num_crf_class
        self.max_len = config.max_len #这是每句话的最大长度
        self.nums = len(data)
        self.config = config
        self.tokenizer = tokenizer
        self.label2id = config.crf_label2id
        self.is_train = True


    def __len__(self):
        return self.nums

    def normal_encoder(self, item):
        '''
        这是对一个数据的encode
        一般中文使用此方式进行encode
        :param item:
        :return:
        '''
        if self.is_train:
            raw_text = item.text

            encoder_res = self.tokenizer.encode_plus(raw_text, truncation=True, max_length=self.max_len)
            input_ids = encoder_res['input_ids']
            token_type_ids = encoder_res['token_type_ids']
            attention_mask = encoder_res['attention_mask']
            return raw_text, input_ids, token_type_ids, attention_mask
        else:
            pass

    def tokenize_encoder(self,item):
        '''
        这是对一个数据(InputExample)的encoder
        :param item:
            raw_text: 这个就是未分词之前的word list
            true_labels:
        :return:
        '''
        if self.is_train:
            raw_text = item.text
            true_labels = item.labels
            true_labels = [self.label2id[label] for label in true_labels]
            subword_tokens,origin_to_subword_index = tokenize_text(self.tokenizer,item,self.max_len)

            ## 这里需要截断，因为因为subword可能太长，超过了config.max_len,需要截断，因此true label和raw_text都会进行截断
            true_labels = true_labels[:len(origin_to_subword_index)]
            raw_text = raw_text[:len(origin_to_subword_index)]

            encoder_res = self.tokenizer.encode_plus(subword_tokens,truncation=True,max_length=self.max_len)
            subword_input_ids = encoder_res['input_ids']
            subword_token_type_ids = encoder_res['token_type_ids']
            subword_attention_mask = encoder_res['attention_mask']
            origin_to_subword_index = [x+1 for x in origin_to_subword_index]

            return raw_text,true_labels,subword_input_ids,subword_token_type_ids,subword_attention_mask,origin_to_subword_index
        else:
            pass

    def collate_fn(self, examples):
        '''
        这个函数用于DataLoader，一次处理一个batch的数据Input example

        :param features: 这个就是一个InputExample,有两个属性
            1. raw_text:['IL-13', ',', 'in', 'the', 'peripheral', 'blood', ';', '(', '2', ')', 'the']
            2. labels:['B-protein', 'O', 'O', 'B-cell_type', 'O']

        :return:
        '''

        features = []
        batch_subword_input_ids = []
        batch_subword_token_type_ids = []
        batch_subword_attention_masks = []
        batch_subwords_labels = []
        batch_labels = []

        for (ex_index, example) in enumerate(examples):

            label_mask = []
            tokens = []
            label_ids = []
            subword_tokens = []
            for word, label in zip(example.text, example.labels):
                word_tokens = self.tokenizer.tokenize(word)
                subword_tokens.extend(word_tokens)
                # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    # 这里竟然只要subword的第一个单词作为有效标签，但我觉得这并不OK，但是好像都是这样使用，我的加权平均也失败了....
                    label_ids.extend([self.config.crf_label2id[label]] + [-1] * (len(word_tokens) - 1))

            label_ids = [-1]+label_ids+[-1]
            batch_labels.append([self.config.crf_label2id[x] for x in example.labels])
            encoder_res = self.tokenizer.encode_plus(subword_tokens, truncation=True, max_length=self.max_len)
            subword_input_ids = encoder_res['input_ids']
            subword_token_type_ids = encoder_res['token_type_ids']
            subword_attention_mask = encoder_res['attention_mask']

            batch_subword_input_ids.append(subword_input_ids)
            batch_subword_token_type_ids.append(subword_token_type_ids)
            batch_subword_attention_masks.append(subword_attention_mask)
            batch_subwords_labels.append(label_ids)



        batch_subword_input_ids = torch.tensor(sequence_padding(batch_subword_input_ids, length=self.max_len)).long()
        batch_subword_token_type_ids = torch.tensor(sequence_padding(batch_subword_token_type_ids, length=self.max_len)).long()
        batch_subword_attention_masks = torch.tensor(sequence_padding(batch_subword_attention_masks, length=self.max_len)).float()

        batch_subwords_labels = torch.tensor(sequence_padding(batch_subwords_labels, value=-1, length=self.max_len)).long()



        return batch_subword_input_ids,batch_subword_token_type_ids,batch_subword_attention_masks,batch_subwords_labels,batch_labels
    def __getitem__(self, index):

        return self.data[index]


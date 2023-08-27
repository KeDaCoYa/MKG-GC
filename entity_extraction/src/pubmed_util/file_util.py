# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2022/01/08
   Copyright:      (c) kedaxia 2022
-------------------------------------------------
   Change Activity:
                   2022/01/08: 
-------------------------------------------------
"""
import os
import json

from nltk.tokenize import sent_tokenize
from nltk.tokenize import wordpunct_tokenize

from ipdb import set_trace

def read_abstract_text(abstract_dir = '/root/code/bioner/ner/pubmed_abstracts/cancer_pubmed/abstracts',type_='json'):

    abstracts_li = []
    file_name_li = os.listdir(abstract_dir)
    for file_name in file_name_li:
        file_path = abstract_dir+'/'+file_name
        if type_ == 'json':
            f = open(file_path,'r',encoding='utf-8')
            json_text = json.load(f)
            f.close()

            try:# 这里面向最基本的情况，就是三部分组成
                raw_text = json_text['title']+'. '+json_text['text']
            except:# 这是因为之前非要给分为多部分，因此额外合并
                raw_text = ''
                for keys in json_text.keys():
                    if keys == 'doi':
                        continue
                    else:
                        raw_text += ' '+json_text[keys]
        else:
            f = open(file_path, 'r', encoding='utf-8')
            raw_text = f.read()
            f.close()

        sents = sent_tokenize(raw_text)
        sent_li = []

        for sent in sents:
            sent_li.append(wordpunct_tokenize(sent))

        abstracts_li.append({
            'pmid':file_name,
            'raw_text':raw_text,
            'file_id': file_name,
            'sent_li':sent_li

        })

    return abstracts_li


if __name__ == '__main__':
    read_abstract_text()

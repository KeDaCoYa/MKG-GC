# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2021/11/25
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/11/25: 
-------------------------------------------------
"""


import logging

from collections import defaultdict
from ipdb import set_trace

from utils.function_utils import  gentl_print
from utils.predicate_utils import align_predicted_labels_with_original_sentence_tokens

logger = logging.getLogger('main.evaluate')




def normal_evaluate_crf(pred_tokens,y,raw_text_li,crf_id2label,type_weight,mode='micro',verbose=False):
    '''

    :param pred_tokens: 这个是预测的结果， 这个结果和真实的长度一样
    :param y:  这是实际的label
    :param raw_text_li:在训练的时候计算p,r,f1,这个是token id并不是raw text，而是token ids
    :return:
    '''

    # tmp_li = []
    # if id2word:
    #
    #     for i,token_id in enumerate(raw_text_li):
    #         tmp = [id2word.get(raw_text_li[i][id]) for id in range(len(pred_tokens[i]))]
    #         tmp_li.append(tmp)
    #     raw_text_li = tmp_li


    lens = [len(i) for i in raw_text_li]

    R_dict = defaultdict(list)
    T_dict = defaultdict(list)

    for i in range(len(raw_text_li)):

        actual_len = lens[i]
        raw_text = raw_text_li[i]
        pred = pred_tokens[i]
        y_true = y[i][:actual_len]

        predict_entities = crf_decode_BIO(raw_text, pred, crf_id2label)

        true_entities = crf_decode_BIO(raw_text, y_true, crf_id2label)

        for key in type_weight.keys():
            R_dict[key].extend(predict_entities[key])
            T_dict[key].extend(true_entities[key])

    f1, p, r = 0., 0., 0.
    if mode == 'micro':
        gentl_li = []
        for key in type_weight.keys():
            R = set(R_dict[key])
            T = set(T_dict[key])
            A = len(R & T)
            B = len(R)
            C = len(T)
            tmp_f1 = 2 * A / (B + C) if (B + C) != 0 else 0
            tmp_P = A / B if B > 1e-10 and A > 1e-10 else 0
            tmp_R = A / C if C > 1e-10 else 0
            f1 += type_weight[key] * tmp_f1
            p += type_weight[key] * tmp_P
            r += type_weight[key] * tmp_R
            gentl_li.append([key, f1, p, r])
        if verbose:
            gentl_print(gentl_li)

        return f1, p, r

    elif mode == 'macro':
        gentl_li = []
        for key in type_weight.keys():
            R = set(R_dict[key])
            T = set(T_dict[key])
            A = len(R & T)
            B = len(R)
            C = len(T)
            tmp_f1 = 2 * A / (B + C) if (B + C) != 0 else 0
            tmp_P = A / B if B > 1e-10 and A > 1e-10 else 0
            tmp_R = A / C if C > 1e-10 else 0
            f1 += tmp_f1
            p += tmp_P
            r += tmp_R
            gentl_li.append([key, f1, p, r])

        if verbose:
            gentl_print(gentl_li)

        return f1 / len(type_weight), p / len(type_weight), r / len(type_weight)




def crf_decode_BIOES(decode_tokens,raw_text,crf_id2label):
    '''
    一次对一条数据进行decode
    将模型的结果tokens进行解码,对多类别的entity进行评估
    :param decode_tokens: 这个是预测之后的结果，由crf decode的结果
    :param raw_text:
    :param id2ent:
    :return: predict_entities,这是返回一个字典，key为entity type，value为list，
        value = [(entity name,start_offset)]
    '''
    predict_entities = {}

    index_ = 0
    while index_ < len(decode_tokens):
        # 把label="B-DRUG" -> ['B','DRUG']
        token_label = crf_id2label[decode_tokens[index_]].split('-')
        if token_label[0].startswith('S'):
            token_type = token_label[1]
            tmp_ent = raw_text[index_]
            if token_type not in predict_entities:
                predict_entities[token_type] = [(tmp_ent,index_)]
            else:
                predict_entities[token_type].append((tmp_ent,index_))
            index_ += 1
        elif token_label[0].startswith("B"):
            token_type = token_label[1]
            start_index = index_
            index_ += 1
            while index_ < len(decode_tokens):
                temp_token_label = crf_id2label[decode_tokens[index_]].split('-')
                if temp_token_label[0].startswith('I') and token_type == temp_token_label[1]:
                    index_ += 1
                elif temp_token_label[0].startswith('E') and token_type == temp_token_label[1]:
                    end_index = index_
                    index_ += 1
                    tmp_ent = raw_text[start_index:end_index+1]
                    if token_type not in predict_entities:
                        predict_entities[token_type] = [(tmp_ent,start_index)]
                    else:
                        predict_entities[token_type].append((tmp_ent,start_index))
                    break
                else:
                    break


        else:
            index_ += 1

    return predict_entities


def crf_decode_BIO(text,pred_tokens,crf_id2label):
    '''
    一次只能对只能对一个句子
    将BIO的结果进行decode,得到json结果
    这个函数也可以看作是BIO数据和json数据的转换
    这是text 评估法，需要依靠实体名称来判断
    :param text:  这是一句话
    :param pred_token: 这是text这句话对应的BIO标签 ['O', 'B-cell_type', 'I-cell_type', 'O']
    :return:
    '''

    pred_token = [crf_id2label[id] for id in pred_tokens]

    res_entities = defaultdict(list)
    start_index = 0
    actual_len = len(text)
    while start_index < actual_len:
        cur_label = pred_token[start_index].split('-')
        if len(cur_label) == 2:
            # 这个entity type和entity type id是全局的数据
            BIO_format, entity_type = cur_label
        else:
            BIO_format = cur_label[0]
            entity_type = 'One'

        if start_index + 1 < actual_len:

            next_label = pred_token[start_index + 1].split('-')

            if len(next_label) == 2:
                BIO_, _ = next_label
            elif len(next_label) == 1:
                BIO_ = next_label[0]

        if BIO_format == 'B' and start_index + 1 < actual_len and BIO_ == 'O':  # 实体是一个单词

            res_entities[entity_type].append((text[start_index],start_index,start_index))

            start_index += 1
        elif BIO_format == 'B' and start_index + 1 >= actual_len:  # 最后只有一个实体，并且只有一个单词，到达了最后

            res_entities[entity_type].append((text[start_index], start_index, start_index))
            break
        elif BIO_format == 'B':
            j = start_index + 1
            while j < actual_len:
                j_label = pred_token[j].split('-')
                if len(j_label) == 2:
                    BIO_, _ = j_label
                elif len(j_label) == 1:
                    BIO_ = j_label[0]

                if BIO_ == 'I':
                    j += 1
                else:
                    res_entities[entity_type].append((" ".join(text[start_index:j]), start_index, j-1))

                    break
            if j >= actual_len:
                j_label = pred_token[j - 1].split('-')
                if len(j_label) == 2:
                    BIO_, _ = j_label
                elif len(j_label) == 1:
                    BIO_ = j_label[0]

                if BIO_ == 'I':
                    res_entities[entity_type].append((" ".join(text[start_index:j]), start_index, j - 1))



            start_index = j
        else:
            start_index += 1
    return res_entities




def crf_decode_BIO_ID(id,text,pred_tokens,crf_id2label):
    '''
    这是ID评估法
    一次只能对只能对一个句子
    将BIO的结果进行decode,得到json结果
    这个函数也可以看作是BIO数据和json数据的转换
    这是text 评估法，需要依靠实体名称来判断
    :param text:  这是一句话
    :param pred_token: 这是text这句话对应的BIO标签 ['O', 'B-cell_type', 'I-cell_type', 'O']
    :return:
    '''

    pred_token = [crf_id2label[id] for id in pred_tokens]

    res_entities = defaultdict(list)
    start_index = 0
    actual_len = len(text)
    while start_index < actual_len:
        cur_label = pred_token[start_index].split('-')
        if len(cur_label) == 2:
            # 这个entity type和entity type id是全局的数据
            BIO_format, entity_type = cur_label
        else:
            BIO_format = cur_label[0]
            entity_type = 'One'

        if start_index + 1 < actual_len:

            next_label = pred_token[start_index + 1].split('-')

            if len(next_label) == 2:
                BIO_, _ = next_label
            elif len(next_label) == 1:
                BIO_ = next_label[0]

        if BIO_format == 'B' and start_index + 1 < actual_len and BIO_ == 'O':  # 实体是一个单词

            res_entities[entity_type].append((id,start_index,start_index))

            start_index += 1
        elif BIO_format == 'B' and start_index + 1 >= actual_len:  # 最后只有一个实体，并且只有一个单词，到达了最后

            res_entities[entity_type].append((id, start_index, start_index))
            break
        elif BIO_format == 'B':
            j = start_index + 1
            while j < actual_len:
                j_label = pred_token[j].split('-')
                if len(j_label) == 2:
                    BIO_, _ = j_label
                elif len(j_label) == 1:
                    BIO_ = j_label[0]

                if BIO_ == 'I':
                    j += 1
                else:
                    res_entities[entity_type].append((id, start_index, j-1))

                    break
            if j >= actual_len:
                j_label = pred_token[j - 1].split('-')
                if len(j_label) == 2:
                    BIO_, _ = j_label
                elif len(j_label) == 1:
                    BIO_ = j_label[0]

                if BIO_ == 'I':
                    res_entities[entity_type].append((id, start_index, j - 1))


            start_index = j
        else:
            start_index += 1
    return res_entities

def bert_evaluate_crf(pred_tokens,y,raw_text_li,crf_id2label,type_weight,mode='micro',verbose=False):
    '''

    macro-评估，这是多实体的实体识别
    这里的评估方法采用的是集合评估法，占用较大的资源，以实体的startoffset和end offset作为判断标准
    在使用bert之后进行评估的话，并且crf会自动加上<start>,<end>,所以需要将pred_tokens取消
    :param pred_tokens:
    :param y: 这个是bert的预测结构，shape = (all_data,seq_len)，这个是填充之后的，需要进行剔除
    :param raw_text_li:


    :param pred_tokens:
    :param y: 这是pad之后的ture lable，
    :param raw_text_li: 原始文本
    :param crf_id2label:
    :param type_weight:
    :param mode:
    :param verbose:
    :return: f1,p,r
    '''

    lens = [len(i) for i in raw_text_li]
    R_dict = defaultdict(list)
    T_dict = defaultdict(list)


    for i in range(len(pred_tokens)):

        #pred = pred_tokens[i][1:-1] # 这里去除的是<sep>和<cls>
        pred = pred_tokens[i] # 这里去除的是<sep>和<cls>,这里是tokenizer的情况，不需要去除，因为在align的时候就已经去除了

        actual_len = lens[i]


        raw_text = raw_text_li[i]

        # 如果true lable填充[CLS],[SEP]了对应的label，则下面的解码
        #y_true = y[i][1:actual_len+1]
        y_true = y[i][:actual_len]

        predict_entities = crf_decode_BIO(raw_text,pred,crf_id2label)

        true_entities = crf_decode_BIO(raw_text,y_true,crf_id2label)
        #predict_entities = crf_decode_BIO_ID(i,raw_text,pred,crf_id2label)


        #true_entities = crf_decode_BIO_ID(i,raw_text,y_true,crf_id2label)

        for key in type_weight.keys():

            R_dict[key].extend(predict_entities[key])
            T_dict[key].extend(true_entities[key])


    f1,p,r = 0.,0.,0.
    if mode == 'micro':
        gentl_li = []
        for key in type_weight.keys():
            R = set(R_dict[key])
            T = set(T_dict[key])
            A = len(R & T)
            B = len(R)
            C = len(T)
            tmp_f1 = 2 * A / (B + C) if (B + C) != 0 else 0
            tmp_P = A / B if B > 1e-10 and A > 1e-10 else 0
            tmp_R = A / C if C > 1e-10 else 0
            f1 += type_weight[key]*tmp_f1
            p += type_weight[key]*tmp_P
            r += type_weight[key]*tmp_R
            gentl_li.append([key,f1,p,r])
        if verbose:
            gentl_print(gentl_li)

        return f1,p,r

    elif mode=='macro':
        gentl_li = []
        for key in type_weight.keys():
            R = set(R_dict[key])
            T = set(T_dict[key])
            A = len(R & T)
            B = len(R)
            C = len(T)
            tmp_f1 = 2 * A / (B + C) if (B + C) != 0 else 0
            tmp_P = A / B if B > 1e-10 and A > 1e-10 else 0
            tmp_R = A / C if C > 1e-10 else 0
            f1 += tmp_f1
            p += tmp_P
            r += tmp_R
            gentl_li.append([key,f1,p,r])

        if verbose:
            gentl_print(gentl_li)


        return f1/len(type_weight),p/len(type_weight),r/len(type_weight)
    else:
        raise ValueError

def bert_evaluate_crf_tokenize(pred_tokens,y,raw_text_li,crf_id2label,type_weight,mode='micro',verbose=False,orig_to_token_indexs=None):
    '''

    这里相当于是按照tokenizer之后的评估，但是这种评估并不是很好..

    macro-评估，这是多实体的实体识别
    这里的评估方法采用的是集合评估法，占用较大的资源，以实体的startoffset和end offset作为判断标准
    在使用bert之后进行评估的话，并且crf会自动加上<start>,<end>,所以需要将pred_tokens取消
    :param pred_tokens: 这是对tokenizer之后的结果，并且有[SEP],[CLS]...
    :param y: 这个是
    :param raw_text_li:原始文本

    :return:f1,p,r
    '''
    pred_tokens = align_predicted_labels_with_original_sentence_tokens(pred_tokens, orig_to_token_indexs,206)


    lens = [len(i) for i in raw_text_li]
    R_dict = defaultdict(list)
    T_dict = defaultdict(list)

    for i in range(len(pred_tokens)):

        #pred = pred_tokens[i][1:-1] # 这里去除的是<sep>和<cls>
        pred = pred_tokens[i] # 这里去除的是<sep>和<cls>,这里是tokenizer的情况，不需要去除，因为在align的时候就已经去除了

        actual_len = lens[i]


        raw_text = raw_text_li[i]

        y_true = y[i][1:actual_len+1]

        predict_entities_text = crf_decode_BIO(raw_text,pred,crf_id2label)
        true_entities_text = crf_decode_BIO(raw_text,y_true,crf_id2label)
        predict_entities = crf_decode_BIO_ID(i,raw_text,pred,crf_id2label)
        #
        #
        true_entities = crf_decode_BIO_ID(i,raw_text,y_true,crf_id2label)

        for key in type_weight.keys():

            R_dict[key].extend(predict_entities[key])
            T_dict[key].extend(true_entities[key])


    f1,p,r = 0.,0.,0.
    if mode == 'micro':
        gentl_li = []
        for key in type_weight.keys():
            R = set(R_dict[key])
            T = set(T_dict[key])
            A = len(R & T)
            B = len(R)
            C = len(T)
            tmp_f1 = 2 * A / (B + C) if (B + C) != 0 else 0
            tmp_P = A / B if B > 1e-10 and A > 1e-10 else 0
            tmp_R = A / C if C > 1e-10 else 0
            f1 += type_weight[key]*tmp_f1
            p += type_weight[key]*tmp_P
            r += type_weight[key]*tmp_R
            gentl_li.append([key,f1,p,r])
        if verbose:
            gentl_print(gentl_li)

        return f1,p,r

    elif mode=='macro':
        gentl_li = []
        for key in type_weight.keys():
            R = set(R_dict[key])
            T = set(T_dict[key])
            A = len(R & T)
            B = len(R)
            C = len(T)
            tmp_f1 = 2 * A / (B + C) if (B + C) != 0 else 0
            tmp_P = A / B if B > 1e-10 and A > 1e-10 else 0
            tmp_R = A / C if C > 1e-10 else 0
            f1 += tmp_f1
            p += tmp_P
            r += tmp_R
            gentl_li.append([key,f1,p,r])

        if verbose:
            gentl_print(gentl_li)


        return f1/len(type_weight),p/len(type_weight),r/len(type_weight)
    else:
        raise ValueError

def bert_evaluate_crf_single(pred_tokens,y,raw_text_li):
    '''
    macro-评估，这是对只有一种实体类别的评估
    这里的评估方法采用的是集合评估法，占用较大的资源，以实体的startoffset和end offset作为判断标准
    在使用bert之后进行评估的话，并且crf会自动加上<start>,<end>,所以需要将pred_tokens取消
    :param pred_tokens:
    :param y: 这个是bert的预测结构，shape = (all_data,seq_len)
    :param lens:
    :return:
    '''

    lens = [len(i) for i in raw_text_li]


    A, B, C = 1e-10, 1e-10, 1e-10
    for i in range(len(pred_tokens)):
        standard_set = set()
        pred_set = set()
        actual_len = lens[i]
        pred = pred_tokens[i][1:-1] # 这里去除的是<sep>和<cls>
        raw_text = raw_text_li[i]
        y_true = y[i][1:actual_len+1]
        start_index = 0
        # 获取正确的set

        while start_index<actual_len:

            if y_true[start_index] == 2 and start_index+1<actual_len and y_true[start_index+1] == 0: #实体是一个单词
                standard_set.add((raw_text[start_index],start_index,start_index))
                start_index += 1
            elif y_true[start_index] == 2 and start_index+1>=actual_len:
                standard_set.add((raw_text[start_index], start_index, start_index))
                break
            elif y_true[start_index] == 2:
                j = start_index+1
                while j < actual_len:
                    if y_true[j] == 1:
                        j += 1
                    else:
                        standard_set.add((" ".join(raw_text[start_index:j]),start_index,j-1))
                        break
                if j >= actual_len:
                    if y_true[j-1] == 1:
                        standard_set.add((" ".join(raw_text[start_index:j]), start_index, j - 1))

                start_index = j
            else:
                start_index += 1

        start_index = 0
        # 获取正确的set
        while start_index < actual_len:
            if pred[start_index] == 2 and start_index + 1<actual_len and pred[start_index + 1] == 0:  # 实体是一个单词
                pred_set.add((raw_text[start_index], start_index, start_index))
                start_index += 1
            elif pred[start_index] == 2 and start_index + 1 >= actual_len: #最后一个实体在最后面
                pred_set.add((raw_text[start_index], start_index, start_index))

                break
            elif pred[start_index] == 2:
                j = start_index + 1
                while j < actual_len:
                    if pred[j] == 1:
                        j += 1
                    else:
                        pred_set.add((" ".join(raw_text[start_index:j]), start_index, j - 1))
                        break
                if j >= actual_len:
                    if pred[j - 1] == 1:
                        pred_set.add((" ".join(raw_text[start_index:j]), start_index, j - 1))

                start_index = j
            else:
                start_index += 1
        # print('标准：',standard_set)
        # print('预测：',pred_set)

        T = set(standard_set)
        A += len(pred_set & T)
        B += len(pred_set)
        C += len(T)
    f1 = 2 * A / (B + C) if (B + C) != 0 else 0
    P = A / B if B > 1e-10 and A>1e-10 else 0
    R = A / C if C >1e-10 else 0

    return P,R,f1

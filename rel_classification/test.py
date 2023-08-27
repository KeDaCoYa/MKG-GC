# -*- encoding: utf-8 -*-
"""
@File    :   test.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/5/14 9:43   
@Description :   None 

"""
import argparse


def get_normal_config():
    parser = argparse.ArgumentParser()


    parser.add_argument('--dataset_name', type=list, help='选择re数据集名称')

    args = parser.parse_args()
    dataset_name = args.dataset_name
    print(eval(''.join(dataset_name)))

get_normal_config()


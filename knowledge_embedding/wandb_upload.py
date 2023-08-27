# -*- encoding: utf-8 -*-
"""
@File    :   wandb_upload.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/5/9 14:23   
@Description :   None 

"""
import wandb
# wandb_name = "五个数据集的未训练结果"
#
# wandb.init(project="多任务实体标准化", name=wandb_name)
#
# # 这是biosyn的结果
# wandb.log({"dev-epoch": 0,
#            'dev-gene-hit@1': 0.925,
#            'dev-gene-hit@5': 0.95745,
#            'dev-disease-hit@1': 0.945263,
#            'dev-disease-hit@5': 0.9625,
#            'dev-chemical-hit@1': 0.93275,
#            'dev-chemical-hit@5': 0.9505,
#            'dev-cell_type-hit@1': 0.89875,
#            'dev-cell_type-hit@5': 0.959,
#            'dev-cell_line-hit@1': 0.94376,
#            'dev-cell_line-hit@5': 0.967,
#
#             },
#           step=0)

#
# wandb_name = "ComplEx"
#
# wandb.init(project="知识嵌入", name=wandb_name)
# wandb.log({"dev-epoch": 0,
#            'MRR': 0.763264,
#            'MR': 56.737879,
#            'hit@1': 0.727945,
#            'hit@3': 0.784850,
#            'hit@5': 0.825019},
#           step=0)

# wandb_name = "transE"
# wandb.init(project="知识嵌入", name=wandb_name)
#
# wandb.log({"dev-epoch": 0,
#            'MRR': 0.711447,
#            'MR': 47.031206,
#            'hit@1': 0.648526,
#            'hit@3': 0.748191,
#            'hit@5': 0.823345},
#           step=0)

# wandb_name = "DistMult"
#
# wandb.init(project="知识嵌入", name=wandb_name)
#
# wandb.log({"dev-epoch": 0,
#            'MRR': 0.732396,
#            'MR': 159.821887,
#            'hit@1': 0.698143,
#            'hit@3': 0.740903,
#            'hit@5': 0.801101},
#           step=0)
#
#
#
# wandb_name = "HoIE"
#
# wandb.init(project="知识嵌入", name=wandb_name)
#
# wandb.log({"dev-epoch": 0,
#            'MRR': 0.704861,
#            'MR': 411.773891,
#            'hit@1': 0.665965,
#            'hit@3': 0.727783,
#            'hit@5': 0.770975},
#           step=0)

#
# wandb_name = "ConvE"
#
# wandb.init(project="知识嵌入", name=wandb_name)
#
# wandb.log({"dev-epoch": 0,
#            'MRR': 0.392975,
#            'MR': 5353.968524,
#            'hit@1': 0.391966,
#            'hit@3': 0.392344,
#            'hit@5': 0.392884},
#           step=0)
#
#
#
# wandb_name = "ConvKB"
#
# wandb.init(project="知识嵌入", name=wandb_name)
#
# wandb.log({"dev-epoch": 0,
#            'MRR': 0.309847,
#            'MR': 366.023378,
#            'hit@1': 0.153169,
#            'hit@3': 0.401145,
#            'hit@5': 0.666559},
#           step=0)
#



# 这是sapbert
# wandb.log({"test-epoch": 0,
#            'test-bc5cdr-disease-hit@1': 0.936,
#            'test-bc5cdr-disease-hit@5': 0.962,
#            'test-bc5cdr-chemical-hit@1': 0.968,
#            'test-bc5cdr-chemical-hit@5': 0.984,
#            'test-ncbi-disease-hit@1': 0.925,
#            'test-ncbi_disease-hit@5': 0.962}, step = 0)

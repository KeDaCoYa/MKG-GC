# -*- encoding: utf-8 -*-
"""
@File    :   BaseModel.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/5/1 20:03   
@Description :   None 

"""
import torch
import torch.nn as nn
class DTIModel(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.linear1 = nn.Linear(config.entity_embedding,100)
        self.linear2 = nn.Linear(config.entity_embedding,100)
        self.classifier = nn.Linear(200,1)
        self.loss_fn = nn.BCELoss()
    def forward(self,ent1,ent2,labels=None):
        e1 = self.linear1(ent1)
        e2 = self.linear2(ent2)
        final_ = torch.cat([e1,e2],axis=1)

        logits = self.classifier(final_)
        if labels:
            loss = self.loss_fn(logits,labels)
            return loss,logits
        return logits
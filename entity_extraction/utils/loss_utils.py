# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  这是保存各种损失函数
   Author :        kedaxia
   date：          2021/11/08
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/11/08:
-------------------------------------------------
"""
import torch
import torch.nn as nn
from ipdb import set_trace


def multilabel_categorical_crossentropy(y_pred, y_true):
    '''
    这是苏神推广的多标签softmax,可以移植到其他之中
    :param y_pred: shape = (batch_size,entity_type,seq_len,seq_len)
    :param y_true:
    :return:
    '''
    batch_size, ent_type_size = y_pred.shape[:2]

    # 这里将其进行转变为shape=(batch_size*entity_type,seq_len*seq_len)
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)

    # 下面这个就相当于分别得到正例的分数和负例的分数
    # (1-2*y_true)看作是一个mask，这个mask的1表示这是非实体，-1表示这是实体
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    # 这是获得negative 对应的预测分数，将对应的true值弄得很小-1e12，这样子在之后e^x 接近0
    y_pred_neg = y_pred - y_true*1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true)*1e12 # mask the pred outputs of neg classes

    # 这个地方加上0是为了统一损失函数计算公式
    zeros = torch.zeros_like(y_pred[..., :1]) #这里相当于获得一个的,zeros.shape=(batch_size*entity type)
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)


    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self,eps = 0.1,reduction='mean',ignore_index=-100):
        super(LabelSmoothingCrossEntropy,self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index
    def forward(self,output,target):
        '''
        在span下，output.shape= [4096,14],target.shape = [4096]
            #这里已经将(batch_size,seq_len)给view了...
        :param output:
        :param target:
        :return:
        '''
        c = output.size()[-1]
        #log_pred.shape = (batch_size*max_seq_len,num_tags)
        log_pred = torch.log_softmax(output,dim=-1)  #就是先对output先取softmax，然后再log一下，这就是crossentropy的前两步操作
        if self.reduction == 'sum':
            # 这里相当于将所有的值给加了
            loss = -log_pred.sum()
        else:
            loss = -log_pred.sum(dim=-1)#torch.Size([4096])
            if self.reduction == 'mean':
                loss = loss.mean()  #这也是和上面的区别为mean

        return loss*self.eps/c+(1-self.eps)*torch.nn.functional.nll_loss(log_pred,target,reduction=self.reduction,ignore_index=self.ignore_index)

class FocalLoss(nn.Module):
    def __init__(self,gamma=2,weight=None,reduction='mean',ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index
    def forward(self,input,target):
        '''

        :param input: [N,C],N=batch_size*max_seq_len,C为实体类别数=14
        :param target:
        :return:
        '''
        log_pt = torch.log_softmax(input,dim=1)
        pt = torch.exp(log_pt) # 这个pt表示的限制正确的概率，这里取exp是为了保证为正数
        log_pt = (1-pt)**self.gamma*log_pt
        loss = torch.nn.functional.nll_loss(log_pt,target,self.weight,reduction=self.reduction,ignore_index=self.ignore_index)
        return loss




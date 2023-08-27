# -*- encoding: utf-8 -*-
"""
@File    :   lpbert_train.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/5/7 16:06   
@Description :   这是用于训练lpbert预训练模型

"""
import os.path
import pickle
import  logging
import time

import torch
import wandb
from ipdb import set_trace
from sklearn.metrics import accuracy_score
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AdamW

from src.models.biolpbert import LpBertForPreTraining
from src.utils.function_utils import get_config, get_logger
from src.utils.train_utils import load_model_and_parallel,build_bert_optimizer_and_scheduler

START_TIME = time.time()

class MyLpBertDataset(Dataset):
    def __init__(self,data,tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.nums = len(data)
    def __len__(self):
        return self.nums
    def __getitem__(self, item):
        data = self.data[item]
        head_mem_res, tail_mem_res, mrm_res = data
        head_mem_input_ids = torch.tensor(head_mem_res['input_ids']).long()
        head_mem_attention_mask = torch.tensor(head_mem_res['attention_mask']).bool()
        head_mem_token_type_ids = torch.tensor(head_mem_res['token_type_ids']).long()
        head_mem_label_ids = torch.tensor(head_mem_res['label_ids']).long()

        tail_mem_input_ids = torch.tensor(tail_mem_res['input_ids']).long()
        tail_mem_attention_mask = torch.tensor(tail_mem_res['attention_mask']).bool()
        tail_mem_token_type_ids = torch.tensor(tail_mem_res['token_type_ids']).long()
        tail_mem_label_ids = torch.tensor(tail_mem_res['label_ids']).long()

        mrm_input_ids = torch.tensor(mrm_res['input_ids']).long()
        mrm_attention_mask = torch.tensor(mrm_res['attention_mask']).bool()
        mrm_token_type_ids = torch.tensor(mrm_res['token_type_ids']).long()
        mrm_label_ids = torch.tensor(mrm_res['label_ids']).long()

        return head_mem_input_ids,head_mem_attention_mask,head_mem_token_type_ids,head_mem_label_ids,tail_mem_input_ids,tail_mem_attention_mask,tail_mem_token_type_ids,tail_mem_label_ids,mrm_input_ids,mrm_attention_mask,mrm_token_type_ids,mrm_label_ids

config = get_config()

if config.use_wandb:
    wandb.init(project="LpBERT", config=vars(config),)
logger = get_logger(config)




tokenizer_file_path = '../embedding/scibert_scivocab_uncased'

tokenizer = BertTokenizer.from_pretrained(tokenizer_file_path)
config.vocab_size = len(tokenizer)
config.type_vocab_size = 3
config.output_dir = './outputs/biolpbert/'
model = LpBertForPreTraining(config)
global_step = 0

if config.use_n_gpu and torch.cuda.device_count() > 1:
    model, device = load_model_and_parallel(model, '0,1', ckpt_path=None, load_type='many2one')
else:
    # model, device = load_model_and_parallel(model, '0,1', ckpt_path=None, load_type='one2one')
    device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')
    model.to(device)

if config.use_fp16:
    scaler = GradScaler()
t_total = config.num_epochs*25000*200
optimizer, scheduler = build_bert_optimizer_and_scheduler(config, model, t_toal=t_total)

for epoch in range(config.num_epochs):
    model.train()
    for i in range(1,200):
        file_path = "./Lpcorpus/{}_150000.pk".format(i)
        if not os.path.exists(file_path):
            break

        with open(file_path,'rb') as f:
            all_data = pickle.load(f)
        train_dataset = MyLpBertDataset(all_data,tokenizer)
        train_dataloader = DataLoader(dataset=train_dataset,batch_size=config.batch_size,shuffle=True)
        train_loss = 0.
        train_mrm_loss = 0.
        train_head_mem_loss = 0.
        train_tail_mem_loss = 0.
        train_mrm_acc = 0.
        train_head_mem_acc = 0.
        train_tail_mem_acc = 0.
        for step,batch_data in enumerate(train_dataloader):
            step += 1
            head_mem_input_ids, head_mem_attention_mask, head_mem_token_type_ids, head_mem_label_ids, tail_mem_input_ids, tail_mem_attention_mask, tail_mem_token_type_ids, tail_mem_label_ids, mrm_input_ids, mrm_attention_mask, mrm_token_type_ids, mrm_label_ids = batch_data
            global_step += 1
            head_mem = {
                'input_ids':head_mem_input_ids.to(device),
                'token_type_ids':head_mem_token_type_ids.to(device),
                'attention_mask':head_mem_attention_mask.to(device),
            }

            tail_mem = {
                'input_ids': tail_mem_input_ids.to(device),
                'token_type_ids': tail_mem_token_type_ids.to(device),
                'attention_mask': tail_mem_attention_mask.to(device),

            }

            mrm = {
                'input_ids': mrm_input_ids.to(device),
                'token_type_ids': mrm_token_type_ids.to(device),
                'attention_mask': mrm_attention_mask.to(device),

            }
            head_mem_label_ids = head_mem_label_ids.to(device)
            tail_mem_label_ids = tail_mem_label_ids.to(device)
            mrm_label_ids = mrm_label_ids.to(device)

            mrm_logits, head_mem_logits, tail_mem_logits, head_mem_loss,tail_mem_loss,mrm_loss = model(head_mem,tail_mem,mrm,head_mem_label_ids,tail_mem_label_ids,mrm_label_ids)
            lr = optimizer.param_groups[0]["lr"]
            loss = mrm_loss + head_mem_loss + tail_mem_loss
            loss = loss.mean()
            train_loss += loss.item()

            mrm_loss = mrm_loss.mean()
            head_mem_loss = head_mem_loss.mean()
            tail_mem_loss = tail_mem_loss.mean()

            train_mrm_loss += mrm_loss.item()
            train_head_mem_loss += head_mem_loss.item()
            train_tail_mem_loss += tail_mem_loss.item()



            _, mrm_predicate = torch.max(mrm_logits, dim=-1)
            _, head_mem_predicate = torch.max(head_mem_logits, dim=-1)

            _, tail_mem_predicate = torch.max(tail_mem_logits, dim=-1)

            mrm_predicate = mrm_predicate.cpu().numpy()
            head_mem_predicate = head_mem_predicate.cpu().numpy()
            tail_mem_predicate = tail_mem_predicate.cpu().numpy()


            head_mem_label_ids = head_mem_label_ids.cpu().numpy()
            tail_mem_label_ids = tail_mem_label_ids.cpu().numpy()
            mrm_label_ids = mrm_label_ids.cpu().numpy()

            head_mem_flag = head_mem_label_ids > 0
            tail_mem_flag = tail_mem_label_ids > 0
            mrm_flag = mrm_label_ids > 0


            mrm_label_ids = mrm_label_ids[mrm_flag]
            mrm_predicate = mrm_predicate[mrm_flag]
            mrm_acc = accuracy_score(mrm_label_ids, mrm_predicate)
            head_mem_label_ids = head_mem_label_ids[head_mem_flag]
            head_mem_predicate = head_mem_predicate[head_mem_flag]
            head_mem_acc = accuracy_score(head_mem_label_ids, head_mem_predicate)
            tail_mem_label_ids = tail_mem_label_ids[tail_mem_flag]
            tail_mem_predicate = tail_mem_predicate[tail_mem_flag]
            tail_mem_acc = accuracy_score(tail_mem_label_ids, tail_mem_predicate)

            train_tail_mem_acc += tail_mem_acc
            train_head_mem_acc += head_mem_acc
            train_mrm_acc += mrm_acc

            logger.info('训练集训练中...: file:{} Epoch {} | Step:{}/{}|{}/{}'.format(i,epoch, step, len(train_dataloader), global_step, t_total))
            logger.info('total loss:{:.5f}|avg_loss:{:.5f}'.format(loss.item(),train_loss/step))
            logger.info('   MRM loss:{:.5f}|avg:{:.5f} Acc:{:.5f}|avg:{:.5f}'.format(mrm_loss.item(),train_mrm_loss/step,mrm_acc,train_mrm_acc/step))
            logger.info('   Head MEM loss:{:.5f}|avg:{:.5f}  Acc:{:.5f}|avg:{:.5f}'.format(head_mem_loss.item(),train_head_mem_loss/step,head_mem_acc,train_head_mem_acc/step))
            logger.info('   Tail MEM loss:{:.5f}|avg:{:.5f} Acc:{:.5f}|avg:{:.5f}'.format(tail_mem_loss.item(),train_tail_mem_loss/step,tail_mem_acc,train_tail_mem_acc/step))
            if config.use_wandb:
                wandb.log(
                    {"train-epoch": epoch,
                    'train-total_loss': loss.item(),
                    'train-total_avg_loss': train_loss/step,
                    'train-head_mem_loss': head_mem_loss.item(),
                    'train-head_mem_avg_loss': train_head_mem_loss/step,
                    'train-tail_mem_loss': tail_mem_loss.item(),
                    'train-tail_mem_avg_loss': train_tail_mem_loss / step,
                    'train-mrm_loss': mrm_loss.item(),
                    'train-mrm_avg_loss': train_mrm_loss / step,

                    'train-head_mem_acc': head_mem_acc,
                    'train-head_mem_avg_acc': train_head_mem_acc / step,
                    'train-tail_mem_acc': tail_mem_acc,
                    'train-tail_mem_avg_acc': train_tail_mem_acc / step,
                    'train-mrm_acc': mrm_acc,
                    'train-mrm_avg_acc': train_mrm_acc / step,

                    'time_s': time.time()-START_TIME,
                    'time_m': (time.time()-START_TIME) / 60,
                    'time_h': (time.time()-START_TIME) / 3600,
                    'lr': lr
                    }, step=global_step
                )
            if config.use_fp16:
                scaler.scale(loss).backward()

                if (step + 1) % config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    # if config.parameter_summary_writer:
                    #     save_parameter_writer(parameter_writer, model, global_step)
                    optimizer.zero_grad()
            else:
                loss.backward()
                # clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                # if config.parameter_summary_writer:
                #     save_parameter_writer(parameter_writer, model, global_step)
                optimizer.zero_grad()
            if global_step % 1000000 == 0:
                if config.save_model:
                    output_dir = os.path.join(config.output_dir, 'epoch_{}_biolpbert'.format(epoch))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)

                    logger.info('-----将模型保存到 {}------'.format(output_dir))

                    torch.save(model.state_dict(), os.path.join(output_dir, 'biolpbert_{}_{}.pt'.format( epoch,global_step)))

    if config.save_model:
        output_dir = os.path.join(config.output_dir, 'epoch_{}_biolpbert'.format(epoch))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        logger.info('-----将模型保存到 {}------'.format(output_dir))

        torch.save(model.state_dict(), os.path.join(output_dir, 'epoch_{}_biolpbert.pt'.format(epoch)))
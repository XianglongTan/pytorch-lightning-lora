from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import torch
import pandas as pd
import lightning.pytorch as pl
from torch import nn
import os
from data import Dataset
from tqdm import *
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold

pl.seed_everything(0)

extra_text = ['网页操作的json中,action 参数的可选值有:',
              '网页操作的json中,dom_type 参数的可选值有:']
extra_label = ['点击、输入',
               '下拉框,下拉框选项,单选框,复选框,导航树下拉框,导航树下拉框选项,按钮,日期输入框,输入框']


# model_path = '../../model/chatglm-6b-int4-v2'
model_path = '../../model/chatglm-6b'
max_length = 1400
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

'''
网页:巨潮资讯。该网页可操作的元素有:['查询', '开始日期', '港股', '债券', '确认', '分类', '标题关键字', '结束日期', '基金', '深沪京', '代码/简称/拼音', '三板']。指令:请搜索代码为000010公司的两年之内深沪京的年报和监事会公告。要在该网页上完成这个指令,用json格式生成操作和相应参数:
{'查询': ['按钮', '', '点击'], 
'代码/简称/拼音': ['输入框', '000010', '输入'], 
'开始日期': ['日期输入框', '2021-03-15', '输入'], 
'结束日期': ['日期输入框', '2023-03-15', '输入'], 
'深沪京': ['按钮', '', '点击'], 
'分类': ['下拉框', '', '点击'], 
'年报': ['下拉框选项', '', '点击'], 
'监事会': ['下拉框选项', '', '点击'], 
'确认': ['按钮', '', '点击']}
'''
test_input_ids = torch.tensor([[     5,  71253,     12,  66738,  65907,  68265,  63823,  64030,  71253,
          63879,  64422,  87053,  63832,     12, 125688,  66961,     22,      6,
             68,  63966,  67913,     22,      6,     68,  84443,     22,      6,
             68,  69402,     22,      6,     68,  65864,     22,      6,     68,
          66787,     22,      6,     68,  69259,  87345,     22,      6,     68,
          65490,  67913,     22,      6,     68,  65155,     22,      6,     68,
          64478,  69982,  65959,     22,      6,     68,  68786,     26,  69267,
             26,  80911,     22,      6,     68,  63903,  64435, 125737,  63823,
          72863,     12,  64157,  67068,  68786,  63834,      8,      8,      8,
              8,      9,      8,  66389,  68138,  71222,  64478,  69982,  65959,
          63825,  78372,  63826, 102198,  67269,  63823,  63858,  77816,  71253,
          63839,  64279,  63907,  72863,      6,  63864,   2031,  69574,  69122,
          64422,  63826,  67042,  67344,     12, 130001, 130004]], dtype=torch.long)
prefix = "要在该网页上完成这个指令,用json格式生成操作和相应参数:"


def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)
    

class ChatGLMModel(pl.LightningModule):

    def __init__(self, model_path=model_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map='auto').half()
        self.model.enable_input_require_grads()
        self.model.is_parallelizable = True
        self.model.model_parallel = True
        self.model.config.use_cache = (
            False  # silence the warnings. Please re-enable for inference!
        )
        self.model.lm_head = CastOutputToFloat(self.model.lm_head)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.lora_model = get_peft_model(self.model, peft_config)
        self.lora_model.print_trainable_parameters()
        torch.cuda.empty_cache()
        self.train_label = []
        self.train_pred = []
        self.eval_label = []
        self.eval_pred = []
        self.epoch_idx = 0
        self.best_eval_epoch_acc = 0
        
    def forward(self, input_ids, label=None):
        return self.lora_model(input_ids, label)
    
    def training_step(self, batch, batch_idx):
        result = self.lora_model(input_ids=batch['input_ids'], labels=batch['labels'])
        self.log('loss', result.loss, on_step=True, prog_bar=True)
        train_pred = torch.argmax(result.logits, dim=-1)
        self.train_pred.append(train_pred)
        self.train_label.append(batch['labels'])
        if batch_idx > 0 and batch_idx % 7000 == 0:
            gen_kwargs = {"max_length": max_length, "num_beams": 3, "do_sample": False, "top_p": 0.8,
                                    "temperature":0.9, 'repetition_penalty':1.1} 
            output = self.lora_model.generate(input_ids=test_input_ids, **gen_kwargs)
            print(tokenizer.decode(test_input_ids[0]))
            print('-'*30)
            res = tokenizer.decode(output[0]) 
            res = res[res.index(prefix)+len(prefix):]
            try:
                res_dict = eval(res)
                print('res is dict')
                for k,v in res_dict.items():
                    print(k,':',v)
            except:
                print('res is not dict!!!')
                print(res)
            print('='*30)
        return {'loss':result.loss}
    
    def on_train_epoch_end(self) -> None:
        batch_acc_lst, batch_token_acc_lst = [], []
        for train_pred, train_label in zip(self.train_pred, self.train_label):
            train_pred = train_pred.cpu()
            train_label = train_label.cpu()
            train_label = torch.cat([train_label[:,1:], 
                                     torch.zeros((train_label.shape[0],1)).fill_(tokenizer.eos_token_id).to(train_label.dtype)], 
                                     dim=-1)  ## fix the label to match the position of pred
            label_mask = torch.where(train_label==-100, 0, 1)
            label_match = (train_label==train_pred) * label_mask
            batch_match = torch.sum(label_match, dim=-1) / torch.sum(label_mask, dim=-1)
            batch_token_acc = torch.sum(batch_match) / label_mask.shape[0]
            batch_match = torch.where(batch_match>=1, 1, 0)
            batch_acc = torch.sum(batch_match) / batch_match.shape[0]
            batch_acc_lst.append(batch_acc.cpu().numpy())
            batch_token_acc_lst.append(batch_token_acc.cpu().numpy())
        self.log('epoch_acc', np.mean(batch_acc_lst))
        self.log('epoch_token_acc', np.mean(batch_token_acc_lst))
        self.epoch_idx += 1
        save_path = 'lora_r16_p2_0526_v1/epoch_{}_epoch_acc_{:.4f}'.format(str(self.epoch_idx), np.mean(batch_acc_lst))
        self.lora_model.save_pretrained(save_path)

    def validation_step(self, batch, batch_idx) :
        result = self.lora_model(input_ids=batch['input_ids'], labels=batch['labels'])
        self.log('eval_loss', result.loss, on_step=False, prog_bar=True, on_epoch=True)
        eval_pred = torch.argmax(result.logits, dim=-1)
        self.eval_pred.append(eval_pred)
        self.eval_label.append(batch['labels'])
        if batch_idx > 0 and batch_idx % 2000 == 0:
            gen_kwargs = {"max_length": max_length, "num_beams": 3, "do_sample": False, "top_p": 0.8,
                                    "temperature":0.9, 'repetition_penalty':1.1} 
            output = self.lora_model.generate(input_ids=test_input_ids, **gen_kwargs)
            print(tokenizer.decode(test_input_ids[0]))
            print('-'*30)
            res = tokenizer.decode(output[0]) 
            res = res[res.index(prefix)+len(prefix):]
            try:
                res_dict = eval(res)
                print('res is dict')
                for k,v in res_dict.items():
                    print(k,':',v)
            except:
                print('res is not dict!!!')
                print(res)
            print('='*30)
    
    def on_validation_epoch_end(self):
        batch_acc_lst, batch_token_acc_lst = [], []
        for eval_pred, eval_label in zip(self.eval_pred, self.eval_label):
            eval_pred = eval_pred.cpu()
            eval_label = eval_label.cpu()
            eval_label = torch.cat([eval_label[:,1:], 
                                     torch.zeros((eval_label.shape[0],1)).fill_(tokenizer.eos_token_id).to(eval_label.dtype)], 
                                     dim=-1)  ## fix the label to match the position of pred
            label_mask = torch.where(eval_label==-100, 0, 1)
            label_match = (eval_label==eval_pred) * label_mask
            batch_match = torch.sum(label_match, dim=-1) / torch.sum(label_mask, dim=-1)
            batch_token_acc = torch.sum(batch_match) / label_mask.shape[0]
            batch_match = torch.where(batch_match>=1, 1, 0)
            batch_acc = torch.sum(batch_match) / batch_match.shape[0]
            batch_acc_lst.append(batch_acc.cpu().numpy())
            batch_token_acc_lst.append(batch_token_acc.cpu().numpy())
        eval_epoch_token_acc = np.mean(batch_token_acc_lst)
        eval_epoch_acc = np.mean(batch_acc_lst)
        self.log('eval_epoch_acc', eval_epoch_acc)
        self.log('eval_epoch_token_acc', eval_epoch_token_acc)
        if eval_epoch_acc < self.best_eval_epoch_acc:
            self.trainer.should_stop = True
        else:
            save_path = 'lora_r16_p2_0524_v0/epoch_{}_eval_epoch_acc_{:.4f}'.format(str(self.epoch_idx), eval_epoch_acc)
            self.lora_model.save_pretrained(save_path)
            self.best_eval_epoch_acc = eval_epoch_acc
        
    def configure_optimizers(self):
        params = [{'params':[p for n,p in self.lora_model.named_parameters() if p.requires_grad], 'lr':2e-5}]
        opt = AdamW(params)
        return opt


def train():

    train_df = pd.read_csv('../data/0525/train_set.csv', nrows=None)

    # skf = StratifiedKFold(n_splits=2)
    # for train_idx, eval_idx in skf.split(train_df, train_df['page']):
    #     train_x, eval_x = train_df.iloc[train_idx], train_df.iloc[eval_idx]
    #     break
    # train_texts = train_x['text'].values.tolist()
    # train_targets = train_x['target'].values.tolist()
    # train_dataset = Dataset(train_texts, train_targets, tokenizer)
    # train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=data_collator)
    # eval_texts = eval_x['text'].values.tolist()
    # eval_targets = eval_x['target'].values.tolist()
    # eval_dataset = Dataset(eval_texts, eval_targets, tokenizer)
    # eval_dataloader = DataLoader(eval_dataset, batch_size=2, shuffle=True, collate_fn=data_collator)

    train_dataset = Dataset(train_df['text'].values.tolist(), train_df['target'].values.tolist(), tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=data_collator)

    pretrain_model = ChatGLMModel()
    checkpoint_callback = ModelCheckpoint(dirpath=f'model_ckp', filename='chatglm6b-{epoch}-{loss:.4f}', 
                                              monitor='loss', mode='min', save_last=False, save_on_train_epoch_end=True,
                                              every_n_epochs=10000)
    accumulate_grad_batches = 32
    trainer = pl.Trainer(accelerator='gpu', devices=1, callbacks=[checkpoint_callback], 
                            max_epochs=50, accumulate_grad_batches=accumulate_grad_batches, 
                            log_every_n_steps=1)
    # trainer.fit(pretrain_model, train_dataloaders=train_dataloader, val_dataloaders=eval_dataloader)
    trainer.fit(pretrain_model, train_dataloaders=train_dataloader)


if __name__ == '__main__':
    train()

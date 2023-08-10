import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
os.environ["WORLD_SIZE"] = "1"

from typing import Any
import lightning.pytorch as pl
from transformers import AutoTokenizer, AutoModel, BloomForCausalLM
import torch
import pandas as pd
import lightning.pytorch as pl
from data import Dataset
from tqdm import *
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import numpy as np
from deepspeed.ops.adam import FusedAdam
from torch.optim import Adam

pl.seed_everything(0)

model_path = '/home/jinxiang/workspace/backbone_models/phoenix-inst-chat-7b'
max_length = 2000
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


class PhoenixModel(pl.LightningModule):

    def __init__(self, model_path=model_path):
        super().__init__()
        # self.model = BloomForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='auto')
        self.model = BloomForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model.enable_input_require_grads()
        self.model.is_parallelizable = True
        self.model.model_parallel = True
        self.model.config.use_cache = (
            False  # silence the warnings. Please re-enable for inference!
        ) 
        torch.cuda.empty_cache()
        self.train_label = []
        self.train_pred = []
        self.eval_label = []
        self.eval_pred = []
        self.epoch_idx = 0
        self.best_eval_epoch_acc = 0
        
    def forward(self, input_ids, attn_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
    
    def training_step(self, batch, batch_idx):
        result = self.model(input_ids=batch['input_ids'], 
                            attention_mask=batch['attn_mask'], 
                            labels=batch['labels'])
        self.log('loss', result.loss, on_step=True, prog_bar=True)
        train_pred = torch.argmax(result.logits, dim=-1)
        self.train_pred.append(train_pred)
        self.train_label.append(batch['labels'])
        # if batch_idx > 0 and batch_idx % 7000 == 0:
        #     gen_kwargs = {"max_length": max_length, "num_beams": 3, "do_sample": False, "top_p": 0.8,
        #                             "temperature":0.9, 'repetition_penalty':1.1} 
        #     output = self.lora_model.generate(input_ids=test_input_ids, **gen_kwargs)
        #     print(tokenizer.decode(test_input_ids[0]))
        #     print('-'*30)
        #     res = tokenizer.decode(output[0]) 
        #     res = res[res.index(prefix)+len(prefix):]
        #     try:
        #         res_dict = eval(res)
        #         print('res is dict')
        #         for k,v in res_dict.items():
        #             print(k,':',v)
        #     except:
        #         print('res is not dict!!!')
        #         print(res)
        #     print('='*30)
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
        # self.epoch_idx += 1
        # save_path = 'lora_r16_p2_0526_v1/epoch_{}_epoch_acc_{:.4f}'.format(str(self.epoch_idx), np.mean(batch_acc_lst))
        # self.lora_model.save_pretrained(save_path)

    def validation_step(self, batch, batch_idx) :
        result = self.lora_model(input_ids=batch['input_ids'], labels=batch['labels'])
        self.log('eval_loss', result.loss, on_step=False, prog_bar=True, on_epoch=True)
        eval_pred = torch.argmax(result.logits, dim=-1)
        self.eval_pred.append(eval_pred)
        self.eval_label.append(batch['labels'])
        # if batch_idx > 0 and batch_idx % 2000 == 0:
        #     gen_kwargs = {"max_length": max_length, "num_beams": 3, "do_sample": False, "top_p": 0.8,
        #                             "temperature":0.9, 'repetition_penalty':1.1} 
        #     output = self.lora_model.generate(input_ids=test_input_ids, **gen_kwargs)
        #     print(tokenizer.decode(test_input_ids[0]))
        #     print('-'*30)
        #     res = tokenizer.decode(output[0]) 
        #     res = res[res.index(prefix)+len(prefix):]
        #     try:
        #         res_dict = eval(res)
        #         print('res is dict')
        #         for k,v in res_dict.items():
        #             print(k,':',v)
        #     except:
        #         print('res is not dict!!!')
        #         print(res)
        #     print('='*30)
    
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
        # if eval_epoch_acc < self.best_eval_epoch_acc:
        #     self.trainer.should_stop = True
        # else:
        #     save_path = 'lora_r16_p2_0524_v0/epoch_{}_eval_epoch_acc_{:.4f}'.format(str(self.epoch_idx), eval_epoch_acc)
        #     self.lora_model.save_pretrained(save_path)
        #     self.best_eval_epoch_acc = eval_epoch_acc
        
    def configure_optimizers(self):
        # opt = FusedAdam(self.model.parameters(), lr=5e-5)
        opt = Adam(self.model.parameters(), lr=5e-5)
        return opt


def train():

    train_df = pd.read_csv('../../data/sql_df.csv', nrows=100)

    train_dataset = Dataset(train_df['question'].values.tolist(), train_df['raw_sql'].values.tolist(), tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=train_dataset.data_collator)

    pretrain_model = PhoenixModel()
    checkpoint_callback = ModelCheckpoint(dirpath=f'model_ckp', 
                                          filename='chatglm6b-{epoch}-{loss:.4f}', 
                                          monitor='loss', 
                                          mode='min', 
                                          save_last=False, 
                                          save_on_train_epoch_end=True,
                                          every_n_epochs=100000)
    trainer = pl.Trainer(accelerator='gpu', 
                         strategy="deepspeed_stage_3", 
                         # strategy='ddp',
                         precision=16, 
                         callbacks=[checkpoint_callback], 
                         max_epochs=50, 
                         log_every_n_steps=1)
    # trainer.fit(pretrain_model, train_dataloaders=train_dataloader, val_dataloaders=eval_dataloader)
    trainer.fit(pretrain_model, train_dataloaders=train_dataloader)


if __name__ == '__main__':
    train()
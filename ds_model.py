import os

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
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from torch.optim import Adam
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.loggers import TensorBoardLogger

'''
the following code must added to model.saved_pretrained:

# Save the model
if state_dict is None:
    if save_ds_trained_model:
        import deepspeed
        with deepspeed.zero.GatheredParameters(model_to_save.parameters()):
            state_dict = model_to_save.state_dict()
            state_dict = {
                key: value.cpu()
                for key, value in state_dict.items()
            }
    else:
        state_dict = model_to_save.state_dict()
'''

logger = TensorBoardLogger("tb_logs", name="gtja_nl2sql")

pl.seed_everything(0)

model_path = '/home/jinxiang/workspace/backbone_models/phoenix-inst-chat-7b'
new_model_path = 'nl2sql_full_params_finetune_final'
max_length = 2000
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

test_input_ids = torch.LongTensor([[     6,  34696,   1023,   4505,  12401,   3549,  46638,  20573,  64871,
            814,   1852, 119854,     42,     45,  17865,  23449,  37259,  19816,
          91828, 161639,    189,  65192,   3721, 119854,     62,   6432,  19816,
         110784,     15,   6432,  19816,  45631,     15,   2235,     57,  13854,
          70857,     15,   4043,  17368,  22478,     36,     15,     53,   5227,
             46,  70857,     15,  34762,   9034,     15,  58835,  22478,     36,
             15,  12905,   1262,  29927,     15,   1508,   9291,   9041,     15,
          12905,   7107,   1495,   5649,     15,  10117,  54298,     15,  37878,
             18,  24679,     15,  14627,  13637,     15, 153868,     15,  32902,
             15,  56779,  62646,     41,  15157,     15, 188331, 204722,     15,
           1528,  14627,   3452,  26906,     15,  10117,  14627,  45613,  27557,
             15, 240330,     15,  12905,  52993,   5699,   4526,  10117,     15,
          14627,     15,  86323,  62378,     15,  12905,   1543,   1557,     15,
           2120,  31476,     64,    189,   7085,   3721, 119854,     62,  21666,
          41862, 153987,     15,   4043,  17368,  22478,     36,     15,  21650,
           7199,  71194,  22015,     15,   1508,   9291,  25389,     15,  95192,
          27134,   1114,     15,   4484,   5360,   1114,     15,   5473,   5360,
           1114,     15,   9291,   1996,   3859,   5649,   1114,     64,    189,
         119858,   3721, 119854,     62,  90435, 194622,  76073,     15,  90435,
             66,  10507,     57,     15,  17110,     48,     15,  90605, 210838,
             15,  17110,     48,  12539,     58,     15,  17110,     48,  12539,
             48,     15,  17110,     48,  36078,     48,     15,  17110,  67615,
             25,     48,     15,  17110,     48,  12539,     60,     15,  83464,
          11335,   5561,   5654,     15,  22116,  11335,   5561,   5654,     15,
         109813,  11335,   5561,   5654,     15, 137498,  11335,   5561,   5654,
             15, 242876,  11335,   5561,   5654,     15,     38,     59,  17479,
         169668,     15,   4170,   2265,   6332,  53764,  17103,   5473,   5360,
             64,    189,   3549,     29,    567,  44740,  21767,  28685, 153868,
          44102,  33539,   9447,  22116,  60654,   2241,  11397,     68,    373,
          21650,   7199,     34,      5]])

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
        result = self.model(**batch)
        self.log('loss', result.loss, on_step=True, prog_bar=True, sync_dist=True)
        train_pred = torch.argmax(result.logits, dim=-1)
        self.train_pred.append(train_pred[:,:-1])  # ignore input of eos token
        self.train_label.append(batch['labels'][:,1:]) # shift labels
        return {'loss':result.loss}
    
    def on_train_epoch_end(self) -> None:
        batch_acc_lst, batch_token_acc_lst = [], []
        for i, (train_pred, train_label) in enumerate(zip(self.train_pred, self.train_label)):
            train_pred = train_pred.cpu()
            train_label = train_label.cpu()
            # train_label = torch.cat([train_label[:,1:], 
            #                          torch.zeros((train_label.shape[0],1)).fill_(tokenizer.eos_token_id).to(train_label.dtype)], 
            #                          dim=-1)  ## fix the label to match the position of pred
            label_mask = torch.where(train_label==-100, 0, 1)
            label_match = (train_label==train_pred) * label_mask
            batch_match = torch.sum(label_match, dim=-1) / torch.sum(label_mask, dim=-1)
            batch_token_acc = torch.sum(batch_match) / label_mask.shape[0]
            batch_match = torch.where(batch_match>=1, 1, 0)
            batch_acc = torch.sum(batch_match) / batch_match.shape[0]
            batch_acc_lst.append(batch_acc.cpu().numpy())
            batch_token_acc_lst.append(batch_token_acc.cpu().numpy())
        self.log('epoch_acc', np.mean(batch_acc_lst), sync_dist=True)
        self.log('epoch_token_acc', np.mean(batch_token_acc_lst), sync_dist=True)
        self.train_pred = []
        self.train_label = []

        # if self.global_rank == 0:
        #     print("\nstart generate")
        #     gen_kwargs = {"max_length": 400, 
        #                 "num_beams": 1,
        #                 "do_sample": False, 
        #                 "top_p": 0.8,
        #                 "temperature":0.9, 
        #                 'repetition_penalty':1.1} 
        #     output = self.model.generate(input_ids=test_input_ids.to(self.model.device), **gen_kwargs)
        #     prefix = tokenizer.decode(test_input_ids[0])
        #     print('prefix:', prefix)
        #     print('-'*30)
        #     res = tokenizer.decode(output[0]) 
        #     res = res[res.index(prefix)+len(prefix):]
        #     print('res:', res)
        #     print('='*30)

    def validation_step(self, batch, batch_idx) :
        result = self.model(**batch)
        self.log('eval_loss', result.loss, on_step=False, prog_bar=True, on_epoch=True, sync_dist=True)
        eval_pred = torch.argmax(result.logits, dim=-1)
        self.eval_pred.append(eval_pred[:,:-1])
        self.eval_label.append(batch['labels'][:,1:])
    
    def on_validation_epoch_end(self):
        batch_acc_lst, batch_token_acc_lst = [], []
        for eval_pred, eval_label in zip(self.eval_pred, self.eval_label):
            eval_pred = eval_pred.cpu()
            eval_label = eval_label.cpu()
            # eval_label = torch.cat([eval_label[:,1:], 
            #                          torch.zeros((eval_label.shape[0],1)).fill_(tokenizer.eos_token_id).to(eval_label.dtype)], 
            #                          dim=-1)  ## fix the label to match the position of pred
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
        self.log('eval_epoch_acc', eval_epoch_acc, sync_dist=True)
        self.log('eval_epoch_token_acc', eval_epoch_token_acc, sync_dist=True)
        self.eval_label = []
        self.eval_pred = []
        
        if self.epoch_idx > 0:
            self.model.save_pretrained('nl2sql_full_params_finetune/epoch_{}_eval_epoch_acc_{:.4f}'.format(str(self.epoch_idx), eval_epoch_acc),
            save_ds_trained_model=True)
            
        self.epoch_idx += 1
        # if eval_epoch_acc < self.best_eval_epoch_acc:
        #     self.trainer.should_stop = True
            # self.model.save_pretrained('nl2sql_full_params_finetune/epoch_{}_eval_epoch_acc_{:.4f}'.format(str(self.epoch_idx), eval_epoch_acc))
        # else:
        #     save_path = 'lora_r16_p2_0524_v0/epoch_{}_eval_epoch_acc_{:.4f}'.format(str(self.epoch_idx), eval_epoch_acc)
        #     self.lora_model.save_pretrained(save_path)
        #     self.best_eval_epoch_acc = eval_epoch_acc
        
    def configure_optimizers(self):
        opt = DeepSpeedCPUAdam(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        # opt = FusedAdam(self.model.parameters(), lr=5e-5)
        # opt = Adam(self.model.parameters(), lr=1e-5, weight_decay=0.01)
        return opt


def train():

    train_df = pd.read_csv('../../data/train.csv', nrows=None)
    dev_df = pd.read_csv('../../data/dev.csv', nrows=None)

    train_dataset = Dataset(train_df['inputs_pretokenized'].values.tolist(), train_df['targets_pretokenized'].values.tolist(), tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=train_dataset.data_collator)
    dev_dataset = Dataset(dev_df['inputs_pretokenized'].values.tolist(), dev_df['targets_pretokenized'].values.tolist(), tokenizer, max_length)
    dev_dataloader = DataLoader(dev_dataset, batch_size=60, shuffle=False, collate_fn=dev_dataset.data_collator)

    model = PhoenixModel()
    checkpoint_callback = ModelCheckpoint(dirpath=f'model_ckp', 
                                          filename='gtjn_nl2sql-{epoch}-{loss:.4f}-{eval_epoch_acc:.4f}', 
                                          monitor='eval_epoch_acc', 
                                          mode='max', 
                                          save_last=False, 
                                          save_on_train_epoch_end=True,
                                          every_n_epochs=10000)
    trainer = pl.Trainer(accelerator='gpu', 
                         strategy="deepspeed_stage_3_offload", 
                         # strategy='ddp',
                         precision=16, 
                         callbacks=[checkpoint_callback], 
                         max_epochs=4, 
                         log_every_n_steps=20,
                         logger=logger,
                         accumulate_grad_batches=1)
    trainer.fit(model, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=dev_dataloader
                )

    model.model.save_pretrained('nl2sql_full_params_finetune_final', save_ds_trained_model=True)



if __name__ == '__main__':
    train()
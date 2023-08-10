from torch.utils.data import Dataset
from tqdm import *
import torch

class Dataset(Dataset):

    label_ignore_idx: int = 100

    def __init__(self, texts, targets, tokenizer, max_length:int=500):
        self.texts = texts
        self.targets = targets
        self.len = len(texts)
        self.text_tokens = []
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.feature = []
        for text, target in zip(texts, targets):
            self.feature.append(self.preprocess(text, target))

    def __getitem__(self, index):
        return self.feature[index]
    
    def __len__(self):
        return self.len

    def preprocess(self, text, target):
        prompt_ids = self.tokenizer.encode(text, max_length=self.max_length, truncation=True)
        target_ids = self.tokenizer.encode(
            target,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False)
        input_ids = prompt_ids + target_ids + [self.tokenizer.eos_token_id]
        return {"input_ids": input_ids, "seq_len": len(prompt_ids)}
    
    def data_collator(self, features: list) -> dict:
        len_ids = [len(feature["input_ids"]) for feature in features]
        longest = max(len_ids)
        input_ids = []
        labels_list = []
        for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
            ids = feature["input_ids"]
            seq_len = feature["seq_len"]
            labels = (
                [self.label_ignore_idx] * (seq_len - 1) + ids[(seq_len - 1) :] + [self.label_ignore_idx] * (longest - ids_l)  # ignore input prompt when calculate loss
            )
            ids = ids + [self.tokenizer.pad_token_id] * (longest - ids_l)
            _ids = torch.LongTensor(ids)
            labels_list.append(torch.LongTensor(labels))
            input_ids.append(_ids)
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels_list)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attn_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
    


from torch.utils.data import Dataset
from tqdm import *

class Dataset(Dataset):

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
    


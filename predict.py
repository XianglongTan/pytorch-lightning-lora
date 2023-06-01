import json
from model import ChatGLMModel
from transformers import AutoTokenizer
from tqdm import *
import torch
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput
import pandas as pd

model = ChatGLMModel()
lora_model = model.lora_model.from_pretrained(model.model, 'lora_r16_p2_0526_v1/epoch_40_epoch_acc_0.6877').eval()
max_length = 1000
model_path = '../../model/chatglm-6b'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
with open('../data/instruction_testB_with_node.json', 'r') as f:
    test_data = json.load(f)
page_node_df = pd.read_csv('../data/0525/page_node_p2_df.csv')
# test_df = pd.read_csv('../data/0523/test_df.csv')
prefix = "要在该网页上完成这个指令,用json格式生成操作和相应参数:"


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


def predict():
    res = []
    for data in test_data:
        page = data['page_source']
        page_res = {'page_source': page, 'instruction_detail':[]}
        for instruction_dict in tqdm(data['instruction_detail']):
            row_text = '网页:{}。指令:{}。请用json格式生成在该网页上的操作:'.format(page, instruction_dict['instruction'])
            input_ids = tokenizer(row_text, return_tensors='pt')['input_ids'].to(lora_model.device)
            gen_kwargs = {"max_length": max_length, "num_beams": 1, "do_sample": False, "top_p": 0.7,
                                    "temperature":1 } 
            output = lora_model.generate(input_ids=input_ids, **gen_kwargs)
            row_res = tokenizer.decode(output[0])
            try:
                row_res = eval(row_res[row_res.index('请用json格式生成在该网页上的操作:')+len('请用json格式生成在该网页上的操作:'):].strip(' '))
            except Exception as e:
                print('error: ', e)
                print(row_res[row_res.index('请用json格式生成在该网页上的操作:')+len('请用json格式生成在该网页上的操作:'):].strip(' '))
                row_res = {}
            page_res['instruction_detail'].append({'instruction':instruction_dict['instruction'],
                                                'key-value': row_res})
        res.append(page_res)
    with open('sub/chatglm6b_0518_v0.json', 'w') as f:
        json.dump(res, f)


def predict_batch(batch_size:int=3):
    res = []
    error_count = 0
    total_count = 0
    for page_idx, data in enumerate(test_data):
        page, page_node = data['page_source'], data['node_name']
        page_res = {'page_source': page, 'instruction_detail':[]}
        for idx in trange(0, len(data['instruction']), batch_size):
            if len(data['instruction'][idx:idx+batch_size]) <= 0:
                break
            row_texts, batch_instruction = [], []
            for instruction in data['instruction'][idx:idx+batch_size]:
                row_text = '网页:{}。该网页可操作的元素有:{}。指令:{}。{}'.format(page, page_node, instruction, prefix)
                row_texts.append(row_text)
                batch_instruction.append(instruction)
            input_ids = tokenizer(row_texts, return_tensors='pt', padding='longest')['input_ids'].to(lora_model.device)
            logits_processor = LogitsProcessorList()
            logits_processor.append(InvalidScoreLogitsProcessor())
            gen_kwargs = {"max_new_tokens": max_length, "num_beams": 3, "do_sample": False, "top_p": 0.8,
                                    "temperature":0.9, 'repetition_penalty':1.1, "logits_processor": logits_processor,} 
            output = lora_model.generate(input_ids=input_ids, **gen_kwargs)
            for i in range(input_ids.shape[0]):
                row_res = tokenizer.decode(output[i])
                try:
                    row_res = eval(row_res[row_res.index(prefix)+len(prefix):].strip(' '))
                    if not isinstance(row_res, dict):
                        error_count += 1
                        print(111, row_res)
                        row_res = {}
                    elif any(isinstance(v, list)==False for v in row_res.values()):
                        error_count += 1
                        print(222, row_res)
                        row_res = {}
                except Exception as e:
                    print('error: ', e)
                    print(batch_instruction[i])
                    print(row_res[row_res.index(prefix)+len(prefix):].strip(' '))
                    error_count += 1
                    row_res = {}
                page_res['instruction_detail'].append({'instruction':batch_instruction[i],
                                                    'key-value': row_res})
                # print(row_res)
                total_count += 1
        res.append(page_res)
        print('page_idx:{}  total_count:{}  error_count:{}'.format(page_idx, total_count, error_count))
    with open('sub/chatglm6b_r16_epoch_40_with_node_prompt_test_mid_sub_0526_v0.json', 'w') as f:
        json.dump(res, f)



if __name__ == '__main__':
    predict_batch(1)
# --*-- encoding=gbk --*--
import os
import json
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torch import Generator

class MyDataset(Dataset):

    def __init__(self, raw_data, label_dict, tokenizer, model_name, method):
        label_list = list(label_dict.keys()) if method not in ['ce', 'scl'] else []
        
        # 记录每个标签词被分成了几个 token
        self.label_word_lengths = []
        if method not in ['ce', 'scl']:
            for lbl in label_list:
                tid = tokenizer.tokenize(lbl)
                ## assert len(tid) == 1, f"标签词 '{lbl}' 被拆分为 {tid}，必须是单token"
                self.label_word_lengths.append(len(tid))
                print(f"标签词 '{lbl}' -> {tid} ({len(tid)} tokens)")
        
        sep_token = ['[SEP]'] if model_name in ['bert', 'bert_chinese'] else ['</s>']
        dataset = list()
        for data in raw_data:
            tokens = data['text'].lower().split(' ')
            label_id = label_dict[data['label']]
            dataset.append((label_list + sep_token + tokens, label_id))
        self._dataset = dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


def my_collate(batch, tokenizer, method, num_classes):
    tokens, label_ids = map(list, zip(*batch))
    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         max_length=256,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    if method not in ['ce', 'scl']:
        #positions = torch.zeros_like(text_ids['input_ids'])
        #positions[:, num_classes:] = torch.arange(0, text_ids['input_ids'].size(1)-num_classes)
        
        prefix_len = num_classes + 1  # label_list + sep_token
        positions = torch.zeros_like(text_ids['input_ids'])
        seq_len = text_ids['input_ids'].size(1)
        if seq_len > prefix_len:
            positions[:, prefix_len:] = torch.arange(0, seq_len - prefix_len)
            
        text_ids['position_ids'] = positions
    return text_ids, torch.tensor(label_ids)


def load_data(dataset, data_dir, tokenizer, train_batch_size, test_batch_size, model_name, method, workers, seed=42):
    if dataset == 'fraud':
        with open(os.path.join(data_dir, f'fraud_Train.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(os.path.join(data_dir, f'fraud_Test.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        label_dict = {'诈骗': 0, '正常': 1}
    elif dataset == 'fraud_after_TextFooler':
        with open(os.path.join(data_dir, f'fraud_Train.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(os.path.join(data_dir, f'fraud_Test_afer_TextFooler.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        label_dict = {'诈骗': 0, '正常': 1}
    elif dataset == 'fraud_3.5_R1':
        with open(os.path.join(data_dir, f'fraud_Train.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(os.path.join(data_dir, f'step-3.5-flash/fraud_Test_R1.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        label_dict = {'诈骗': 0, '正常': 1}
    elif dataset == 'fraud_3.5_R2':
        with open(os.path.join(data_dir, f'fraud_Train.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(os.path.join(data_dir, f'step-3.5-flash/fraud_Test_R2.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        label_dict = {'诈骗': 0, '正常': 1}
    elif dataset == 'fraud_3.5_R3':
        with open(os.path.join(data_dir, f'fraud_Train.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(os.path.join(data_dir, f'step-3.5-flash/fraud_Test_R3.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        label_dict = {'诈骗': 0, '正常': 1}
    elif dataset == 'fraud_3.7_R1':
        with open(os.path.join(data_dir, f'fraud_Train.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(os.path.join(data_dir, f'step-3.7-flash/fraud_Test_R1.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        label_dict = {'诈骗': 0, '正常': 1}
    elif dataset == 'fraud_3.7_R2':
        with open(os.path.join(data_dir, f'fraud_Train.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(os.path.join(data_dir, f'step-3.7-flash/fraud_Test_R2.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        label_dict = {'诈骗': 0, '正常': 1}
    elif dataset == 'fraud_3.7_R3':
        with open(os.path.join(data_dir, f'fraud_Train.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(os.path.join(data_dir, f'step-3.7-flash/fraud_Test_R3.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        label_dict = {'诈骗': 0, '正常': 1}
    else:
        raise ValueError('unknown dataset')
    trainset = MyDataset(train_data, label_dict, tokenizer, model_name, method)
    testset = MyDataset(test_data, label_dict, tokenizer, model_name, method)
    collate_fn = partial(my_collate, tokenizer=tokenizer, method=method, num_classes=len(label_dict))
    
    #train_dataloader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn, pin_memory=True)
    train_g = Generator()
    train_g.manual_seed(seed)
    train_dataloader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn, pin_memory=True, generator=train_g)
    
    test_dataloader = DataLoader(testset, test_batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn, pin_memory=True)
    
    label_word_lengths = trainset.label_word_lengths
    return train_dataloader, test_dataloader, label_word_lengths

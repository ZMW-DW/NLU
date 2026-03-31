import torch
import json
from transformers import BertTokenizer
from typing import List, Dict
from .config import intent2id, slot2id
from torch.utils.data import Dataset

class NLUDataset(Dataset):
    def __init__(self, file_path):
        """
        在初始化时直接加载并解析文件
        """
        self.file_path = file_path
        self.data = self._load_and_format(file_path)

    def _load_and_format(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        formatted_data = []
        for item in raw_data:
            formatted_data.append({
                "text": item.get("question", ""),
                "tokens": item.get("tokens", []),
                "slots": item.get("slots", []),
                "intent": item.get("function", {}).get("name", "")
            })
        return formatted_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def collate_fn_factory(tokenizer, max_len):
    """
    这是一个闭包，用来给 DataLoader 提供自定义的 batch 处理逻辑
    即你提到的 map 操作，但在 PyTorch 中通常在 collate_fn 里做动态 padding
    """
    def collate_fn(batch_list):
        return prepare_batch(batch_list, tokenizer, max_len)
    return collate_fn

def prepare_labels(tokens, slot_tags, tokenizer, max_len):
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        padding='max_length',
        max_length=max_len,
        truncation=True,
        return_tensors="pt"
    )
    word_ids = encoding.word_ids(batch_index=0)
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        else:
            tag_id = slot2id[slot_tags[word_idx]]
            label_ids.append(tag_id)
    
    return torch.tensor([label_ids], dtype=torch.long)

def prepare_batch(batch_data: List[Dict], tokenizer: BertTokenizer, max_len: int = 128):
    input_ids_list = []
    attention_mask_list = []
    intent_labels_list = []
    slot_labels_list = []
    
    for item in batch_data:
        # 1. 处理文本输入
        text_encoding = tokenizer(
            item['text'],
            padding='max_length',
            max_length=max_len,
            truncation=True,
            return_tensors="pt"
        )
        input_ids_list.append(text_encoding['input_ids'].squeeze(0))
        attention_mask_list.append(text_encoding['attention_mask'].squeeze(0))
        
        # 2. 处理意图标签
        intent_id = intent2id[item['intent']]
        intent_labels_list.append(intent_id)
        
        # 3. 处理槽位标签
        slot_labels = prepare_labels(item['tokens'], item['slots'], tokenizer, max_len)
        slot_labels_list.append(slot_labels.squeeze(0))
    
    # 堆叠成 Batch Tensor
    return {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_mask_list),
        'intent_labels': torch.tensor(intent_labels_list, dtype=torch.long),
        'slot_labels': torch.stack(slot_labels_list)
    }
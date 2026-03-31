import torch
import torch.nn as nn
from .model import NLUModel, compute_loss
from tqdm import tqdm

class NLUTrainer:
    def __init__(self, model: NLUModel, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def train_step(self, batch, optimizer, intent_weight, ner_weight):
        """
        优化后的单步训练：直接接收 DataLoader 产生的 Tensor Batch
        """
        self.model.train()
        
        # 批量移动到设备
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        
        # forward
        outputs = self.model(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask']
        )
        
        # 计算损失
        total_loss, intent_loss, slot_loss = compute_loss(
            outputs, 
            inputs['intent_labels'], 
            inputs['slot_labels'],
            intent_weight,
            ner_weight
        )
        
        # backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        
        return total_loss.item(), intent_loss.item(), slot_loss.item()

    @torch.no_grad()
    def evaluate(self, data_loader):
        """简单的评估逻辑"""
        self.model.eval()
        correct_intent = 0
        total = 0
        for batch in data_loader:
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            
            preds = torch.argmax(outputs.intent_logits, dim=-1)
            correct_intent += (preds == inputs['intent_labels']).sum().item()
            total += inputs['intent_labels'].size(0)
            
        return correct_intent / total
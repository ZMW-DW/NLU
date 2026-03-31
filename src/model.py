import torch
import math
import torch.nn as nn
from transformers import BertModel
from dataclasses import dataclass
from .config import NLU_config, INTENT_LIST, SLOT_LABELS

@dataclass
class NLUModelOutput:
    intent_logits: torch.Tensor
    slot_logits: torch.Tensor
    hidden_states: torch.Tensor = None  

class NLUModel(nn.Module):
    def __init__(self, config: NLU_config):
        super(NLUModel, self).__init__()
        self.config = config
        # 如果本地没有模型，可以先下载或使用在线标识
        self.bert = BertModel.from_pretrained(config.model_path)
        
        self.intent_classifier = nn.Linear(config.hidden_dimension, config.classification_numbers)
        self.slot_classifier = nn.Linear(config.hidden_dimension, config.slots_numbers)
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        # BERT 编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # 意图分类 (使用 [CLS] token)
        cls_hidden = last_hidden_state[:, 0, :]  # [Batch, Hidden]
        cls_hidden = self.dropout(cls_hidden)
        intent_logits = self.intent_classifier(cls_hidden)
        
        # 槽位填充 (使用序列所有 Token)
        seq_hidden = self.dropout(last_hidden_state)  # [Batch, Seq, Hidden]
        slot_logits = self.slot_classifier(seq_hidden)  # [Batch, Seq, Num_Slots]
        
        return NLUModelOutput(
            intent_logits=intent_logits,
            slot_logits=slot_logits,
            hidden_states=last_hidden_state
        )

# ==========================================
# 损失函数 (Loss Function)
# ==========================================
def compute_loss(
    outputs: NLUModelOutput, 
    intent_labels: torch.Tensor, 
    slot_labels: torch.Tensor, 
    intent_weight: float,
    ner_weight: float,
):
    """
    计算联合损失 (Joint Loss)
    """
    # 1. 意图损失
    intent_loss_fct = nn.CrossEntropyLoss()
    intent_loss = intent_loss_fct(outputs.intent_logits, intent_labels)

    # 2. 槽位损失
    # 展平张量以适应 CrossEntropyLoss (Batch*Seq, Num_Slots)
    batch_size, seq_len, num_slots = outputs.slot_logits.shape
    slot_logits_flat = outputs.slot_logits.view(-1, num_slots)
    slot_labels_flat = slot_labels.view(-1)

    slot_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # 忽略 -100 (PAD/SEP/CLS)
    slot_loss = slot_loss_fct(slot_logits_flat, slot_labels_flat)

    # 3. 加权总损失
    total_loss = intent_weight * intent_loss + ner_weight * slot_loss
    
    return total_loss, intent_loss, slot_loss


def predict(model, tokenizer, text, device, config):
    model.eval()
    
    # 1. 对输入文本进行编码
    inputs = tokenizer(
        text, 
        return_offsets_mapping=True, 
        padding='max_length', 
        max_length=config.max_tokens, 
        truncation=True, 
        return_tensors="pt"
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    offset_mapping = inputs['offset_mapping'][0] 
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    # --- A. 解析意图 (Intent) 与 资信度 ---
    # outputs.intent_logits: [batch_size, num_intents]
    intent_probs = torch.softmax(outputs.intent_logits, dim=-1)
    intent_conf, intent_pred_idx = torch.max(intent_probs, dim=-1)
    
    intent_id = intent_pred_idx.item()
    intent_name = INTENT_LIST[intent_id]
    intent_confidence = intent_conf.item()

    # --- B. 槽位解析与局部置信度 ---
    slot_probs_dist = torch.softmax(outputs.slot_logits, dim=-1)
    slot_preds = torch.argmax(outputs.slot_logits, dim=-1)[0].cpu().numpy()
    
    # 获取每个 Token 被选中标签的具体概率
    # slot_probs_dist: [1, seq_len, num_slots]
    token_best_probs = torch.max(slot_probs_dist, dim=-1)[0][0] 

    slots = {}
    current_slot = None
    current_value = ""
    key_token_confidences = []

    last_end = -1

    for i, slot_id in enumerate(slot_preds):
        tag = SLOT_LABELS[slot_id]
        if tag != "O": # 💡 只关注关键槽位
            key_token_confidences.append(token_best_probs[i].item())

        # 确保 offset_mapping 是 tensor 或者是可索引对象
        start, end = offset_mapping[i][0].item(), offset_mapping[i][1].item()
        
        # 过滤掉 BERT 自动补全的无效 mapping (0,0)
        if start == 0 and end == 0:
            continue

        if tag.startswith("B-"):
            if current_slot:
                slots[current_slot] = current_value.strip()
            current_slot = tag[2:]
            current_value = text[start:end]
            last_end = end # 【关键】B- 也要更新 last_end
            
        elif tag.startswith("I-") and current_slot == tag[2:]:
            # 【关键】判断是否需要补空格：如果当前 start 大于上一个 end
            if start > last_end and last_end != -1:
                current_value += " "
            
            current_value += text[start:end]
            last_end = end # 【关键】更新结束位置
            
        else:
            if current_slot:
                slots[current_slot] = current_value.strip()
            current_slot = None
            current_value = ""
            last_end = -1
    
    # 循环结束后处理最后一个槽位
    if current_slot:
        slots[current_slot] = current_value.strip()

    # --- C. 计算综合可靠性 ---
    avg_slot_conf = sum(key_token_confidences) / len(key_token_confidences) if key_token_confidences else 1.0
    overall_reliability = avg_slot_conf * intent_confidence

    return {
        "intent": intent_name,
        "intent_confidence": intent_confidence,
        "slots": slots,
        "slot_confidences": avg_slot_conf,
        "overall_reliability": overall_reliability
    }
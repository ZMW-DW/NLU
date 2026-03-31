import torch
import os
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.optim import AdamW
from tqdm import tqdm

from src.model import NLUModel
from src.trainer import NLUTrainer
from src.config import NLU_config
from src.data_utils import NLUDataset, collate_fn_factory

def main():
    config = NLU_config()
    # 确保保存目录存在
    os.makedirs("checkpoints", exist_ok=True)
    
    tokenizer = BertTokenizer.from_pretrained(config.model_path)
    model = NLUModel(config)
    trainer = NLUTrainer(model, tokenizer)
    
    # 1. 加载训练集与测试集 (验证集)
    train_path = '/data/home/daiwei/NLU_Model/datasets/scripts/train.json'
    test_path = '/data/home/daiwei/NLU_Model/datasets/scripts/test.json'
    
    collate_fn = collate_fn_factory(tokenizer, config.max_tokens)
    
    train_loader = DataLoader(
        NLUDataset(train_path), 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        NLUDataset(test_path), 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    optimizer = AdamW(model.parameters(), lr=5e-5)

    print(f"🚀 开始训练 | 设备: {trainer.device}")
    print(f"📊 训练样本: {len(train_loader)} | 验证样本: {len(test_loader)}")

    for epoch in range(config.epochs):
        # --- 训练阶段 ---
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs} [Train]")
        
        for batch_data in pbar:
            loss, i_loss, s_loss = trainer.train_step(
                batch_data, 
                optimizer, 
                config.intent_weight, 
                config.ner_weight
            )
            total_train_loss += loss
            pbar.set_postfix({'loss': f"{loss:.3f}", 'i_loss': f"{i_loss:.3f}", "s_loss" : f"{s_loss:.3f}"})

        avg_loss = total_train_loss / len(train_loader)
        
        # --- 验证阶段 ---
        val_acc = trainer.evaluate(test_loader)
        
        print(f"✅ Epoch {epoch+1} 结束 | 平均 Loss: {avg_loss:.4f} | 验证集意图准确率: {val_acc:.2%}")

        # 保存模型
        save_path = f"checkpoints/nlu_model_e{epoch+1}_acc{val_acc:.3f}.bin"
        torch.save(model.state_dict(), save_path)
        print(f"💾 模型已保存至: {save_path}\n")

if __name__ == "__main__":
    main()
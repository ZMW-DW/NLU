import torch
import json
import os
from transformers import BertTokenizer
from src.model import NLUModel
from src.config import NLU_config
from src.pipeline import NLUPipeline

def run_evaluation():
    # 1. 基础配置
    config = NLU_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 加载基础设施
    tokenizer = BertTokenizer.from_pretrained(config.model_path)
    model = NLUModel(config)
    
    model_checkpoint = "/data/home/daiwei/NLU_Model/checkpoints/nlu_model_e4_acc0.954.bin"
    if not os.path.exists(model_checkpoint):
        print(f"❌ 未找到模型文件: {model_checkpoint}")
        return

    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. 初始化 Pipeline
    tools_path = '/data/home/daiwei/NLU_Model/datasets/tools.json'
    with open(tools_path, "r") as f:
        tools_data = json.load(f)
        
    pipeline = NLUPipeline(model, tokenizer, config, tools_data)
    print(f"✅ Pipeline 就绪 | 权重: {os.path.basename(model_checkpoint)}\n")

    # 4. 加载评估数据
    eval_path = '/data/home/daiwei/NLU_Model/datasets/evaluate.json'
    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    # 5. 评测统计变量
    correct_count = 0
    # total_count = len(eval_data)
    total_count = 0
    low_rel_threshold = 0.6  # 设定置信度阈值
    ignore_keys = ["value"]  # 你提到的不需要管的参数

    print(f"{'Question':<35} | {'Match':<8} | {'Rel':<8} | {'Result'}")
    print("-" * 120)

    for item in eval_data:
        text = item['question']
        gt_func = item['function'] # Ground Truth
        
        # 模型推理
        result = pipeline.execute(text)
        pred_func = result['function_call']

        if pred_func['name'] == "manage_casting_session": continue
            
        total_count += 1
        rel = result['reliability']

        # --- 核心逻辑：自动对齐校验 ---
        is_match = True
        # 1. 校验函数名
        if pred_func['name'] != gt_func['name']: is_match = False
        else:
            # 2. 校验参数（排除 ignore_keys）
            gt_params = gt_func.get('parameters', {})
            pred_params = pred_func.get('parameters', {})
            
            for k, v in gt_params.items():
                if k in ignore_keys: continue 
                if k not in pred_params or pred_params[k] != v:
                    is_match = False
                    break
        
        if is_match: correct_count += 1

        # --- 打印展示 ---
        match_str = "✅ PASS" if is_match else "❌ FAIL"
        warning = "⚠️ LOW" if rel < low_rel_threshold else "    "
        
        # 截断长文本方便对齐观察
        display_text = text[:33] + ".." if len(text) > 33 else text
        res_json = json.dumps(pred_func['parameters'])
        
        print(f"{display_text:<35} | {match_str:<8} | {rel:<8.3f} {warning} | {pred_func['name']}: {res_json}")

    # 6. 最终汇总报告
    accuracy = correct_count / total_count
    print("-" * 120)
    print(f"📊 评测汇总:")
    print(f"   - 样本总数: {total_count}")
    print(f"   - 匹配成功: {correct_count}")
    print(f"   - 准确率 (Accuracy): {accuracy:.2%}")
    print(f"   - 置信度阈值: {low_rel_threshold} (低于此值建议人工复核)")

if __name__ == "__main__":
    run_evaluation()
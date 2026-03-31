from dataclasses import dataclass
from typing import Dict, List

@dataclass
class NLU_config:
    # BERT 路径
    model_path: str = "/data/home/daiwei/models/google-bert/bert-base-uncased"  # 建议本地路径或 huggingface 标识
    hidden_dimension: int = 768
    max_tokens: int = 128
    
    # 分类数量（根据你的数据调整）
    classification_numbers: int = 10  # INTENT_LIST 长度
    @property
    def slots_numbers(self):
        return len(SLOT_LABELS)
    # slots_numbers: int = 21           # SLOT_LABELS 长度
    dropout_rate: float = 0.1

    intent_weight: float = 0.3
    ner_weight: float = 0.7
    epochs: int = 4
    batch_size: int = 16
    

# ==========================================
# 标签映射 (LABEL MAPPING)
# ==========================================
INTENT_LIST = [
    "control_media_devices",
    "configure_display_behavior",
    "configure_casting_protocol",
    "configure_screen_sharing",
    "set_access_policy",
    "set_reverse_control_policy",
    "configure_system_limits",
    "manage_casting_session",
    "configure_device_settings",
    "handle_ui_action"
]

SLOT_LABELS = [
    "O",
    "B-DEV", "I-DEV",     # 媒体设备 (microphone, camera)
    "B-FEAT", "I-FEAT",   # 显示功能 (auto_fullscreen, floating_window)
    "B-PROT", "I-PROT",   # 投屏协议 (airplay, miracast)
    "B-POL_T", "I-POL_T", # 策略类型 (policy_type)
    "B-MODE", "I-MODE",   # 模式 (allow_all, require_password)
    "B-LIM_T", "I-LIM_T", # 限制类型 (split_screen, refresh_interval)
    "B-ACT", "I-ACT",     # 函数特定动作 (start_cast, disconnect_all)
    "B-SET", "I-SET",     # 设置项 (device_name, display_mode)
    "B-STATE", "I-STATE", # 💡 新增：逻辑状态 (on, off, mute, active) -> 对应 enable
    "B-TGT", "I-TGT",     # 💡 新增：操控目标 (device 1, all, CEO's laptop) -> 对应 targets
    "B-VAL", "I-VAL"      # 纯净数值 (password, numeric values)
]

# 构建映射字典
intent2id = {intent: idx for idx, intent in enumerate(INTENT_LIST)}
slot2id = {slot: idx for idx, slot in enumerate(SLOT_LABELS)}
id2intent = {idx: intent for intent, idx in intent2id.items()}
id2slot = {idx: slot for slot, idx in slot2id.items()}
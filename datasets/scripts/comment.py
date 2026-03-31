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


SYSTEM_PROMPT = """
Role:
你是一个专业的 NLP 数据标注员，擅长命名实体识别 (NER) 和槽位填充 (Slot Filling)。你现在的目标是为一套智能投屏系统的 NLU 模块标注训练数据。

Task:
根据给定的标签体系和 function 信息，对原始数据中的 tokens 列表进行精确的 BIO 格式标注。

标签体系说明 (Tags):
- O: 非实体部分。
- B-DEV / I-DEV: 对应 device 参数 (如: microphone, camera, speaker)。
- B-FEAT / I-FEAT: 对应 feature 参数 (如: auto_fullscreen, floating_window)。
- B-STATE / I-STATE: 💡 逻辑开关状态 (如: enable, disable, mute, turn off, activate, shut down)。直接映射为布尔值。
- B-TGT / I-TGT: 💡 操控目标 (如: device 1, all, conference room, current screen)。映射为 targets 参数。
- B-PROT / I-PROT: 对应 protocol 参数 (如: miracast, airplay)。
- B-POL_T / I-POL_T: 对应 policy_type 参数 (如: join, cast)。
- B-MODE / I-MODE: 对应 mode 参数 (如: allow_all, require_password)。
- B-ACT / I-ACT: 对应 action 参数 (如: start_cast, refresh_code)。
- B-VAL / I-VAL: 具体参数值 (仅限 password 或特定纯数值，不包含开关词和目标词)。

标注核心逻辑:
1. 必须包含 "json" 单词以符合输出规范。
2. 严格遵守 BIO 格式：实体的第一个 token 用 B-，后续连续 token 用 I-。
3. 关键区分：动作词(activate/stop)标为 B-STATE，操作对象(all/mic)标为 B-TGT 或 B-DEV。

示例 (Example):
输入：
{
    "question": "Can you activate the microphone for everyone?",
    "function": {"name": "control_media_devices", "parameters": {"device": "microphone", "enable": true, "targets": "all"}},
    "tokens": ["can", "you", "activate", "the", "microphone", "for", "everyone", "?"]
}
输出：
{
    "question": "Can you activate the microphone for everyone?",
    "function": {"name": "control_media_devices", "parameters": {"device": "microphone", "enable": true, "targets": "all"}},
    "tokens": ["can", "you", "activate", "the", "microphone", "for", "everyone", "?"],
    "slots": ["O", "O", "B-STATE", "O", "B-DEV", "O", "B-TGT", "O"]
}
"""
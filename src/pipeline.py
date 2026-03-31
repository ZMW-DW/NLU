import re
import torch
from .model import predict  
from sentence_transformers import SentenceTransformer, util

class SemanticRouter:
    def __init__(self, tools_config, model_path='/data/home/daiwei/models/bge-small-en-v1.5'):
        self.embedder = SentenceTransformer(model_path)
        self.param_types = {} 
        self.index_bank = self._build_index(tools_config)

    def _build_index(self, tools_config):
        index = {}
        for tool in tools_config:
            func = tool['function']
            f_name = func['name']
            properties = func['parameters']['properties']
            index[f_name] = {}
            self.param_types[f_name] = {}

            for p_name, p_info in properties.items():
                self.param_types[f_name][p_name] = p_info.get('type')
                # 只有枚举类型才进入向量库进行语义对齐
                if 'enum' in p_info:
                    enums = p_info['enum']
                    # 💡 描述增强：让向量包含函数的功能上下文
                    descs = [f"{e}: {func['description']}" for e in enums]
                    embeddings = self.embedder.encode(descs, convert_to_tensor=True)
                    index[f_name][p_name] = {"enums": enums, "vecs": embeddings}
        return index

    def match_parameters(self, intent_name, extracted_slots, full_text):
        """
        返回: (final_params, max_router_similarity, logic_conflict)
        """
        if intent_name not in self.param_types:
            return extracted_slots, 0.0, False
            
        final_params = {}
        text_lower = full_text.lower()
        max_sim = 0.0
        logic_conflict = False

        # 1. 💡 语义枚举匹配 (维持原有逻辑，处理 enum 类型参数)
        if intent_name in self.index_bank:
            query_text = " ".join(extracted_slots.values()) if extracted_slots else full_text
            query_vec = self.embedder.encode(query_text, convert_to_tensor=True)
            
            for p_name, data in self.index_bank[intent_name].items():
                scores = util.cos_sim(query_vec, data['vecs'])[0]
                best_idx = torch.argmax(scores).item()
                current_score = scores[best_idx].item()
                max_sim = max(max_sim, current_score)
                
                if current_score > 0.35: 
                    final_params[p_name] = data['enums'][best_idx]

        # 2. 💡 针对新标签 B-STATE 的布尔判定 (核心修复)
        # 提取所有被标注为 STATE 的文本内容
        state_content = extracted_slots.get('STATE', "").lower()
        
        # 强逻辑词库
        neg_words = ['off', 'disable', 'stop', 'terminate', 'close', 'hide', 'mute', 'shut down', 'deny']
        pos_words = ['on', 'enable', 'start', 'activate', 'show', 'open', 'unmute', 'allow']

        if 'enable' in self.param_types[intent_name]:
            # 优先检查是否存在 STATE 标签，如果没有则看全句
            check_text = state_content if state_content else text_lower
            
            if any(w in check_text for w in neg_words):
                final_params['enable'] = False
            elif any(w in check_text for w in pos_words):
                final_params['enable'] = True

        # 3. 💡 针对新标签 B-TGT 的目标归一化
        target_content = extracted_slots.get('TGT', "").lower()
        if target_content:
            if any(kw in target_content for kw in ['all', 'every', 'everyone']):
                final_params['targets'] = "all"
            else:
                nums = re.findall(r'\d+', target_content)
                final_params['targets'] = nums[0] if nums else target_content
        
        # 4. 逻辑冲突检查 (针对 Mute 但 enable 为 True 的防御)
        if 'mute' in text_lower and final_params.get('enable') is True:
            logic_conflict = True

        return final_params, max_sim, logic_conflict

class NLUPipeline:
    def __init__(self, model, tokenizer, config, tools_config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        self.router = SemanticRouter(tools_config)
        
    def execute(self, text):
        nlu_result = predict(self.model, self.tokenizer, text, self.device, self.config)
        
        intent = nlu_result['intent']
        # if nlu_result['overall_reliability'] < 0.4:
        #     if any(kw in text.lower() for kw in ['cast', 'projection', 'mirror']):
        #         intent = "manage_casting_session"

        # 3. 语义参数匹配
        final_params, router_sim, _ = self.router.match_parameters(
            intent, 
            nlu_result['slots'], 
            text
        )
        
        # 4. 💡 综合可靠性公式重构
        slot_conf = nlu_result.get('slot_confidence', nlu_result['overall_reliability'])
        combined_rel = (nlu_result['intent_confidence'] + slot_conf + router_sim) / 3
        
        return {
            "text": text,
            "function_call": {
                "name": intent,
                "parameters": final_params
            },
            "reliability": combined_rel
        }
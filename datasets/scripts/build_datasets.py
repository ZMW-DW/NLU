import json
import asyncio
from openai import AsyncClient
from asyncio import Semaphore

from comment import SYSTEM_PROMPT

semaphore = Semaphore(20)

async def process(data: dict, client: AsyncClient):
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="deepseek-chat",  # 或你使用的具体模型
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(data, ensure_ascii=False)}
                ],
                response_format={"type": "json_object"} # 强制返回 JSON 格式
            )
            
            # 解析模型返回的内容
            result_content = response.choices[0].message.content
            return json.loads(result_content)
        except Exception as e:
            print(f"Error processing item: {e}")
            # 出错时返回带空槽位的原始数据，方便后期追溯
            data["slots"] = ["O"] * len(data.get("tokens", []))
            return data
        
async def main():
    client = AsyncClient(api_key="sk-xx", base_url="https://api.deepseek.com")
    
    input_path = "/data/home/daiwei/NLU_Model/datasets/scripts/process_train.json"
    output_path = "/data/home/daiwei/NLU_Model/datasets/scripts/train.json"

    # 1. 加载数据
    with open(input_path, 'r', encoding='utf-8') as f:
        datasets = json.load(f)

    # 2. 创建并发任务
    tasks = [process(item, client) for item in datasets]
    
    # 3. 执行任务并获取进度
    print(f"开始标注，共 {len(datasets)} 条数据...")
    labeled_results = await asyncio.gather(*tasks)

    # 4. 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labeled_results, f, ensure_ascii=False, indent=4)
    
    print(f"标注完成，结果已保存至: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())


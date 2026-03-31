import json
from transformers import BertTokenizer

def process(text: str, tokenizer: BertTokenizer) -> dict:
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        return_tensors=None
    )
    tokens = encoding.tokens()
    return tokens[1:-1]


def main(model_path:str, datasets_path: str):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    with open(datasets_path, "r") as f:
        data = json.load(f)
    
    buffer = []
    for item in data:
        item['tokens'] = process(item['question'], tokenizer)
        buffer.append(item)

    with open("/data/home/daiwei/NLU_Model/datasets/scripts/process_test.json", "w") as f:
        json.dump(buffer, f, indent=4)


def final_datasets(file_path: str):
    with open(file_path, "r") as f:
        datasets = json.load(f)
    
    buffer = 
    for data in datasets:
        
        
        
if __name__ == "__main__":
    model_path = "/data/home/daiwei/models/google-bert/bert-base-uncased"
    datasets_path = "/data/home/daiwei/NLU_Model/datasets/evaluate.json"

    main(model_path, datasets_path)
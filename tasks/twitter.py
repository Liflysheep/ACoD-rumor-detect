import json
from typing import List
from llm_client import LLMClient
from tasks.base import Task
from utils import Example, sample_data
import pandas as pd
import numpy as np

SAMPLE_SIZE = 100  # 每个标签的样本数量  

class TwitterRumor(Task):
    def __init__(self, llm: LLMClient):
        super().__init__("twitter", llm)

    def load_data(self) -> List[Example]:
        """加载推特谣言检测数据（简化版，假设数据格式正确）"""
        data_rows = [] 
        return_data = []
        with (
            open("./data/twitter_lines.txt", "r", encoding="utf-8") as text_file,
            open("./data/twitter_label 0 for fake, 1 for true.txt", "r", encoding="utf-8") as label_file,
            open("./data/twitter_token.txt", "r", encoding="utf-8") as token_file
        ):
            for text_line, label_line, token_line in zip(text_file, label_file, token_file):
                text = text_line.strip()
                label = int(label_line.strip())
                token_count = int(token_line.strip())  # 直接读取token数量
                
                row = {
                    'text': text,
                    'label': label,
                    'token_count': token_count  # 直接存储token数量
                }
                data_rows.append(row)
        df = pd.DataFrame(data_rows)
        df = df[(df['token_count'] <= 100) & (df['token_count'] >= 4)]  # 正确写法
        for i in range(len(df)):
            return_data.append(Example(question=str(df.iloc[i]['text']), answer=str(df.iloc[i]['label'])))
        return return_data
        # return sample_data(df, SAMPLE_SIZE)

    def extract_answer(self, raw_response: str) -> int:
            """提取并标准化模型输出结果（简化版）"""
            raw_response = raw_response.strip().lower()
            
            try:
                score = float(raw_response)
                return 1 if float(score) > 0.5 else 0
            except ValueError:
                pass
            # 尝试从"####"分隔符后提取答案
            if "####" in raw_response:
                return self.extract_answer(raw_response.split("####")[-1])
                
            return 0  # 默认返回0（假新闻）

    def equal(self, predicted_answer: int, expected_answer: int) -> bool:
        """比较预测答案和预期答案"""
        return predicted_answer == expected_answer
    

    
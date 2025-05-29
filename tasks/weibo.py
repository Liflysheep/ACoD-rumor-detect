import json
from typing import List
from llm_client import LLMClient
from tasks.base import Task
from utils import Example,clean_weibo_text,sample_data
import pandas as pd
import numpy as np
import os

SAMPLE_SIZE = 100  # 每个标签的样本数量  

class WeiboRumor(Task):
    def __init__(self, llm: LLMClient):
        super().__init__("weibo", llm)

    def load_data(self) -> List[Example]:
        """加载推特谣言检测数据（简化版，假设数据格式正确）"""
        data_rows = []  # 用于存储所有数据的列表
        
        # 定义要处理的文件列表及其对应标签
        tweet_files = [
            ('test_nonrumor.txt', 1),
            ('test_rumor.txt', 0),
            ('train_nonrumor.txt', 1),
            ('train_rumor.txt', 0)
        ]
    
        for filename, label in tweet_files:
            text_filepath = os.path.join('./data/tweets/', filename)
            token_filepath = os.path.join('./data/tweets/', f'token_count_{filename}')

            with open(text_filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                texts = [clean_weibo_text(lines[i].strip()) for i in range(2, len(lines), 3)]
                
            # 读取对应的token计数文件
            with open(token_filepath, 'r', encoding='utf-8') as f:
                tokens = [int(line.strip()) for line in f.readlines()]
                
            # 确保文本和token数量匹配
            min_len = min(len(texts), len(tokens))
            for i in range(min_len):
                data_rows.append({
                    'text': texts[i],    
                    'label': label,
                    'token_count': tokens[i],  
                })
        df = pd.DataFrame(data_rows)
        df = df[df['token_count'] >= 30]

        return sample_data(df, SAMPLE_SIZE)


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
    

    
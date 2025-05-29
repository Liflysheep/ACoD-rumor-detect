import re
from typing import List, Literal, Union
import transformers
from typing import List
import yaml
from pydantic import BaseModel
import numpy as np
import pandas as pd


class Example(BaseModel):
    question: str
    answer: str


class Config(BaseModel):
    system_prompt: str
    format: str
    fewshot: List[Example]


_tokenizer = None

def get_tokenizer():
    """
    获取或初始化分词器。
    分词器只加载一次，以提高效率。
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = transformers.AutoTokenizer.from_pretrained(
            "./utils/",
            trust_remote_code=True,
            use_fast=True  # 强制使用快速分词器
        )
    return _tokenizer

def token_count(texts: List[str]) -> List[int]:
    """
    计算输入文本列表中每个字符串的token数量。

    Args:
        texts (List[str]): 待处理的字符串列表，每个字符串可以是一个句子或一个段落。

    Returns:
        List[int]: 每个字符串的token数量列表。
    """
    tokenizer = get_tokenizer()

    all_token_counts = []
    for text_item in texts:
        # 对列表中的每个字符串进行分词
        tokens = tokenizer.encode(text_item, add_special_tokens=False)
        all_token_counts.append(len(tokens))

    return all_token_counts

def clean_weibo_text(text: str) -> str:
    """
    微博文本专用清洗：
    1. 保留中文、英文、数字
    2. 移除URL、@提及、话题标签
    3. 移除所有标点符号（包括中文标点）
    4. 合并连续空白
    """
    if not isinstance(text, str):
        return ""
    
    # 移除URL和@提及
    text = re.sub(r'http\S+|@\S+|#\S+', '', text)
    
    # 只保留中文、英文、数字和空格
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    
    # 合并连续空白并去除首尾空白
    text = ' '.join(text.split()).strip()
    
    return text

def sample_data(data, num_samples: int) -> List[Example]:
        return_data = []
        df_0 = data[data['label'] == 0].copy()
        df_1 = data[data['label'] == 1].copy()      
        df_0 = df_0.sort_values(by='token_count')
        df_1 = df_1.sort_values(by='token_count')

        # 生成均匀分布的索引
        indices = np.linspace(0, len(df_0)-1, num=num_samples, dtype=int)
        sample_0 = df_0.iloc[indices]
        indices = np.linspace(0, len(df_1)-1, num=num_samples, dtype=int)
        sample_1 = df_1.iloc[indices]

        concat = pd.concat([sample_0, sample_1], axis=0).reset_index(drop=True)

        for i in range(len(concat)):
            return_data.append(Example(question=str(concat.iloc[i]['text']), answer=str(concat.iloc[i]['label'])))
            
        return return_data

def load_config(task: Literal["twitter", "weibo"], config: Literal["baseline", "cot", "cod","cod_ablation","basebaseline"]) -> Config:
    with open(f"./configs/{task}_{config}.yaml") as f:
        return Config.model_validate(yaml.safe_load(f))


def compose_request(config: Config, shot: int, question: str) -> str:
    request = config.system_prompt + "\n"
    if shot is None:
        shot = len(config.fewshot)
    if shot != 0:
        fewshot = [config.format.format(question=ex.question, answer=ex.answer) for ex in config.fewshot[:shot]]
        request += "\n".join(fewshot) + "\n"
    request += config.format.format(question=question, answer="")
    return request


def nth_percentile(values: list[float], percentile: float) -> float:
    values = sorted(values)
    index = min(round(percentile * len(values)), len(values)) - 1
    return values[index]


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def trimmed_average(values: list[float], percentile: float) -> float:
    values = sorted(values)
    count = round(len(values) * percentile)
    trimmed = values[count : len(values) - count]
    return average(trimmed)


def extract_number_from_string(s: str) -> Union[int, float]:
    match = re.search(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?", s)
    if match:
        number_str = match.group().replace(",", "")  # Remove commas
        return float(number_str) if "." in number_str else int(number_str)
    return None

import time
from abc import ABC, abstractmethod
from typing import List, Literal
from tqdm import tqdm
from llm_client import LLMClient
from utils import Config, Example, compose_request, load_config
import random
import json
import os



RANDOM_SEED = 42
random.seed(RANDOM_SEED)


class Task(ABC):
    def __init__(self, name: str, llm: LLMClient):
        self.name = name
        self.llm = llm
        self.token_count_tracker = []
        self.latency_tracker = []
        self.question_log = []
        self.answer_log = []
        self.prediction_log = []
        self.gt_log = []
        self.if_log = False
        self.num_log = []
        self.load_path = None  # 用于加载状态的路径

    @abstractmethod
    def load_data(self) -> List[Example]:
        """加载数据"""
        # 示例实现
        return [Example(f"Q{i}", f"A{i}") for i in range(100)]


    @abstractmethod
    def extract_answer(self, raw_response: str) -> any:
        """从原始响应中提取答案"""
        # 示例实现
        return raw_response.split(" ")[-1]

    @abstractmethod
    def equal(self, predicted_answer: any, expected_answer: any) -> bool:
        """比较预测答案与期望答案是否相等"""
        # 示例实现
        return predicted_answer == expected_answer

    def evaluate_example(self, model: str, config: Config, shot: int, example: Example) -> bool:
        """评估单个样本 (保持原样，它会更新内部列表)"""
        payload = compose_request(config, shot, example.question)
        start_time = time.time()
        response, token_count = self.llm.request(payload, model)
        end_time = time.time()
        
        # 记录日志 (重要：必须先更新内部列表)
        self.token_count_tracker.append(token_count)
        self.latency_tracker.append(end_time - start_time)
        self.question_log.append(example.question)
        self.answer_log.append(response.strip().replace("\n", " "))
        predicted_answer = self.extract_answer(response)
        expected_answer = self.extract_answer(example.answer)
        self.prediction_log.append(predicted_answer)
        self.gt_log.append(expected_answer)
        
        is_correct = self.equal(predicted_answer, expected_answer)
        return is_correct

    def _append_log(self, entry: dict):
        """将单个结果追加到 JSON Lines 日志文件。"""
        try:
            # 使用 'a' 模式追加, 并确保每次写入后都换行
            with open(self.load_path, 'a', encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except IOError as e:
            print(f"警告：无法追加到日志文件 {self.load_path}: {e}")

    def _load_state_from_jsonl(self):
        """从 JSON Lines 文件加载状态并重建日志。"""
        correct = 0
        last_index = 0
        # 清空当前日志，准备从文件重建
        self.token_count_tracker.clear()
        self.latency_tracker.clear()
        self.question_log.clear()
        self.answer_log.clear()
        self.prediction_log.clear()
        self.gt_log.clear()
        self.num_log.clear()

        try:
            with open(self.load_path, 'r', encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line: continue  # 跳过空行

                    try:
                        entry = json.loads(line)
                        # 重建日志列表
                        self.token_count_tracker.append(entry['token_count'])
                        self.latency_tracker.append(entry['latency'])
                        self.question_log.append(entry['question'])
                        self.answer_log.append(entry['answer'])
                        self.prediction_log.append(entry['prediction'])
                        self.gt_log.append(entry['gt'])
                        self.num_log.append(entry['num'])
                        
                        if entry['is_correct']:
                            correct += 1
                        last_index = entry['num'] # 使用记录的编号作为索引

                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"警告：日志文件 {self.load_path} 第 {i+1} 行格式错误或不完整 ({e})。将从该行之前的位置恢复。")
                        # 返回上一个有效状态
                        return correct, last_index
            return correct, last_index
        except FileNotFoundError:
            print("日志文件未找到，将从头开始。")
            return 0, 0
        except Exception as e:
            print(f"加载日志时发生未知错误: {e}。将从头开始。")
            return 0, 0


    def evaluate(self, model: str, config: Literal["baseline", "cot", "cod", "cod_ablation", "basebaseline"], shot: int = None, test_set_size: int = -1, if_log: bool = False, load_path: str = None) -> float:
        self.if_log = if_log
        self.load_path = load_path 
        correct = 0
        start_index = 0

        config_obj = load_config(self.name, config)
        test_set = self.load_data()

        if test_set_size != -1: # -1 表示使用全集
            random.seed(RANDOM_SEED) 
            test_set = random.sample(test_set, test_set_size)

        total_examples = len(test_set)
        if total_examples == 0:
            return 0.0

        # 如果启用日志且文件存在，则加载中间结果
        if self.if_log and os.path.exists(self.load_path):
            print(f"检测到中间结果文件 {self.load_path}，正在加载...")
            correct, start_index = self._load_state_from_jsonl()
            if start_index > 0:
                 print(f"已成功加载 {start_index} 个结果，将从该位置继续评估。")
            
            if start_index >= total_examples:
                print("评估已完成，正在清理并返回结果。")
                if os.path.exists(self.load_path):
                    os.remove(self.load_path)
                return correct / total_examples

        # 循环处理测试集，从 start_index 开始
        for i in tqdm(range(start_index, total_examples), initial=start_index, total=total_examples, desc="评估进度"):
            example = test_set[i]
            current_index = i + 1  # 当前处理完的样本编号 (1-based)
            
            # 执行评估 (这会更新内部列表)
            is_correct = self.evaluate_example(model, config_obj, shot, example)
            
            # evaluate_example 内部已经更新了 correct，但我们这里需要明确
            if is_correct:
                correct += 1 # 确保 correct 与加载/循环一致

            # 只有在启用日志时才追加到文件
            if self.if_log:
                # 从内部列表中获取刚刚添加的条目
                log_entry = {
                    'num': current_index,
                    'token_count': self.token_count_tracker[-1],
                    'latency': self.latency_tracker[-1],
                    'question': self.question_log[-1],
                    'answer': self.answer_log[-1],
                    'prediction': self.prediction_log[-1],
                    'gt': self.gt_log[-1],
                    'is_correct': is_correct,
                }
                self._append_log(log_entry)

        # 评估完成后，如果启用了日志，则删除中间文件
        if self.if_log and os.path.exists(self.load_path):
            try:
                print(f"评估完成，正在删除中间结果文件 {self.load_path}...")
                os.remove(self.load_path)
            except OSError as e:
                print(f"警告：无法删除中间结果文件 {self.load_path}: {e}")

        return correct / total_examples
import time
from abc import ABC, abstractmethod
from typing import List, Literal
from tqdm import tqdm
from llm_client import LLMClient
from utils import Config, Example, compose_request, load_config
import random
import json

RANDOM_SEED = 42
random.seed(RANDOM_SEED)  # Set the seed first


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

    @abstractmethod
    def load_data(self) -> List[Example]:
        pass

    @abstractmethod
    def extract_answer(self, raw_response: str) -> any:
        pass

    @abstractmethod
    def equal(self, predicted_answer: any, expected_answer: any) -> bool:
        pass

    def evaluate_example(self, model: str, config: Config, shot: int, example: Example) -> bool:
        # prepare payload
        payload = compose_request(config, shot, example.question)

        # run inference
        start_time = time.time()
        response, token_count = self.llm.request(payload, model)
        end_time = time.time()
        self.token_count_tracker.append(token_count)
        self.latency_tracker.append(end_time - start_time)

        # log
        self.question_log.append(example.question)      
        self.answer_log.append(response.strip().replace("\n", " "))

        # check result
        predicted_answer = self.extract_answer(response)
        expected_answer = self.extract_answer(example.answer)
        self.prediction_log.append(predicted_answer)
        self.gt_log.append(expected_answer)
        equal = self.equal(predicted_answer, expected_answer)
        # if not equal:
        #     print(f"Example: {example.question}")
        #     print(f"Expected: {expected_answer}, Predicted: {predicted_answer}")
        #     print(f"Full response: {response}")
        return equal

    def evaluate(self, model: str, config: Literal["baseline", "cot", "cod","cod_ablation","basebaseline"], shot: int = None, test_set_size: int = None, if_log: bool = False) -> float:
        self.if_log = if_log
        last_index = 0
        if self.if_log:
            with open(r'.\log.json','r',encoding="utf-8") as f:
                data = json.load(f)            
        correct = 0
        config = load_config(self.name, config)
        test_set = self.load_data()
        if test_set_size is not None:
            test_set = random.sample(test_set, test_set_size)
        for example in tqdm(test_set):
            last_index += 1
            self.num_log.append(last_index)
            if self.evaluate_example(model, config, shot, example):
                correct += 1
        return correct / len(test_set)

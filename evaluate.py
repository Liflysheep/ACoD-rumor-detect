import argparse
import csv
import os
import json

from llm_client import LLMClient
from tasks.twitter import TwitterRumor
from tasks.weibo import WeiboRumor
from utils import token_count

MODEL_MAPPING = {
    "gpt-4o": "gpt-4o-2024-11-20",
    "gpt-4o-mini": "gpt-4o-mini",
    "sonnet": "claude-3-5-sonnet-20240620",
    "haiku": "claude-3-5-haiku-20241022",
    "deepseek-v3": "deepseek-chat",
    "qwen2-14b" : "qwen2.5-14b-instruct",
    "qwen3-32b" : "Qwen/Qwen3-32B",
    "qwen2.5-32b" : "qwen2.5-32b-instruct",
    "qwen3-turbo" :"qwen-turbo-2025-04-28",
    "qwen2.5-3b" : "qwen2.5-3b-instruct"
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["twitter",'weibo'])
    parser.add_argument("--model", default="deepseekv3")
    parser.add_argument(
        "--prompt",
        choices=["baseline", "cod", "cot","cod_ablation","basebaseline"],
        default="cod",
        help="Prompting strategy",
    )
    parser.add_argument(
        "--shot",
        type=int,
        default=None,
        help="Number of fewshot to be included, by default, include all fewshot examples",
    )
    parser.add_argument(
        "--url",
        default='https://api.deepseek.com/v1',
        help="Base url for llm model endpoint",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for model access, will use api keys in environment variables for openai and claude models.",
    )
    parser.add_argument(
        "--test-set-size",
        type=int,
        default=-1,
        help="Number of examples to be test, by default, if not specified, use all examples",
    )
    parser.add_argument(
        "--if-log",
        default=False,
        help="Whether to log middle results, default is False",
    )
    parser.add_argument(
        "--load-path",
        default=None,
        help="Path to load previous results, if specified, will load from this path",
    )

    args = parser.parse_args()
    llm_client = LLMClient(args.url, args.api_key)
    match args.task:
        case "weibo":
            task = WeiboRumor(llm_client)
        case "twitter":
            task = TwitterRumor(llm_client)
        case _:
            raise ValueError("Invalid task")

    model = MODEL_MAPPING.get(args.model, args.model)
    accuracy = task.evaluate(model, args.prompt, args.shot, args.test_set_size, args.if_log, args.load_path)
    results = [
        [
            "qs_token",
            "out_token",
            "predict",
            "gt",
            "time",
            "is_correct",
            "question",
            "answer"
        ]
    ]
    token_log = token_count(task.question_log)
    # Add all the logged data to results
    for i in range(len(task.question_log)):
        results.append([
            token_log[i],  # qs_token (approximate word count)
            task.token_count_tracker[i],    # out_token (actual token count from LLM)
            task.prediction_log[i],         # predict
            task.gt_log[i],                 # gt
            task.latency_tracker[i],       # time
            task.equal(task.prediction_log[i], task.gt_log[i]),
            task.question_log[i],          # question
            task.answer_log[i],             # answer                # is_correct
        ])

    if not os.path.exists("./results"):
        os.makedirs("./results")

    model_name = args.model.split(":")[1] if ":" in args.model else args.model
    model_name = model_name.replace("/", "_")
    fname = (
        f"{args.task}-{model_name}-{args.prompt}-{args.shot}"
        if args.shot is not None
        else f"{args.task}-{model_name}-{args.prompt}"
    )
    with open(f"./results/{fname}.csv", "w", newline="") as f:
        writer = csv.writer(f)  # 用制表符分隔，方便对齐
        writer.writerows(results)

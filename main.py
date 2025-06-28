import os
import subprocess
from pathlib import Path

# 环境配置
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_DATASETS_CACHE"] = "./data_cache"

# 实验参数配置
TASK = ["twitter"]  # 任务名称
MODEL = ["deepseek-v3"]  # 模型名称
PROMPTS =["cot"]  # 改为列表，测试多种提示策略
# PROMPTS = ["cot"]  # 改为列表，测试多种提示策略
# URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# API_KEY = "	sk-41c9010ff48d4c0b941f152cadc5c41a"
# URL = "https://api.siliconflow.cn/v1"
# API_KEY = "sk-qefgglbcwnvvixmplmgyxshbeysskuynnzwavwcbbodmdhiq"
URL = "https://api.deepseek.com/v1"
# API_KEY = "sk-8e4594ee04cd4473b99b6880190120f6" # 保密性低的key
API_KEY = "sk-3f2561510dc44e3086ab0e980aa746fa" # Cot-rumor-detect
TEST_SET_SIZE = -1  # 测试集大小
SHOTS = [2]  # 测试的shot数量范围
LOAD_PATH = "intermediate_data_deepseek.jsonl"  # 日志文件路径

def run_experiments():
    """运行所有prompt和shot组合的实验"""
    print("=" * 40)
    print(f"开始实验任务: {TASK}")
    print(f"使用模型: {MODEL} | 测试提示策略: {PROMPTS}")
    print("=" * 40 + "\n")

    # 创建结果目录
    Path("./results").mkdir(exist_ok=True)

    for task in TASK:
        for model in MODEL:
            for prompt in PROMPTS:
                for shot in SHOTS:
                    print(f"▶ 正在运行 prompt={prompt}, shot={shot} 的实验...")
                    
                    # 构建命令行参数
                    cmd = [
                        "python", "evaluate.py",
                        "--task", task,
                        "--model", model,
                        "--prompt", prompt,
                        "--shot", str(shot),
                        "--test-set-size", str(TEST_SET_SIZE),
                        "--url", URL,
                        "--api-key", API_KEY,
                        "--if-log", "True",
                        "--load-path", LOAD_PATH
                    ]
                    
                    try:
                        # 运行实验（继承当前环境变量）
                        subprocess.run(cmd, check=True)
                        print(f"✓ [prompt={prompt}, shot={shot}] 实验完成\n")
                    except subprocess.CalledProcessError as e:
                        print(f"✕ [prompt={prompt}, shot={shot}] 实验失败: {str(e)}\n")


if __name__ == "__main__":
    # 运行所有实验组合
    run_experiments()
    print("\n实验流程全部完成！")
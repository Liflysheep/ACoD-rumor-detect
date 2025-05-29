#!/bin/bash

# 1. 激活 Conda 环境
source r"C:\miniconda3\etc\profile.d\conda.sh"  # 加载 Conda 初始化脚本（路径可能需要调整）
conda activate mc  # 激活目标环境

# 2. 运行 Python 脚本
python "D:\Study\大三\机器学习初步\2025MachineLearning\大作业\chain-of-draft\evaluate.py" \
    --task gsm8k \
    --model gpt-4o \
    --prompt cod \
    --shot 2 \
    --url https://api2.aigcbest.top/v1 \
    --api-key sk-17UeXf8m7e61JOu9fIf8OD955YNtT1KW2wVl0E5zAgBW9W7v \
    --test-set-size 10
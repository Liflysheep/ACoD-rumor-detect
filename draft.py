import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
import os
from itertools import product

plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

PROMPTS = ["cod","cot","baseline","basebaseline"]  # 提示策略
DATASETS = ["weibo","twitter"]  # 数据集
MODEL = "deepseek-v3"  # 模型名称
SHOT = 2  # few-shot数量
PERCENTILES = [0, 20, 40, 60, 80, 100]  # 分段百分位数
OUTPUT_DIR = "./analysis/"

def analyze_dataset(dataset):
    plt.figure(figsize=(12, 8))
    
    # 精心设计的视觉方案
    color_palette = {
        "cod": "#1f77b4",      # 蓝色系 (组1)
        "cot": "#aec7e8",       # 浅蓝色 (组1)
        "baseline": "#ff7f0e",  # 橙色系 (组2)
        "basebaseline": "#ffbb78"  # 浅橙色 (组2)
    }
    
    marker_style = {
        "cod": "o",      # 圆形
        "cot": "s",       # 方形
        "baseline": "^",  # 三角形
        "basebaseline": "D"  # 菱形
    }
    
    line_style = {
        "cod": "-",       # 实线
        "cot": "--",       # 虚线
        "baseline": "-",   # 实线
        "basebaseline": "--"  # 虚线
    }
    
    all_data = []  # 用于存储所有策略的数据
    
    for prompt in PROMPTS:
        try:
            # 1. 数据加载和预处理
            file_path = f"./results/{dataset}-{MODEL}-{prompt}-{SHOT}.csv"
            df = pd.read_csv(file_path, delimiter='\t')
            
            # 2. 按 qs_token 排序
            df = df.sort_values("qs_token").reset_index(drop=True)

            # 3. 动态分段策略（基于百分位数）
            bins = np.percentile(df["qs_token"], PERCENTILES)
            bins = np.unique(bins)  # 确保边界唯一
            labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]

            # 4. 计算每个区间的准确率
            df["token_bin"] = pd.cut(df["qs_token"], bins=bins, labels=labels, include_lowest=True)
            accuracy_df = df.groupby("token_bin", observed=False)["is_correct"].mean().reset_index()
            accuracy_df.columns = ["token_range", "accuracy"]
            accuracy_df["prompt"] = prompt  # 添加提示策略列
            
            all_data.append(accuracy_df)
            
            # 5. 绘制折线图（使用精心设计的视觉参数）
            plt.plot(
                accuracy_df["token_range"],
                accuracy_df["accuracy"],
                marker=marker_style[prompt],
                color=color_palette[prompt],
                linestyle=line_style[prompt],
                linewidth=2.5,
                markersize=9,
                markeredgecolor='black',  # 标记边框
                markeredgewidth=0.8,
                label=f"{prompt} (Acc: {df['is_correct'].mean():.1%})"
            )
            
        except FileNotFoundError:
            print(f"文件不存在: {file_path}")
            continue
        except Exception as e:
            print(f"处理 {dataset}-{prompt} 时出错: {str(e)}")
            continue
    
    if not all_data:
        print(f"没有找到 {dataset} 的有效数据")
        return False
    
    # 图表装饰
    title = f"Model Accuracy by Question Token Length\nDataset: {dataset}, Model: {MODEL}"
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("Token Length Range", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))  # 转换为百分比格式
    
    # 添加网格线和图例
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", bbox_to_anchor=(1.18, 1), framealpha=1)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"{dataset}_{MODEL}_all_prompts_accuracy_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 合并所有数据并打印
    combined_df = pd.concat(all_data)
    print(f"\n=== {dataset} - 所有提示策略统计摘要 ===")
    print(f"模型: {MODEL}")
    print("\n按 token 长度和提示策略分段的准确率:")
    print(combined_df.pivot(index="token_range", columns="prompt", values="accuracy").to_string())
    
    return True

if __name__ == "__main__":
    # 每个数据集单独处理，生成一张包含所有提示策略的图
    for dataset in DATASETS:
        print(f"\n正在处理: {dataset} 数据集")
        analyze_dataset(dataset)
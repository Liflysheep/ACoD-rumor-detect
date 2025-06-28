# Chain of Draft: Thinking Faster by Writing Less
[[Paper]]()

The rapid dissemination of misinformation on social media platforms has underscored the urgent need for effective and interpretable fake news detection methods. While small language models (SLMs) like BERT have been extensively applied in this domain, they often struggle with reasoning and lack transparency in decision-making. Recently, large language models (LLMs) have demonstrated impressive reasoning capabilities when guided by prompting strategies such as Chain-of-Thought (CoT). However, CoT often suffers from inefficiency and excessive token usage, which hampers its practicality in real-world applications. 
In this paper, we explore an emerging prompting strategy—Chain-of-Draft (CoD)—and introduce an Aspect-aware Chain-of-Draft (ACoD) Prompt specifically designed for fake news detection. ACoD encourages LLMs to perform concise, aspect-specific reasoning based on three critical dimensions: source credibility, multiple confirmation, and evidence support. We evaluate ACoD against Standard Prompting, CoT, and aspect-aware variants using two widely-used datasets, Weibo21 (Chinese) and Twitter16 (English), and two state-of-the-art LLMs, Deepseek-v3 and Qwen2.5-32b. Experimental results reveal that ACoD not only improves detection accuracy but also significantly reduces token consumption and inference latency. Notably, we observe that CoT prompting can degrade performance in this domain, likely due to the noisy nature of long reasoning chains. In contrast, ACoD produces compact drafts that enhance interpretability while preserving efficiency. Our findings suggest that CoD-style prompting holds great promise for balancing accuracy, interpretability, and efficiency in LLM-based fake news detection.


## Usage
To run evaluation:
```bash
python evaluate.py \
    --task twitter \      # Task to evaluate (options: twitter,weibo)
    --model deepseek-v3 \    # Model to be evaluated
    --prompt cod \      # Prompting strategy (options: baseline, cod, cot)
    --shot 2 \          # [Optional] Number of few-shot examples to include in the prompt (uses all available examples by default if omitted)
    --url $BASE_URL \   # [Optional] Base URL for an OpenAI-compatible interface (e.g., locally hosted models)
    --api-key $KEY \    # [Optional] API key for model access (automatically loads from environment variables for Claude and OpenAI models if not provided)
```
Currently, the script supports Claude models, OpenAI models, as well as any model that uses an OpenAI-compatible interface.
The evaluation results will be stored under `./results/`.

All prompts and fewshot examples are stored under `./configs/{task}-{prompt}.yaml`. 

## Citation

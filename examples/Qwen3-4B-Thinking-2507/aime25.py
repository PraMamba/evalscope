
import os
for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(k, None)

from evalscope import TaskConfig, run_task  
  
task_cfg = TaskConfig(  
    model='Qwen/Qwen3-4B-Thinking-2507',
    api_url='http://0.0.0.0:30005/v1/chat/completions',
    eval_type="openai_api",
    
    # 🎯 指定保存路径
    work_dir='/home/scbjtfy/evalscope/examples/Qwen3-4B-Thinking-2507/aime25',  # 自定义目录名
    no_timestamp=False,  # 保留时间戳（推荐，避免覆盖）

    # Pass@k 配置
    repeats=16,  # 生成 64 次用于计算 pass@64

    datasets=["aime25"],
    dataset_args={
        "aime25": {
            "aggregation": "mean_and_pass_at_k",  # 计算 pass@k
            "few_shot_num": 0,
            "filters": {"remove_until": "</think>"}  # 🔧 新增：过滤思考内容
        }
    },

    eval_batch_size=128,  # 根据你的 API 并发能力调整

    # 🔧 优化：使用官方推荐参数
    generation_config={
        "max_tokens": 81920,         # 改为 81920（高难度任务）
        "temperature": 0.6,          # 改为 0.6
        "top_p": 0.95,              # 改为 0.95
        "top_k": 20,
        "presence_penalty": 1.5     # 保持 1.5（在合理范围内）
    },

    timeout=120000,  # 🔧 增加到 120 秒
    stream=True,
    limit=None
)  
  
res = run_task(task_cfg=task_cfg)

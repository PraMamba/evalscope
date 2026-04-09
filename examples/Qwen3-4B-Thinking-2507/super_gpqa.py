
import os
for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(k, None)

from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen/Qwen3-4B-Thinking-2507',
    api_url='http://0.0.0.0:30005/v1/chat/completions',
    eval_type="openai_api",

    # 📊 SuperGPQA 配置（大规模研究生级别问题，26K+ 题目）
    datasets=["super_gpqa"],
    dataset_args={
        "super_gpqa": {
            "few_shot_num": 0,
            "filters": {"remove_until": "</think>"}
            # 可选：指定子集 "subset_list": ["Mathematics", "Physics", "Computer Science and Technology"]
        }
    },

    eval_batch_size=128,

    generation_config={
        "max_tokens": 8192,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5
    },

    timeout=60000,
    stream=True,

    work_dir='/home/scbjtfy/evalscope/examples/Qwen3-4B-Thinking-2507/super_gpqa/',
    no_timestamp=False,
)

res = run_task(task_cfg=task_cfg)

print(f"\n✅ SuperGPQA 评测完成！")
print(f"📁 结果保存在: {task_cfg.work_dir}")

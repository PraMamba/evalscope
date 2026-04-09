
import os
for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(k, None)

from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen/Qwen3-4B-Thinking-2507',
    api_url='http://0.0.0.0:30005/v1/chat/completions',
    eval_type="openai_api",

    # 📊 GPQA-Diamond 配置（研究生级别科学问题）
    datasets=["gpqa_diamond"],
    dataset_args={
        "gpqa_diamond": {
            "few_shot_num": 0,  # 0-shot（默认）或 5-shot
            "filters": {"remove_until": "</think>"}
        }
    },

    eval_batch_size=128,

    # GPQA 问题较难，需要较长推理
    generation_config={
        "max_tokens": 8192,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5
    },

    timeout=60000,
    stream=True,

    work_dir='/home/scbjtfy/evalscope/examples/Qwen3-4B-Thinking-2507/gpqa_diamond/',
    no_timestamp=False,
)

res = run_task(task_cfg=task_cfg)

print(f"\n✅ GPQA-Diamond 评测完成！")
print(f"📁 结果保存在: {task_cfg.work_dir}")

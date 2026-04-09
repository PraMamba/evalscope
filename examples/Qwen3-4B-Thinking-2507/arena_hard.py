
import os
for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(k, None)

from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen/Qwen3-4B-Thinking-2507',
    api_url='http://0.0.0.0:30005/v1/chat/completions',
    eval_type="openai_api",

    # 📊 Arena-Hard 配置（竞技场对战评测）
    # 注意：需要配置 judge 模型（默认 gpt-4-1106-preview）
    datasets=["arena_hard"],
    dataset_args={
        "arena_hard": {
            "few_shot_num": 0,
            "filters": {"remove_until": "</think>"}
        }
    },

    eval_batch_size=64,  # Arena 评测需要 judge 模型，适当降低并发

    generation_config={
        "max_tokens": 8192,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5
    },

    # Judge 模型配置（如使用其他模型作为评判）
    # judge_model_config={
    #     "api_url": "https://api.openai.com/v1/chat/completions",
    #     "api_key": "your-api-key",
    #     "model": "gpt-4-1106-preview"
    # },

    timeout=60000,
    stream=True,

    work_dir='/home/scbjtfy/evalscope/examples/Qwen3-4B-Thinking-2507/arena_hard/',
    no_timestamp=False,
)

res = run_task(task_cfg=task_cfg)

print(f"\n✅ Arena-Hard 评测完成！")
print(f"📁 结果保存在: {task_cfg.work_dir}")

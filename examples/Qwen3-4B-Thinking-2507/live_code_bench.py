
import os
for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(k, None)

from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen/Qwen3-4B-Thinking-2507',
    api_url='http://0.0.0.0:30005/v1/chat/completions',
    eval_type="openai_api",

    # 📊 LiveCodeBench 配置（代码生成评测）
    # 使用 Pass@k 评测，需要多次采样
    datasets=["live_code_bench"],
    dataset_args={
        "live_code_bench": {
            "aggregation": "mean_and_pass_at_k",
            "filters": {"remove_until": "</think>"},
            # 可选：日期过滤
            # "start_date": "2025-02-01",
            # "end_date": "2025-05-31"
        }
    },

    # Pass@k 配置（代码评测通常用 pass@1 或 pass@10）
    repeats=10,

    eval_batch_size=64,  # 代码执行需要资源，适当降低并发

    generation_config={
        "max_tokens": 32768,  # 代码生成需要足够空间
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5
    },

    timeout=120000,
    stream=True,

    work_dir='/home/scbjtfy/evalscope/examples/Qwen3-4B-Thinking-2507/live_code_bench/',
    no_timestamp=False,
)

res = run_task(task_cfg=task_cfg)

print(f"\n✅ LiveCodeBench 评测完成！")
print(f"📁 结果保存在: {task_cfg.work_dir}")

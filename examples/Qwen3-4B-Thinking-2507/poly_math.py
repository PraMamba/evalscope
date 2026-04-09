
import os
for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(k, None)

from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen/Qwen3-4B-Thinking-2507',
    api_url='http://0.0.0.0:30005/v1/chat/completions',
    eval_type="openai_api",

    # 📊 PolyMath 配置（多语言数学推理，18 种语言，9K 题目）
    # 高难度任务，推荐使用 Pass@k
    datasets=["poly_math"],
    dataset_args={
        "poly_math": {
            "aggregation": "mean_and_pass_at_k",
            "filters": {"remove_until": "</think>"}
            # 可选：指定语言和难度
            # "subset_list": ["en", "zh", "ja", "ko", "de", "fr"]
        }
    },

    # Pass@k 配置
    repeats=16,  # PolyMath 题目较多，可适当减少

    eval_batch_size=128,

    # 数学任务需要长推理链
    generation_config={
        "max_tokens": 32768,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5
    },

    timeout=120000,
    stream=True,

    work_dir='/home/scbjtfy/evalscope/examples/Qwen3-4B-Thinking-2507/poly_math/',
    no_timestamp=False,
)

res = run_task(task_cfg=task_cfg)

print(f"\n✅ PolyMath 评测完成！")
print(f"📁 结果保存在: {task_cfg.work_dir}")

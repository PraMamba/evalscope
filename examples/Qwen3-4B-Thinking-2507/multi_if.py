
import os
for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(k, None)

from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen/Qwen3-4B-Thinking-2507',
    api_url='http://0.0.0.0:30005/v1/chat/completions',
    eval_type="openai_api",

    # 📊 Multi-IF 配置（多语言多轮指令遵循）
    datasets=["multi_if"],
    dataset_args={
        "multi_if": {
            "few_shot_num": 0,
            "max_turns": 3,  # 最大交互轮数（1-3）
            "filters": {"remove_until": "</think>"}
            # 可选：指定语言子集
            # "subset_list": ["Chinese", "English", "French", "German"]
        }
    },

    eval_batch_size=64,  # 多轮对话需要更多资源

    generation_config={
        "max_tokens": 4096,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5
    },

    timeout=90000,  # 多轮对话可能需要更长时间
    stream=True,

    work_dir='/home/scbjtfy/evalscope/examples/Qwen3-4B-Thinking-2507/multi_if/',
    no_timestamp=False,
)

res = run_task(task_cfg=task_cfg)

print(f"\n✅ Multi-IF 评测完成！")
print(f"📁 结果保存在: {task_cfg.work_dir}")

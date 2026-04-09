
import os
for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(k, None)

from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen/Qwen3-4B-Thinking-2507',
    api_url='http://0.0.0.0:30005/v1/chat/completions',
    eval_type="openai_api",

    # 📊 IFEval 配置（指令遵循评测）
    datasets=["ifeval"],
    dataset_args={
        "ifeval": {
            "few_shot_num": 0,
            "filters": {"remove_until": "</think>"}
        }
    },

    eval_batch_size=128,

    generation_config={
        "max_tokens": 4096,  # 指令遵循任务不需要太长
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5
    },

    timeout=60000,
    stream=True,

    work_dir='/home/scbjtfy/evalscope/examples/Qwen3-4B-Thinking-2507/ifeval/',
    no_timestamp=False,
)

res = run_task(task_cfg=task_cfg)

print(f"\n✅ IFEval 评测完成！")
print(f"📁 结果保存在: {task_cfg.work_dir}")

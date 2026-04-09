
import os
for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(k, None)

from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen/Qwen3-4B-Thinking-2507',
    api_url='http://0.0.0.0:30015/v1/chat/completions',
    eval_type="openai_api",

    datasets=["mmlu_redux"],
    dataset_args={
        "mmlu_redux": {
            "few_shot_num": 5,
            "filters": {"remove_until": "</think>"}
        }
    },

    eval_batch_size=256,

    generation_config={
        "max_tokens": 4096,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5
    },

    timeout=60000,
    stream=True,

    # 断点续测：指向中断的输出目录（时间戳目录）
    use_cache='/home/scbjtfy/evalscope/examples/Qwen3-4B-Thinking-2507_Manual_Resize_Block-8_Size-32_Num-256_ExplicitTokens_Continued-Pretraining-ALL_Augmented-V1_SFT-ALL_DFT_RL-GSPO_Judge_Reward_MultiTask-V2_9-CELL_TYPE_HOMOGENEOUS-Plus/global_step_765/mmlu_redux/20260309_151828/',
)

res = run_task(task_cfg=task_cfg)

print(f"\n✅ MMLU-Redux 评测完成！")
print(f"📁 结果保存在: {task_cfg.work_dir}")

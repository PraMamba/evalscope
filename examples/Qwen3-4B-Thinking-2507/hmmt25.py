
import os
for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(k, None)

from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen/Qwen3-4B-Thinking-2507',
    api_url='http://0.0.0.0:30015/v1/chat/completions',
    eval_type="openai_api",

    # 📊 HMMT25 配置（哈佛-MIT 数学竞赛，高难度）
    # 类似 AIME25，需要 Pass@k 评测
    datasets=["hmmt25"],
    dataset_args={
        "hmmt25": {
            "aggregation": "mean_and_pass_at_k",
            "few_shot_num": 0,
            "filters": {"remove_until": "</think>"}
        }
    },

    # Pass@k 配置
    repeats=16,

    eval_batch_size=128,

    # 高难度数学任务，需要长推理链
    generation_config={
        "max_tokens": 81920,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5
    },

    timeout=120000,  # 2 分钟超时
    stream=True,

    work_dir='/home/scbjtfy/evalscope/examples/Qwen3-4B-Thinking-2507_Manual_Resize_Block-8_Size-32_Num-256_ExplicitTokens_Continued-Pretraining-ALL_Augmented-V1_SFT-ALL_DFT_RL-GSPO_Judge_Reward_MultiTask-V2_9-CELL_TYPE_HOMOGENEOUS-Plus/global_step_765/hmmt25/',
    no_timestamp=False,
)

res = run_task(task_cfg=task_cfg)

print(f"\n✅ HMMT25 评测完成！")
print(f"📁 结果保存在: {task_cfg.work_dir}")

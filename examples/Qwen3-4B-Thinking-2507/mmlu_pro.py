
import os
for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(k, None)

from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen/Qwen3-4B-Thinking-2507',
    api_url='http://0.0.0.0:30015/v1/chat/completions',
    eval_type="openai_api",

    # 📊 MMLU-Pro 配置
    datasets=["mmlu_pro"],
    dataset_args={
        "mmlu_pro": {
            "few_shot_num": 5,  # 5-shot learning（默认推荐）
            "filters": {"remove_until": "</think>"}  # 过滤思考内容
            # "subset_list": ["math", "physics", "computer science"]
        }
    },

    # 🚀 并发配置
    eval_batch_size=256,  # MMLU-Pro 问题较短，可以高并发

    # 🎯 生成参数（针对选择题优化）
    generation_config={
        "max_tokens": 4096,          # 选择题不需要太长（降低成本）
        "temperature": 0.6,          # 官方推荐
        "top_p": 0.95,              # 官方推荐
        "top_k": 20,
        "presence_penalty": 1.5
    },

    timeout=60000,  # 60秒足够（选择题推理较快）
    stream=True,

    # 📁 保存路径
    work_dir='/home/scbjtfy/evalscope/examples/Qwen3-4B-Thinking-2507_Manual_Resize_Block-8_Size-32_Num-256_ExplicitTokens_Continued-Pretraining-ALL_Augmented-V1_SFT-ALL_DFT_RL-GSPO_Judge_Reward_MultiTask-V2_9-CELL_TYPE_HOMOGENEOUS-Plus/global_step_765/mmlu_pro/',
    no_timestamp=False,  # 保留时间戳

    # limit=100  # 可选：测试时限制样本数
)

res = run_task(task_cfg=task_cfg)

print(f"\n✅ MMLU-Pro 评测完成！")
print(f"📁 结果保存在: {task_cfg.work_dir}")

#!/usr/bin/env python3
"""
Qwen3-4B-Thinking-2507 全量评测脚本
运行 EvalScope 支持的所有 11 个 Benchmark

使用方法：
    # 运行所有评测
    python run_all.py

    # 运行指定类别
    python run_all.py --category knowledge
    python run_all.py --category reasoning
    python run_all.py --category coding
    python run_all.py --category alignment
    python run_all.py --category multilingual

    # 运行指定数据集
    python run_all.py --datasets mmlu_pro gpqa_diamond
"""

import os
import sys
import argparse
from datetime import datetime

# 清除代理
for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(k, None)

from evalscope import TaskConfig, run_task

# ============ 基础配置 ============
BASE_CONFIG = {
    "model": "Qwen/Qwen3-4B-Thinking-2507",
    "api_url": "http://0.0.0.0:30005/v1/chat/completions",
    "eval_type": "openai_api",
    "stream": True,
}

BASE_GENERATION_CONFIG = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 1.5
}

BASE_DIR = "/home/scbjtfy/evalscope/examples/Qwen3-4B-Thinking-2507"

# ============ 数据集配置 ============
BENCHMARK_CONFIGS = {
    # === Knowledge 类别 ===
    "mmlu_pro": {
        "category": "knowledge",
        "datasets": ["mmlu_pro"],
        "dataset_args": {
            "mmlu_pro": {
                "few_shot_num": 5,
                "filters": {"remove_until": "</think>"}
            }
        },
        "eval_batch_size": 256,
        "generation_config": {**BASE_GENERATION_CONFIG, "max_tokens": 4096},
        "timeout": 60000,
    },
    "mmlu_redux": {
        "category": "knowledge",
        "datasets": ["mmlu_redux"],
        "dataset_args": {
            "mmlu_redux": {
                "few_shot_num": 5,
                "filters": {"remove_until": "</think>"}
            }
        },
        "eval_batch_size": 256,
        "generation_config": {**BASE_GENERATION_CONFIG, "max_tokens": 4096},
        "timeout": 60000,
    },
    "gpqa_diamond": {
        "category": "knowledge",
        "datasets": ["gpqa_diamond"],
        "dataset_args": {
            "gpqa_diamond": {
                "few_shot_num": 0,
                "filters": {"remove_until": "</think>"}
            }
        },
        "eval_batch_size": 128,
        "generation_config": {**BASE_GENERATION_CONFIG, "max_tokens": 8192},
        "timeout": 60000,
    },
    "super_gpqa": {
        "category": "knowledge",
        "datasets": ["super_gpqa"],
        "dataset_args": {
            "super_gpqa": {
                "few_shot_num": 0,
                "filters": {"remove_until": "</think>"}
            }
        },
        "eval_batch_size": 128,
        "generation_config": {**BASE_GENERATION_CONFIG, "max_tokens": 8192},
        "timeout": 60000,
    },

    # === Reasoning 类别 ===
    "aime25": {
        "category": "reasoning",
        "datasets": ["aime25"],
        "dataset_args": {
            "aime25": {
                "aggregation": "mean_and_pass_at_k",
                "few_shot_num": 0,
                "filters": {"remove_until": "</think>"}
            }
        },
        "repeats": 64,
        "eval_batch_size": 128,
        "generation_config": {**BASE_GENERATION_CONFIG, "max_tokens": 81920},
        "timeout": 120000,
    },
    "hmmt25": {
        "category": "reasoning",
        "datasets": ["hmmt25"],
        "dataset_args": {
            "hmmt25": {
                "aggregation": "mean_and_pass_at_k",
                "few_shot_num": 0,
                "filters": {"remove_until": "</think>"}
            }
        },
        "repeats": 64,
        "eval_batch_size": 128,
        "generation_config": {**BASE_GENERATION_CONFIG, "max_tokens": 81920},
        "timeout": 120000,
    },

    # === Coding 类别 ===
    "live_code_bench": {
        "category": "coding",
        "datasets": ["live_code_bench"],
        "dataset_args": {
            "live_code_bench": {
                "aggregation": "mean_and_pass_at_k",
                "filters": {"remove_until": "</think>"}
            }
        },
        "repeats": 10,
        "eval_batch_size": 64,
        "generation_config": {**BASE_GENERATION_CONFIG, "max_tokens": 32768},
        "timeout": 120000,
    },

    # === Alignment 类别 ===
    "ifeval": {
        "category": "alignment",
        "datasets": ["ifeval"],
        "dataset_args": {
            "ifeval": {
                "few_shot_num": 0,
                "filters": {"remove_until": "</think>"}
            }
        },
        "eval_batch_size": 128,
        "generation_config": {**BASE_GENERATION_CONFIG, "max_tokens": 4096},
        "timeout": 60000,
    },
    "arena_hard": {
        "category": "alignment",
        "datasets": ["arena_hard"],
        "dataset_args": {
            "arena_hard": {
                "few_shot_num": 0,
                "filters": {"remove_until": "</think>"}
            }
        },
        "eval_batch_size": 64,
        "generation_config": {**BASE_GENERATION_CONFIG, "max_tokens": 8192},
        "timeout": 60000,
    },

    # === Multilingualism 类别 ===
    "multi_if": {
        "category": "multilingual",
        "datasets": ["multi_if"],
        "dataset_args": {
            "multi_if": {
                "few_shot_num": 0,
                "max_turns": 3,
                "filters": {"remove_until": "</think>"}
            }
        },
        "eval_batch_size": 64,
        "generation_config": {**BASE_GENERATION_CONFIG, "max_tokens": 4096},
        "timeout": 90000,
    },
    "poly_math": {
        "category": "multilingual",
        "datasets": ["poly_math"],
        "dataset_args": {
            "poly_math": {
                "aggregation": "mean_and_pass_at_k",
                "filters": {"remove_until": "</think>"}
            }
        },
        "repeats": 16,
        "eval_batch_size": 128,
        "generation_config": {**BASE_GENERATION_CONFIG, "max_tokens": 32768},
        "timeout": 120000,
    },
}


def run_benchmark(name: str, config: dict):
    """运行单个 Benchmark 评测"""
    print(f"\n{'='*60}")
    print(f"🚀 开始评测: {name}")
    print(f"{'='*60}")

    task_config = TaskConfig(
        **BASE_CONFIG,
        datasets=config["datasets"],
        dataset_args=config["dataset_args"],
        eval_batch_size=config["eval_batch_size"],
        generation_config=config["generation_config"],
        timeout=config["timeout"],
        repeats=config.get("repeats"),
        work_dir=f"{BASE_DIR}/{name}/",
        no_timestamp=False,
    )

    try:
        result = run_task(task_cfg=task_config)
        print(f"✅ {name} 评测完成！")
        return True, result
    except Exception as e:
        print(f"❌ {name} 评测失败: {e}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-4B-Thinking-2507 Benchmark 评测")
    parser.add_argument("--category", type=str, choices=["knowledge", "reasoning", "coding", "alignment", "multilingual"],
                        help="指定评测类别")
    parser.add_argument("--datasets", nargs="+", type=str, help="指定评测数据集")
    parser.add_argument("--list", action="store_true", help="列出所有可用的数据集")
    args = parser.parse_args()

    # 列出所有数据集
    if args.list:
        print("\n📊 可用的 Benchmark 数据集:\n")
        for cat in ["knowledge", "reasoning", "coding", "alignment", "multilingual"]:
            print(f"【{cat.upper()}】")
            for name, cfg in BENCHMARK_CONFIGS.items():
                if cfg["category"] == cat:
                    print(f"  - {name}")
            print()
        return

    # 确定要运行的数据集
    benchmarks_to_run = []

    if args.datasets:
        # 运行指定的数据集
        for ds in args.datasets:
            if ds in BENCHMARK_CONFIGS:
                benchmarks_to_run.append(ds)
            else:
                print(f"⚠️ 未知数据集: {ds}")
    elif args.category:
        # 运行指定类别的所有数据集
        for name, cfg in BENCHMARK_CONFIGS.items():
            if cfg["category"] == args.category:
                benchmarks_to_run.append(name)
    else:
        # 运行所有数据集
        benchmarks_to_run = list(BENCHMARK_CONFIGS.keys())

    if not benchmarks_to_run:
        print("❌ 没有可运行的数据集")
        return

    # 开始评测
    print(f"\n📋 将要运行 {len(benchmarks_to_run)} 个 Benchmark 评测:")
    for name in benchmarks_to_run:
        print(f"  - {name}")

    start_time = datetime.now()
    results = {}

    for name in benchmarks_to_run:
        success, result = run_benchmark(name, BENCHMARK_CONFIGS[name])
        results[name] = {"success": success, "result": result}

    # 汇总结果
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'='*60}")
    print(f"📊 评测汇总")
    print(f"{'='*60}")
    print(f"总耗时: {duration}")
    print(f"\n结果:")

    success_count = sum(1 for r in results.values() if r["success"])
    for name, r in results.items():
        status = "✅" if r["success"] else "❌"
        print(f"  {status} {name}")

    print(f"\n成功率: {success_count}/{len(results)}")


if __name__ == "__main__":
    main()

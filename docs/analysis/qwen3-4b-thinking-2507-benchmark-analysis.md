# Qwen3-4B-Thinking-2507 Benchmark 可用性与评测时间分析

> 基于 EvalScope 框架代码库深度分析，对照 Qwen3-4B-Thinking-2507 官方技术报告中的 24 个 Benchmark。

---

## 1. Benchmark 可用性总览

| 类别 | Benchmark | 框架中名称 | 状态 | 说明 |
|------|-----------|-----------|------|------|
| **Knowledge** | MMLU-Pro | `mmlu_pro` | ✅ 有 | 14 子集, ~12K 样本, MCQ |
| | MMLU-Redux | `mmlu_redux` | ✅ 有 | 57 子集, ~3K 样本, MCQ |
| | GPQA | `gpqa_diamond` | ✅ 有 | ~198 样本, MCQ |
| | SuperGPQA | `super_gpqa` | ✅ 有 | 87 子集, ~26K 样本, 10选项MCQ |
| **Reasoning** | AIME25 | `aime25` | ✅ 有 | 2 子集, 30 题, 数学生成 |
| | HMMT25 | `hmmt25` | ✅ 有 | 1 子集, ~30 题, 数学生成 |
| | LiveBench 20241125 | - | ❌ 无 | 框架中不存在 |
| **Coding** | LiveCodeBench v6 | `live_code_bench` | ✅ 有 | 支持日期过滤, 需 sandbox 执行代码 |
| | CFEval | - | ❌ 无 | 框架中不存在 |
| | OJBench | - | ❌ 无 | 框架中不存在 |
| **Alignment** | IFEval | `ifeval` | ✅ 有 | ~540 样本, 规则评分 |
| | Arena-Hard v2 | `arena_hard` | ⚠️ 有(v1) | 需要 Judge LLM, 版本可能不一致 |
| | Creative Writing v3 | - | ❌ 无 | 框架中不存在 |
| | WritingBench | - | ❌ 无 | 框架中不存在 |
| **Agent** | BFCL-v3 | `bfcl_v3` | ✅ 有 | 16 子集, 多轮函数调用 |
| | TAU1-Retail | `tau_bench` (retail) | ✅ 有 | 多轮 Agent, 需 user sim LLM |
| | TAU1-Airline | `tau_bench` (airline) | ✅ 有 | 同上 |
| | TAU2-Retail | `tau2_bench` (retail) | ✅ 有 | TAU 增强版, 需 user sim LLM |
| | TAU2-Airline | `tau2_bench` (airline) | ✅ 有 | 同上 |
| | TAU2-Telecom | `tau2_bench` (telecom) | ✅ 有 | 同上 |
| **Multilingual** | MultiIF | `multi_if` | ✅ 有 | 11 语言, 多轮(最多3轮) |
| | MMLU-ProX | - | ❌ 无 | 框架中不存在 |
| | INCLUDE | - | ❌ 无 | 框架中不存在 |
| | PolyMATH | `poly_math` | ✅ 有 | 18 语言, ~9K 样本 |

**统计：24 个 Benchmark 中，可用 17 个，缺失 7 个。**

---

## 2. 按评测时间排序（从快到慢）

以 API 服务 (batch_size=256, ~20-50 tok/s) 为基准估算：

| 排名 | Benchmark | 预估时间 | 评分方式 | 关键瓶颈 |
|------|-----------|---------|---------|---------|
| 1 | GPQA (`gpqa_diamond`) | **1-3 分钟** | MCQ 规则评分 | 仅 198 样本 |
| 2 | AIME25 (`aime25`) | **2-5 分钟** | 数学规则评分 | 仅 30 题，但每题长推理 |
| 3 | HMMT25 (`hmmt25`) | **2-5 分钟** | 数学规则评分 | 仅 ~30 题 |
| 4 | IFEval (`ifeval`) | **3-6 分钟** | 规则约束检查 | ~540 样本，无 Judge |
| 5 | MMLU-Redux (`mmlu_redux`) | **5-8 分钟** | MCQ 规则评分 | ~3K 样本, 0-shot |
| 6 | MMLU-Pro (`mmlu_pro`) | **15-25 分钟** | MCQ 规则评分 | ~12K 样本, 5-shot 长 prompt |
| 7 | MultiIF (`multi_if`) | **15-30 分钟** | 多轮规则评分 | ~3K 样本 x 3 轮 = ~9K 次调用 |
| 8 | PolyMATH (`poly_math`) | **30-60 分钟** | 数学规则评分 | 9K 数学题，长推理输出 |
| 9 | SuperGPQA (`super_gpqa`) | **30-60 分钟** | MCQ 规则评分 | 26K+ 样本，10 选项 |
| 10 | Arena-Hard (`arena_hard`) | **30-90 分钟** | **LLM Judge** (x2) | 500 样本 + 1000 次 Judge 调用 |
| 11 | LiveCodeBench (`live_code_bench`) | **2-4 小时** | **代码执行** (sandbox) | ~500 题，每题需执行代码 |
| 12 | BFCL-v3 (`bfcl_v3`) | **2-4 小时** | 多轮 Agent + AST 检查 | 多轮子集无法并行 |
| 13 | TAU1 (`tau_bench`) | **4-8+ 小时** | **双 LLM** 多轮 Agent | 每个任务 5-20+ 轮对话 |
| 14 | TAU2 (`tau2_bench`) | **4-8+ 小时** | **双 LLM** 多轮 Agent | 3 个域，同上 |

### 关键说明

- **最快的 5 个**（GPQA、AIME25、HMMT25、IFEval、MMLU-Redux）加起来总共不到 **30 分钟**，全部是规则评分
- **中间层**（MMLU-Pro、MultiIF、PolyMATH、SuperGPQA）主要瓶颈是样本量大，但评分仍是规则，共约 **2-3 小时**
- **Arena-Hard** 是分界点 — 需要 Judge LLM，速度取决于 Judge API 的吞吐
- **最慢的 4 个**（LiveCodeBench、BFCL-v3、TAU1、TAU2）需要代码执行或多轮 Agent 交互，单个可能跑 **半天**

---

## 3. 缺失的 7 个 Benchmark

| 缺失 | 类型 | 备注 |
|------|------|------|
| LiveBench | 通用推理 | 需自行集成或使用原版工具 |
| CFEval | Codeforces 编码 | 需自行集成 |
| OJBench | OJ 编码 | 需自行集成 |
| Creative Writing v3 | 写作对齐 | 需自行集成 |
| WritingBench | 写作对齐 | 需自行集成 |
| MMLU-ProX | 多语言知识 | 需自行集成 |
| INCLUDE | 多语言知识 | 需自行集成 |

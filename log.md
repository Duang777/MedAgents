# 实验日志

## 模板
- 日期：
- 实验/改进项：
- 做了什么：
- 结果：
- 备注：

---
## 2026-05-14
- 实验/改进项：下载并准备 MedAgents 基线实验的完整数据集
- 做了什么：
  - 创建独立虚拟环境 `.venvdata`（Python 3.11）用于数据处理。
  - 从 Hugging Face 下载完整数据并转换为项目可直接运行的 `test.jsonl` 格式。
  - 新增/更新：
    - `datasets/MedMCQA/test.jsonl`（4183）
    - `datasets/PubMedQA/test.jsonl`（1000）
    - `datasets/MMLU/anatomy/test.jsonl`（135）
    - `datasets/MMLU/clinical_knowledge/test.jsonl`（265）
    - `datasets/MMLU/college_medicine/test.jsonl`（173）
    - `datasets/MMLU/medical_genetics/test.jsonl`（100）
    - `datasets/MMLU/professional_medicine/test.jsonl`（272）
  - 保留已有 `datasets/MedQA/test.jsonl`（1273）。
- 结果：
  - 目标 benchmark 数据已全部就绪，且均为可运行的 `test.jsonl`。
  - 当前这些任务合计可用测试样本数：7401。
- 备注：
  - 作者 Google Drive 提供的是 `datasets_sample`（每任务约 100 条），并非完整实验集。
  - Hugging Face 未配置登录 token，仍成功下载。

---
## 2026-05-14
- 实验/改进项：方向一（idea1）首版开发：记忆增强与反思闭环接入 MedAgents
- 做了什么：
  - 新增 `idea1.md`，固化方向一方案与分阶段路线图。
  - 新增 `memory_bank.py`：实现 `CaseBank`（JSON 持久化、基于余弦相似度的 Top-K 检索、成功权重、时序衰减、剪枝）。
  - 新增 `reflection_agent.py`：实现成功案例反思提炼接口 `reflect()`。
  - 修改 `prompt_generator.py`：为最终答案阶段增加记忆注入位 `memory_context`。
  - 修改 `utils.py`：新增记忆上下文构建函数，并在 `syn_only/syn_verif` 回答阶段注入记忆。
  - 修改 `run.py`：增加参数 `--enable_memory --memory_path --memory_top_k --memory_max_size --enable_reflection`，串联“检索 -> 推理 -> 反思 -> 记忆更新”闭环。
- 结果：
  - 代码层面完成首版可运行闭环（不依赖额外训练）。
  - 已通过语法编译校验：`run.py/utils.py/prompt_generator.py/memory_bank.py/reflection_agent.py`。
- 备注：
  - 当前向量化实现为轻量 bag-of-words（保证先跑通）；后续可替换为 `sentence-transformers`。
  - 严格按规则未加入广义兜底逻辑，仅保留必要失败分支。

---

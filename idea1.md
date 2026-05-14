# Idea 1: 记忆增强与持续学习的 MedAgents 改造方案

## 目标
基于 MedAgents 构建“方向一”记忆增强方案，让系统在不做额外训练的前提下，通过架构优化实现持续进化：
- 借鉴成功案例
- 反思失败与成功轨迹
- 在后续病例中复用可迁移策略

## 总体流程
1. 记忆构建阶段
- 回顾历史案例
- 质量评估（是否进入精华记忆库）
- 反思提炼（Reflection Agent）
- 写入 CaseBank

2. 记忆推理阶段
- 新问题到来
- 向量检索 Top-K 相关经验
- 将经验注入提示词
- 进入 MedAgents 主流程进行协作推理

## 本项目第一版实现范围
- 新增 `memory_bank.py`
  - `CaseBank` 数据结构与 JSON 持久化
  - 向量化与余弦相似度检索 Top-K
  - 时序衰减与成功权重
  - 记忆剪枝策略
- 新增 `reflection_agent.py`
  - `ReflectionAgent.reflect()` 提炼“与具体病例无关”的可迁移策略
- 接入主流程
  - 推理前：检索记忆并注入最终回答 Prompt
  - 推理后：基于正确性判断触发反思并更新记忆库

## 关键设计
- 记忆条目字段（最小闭环）：
  - `id`, `question`, `options`, `pred_answer`, `gold_answer`
  - `syn_report`, `reasoning_trace`, `reflection`
  - `timestamp`, `success`, `success_weight`, `embedding`
- 检索排序分数：
  - `score = cosine_similarity * success_weight * time_decay`

## 后续迭代方向
- 引入失败模式记忆（错误归因）
- 按任务类型分桶检索（MedQA / PubMedQA / MMLU）
- 增加多样性约束，降低记忆偏置
- 增加异步反思队列，降低在线延迟

# Idea 1 升级版：反思驱动进化 + 自适应双阶段检索

## 1. 研究目标
在不做额外训练（no fine-tuning）的前提下，升级 MedAgents 的记忆机制，使系统具备：
- 从成功案例中提炼可迁移规则（反思驱动进化）
- 通过双阶段检索降低噪声注入（熟悉度筛选 + 回忆重排序）

本升级明确不引入图结构，聚焦可控实现与可复现实验。

## 2. 核心创新

### 创新A：反思驱动的自我进化
- 仅对“正确且高质量”病例触发反思。
- 通过 LLM 生成实体无关规则，写入 `RuleBank`。
- 推理时联合使用病例记忆与规则记忆。

### 创新B：自适应双阶段检索
- 阶段1（熟悉度筛选）：使用 `text-embedding-3-large` 向量召回候选病例。
- 阶段2（回忆重排序）：融合语义相似度、决策路径相似度、医学实体重叠度进行重排。
- 同时检索规则库（规则语义 + 规则触发实体）。

## 3. 系统模块

### 3.1 CaseBank（病例记忆库）
字段：
- `id`, `question`, `options`, `pred_answer`, `gold_answer`
- `syn_report`, `reasoning_trace`
- `timestamp`, `success`, `confidence`, `success_weight`
- `embedding`（`text-embedding-3-large`）
- `entities`（轻量实体抽取）

### 3.2 RuleBank（规则记忆库）
字段：
- `id`, `rule_text`, `trigger_entities`
- `confidence`, `timestamp`, `usage_count`
- `embedding`（`text-embedding-3-large`）

### 3.3 EvolutionReflector（两阶段反思器）
- 阶段1：质量门控（正确性 + 置信度阈值）
- 阶段2：规则提炼（输出 JSON 规则列表）

### 3.4 DualStageRetriever（双阶段检索器）
- 阶段1：Top-M 语义候选召回
- 阶段2：`score = w1*sim_semantic + w2*sim_path + w3*sim_entity`
- 输出：Top-K 病例 + Top-Kr 规则

## 4. 与现有代码集成点
- `api_utils.py`：新增 embedding API 调用（走同一网关）
- `memory_bank.py`：改为真实 embedding 存储与召回
- `rule_bank.py`：新增规则库
- `reflection_agent.py`：升级为两阶段反思并写入规则库
- `dual_stage_retriever.py`：新增双阶段检索
- `run.py`：接入检索注入 + 反思进化循环
- `utils.py`：增强记忆注入格式

## 5. 关键运行参数（建议默认）
- `embedding_model=text-embedding-3-large`
- `familiarity_top_m=15`
- `familiarity_threshold=0.6`
- `case_top_k=3`
- `rule_top_kr=2`
- `w1/w2/w3 = 0.5/0.3/0.2`
- `reflection_confidence_threshold=0.9`

## 6. 实验计划（消融）
1. 原始 MedAgents（无记忆）
2. 单阶段病例记忆（无反思）
3. 病例+规则（无双阶段）
4. 双阶段（无反思）
5. 完整方案（双阶段+反思）

## 7. 本次工程目标
- 先实现“完整方案”最小可运行版本（MVP）
- 以 MedQA 小样本完成冒烟与初步对照
- 记录全部结果到 `log.md`（中文）

## 8. 执行顺序约束（当前阶段）
- 先完成“外层创新”实验验证：反思驱动进化 + 自适应双阶段检索。
- 在外层对照实验跑完并记录结论前，不进行“内层工作流重构”（问题分域/选项分域/专家协作/综合/修订流程改造）。
- 内层优化将作为下一阶段（idea1-内层增强）单独推进，以避免实验变量混淆。

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
## 2026-05-14
- 实验/改进项：API 配置升级为环境变量驱动，并接入 dmxapi 指定模型
- 做了什么：
  - 修改 `api_utils.py`，移除代码内写死的 `api_base/api_version/api_key`。
  - 改为从环境变量读取：`OPENAI_API_TYPE`、`OPENAI_API_BASE`、`OPENAI_API_VERSION`、`OPENAI_API_KEY`。
  - 增加 OpenAI-compatible 路由（非 Azure 时走 `model=`，Azure 时走 `engine=`）。
  - 增加模型别名支持：`deepseek-v4-pro-guan`。
  - 未在代码和日志中写入明文 key。
- 结果：
  - 配置方式已升级，可直接切换到 dmxapi 与指定模型运行。
  - `api_utils.py` 已通过语法编译校验。
- 备注：
  - 当前项目依赖 `openai==0.27.4`（旧 SDK），本次保持兼容，仅做配置层升级。

---
## 2026-05-14
- 实验/改进项：idea1 三组冒烟对照实验（deepseek-v4-pro-guan, MedQA 0-3）
- 做了什么：
  - 新建实验虚拟环境 `.venvexp` 并安装运行依赖。
  - 配置 dmxapi 环境变量后，运行三组对照：
    - baseline（无记忆）
    - memory（记忆检索）
    - memory+reflection（记忆检索+反思）
  - 过程中修复两处阻塞问题：
    - `run.py` 模型白名单限制导致新模型被拒绝，改为允许任意 `model_name`。
    - `utils.py` 的记忆上下文函数命名导致 `NameError`，改为可导入函数名。
  - 另对综合报告格式做了定向兼容：当模型未输出 `Total Analysis:` 时，追加一次“仅重排格式”请求再解析。
- 结果：
  - baseline: 3/3 = 1.000
  - memory: 2/3 = 0.667
  - memory+reflection: 3/3 = 1.000
  - 三组均已端到端跑通，记忆检索字段写入输出成功。
- 备注：
  - 该轮为冒烟样本（n=3），只用于验证流程与稳定性，不用于最终结论。
  - memory 组出现一次 API 超时重试，最终完成。

---
## 2026-05-14
- 实验/改进项：idea1 升级实现（真实 embedding + 双阶段检索 + 反思进化）
- 做了什么：
  - 更新 `idea1.md` 为升级版方案文档（去图结构，聚焦反思+双阶段检索）。
  - 新增 `rule_bank.py`（规则库存储与剪枝）。
  - 新增 `dual_stage_retriever.py`（熟悉度召回 + 回忆重排 + 规则检索）。
  - 升级 `api_utils.py`：新增 embedding API 调用，默认 `text-embedding-3-large`（环境变量可配置）。
  - 重写 `memory_bank.py`：病例存储改为真实 embedding 与实体字段。
  - 重写 `reflection_agent.py`：升级为 `EvolutionReflector`（质量门控 + JSON 规则提炼 + 写 RuleBank）。
  - 修改 `run.py`：接入 CaseBank + RuleBank + DualStageRetriever + EvolutionReflector，并新增相关参数。
  - 修改 `utils.py`：记忆注入升级为“病例记忆 + 规则记忆”双通道。
- 结果：
  - Embedding 连通性验证通过：`text-embedding-3-large` 返回向量长度 3072。
  - 升级版端到端冒烟跑通（MedQA 2题，memory+reflection开启）。
  - 当前冷启动结果：CaseBank 已写入 2 条，RuleBank 未触发（规则数 0）。
- 备注：
  - 冷启动前几题检索命中为 0 属正常现象；需先积累足量成功病例再体现双阶段检索收益。
  - 规则提炼受“正确性+置信度阈值”控制，小样本下可能不触发。

---
## 2026-05-14
- 实验/改进项：外层方案正式小样本对照（MedQA n=50）
- 做了什么：
  - 在同一模型与同一配置下完成三组对照：
    - baseline（无记忆）
    - memory（双阶段病例检索）
    - memory+reflection（双阶段检索 + 反思规则库）
  - 使用 `text-embedding-3-large` 作为统一嵌入模型。
- 结果：
  - baseline: 44/50 = 0.88
  - memory: 44/50 = 0.88
  - memory+reflection: 46/50 = 0.92
  - memory 组平均病例检索命中：0.18 条/题
  - memory+reflection 组平均规则检索命中：0.96 条/题
  - memory+reflection 组新增规则：8 条（RuleBank 总数 8）
- 备注：
  - 本轮提升主要来自“反思规则库”引入后的规则注入，而非病例检索命中提升。
  - 样本量为 50，结论为阶段性结果；建议下一步扩展至 n=200 做稳定性验证。

---
## 2026-05-14（补充：可复现实验记录）
- 实验名称：idea1 外层方案对照（MedQA 前50题）
- 代码分支：`idea1`
- 模型：`deepseek-v4-pro-guan`
- 网关：`https://www.dmxapi.cn/v1`
- 嵌入模型：`text-embedding-3-large`
- 数据范围：`start_pos=0, end_pos=50`
- 统一推理配置：`method=syn_verif`, `max_attempt_vote=1`

- 组别配置：
  - baseline：不启用记忆
  - memory：`--enable_memory`，`memory_top_k=3`，`rule_top_k=2`，`familiarity_top_m=15`，`familiarity_threshold=0.6`
  - memory+reflection：在 memory 基础上增加 `--enable_reflection`，`reflection_confidence_threshold=0.9`

- 结果汇总：
  - baseline：44/50，accuracy=0.88
  - memory：44/50，accuracy=0.88
  - memory+reflection：46/50，accuracy=0.92

- 过程指标：
  - memory 平均病例命中：0.18 条/题
  - memory+reflection 平均规则命中：0.96 条/题
  - memory+reflection 新增规则：8 条

- 产物路径：
  - baseline 输出：`outputs/idea1_eval50/baseline/deepseek-v4-pro-guan-syn_verif`
  - memory 输出：`outputs/idea1_eval50/memory/deepseek-v4-pro-guan-syn_verif`
  - memory+reflection 输出：`outputs/idea1_eval50/memory_reflect/deepseek-v4-pro-guan-syn_verif`
  - case bank（memory）：`memory/idea1_eval50_case_bank.json`
  - case bank（memory+reflection）：`memory/idea1_eval50_case_bank_reflect.json`
  - rule bank（memory+reflection）：`memory/idea1_eval50_rule_bank_reflect.json`

- 稳定性说明：
  - 运行过程中曾出现网关 DNS/超时波动；已通过重试与解析保护确保实验可继续。
  - `data_utils.py` 对 `Total Analysis:` 缺失增加了安全解析，避免中断。

---
## 2026-05-14（补充：外层稳定快照与回退点）
- 实验/改进项：创建“仅外层创新”稳定版本快照，供后续内层优化前回退
- 做了什么：
  - 撤销了本轮内层第一步尝试（专家显式引用相关改动），恢复到“未做内层优化”的代码状态。
  - 提交外层稳定版本到 `idea1` 分支。
  - 创建并推送可回退标签（annotated tag）。
- 代码快照信息：
  - commit: `571f774`
  - branch: `idea1`
  - tag: `idea1-outerloop-stable-20260514`
- 快照内容范围：
  - 包含：CaseBank / RuleBank / DualStageRetriever / EvolutionReflector（外层创新）
  - 不包含：内层工作流优化（问题分域/专家显式引用/聚合器改造等）
- 作用：
  - 作为后续内层优化前的稳定基线，确保可一键回退与对照。

---
## 2026-05-14
- 实验/改进项：内层全链路优化首版（分域/分析/综合/修订）
- 做了什么：
  - `prompt_generator.py`：
    - 问题分析/选项分析 prompt 新增 `memory_brief` 注入位。
    - 综合报告 prompt 增加 `Confidence` 字段输出要求。
  - `utils.py`：
    - 新增记忆引用计数、冲突检测、综合置信度解析。
    - 在 `enable_inner_enhancement` 下将记忆简报注入专家分析与综合阶段。
    - 在“冲突或低置信”时向修订提示补充记忆简报，执行记忆引导修订。
    - 输出新增指标：`memory_citation_rate_question`、`memory_citation_rate_option`、`conflict_detected`、`syn_confidence`。
  - `run.py`：
    - 新增参数：`--enable_inner_enhancement`、`--inner_low_confidence_threshold`。
    - 聚合并写入记忆引用率指标。
- 结果（冒烟，MedQA 0-5，memory+reflection+inner 开启）：
  - n=5，accuracy=0.60
  - 平均病例命中=0.00，平均规则命中=0.00
  - 平均问题引用率=0.00，平均选项引用率=0.00
  - 冲突触发次数=0，低置信次数=2
- 备注：
  - 本轮受网关连接重置与DNS波动影响显著，embedding与chat均出现多次重试失败。
  - 冷启动命中不足导致“记忆注入与引用”未被有效触发；需在有预热库条件下再次验证内层效果。

---
## 2026-05-15
- 实验/改进项：实验脚本可观测性与可恢复性升级（实时指标 + 断点续跑）
- 做了什么：
  - `run.py` 新增实时运行指标：
    - 每题后动态显示 running accuracy（`acc`）
    - 预计剩余时间（`eta_sec`）
    - 当前完成数（`done`）
  - `run.py` 新增断点续跑能力：
    - 新增参数 `--resume` 和 `--overwrite_output`
    - 自动统计已完成样本数并计算 `effective_start`
    - 自动修复输出文件尾部损坏 JSON（截断到最后一条有效记录）
  - 性能优化（不改算法）：
    - `utils.py` 问题分析与选项分析改为同题内并发（ThreadPool）
    - `api_utils.py` 增加 embedding 内存缓存
    - 去除 `generate_response_multiagent` 在并发下的误杀超时装饰（保留重试）
- 结果：
  - 断点续跑验证通过：
    - 先跑 60~62（2条）
    - 再跑 60~64 且 `--resume`，自动从 62 继续，最终共 4 条
  - 提速对照（同切片 50~53，3题）：
    - 串行：492s，acc=1.00
    - 并发+cache：278s，acc=1.00
    - 提速约 1.77x，准确率未观察到下降
- 备注：
  - 当前主要耗时仍受网关波动与重试影响，并发已显著降低纯流程开销。

---

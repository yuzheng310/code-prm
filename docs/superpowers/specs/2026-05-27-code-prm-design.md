# Code-PRM: 面向代码 Agent 的过程奖励模型

**Spec 日期**: 2026-05-27
**作者**: xiaoyuzheng
**Scope**: 1.5–2 个月单人项目,基于现有 TypeScript codeAgent 扩展研究模块
**目标产出**: GitHub repo + 训练好的 PRM checkpoint + SWE-bench 评测数字 + 技术报告

---

## 1. 一句话定义

在现有 TypeScript codeAgent 之上,搭建一个 **Code Process Reward Model (PRM)**,对代码 Agent 多轮工具调用 trajectory 的每一个 step 打质量分,通过 Best-of-N 推理时算法在 SWE-bench Lite 上提升 base agent 的 pass@1。

---

## 2. 背景与动机

### 2.1 个人目标
强化简历以匹配校招算法/RL 方向岗位,具体命中以下 JD 关键词:
- **Reward System**(岗位:大语言模型应用算法工程师,JD-3)
- **Critic 能力 / 生成-判别-训练自闭环**(同岗位 JD-1)
- **CodeAgent / long-horizon tasks**(算法-机器学习岗 JD-2)
- **RLHF / LoRA / PyTorch**(职位要求第 3 条)

### 2.2 现有筹码
- 自研 TypeScript codeAgent:21 工具、权限两阶段分类、Prompt Cache、MCP、Skills/Hooks、LSP、worktree 隔离
- 工程完成度高,但**算法/训练侧产出为零**——这是简历断层

### 2.3 技术动机
代码 Agent 训练存在**稀疏奖励**问题:一次 15 步的 bug fix,只有最后 pytest 给出 0/1 信号,中间 14 步对错完全无监督。PRM 通过对每个 step 打分,提供密集信号,用于:
1. 推理时 Best-of-N / Beam Search / Tree Search 选轨迹
2. 早停剪枝节省 API 成本
3. 未来作为 dense reward 接入 GRPO/PPO

### 2.4 研究缝隙
- 数学 PRM(Math-Shepherd, PRM800K)已卷烂
- 视觉 PRM(VisualPRM)2025 年起步
- **代码 Agent PRM 仍是开放问题**,2024-2025 是合理时机窗口

---

## 3. 目标与非目标

### 3.1 目标(Must Have)
- [G1] 一个训好的 Code PRM checkpoint(基于 Qwen2.5-Coder-1.5B + LoRA + scalar head)
- [G2] 一个 2400 条 trajectory 的自采数据集,带 step-level MC 软标签
- [G3] 一个自建 CodeProcessBench(200 trajectory / ~1500 step 级人工标签)
- [G4] 在 SWE-bench Lite 上端到端 Best-of-N 评测,**目标提升 ≥ 8 pp**
- [G5] PRM step-level F1 ≥ 70%
- [G6] 一份 8-12 页技术报告 + 中文博客
- [G7] 可演示的 demo:在终端 `codeagent fix-bug ...` 跑出 Best-of-N 选轨效果

### 3.2 非目标(Won't Do in This Scope)
- [N1] **不做 policy 训练**(GRPO/PPO),只做 reward model;policy 仍用 Claude Sonnet
- [N2] 不替换现有 codeAgent 主力 LLM,PRM 是辅助评分器,与 LLM 并存
- [N3] 不冲 NeurIPS main track,以 GitHub repo + 技术报告为主要交付物
- [N4] 不做多模态 / 不做工具发现 / 不做自我改进
- [N5] 不在 1.5-2 月 scope 内做 ORM 兜底(全力压 PRM)

---

## 4. 系统架构

### 4.1 顶层数据流

```
┌──────────────────────────┐
│  TypeScript codeAgent    │
│  (现有,21 工具,MCP/LSP) │
└────────────┬─────────────┘
             │ 跑任务,记录 step
             ↓
       trajectory.jsonl
             │
             ↓
┌──────────────────────────┐
│  MC Labeler (Python)     │
│  对每个 step 做 K 次     │
│  rollout 估 mc_i ∈ [0,1] │
└────────────┬─────────────┘
             │
             ↓
       labeled.jsonl
             │
             ↓
┌──────────────────────────┐
│  PRM Trainer (Python)    │
│  Qwen2.5-Coder-1.5B +    │
│  LoRA + scalar head      │
│  masked-MSE loss         │
└────────────┬─────────────┘
             │
             ↓
        checkpoint
             │
             ↓
┌──────────────────────────┐    ┌──────────────────────┐
│  PRM Inference Service    │←───│  TypeScript codeAgent│
│  (Python FastAPI, 3090)  │    │  Best-of-N plugin    │
└──────────────────────────┘    └──────────────────────┘
```

### 4.2 模块划分

| 模块 | 语言 | 职责 | 复用源 |
|---|---|---|---|
| Trajectory Collector | TS | 在 codeAgent 里加 logging hook,输出 jsonl | 现有 codeAgent 扩展 |
| MC Labeler | Python | 对中间 step 做 K 次续跑,统计成功率 | RLHFlow Math-Shepherd MC 脚本 |
| PRM Model | Python | Qwen-Coder backbone + LoRA + scalar head | VLM-PRM repo 的 head 实现 |
| Trainer | Python | masked-MSE 训练循环 | OpenR `train/mat/` + HF TRL |
| CodeProcessBench Eval | Python | step-level F1 / P / R | 仿 VisualProcessBench |
| End-to-End Eval | Python | SWE-bench Lite + Best-of-N | SWE-bench harness + OpenR `reason/evaluation/` |
| PRM Service | Python (FastAPI) | HTTP 接口,接收 trajectory 返回分数 | OpenR `reason/llm_service/` |
| Best-of-N Plugin | TS | codeAgent 侧调 PRM 服务,选最高分 | 自写 |

### 4.3 接口契约

**Trajectory Collector → jsonl** (每行一条 trajectory,字段权威定义见 `src/labeler/trajectory_schema.py`):
```json
{
  "task_id": "django__django-12345",
  "task_type": "swe-bench-lite",
  "run_id": "uuid-v4-string",
  "rollout_id": 0,
  "repo": "django/django",
  "base_commit": "abc123...",
  "final_diff": "diff --git a/...\n...",
  "trajectory": [
    {
      "step": 0,
      "role": "assistant",
      "thought": "...",
      "tool": "read_file",
      "tool_args": {"path": "..."},
      "tool_result": "..."
    }
  ],
  "outcome": 1,
  "test_result": {"passed": true, "command": "pytest", "exit_code": 0, "duration_sec": 12.5},
  "policy_model": "claude-sonnet-4-5",
  "timestamp": "2026-...",
  "token_usage": {"input_tokens": 25000, "output_tokens": 5000, "cost_usd": 0.15}
}
```

**Step Labeler → labeled.jsonl** (在 trajectory 上增加每 step 标签 + 标注 `label_method`):
```json
{
  "task_id": "...",
  "trajectory": [
    {"step": 0, "...": "...", "step_label": 0.625},
    {"step": 1, "...": "...", "step_label": 0.875}
  ],
  "outcome": 1,
  "label_method": "llm_judge"
}
```

**PRM Service HTTP API**:
- `POST /score` → body: `{"trajectory": [...]}` → response: `{"step_scores": [0.5, 0.78, ...], "trajectory_score": 0.71}`

---

## 5. 关键技术细节

### 5.1 PRM 模型架构

```python
class CodePRM(nn.Module):
    def __init__(self, backbone_name="Qwen/Qwen2.5-Coder-1.5B-Instruct"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        # LoRA 注入(rank=16,target = q_proj/k_proj/v_proj/o_proj)
        self.backbone = peft.get_peft_model(self.backbone, lora_cfg)
        h = self.backbone.config.hidden_size  # 1536 for Qwen-Coder-1.5B
        self.reward_head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(h // 4, 1),
        )

    def forward(self, input_ids, attention_mask, step_spans):
        hidden = self.backbone(input_ids, attention_mask).last_hidden_state
        token_rewards = self.reward_head(hidden).squeeze(-1)
        step_rewards = pool_by_span(token_rewards, step_spans, mode="mean")
        return step_rewards
```

Reward head 设计完全照搬 VLM-PRM repo,经过验证可工作。

### 5.2 损失函数

Masked-MSE(Math-Shepherd 风格,但标签来源见 §5.3):

$$\mathcal{L} = \frac{1}{\sum_i m_i} \sum_i m_i \cdot (r_i - y_i)^2$$

其中 `r_i` 是 PRM 第 i 步预测,`y_i` 是 step_label(Phase 1 来自 LLM judge surrogate;Phase 2 future 可换成真 MC rollout),`m_i` 是 valid step mask。

### 5.3 Step 标签生成 — Phase 1: LLM-judge surrogate

**这一节诚实声明:Phase 1 的 `step_label` 不是真实 Monte Carlo rollout,而是 LLM-judge surrogate(weak supervision)。**

**算法**:
- 对每条 outcome=1 的训练 trajectory,对其中**有工具调用的 step**(忽略纯 thought)做 K=4 次 LLM-judge call
- LLM(Claude Opus,via DeepSeek 映射为 V4-Pro)接收 trajectory prefix,被要求预测"final outcome PASS/FAIL"。Judge 比 policy(Sonnet→V4-Flash)更强,避免 self-evaluation 偏差(Math-Shepherd 经典 "大模型 supervise 小模型" 设置)
- `step_label_i = (K 次 judge 中预测 PASS 的数) / K`
- outcome=0 的 trajectory:整条按 `step_label = 0` 标(outcome-only 简化,避免噪声)
- 输出 trajectory 标注 `label_method = "llm_judge"`

**为什么不做真 MC**:
- 真 MC 要求 step-level 状态恢复(repo checkpoint / agent state replay)+ 沙箱化 pytest 执行
- 工程量 3-4 周,标注 API 成本翻 4-8 倍(单条 trajectory 从 ~$0.05 涨到 ~$0.3)
- 1.5-2 月时间窗口外
- Phase 2 Future Work 列了升级路径

**这种 surrogate 的 known limitations**:
1. **Judge bias**:Haiku 对 partial code trajectory 的"PASS 概率"判断有 calibration 误差,可能系统性高估或低估
2. **粒度损失**:LLM 是"预测最终结果",不是"step-level 因果归因"——它可能把好步骤错判成坏,或反之
3. **outcome=0 全 0 简化**:失败 trajectory 的所有 step 都标 0,丢失了"中间步骤可能是对的"的信号(Math-Shepherd 也这么做,所以可接受)

**Phase 1 这个 surrogate 仍然足够支撑 PRM 训练**——只要 judge 比 random 强,PRM 学到的 ranking 信号就有用。Best-of-N 评测会直接验证。

**如果 Phase 1 PRM 在 Best-of-N 上没涨点,优先怀疑 surrogate 信号弱,升 K 值到 8 或切换 judge 模型到 Sonnet,而不是立即上真 MC**。

### 5.4 训练超参数

| 项 | 值 | 备注 |
|---|---|---|
| Backbone | Qwen2.5-Coder-1.5B-Instruct | 3090 + LoRA 充裕 |
| LoRA rank | 16 | 标准配置 |
| LoRA target | q/k/v/o_proj | |
| Learning rate | 5e-5(scalar head),1e-4(LoRA) | 分组 |
| Batch size | 4(梯度累积到 16) | 3090 显存约束 |
| Max seq len | 4096 | trajectory 平均 ~3k token |
| Optimizer | AdamW + cosine schedule | |
| Epochs | 3 | val loss 早停 |
| Warmup | 100 steps | |

### 5.5 Best-of-N 推理时算法

1. base policy 用 Claude Sonnet,温度 0.8,采 N=8 条 trajectory
2. PRM service 给每条打分,取 `trajectory_score = mean(step_scores[-3:])`(后段权重大)
3. 选最高分 trajectory 返回
4. 可选:加 early stopping,某条 trajectory 在第 t 步 score < 0.2 时 abort

---

## 6. 数据策略

### 6.1 训练数据

| 来源 | 任务数 | rollout 数 | 总 trajectory |
|---|---|---|---|
| SWE-bench Lite | 300 | 4 | 1200 |
| BigCodeBench-Hard | 300 | 4 | 1200 |
| **合计** | 600 | — | **2400** |

每条 trajectory 平均 8-15 step,总 step 数约 24k-36k。

### 6.2 评测数据

| Benchmark | 用途 | 规模 |
|---|---|---|
| CodeProcessBench(自建) | step-level F1/P/R | 200 trajectory / ~1500 step 标签 |
| SWE-bench Lite | 端到端 pass@1 | 全部 300 题 |
| BigCodeBench-Hard | 端到端 pass@1 | 全部 300 题 |

CodeProcessBench 的 step 标签获取:
- 从训练集留出 200 条 outcome=1 的 trajectory
- 用 Claude Sonnet 当标注员,给每个 step 打"对/错/中立" 三元标签(prompt 工程 + 校验)
- 抽样 10% 人工 review,准确率 > 85% 才发布

### 6.3 API 成本预估

| 阶段 | 模型 | 调用量 | 预估成本 |
|---|---|---|---|
| 主轨迹采集 | Claude Sonnet | 2400 trajectory × 30k token | ~$220 |
| MC rollout 标注 | Claude Haiku | 30k step × 4 rollout × 5k token | ~$120 |
| Best-of-N 评测 | Claude Sonnet | 600 题 × 8 rollout × 30k token | ~$430 |
| CodeProcessBench 标注 | Claude Sonnet | 1500 step × 3k token | ~$30 |
| **总计** | | | **~$800** |

预算控制:必要时把 Sonnet 降级到 Haiku / GPT-4o-mini,可压到 $300 以内。

---

## 7. 评测方案

### 7.1 Step-Level Metrics(PRM 自身能力)

在 CodeProcessBench 上做 threshold-scan:
- 对 PRM 输出 step_score,扫描 threshold ∈ [0, 1] 步长 0.05
- 在每个 threshold 下计算 P / R / F1
- 报告 best-F1 threshold 下的 P / R / F1
- 分子集报告:bug-fix / algorithm / other

**目标数字**:F1 ≥ 70%(对比基线:数学 PRM 在 code 上预计 < 50%)

### 7.2 End-to-End Metrics(下游应用价值)

| Method | SWE-bench Lite pass@1 | BigCodeBench-Hard pass@1 |
|---|---|---|
| Sonnet base | W7 实测 | W7 实测 |
| + Self-Consistency Best-of-8 | W7 实测 | W7 实测 |
| + **PRM-guided Best-of-8** | **目标 ≥ baseline + 8pp** | **目标 ≥ baseline + 6pp** |
| + PRM with early-stop | 同上,成本降低 ≥ 30% | 同上 |

注:baseline 绝对数值依赖于实测,本 spec 不预设具体数;成功标准锚在"相对 baseline 的 ΔPP"上。参考量级:Anthropic 公开数据中 Claude Sonnet 在 SWE-bench Lite 上 pass@1 约 30–50% 区间,我们的 baseline 预期落在此范围。

### 7.3 对照基线

| Baseline | 用途 |
|---|---|
| Random selection | 8 选 1 随机选,验证 PRM 是否真的学到东西 |
| Self-consistency vote | 投票法基线 |
| Length-based heuristic | 选最短/最长 trajectory,验证 PRM 不是在学长度 |
| Skywork-o1-Open-PRM-Qwen-2.5-7B | 数学 PRM 在 code 上的迁移效果 |

### 7.4 Ablation 实验

- Backbone 大小:Qwen2.5-Coder-1.5B vs 3B(QLoRA)
- LoRA rank:8 vs 16 vs 32
- Pooling 策略:mean vs last-token vs attention-pool
- MC K 值:K=2 vs K=4 vs K=8
- outcome=0 trajectory 是否纳入训练
- Step 切分粒度:每个 tool call 一个 step vs 每个 assistant turn 一个 step

---

## 8. 时间线(8 周)

| 周 | 任务 | Deliverable | 风险点 |
|---|---|---|---|
| W1 | Fork OpenR + 验证 lab box 环境(pytest 全过即可,不再跑 OpenR baseline 训练 — 见 §13 决策) | 环境验证 + 项目骨架 ready | 环境问题 |
| W2 | TS codeAgent 加 trajectory logging,跑 100 个 SWE-bench Lite 任务采主轨迹 | 第一批 trajectory.jsonl | codeAgent 集成问题 |
| W3 | 实现 MC labeler,对全量 2400 trajectory 做 MC 标注;数据清洗 + token-level alignment | code_prm_train.jsonl 就绪 | MC 标签信号弱 |
| W4 | 训 Qwen2.5-Coder-1.5B PRM,跑通收敛,初步 val F1 | 第一版 checkpoint | 不收敛 |
| W5 | 调超参 + ablation(backbone/lr/rank/pooling) | 最终 PRM checkpoint | — |
| W6 | 构建 CodeProcessBench,跑 step-level 评测 | F1/P/R 数字 | 标注质量 |
| W7 | PRM service + TS plugin 集成;端到端 SWE-bench Best-of-N 评测 | 端到端数字 | 集成 bug |
| W8 | 写 README + 技术报告 + 中文博客 + 简历点;录 demo 视频 | GitHub repo public + report.pdf | — |

**Buffer**: 1 周(W4-W5 之间任何一周不收敛,就用 buffer 调)

---

## 9. 风险与对冲

| 风险 | 概率 | 影响 | 对冲 |
|---|---|---|---|
| API 钱超预算 | 中 | 中 | 降级到 Haiku/GPT-4o-mini,削减任务数到 300 |
| MC 标注噪声大,PRM 学不到 | 中 | 高 | 加大 K 值;只用 outcome=1 的轨迹;对照数学 PRM 在 code 上的迁移效果作为下界 |
| Qwen-Coder-1.5B 太弱 | 低 | 中 | 升 3B QLoRA;或换 DeepSeek-Coder-1.3B |
| SWE-bench docker 跑不稳 | 中 | 中 | 用 SWE-bench Lite 子集(50 题);备选 BigCodeBench(纯 Python pip 装) |
| Best-of-N 没涨点 | 低 | 高 | 兜底:step-level F1 数字本身已是 contribution;加 tree search 试试 |
| codeAgent (TS) 与 Python 集成出问题 | 低 | 中 | 用 jsonl 文件而不是实时 HTTP,解耦时间紧迫性 |
| 训练不收敛 | 中 | 高 | 先在 PRM800K 上验证 pipeline 没问题;数据规模分级(100 → 500 → 2400) |

**关键防卡死策略**:
1. W1 不改任何代码,只跑 OpenR 默认配置,确认环境
2. 数据规模 100 → 500 → 2400 三段式,任一段出问题早暴露
3. 每周一个 checkpoint，可回退

---

## 10. 简历交付物

### 10.1 简历项目描述(预演)

```
项目名: Code-PRM: Process Reward Model for Code Agent Trajectories
栈: PyTorch / Transformers / PEFT / TypeScript / OpenR / SWE-bench
─────────────────────────────────────────────────────────────────
• 基于自研 TypeScript codeAgent (21 工具,MCP/LSP/worktree) 构建多轮代码
  trajectory 数据流水线,采集 SWE-bench Lite + BigCodeBench 共 2400+ 条
  工具调用轨迹

• 复现并扩展 OpenR / Math-Shepherd PRM 训练框架至代码 Agent 场景:
  蒙特卡洛 rollout 生成 step-level 软标签,基于 Qwen2.5-Coder-1.5B
  + LoRA + scalar head 训练 Process Reward Model,采用 masked MSE 损失

• 自建 CodeProcessBench (200 trajectory / 1.5k step 级标签),PRM 在
  step-level 上达到 F1 = XX%;在 SWE-bench Lite 上将 Claude Sonnet
  base agent 的 pass@1 从 X% 提升到 PRM-guided Best-of-8 的 Y%
  (+Z pp)

• 设计 Best-of-N + early-stop 推理时算法,在保持精度的同时降低 30% API
  成本
```

### 10.2 GitHub Repo 结构

```
code-prm/
├── README.md                  ← 项目主页 + 评测表 + 一键复现
├── docs/
│   ├── report.pdf             ← 8-12 页技术报告
│   ├── blog.md                ← 中文博客
│   └── design.md              ← 本文档
├── data/
│   ├── code-trajectory-2.4k/  ← 训练/验证/测试 jsonl
│   └── CodeProcessBench/      ← 评测集 + eval.py
├── checkpoints/
│   └── code-prm-qwen-coder-1.5b/  ← LoRA + reward head
├── src/
│   ├── collector/             ← TS trajectory collector(codeAgent 插件)
│   ├── labeler/               ← MC labeler(Python)
│   ├── prm/                   ← 模型 + loss
│   ├── train/                 ← 训练入口
│   ├── eval/                  ← 评测脚本
│   └── service/               ← FastAPI PRM 服务
├── scripts/
│   ├── run_swebench.sh
│   └── reproduce.sh           ← 一键复现
└── demo/
    └── best-of-n.mp4          ← 终端 demo 视频
```

### 10.3 面试 talking points

1. **生成-判别-训练闭环**:"我的 codeAgent 当生成器,跑 trajectory 当数据 → MC 标注当判别监督信号 → PRM 训练 → 闭环回到 codeAgent 做 Best-of-N"
2. **为什么是 PRM 不是 ORM**:"代码任务长 horizon,稀疏奖励对 Best-of-N / 早停剪枝都不够用,需要密集信号"
3. **为什么 1.5B 够**:"PRM 是判别任务不是生成任务,小模型完全够,OpenR 论文 1.5B PRM 给 72B policy 当 critic 也 OK"
4. **DeepSeek-R1 说 PRM 不必要怎么看**:"R1 在 outcome 干净的数学场景下做了简化,代码场景测试套件可能 flaky,PRM 价值反而更大;且小算力场景 PRM 仍是性价比之选"

---

## 11. Future Work(超出本 spec scope)

- **Real Monte-Carlo rollout 标注**(替换 Phase 1 的 LLM-judge surrogate):
  - 给 TS codeAgent 加 state checkpoint / replay 能力
  - Python 端实现沙箱化 pytest 执行(SWE-bench docker harness)
  - 每个 step 真的让 agent 续跑 K 次,真的跑 pytest,真的统计成功率
  - 预计工程量:3-4 周,API 成本 4-8x
  - `Trajectory.repo / base_commit / final_diff` 字段已预留

- **二期 GRPO**:用本 PRM 作为 dense reward,在 Qwen2.5-Coder-1.5B 上做 GRPO policy 训练(参考 veRL / OpenR 的 RL 模块)
- **Tree Search**:接入 MCTS,把 PRM 当 value function
- **Self-improvement loop**:GRPO 出来的更强 policy 重新跑 trajectory → 重训 PRM → 再 GRPO,形成迭代
- **多模态扩展**:接入截图 / 图表理解任务,做 VLM-Code-PRM(对标 VisualPRM)

---

## 12. 开放问题(实施阶段可调)

1. Step 切分粒度:每 tool call 一个 step,还是合并 thought+tool 当一个 step?(倾向前者,粒度更细)
2. CodeProcessBench 是否开源?(倾向是,但要先确保标注质量)
3. 是否在 README 里同时挂 ORM 版本作为 baseline?(倾向不,聚焦)
4. 报告投不投 workshop?(若 W8 还有余力可投 NeurIPS/ICLR workshop;非必须)

---

## 13. 决策记录

| 决策 | 选择 | 备选 | 日期 |
|---|---|---|---|
| 方向 | Code-PRM | RFT 蒸馏 / mini Agent RL | 2026-05-27 |
| 主代码源 | Fork OpenR | RLHFlow / AceCoder / 从零拼装 | 2026-05-27 |
| Backbone | Qwen2.5-Coder-1.5B | 3B / DeepSeek-Coder-1.3B / 通用 1.5B | 2026-05-27 |
| Rollout LLM | Sonnet (主) + Haiku (MC) | 全 Sonnet / 全 GPT-4o-mini | 2026-05-27 |
| 任务集 | SWE-bench Lite + BigCodeBench-Hard | 只 SWE / 只 BigCode | 2026-05-27 |
| ORM 兜底 | 不做 | 做 | 2026-05-27(用户决定) |
| 二期 GRPO | 列 Future,不在本 scope | 现在做 | 2026-05-27 |
| PRM 范式 | **scalar head + masked MSE**(VLM-PRM 同源) | OpenR `+/-` token 预测(Math-Shepherd 同源) | 2026-05-27 |
| OpenR baseline 训练 | **跳过**(范式不同,且 OpenR 预处理 submodule broken) | 跑通 OpenR 数学 PRM | 2026-05-27 |
| Step 标签方法 | **LLM-judge surrogate**(诚实命名,Phase 2 可升 real MC) | 真实 Monte Carlo rollout(3-4 周工程量+成本翻倍) | 2026-05-27(review 后) |
| Trajectory schema | 扩展含 `run_id` / `rollout_id` / `repo` / `base_commit` / `final_diff` / `test_result` / `token_usage` / `label_method` | 最小 schema(只 task_id/trajectory/outcome) | 2026-05-27(review 后) |
| 数据目录布局 | flat:所有 rollout 同一目录,trajectory 自带 `rollout_id` | nested `rollout_k/` 子目录(易导致 glob 漏读) | 2026-05-27(review 后) |
| 成本追踪 | TS 上报真实 `token_usage` → Python 端 `cost_aggregator` 汇总;collect_batch 仅做粗估软上限 | Python 估算 input/output token 作硬上限 | 2026-05-27(review 后) |

### 决策说明:PRM 范式选择(2026-05-27)

在实际 inspect OpenR 训练代码 (`prm/code/finetune_qwen_single_gpu.py`) 后发现,
OpenR 使用的是 **next-token prediction on `+`/`-` 标记** 的训练范式(Math-Shepherd 同源),
而非我们 spec §5.1 写的 **scalar reward head + masked MSE**(VLM-PRM 同源)。

两种范式都合法,但实施差异较大。决策选择 scalar head,理由:
1. 软标签信号无损(MC `mc_i ∈ [0,1]` 直接 MSE 回归;`+/-` 需离散化损失信号)
2. 与 VLM-PRM repo 同源,简历叙事更"原创"("扩展 VLM-PRM 到 multi-turn agent")
3. OpenR 的训练数据格式(`process` 字符串 + 步骤分隔 token)对 multi-turn tool trajectory 适配性差

结果:**OpenR 不再作为"训练框架的主干",降级为"参考代码 + Best-of-N 推理算法源"**。
- 不再跑 OpenR baseline 训练(原 Phase 1 Task 6-7,改为跳过)
- Phase 2 训练代码以 VLM-PRM repo 的 scalar head 实现为模板,自写
- OpenR 的 `reason/evaluation/` Best-of-N 推理算法仍在 Phase 3 复用

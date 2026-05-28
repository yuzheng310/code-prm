# Code-PRM 当前工作状况

最近更新:commit `b9360ae`(round 9 review 已完成,pilot 数据流水线已端到端跑通)

---

## 一句话概括

**为代码 Agent 的多轮 trajectory 训一个 step-level Process Reward Model**。当前在 **Phase 1**(造数据)收尾阶段,Phase 2(训 PRM)、Phase 3(评测 + Best-of-N + 写报告)未启动。

完整 design:`docs/superpowers/specs/2026-05-27-code-prm-design.md`
Phase 1 plan:`docs/superpowers/plans/2026-05-27-code-prm-phase1-foundation.md`

---

## 关键决策(已锁定)

| 决策 | 选择 | 为什么 |
|---|---|---|
| PRM 范式 | scalar head + masked MSE(VLM-PRM 同源) | 软标签无损;OpenR 的 `+/-` 离散化损失信号 |
| Step 标签来源 | **LLM-judge surrogate**(不是真 MC rollout) | 真 MC 要 sandbox 状态 replay,3-4 周工程量 |
| outcome=0 路径 | "outcome_zero_simplification"(所有 tool step = 0) | Math-Shepherd 简化版,省 API |
| 任务集 | **只用 BigCodeBench-Hard** | SWE-bench 要 docker harness,Phase 2 future work |
| TS Agent | `yuzheng310/pi`(fork of earendil-works/pi) | hooks 系统干净,免侵入 |
| LLM 后端 | DeepSeek Anthropic 兼容端点 | API key 限制,直连 Anthropic 403。`claude-*` 模型名自动映射到 `deepseek-v4-flash/pro` |
| Collection 模型 | `claude-sonnet-4-5` → `deepseek-v4-flash` | 便宜的策略模型 |
| **Judge 模型** | **`claude-opus-4-7` → `deepseek-v4-pro`** | **避免 self-evaluation 偏差**(judge 比 policy 强,Math-Shepherd 经典设置)。早期用 haiku→flash 时 mean 偏低(0.276,61% 全 0),换 opus→pro 期望 mean 上移 |
| 数据语义 | **诚实命名**:`label_method` 区分 `llm_judge` / `outcome_zero_simplification` / 未来 `mc_rollout` / `ground_truth` | 防止下游报告写"MC labels"被戳穿 |

---

## 架构

```
任务集 (HF) ─→ Python collect_batch.py
                  ├─ 设 env: CODE_PRM_LOG_DIR / TASK_JSON / ROLLOUT_ID
                  └─ 子进程: node pi/dist/cli.js -p "<problem>"
                              └─ pi 加载 ~/.pi/agent/extensions/trajectory_logger.ts
                                  └─ 写 $LOG_DIR/<task_type>_<date>.jsonl  ← raw trajectory
                                      
raw jsonl ─→ Python label_all.py
                  ├─ 每条 trajectory:
                  │   ├─ outcome=0 → step_label=0.0 (no API)
                  │   └─ outcome=1 → K=4 次调 Haiku(via DeepSeek)judge
                  ├─ stamp label_method
                  └─ 原子写 $LABEL_DIR/<file>.jsonl + labeling_manifest.json

labeled jsonl ─→ scripts/30_assemble_dataset.py
                    ├─ 6 个 hard checks(label_method / task_prompt 覆盖率 / 
                    │   step_label 覆盖率 / token_usage 覆盖率 / 
                    │   outcome_zero 标签一致性 / 分布非退化)
                    └─ train/val/test 分割 → data/code-trajectory-2.4k/
```

---

## 当前完成度

| 阶段 | 状态 |
|---|---|
| 代码骨架(Phase 1) | ✅ 18 Python files, 6 shell scripts, 1 TS extension |
| 单元测试 | ✅ ~110 tests,语法/逻辑全过 |
| Lab box 环境 | ✅ AutoDL vGPU-48GB, Node 20, pi 已 build, conda env 完整 |
| pi extension(trajectory_logger.ts) | ✅ 字段全部对齐 pi 真实 API,pilot 数据 schema 正确 |
| Trajectory collection 流水线 | ✅ 10 任务 pilot 跑通,token_usage / tool_result / thought 全部捕获 |
| LLM-judge labeler 流水线 | ⚠️ 代码完成,但 **judge 路径在真实数据上的非退化分布尚未验证**(下一步) |
| 全量数据采集 | ❌ 未开始(~$80,~8h) |
| 全量 labeling | ❌ 未开始(~$60,~12h) |
| Dataset assembly + Phase 1 报告 | ❌ 未开始 |

---

## 立即下一步:验证 judge 真的工作

```bash
cd ~/code-prm
git pull origin main
bash scripts/07_label_pilot_judge.sh
```

这一步用 force-pass(把 pilot 数据 outcome 改为 1)走 judge 分支,验证:
- DeepSeek 真能调通(URL / API key / 模型映射正确)
- 标签分布 non-degenerate(`mean ∈ [0.2, 0.8]`,至少 2 个 distinct value)

**成本:~$0.5,~5-10 分钟。**

**成功标准**:输出最后看到 `→ HEALTHY: True`

如果 `HEALTHY: False`(全 0 或全 1),需要调:
- judge prompt(`src/labeler/step_labeler.py:_build_continuation_prompt`)
- K 值(默认 4,可升到 8)
- 模型(可强制 `deepseek-v4-pro` 试更强 judge)

---

## 全部跑通后的剩余命令(顺序)

```bash
# ── 全量采集(~6-8h, ~$80) ────────────────
tmux new -s collect
bash scripts/11_collect_bigcodebench.sh
# Ctrl-B d 脱离,期间可关 ssh

# ── 真实成本统计 ────────────────
python -m src.utils.cost_aggregator --dir data/raw/bigcodebench-hard

# ── 全量 labeling(~12h, ~$60) ────────────────
tmux new -s label
bash scripts/20_label_steps.sh

# ── 组装最终数据集(2 分钟,会跑 6 个 hard checks) ────────────────
python scripts/30_assemble_dataset.py
# 任何 check 不过会 SystemExit(2) 并指出哪条不达标。

# ── 填 Phase 1 报告 ────────────────
# 用 cost_aggregator + label distribution 数字填:
$EDITOR docs/phase1-report.md

# ── 打标签发布 ────────────────
git tag phase1-complete && git push --tags
```

---

## Phase 2 / 3 准备工作

**Phase 1 出来后,告诉我:**
> "Phase 1 done, commit `<sha>`,数据集在 `data/code-trajectory-2.4k/`,labeling manifest 在 `data/labeled/bigcodebench-hard/labeling_manifest.json`,Phase 1 report 在 `docs/phase1-report.md`"

我会用 `writing-plans` skill 生成 Phase 2 plan(训练 PRM)。预计内容:
- Backbone: Qwen2.5-Coder-1.5B + LoRA + scalar head(spec §5.1)
- Loss: masked-MSE(spec §5.2)
- Eval: step-level F1(spec §7.1)+ Best-of-N(spec §7.2)
- 时间:~5-7 天 lab box GPU 时间

---

## 重要文件 / 路径

| 路径 | 用途 |
|---|---|
| `src/collector/trajectory_logger.ts` | pi extension(核心) |
| `src/labeler/trajectory_schema.py` | Pydantic 数据契约 |
| `src/labeler/step_labeler.py` | LLM-judge 实现 |
| `src/eval/swebench_runner.py` | TS 子进程启动器 |
| `src/eval/collect_batch.py` | 异步并发采集驱动 |
| `src/utils/cost_aggregator.py` | 从 trajectory.token_usage 算真实成本 |
| `scripts/05_collect_pilot.sh` | 10-task pilot |
| `scripts/06_label_pilot.sh` | labeling pilot(outcome=0 → simplification path) |
| `scripts/07_label_pilot_judge.sh` | force-pass pilot(验证 judge 路径,**下一步要跑这个**) |
| `scripts/11_collect_bigcodebench.sh` | 全量采集 |
| `scripts/20_label_steps.sh` | 全量 labeling |
| `scripts/30_assemble_dataset.py` | 组装 + 6 hard checks |
| `~/.pi/agent/extensions/trajectory_logger.ts` | 软链到 src/collector 的 extension(由 setup 脚本创建) |
| `~/pi/packages/coding-agent/dist/cli.js` | pi CLI |

---

## 已知 gotcha(踩过的坑,留个记号)

1. **HuggingFace 国内不通**:脚本默认 `HF_ENDPOINT=https://hf-mirror.com`
2. **Anthropic 国内 403**:脚本默认 `ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic`
3. **`.ipynb_checkpoints/` 干扰**:collect/label 已过滤 hidden dir
4. **SWE-bench 任务需要 docker harness**:Phase 1 跳过,只用 BigCodeBench
5. **pi 字段名跟 Anthropic 不同**:`usage.input` 不是 `usage.input_tokens`(extension 已对齐)
6. **node ≥ 20 + 系统 lib**:setup_pi.sh 有检测
7. **`thought` 来自 `ThinkingContent`**:extended thinking 模式下推理在这,extension 已读

---

## 总体预算

- 已花:~$1(pilots)
- 待花:~$140(全量采集 + labeling)
- Phase 1 上限:$500(spec §6.3)
- 剩余 buffer 充裕

---

## 下一步只看这两条

1. **跑 `bash scripts/07_label_pilot_judge.sh`**,看 `→ HEALTHY: True`
2. **OK 后跑 `bash scripts/11_collect_bigcodebench.sh`(tmux 后台)**

之后顺着 "全部跑通后的剩余命令" 章节走。

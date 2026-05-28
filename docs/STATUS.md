# Code-PRM 当前工作状况

最近更新:commit `445a94f`(加入 BigCodeBench 真实 grader,移除预算硬上限)

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
| **Judge 模型** | **`claude-opus-4-7` → `deepseek-v4-pro`** | **避免 self-evaluation 偏差**(judge 比 policy 强,Math-Shepherd 经典设置) |
| **Outcome 来源** | **trajectory_logger.ts 内置 BigCodeBench grader**(`agent_end` 时跑 `python <task.test>` 60s timeout) | 早期 outcome 默认 0 导致 100% 走 `outcome_zero_simplification` 路径,LLM-judge 从未被真实数据触发。grader 给出真实 pass/fail,outcome=1 才走 judge → step_label 才有意义 |
| 数据语义 | **诚实命名**:`label_method` 区分 `llm_judge` / `outcome_zero_simplification` / 未来 `mc_rollout` / `ground_truth` | 防止下游报告写"MC labels"被戳穿 |
| 预算控制 | **代码层不强制预算**(`--budget_usd 1000000` ≈ 无限);成本看 relay dashboard / `cost_aggregator` 后置统计 | Opus 节奏下 hard cap 会原子写回滚已完成工作;改成"花了再说"的策略 |

---

## 架构

```
任务集 (HF) ─→ Python collect_batch.py
                  ├─ 设 env: CODE_PRM_LOG_DIR / TASK_JSON / ROLLOUT_ID
                  └─ 子进程: node pi/dist/cli.js -p "<problem>"
                              └─ pi 加载 ~/.pi/agent/extensions/trajectory_logger.ts
                                  └─ 写 $LOG_DIR/<task_type>_<date>.jsonl  ← raw trajectory
                                      
raw jsonl(已含真实 outcome,grader 跑过 BigCodeBench test)
              ─→ Python label_all.py
                  ├─ 每条 trajectory:
                  │   ├─ outcome=0 → step_label=0.0 (no API,simplification)
                  │   └─ outcome=1 → K=4 次调 Opus → V4-Pro judge
                  ├─ stamp label_method ∈ {"outcome_zero_simplification","llm_judge"}
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
| pi extension(trajectory_logger.ts) | ✅ 字段全部对齐 pi 真实 API,**含 BigCodeBench grader** |
| Trajectory collection 流水线 | ✅ 10 任务 pilot 跑通,token_usage / tool_result / thought 全部捕获 |
| 真实 outcome 标签 | ⚠️ grader 代码完成,**真实分布尚未验证(下一步:重跑 pilot 看 outcome 分布)** |
| LLM-judge labeler 流水线 | ⚠️ 代码完成,verification 等真实 outcome 数据(force-pass 测试已被废弃,逻辑见 commit `445a94f`) |
| 全量数据采集 | ❌ 未开始(~$80,~8h) |
| 全量 labeling | ❌ 未开始(~$60,~12h) |
| Dataset assembly + Phase 1 报告 | ❌ 未开始 |

---

## 立即下一步:重跑 pilot,看真实 outcome 分布

```bash
cd ~/code-prm
git pull origin main

# pilot 现在会真跑 BigCodeBench 测试 → 真实 outcome
bash scripts/05_collect_pilot.sh

# 看 outcome 分布
python -c "
import json
from collections import Counter
import glob
outcomes = Counter()
for f in glob.glob('data/raw/pilot/*.jsonl'):
    if '.ipynb_checkpoints' in f: continue
    for line in open(f):
        t = json.loads(line)
        outcomes[t['outcome']] += 1
print('outcome distribution:', dict(outcomes))
"
```

**成功标准**:`outcome distribution: {0: ~6, 1: ~4}`(或类似的 30-60% pass 率)。

完全没有 outcome=1 → 说明 grader 没工作 / agent 写错路径 / 任务太难。贴 raw 数据 stderr_tail 排查。

**outcome 分布正常后**:
```bash
bash scripts/06_label_pilot.sh
# 这次会真正调 Opus judge,在 outcome=1 trajectories 上产生有意义的 step_label
```

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

## 运行环境

### 硬件 / 镜像(AutoDL 租用)

| 项 | 配置 |
|---|---|
| GPU | `vGPU-48GB-350W ×1`(L40S/RTX 6000 Ada 同档,48 GB VRAM) |
| CPU | 12 核 Xeon Platinum 8260 |
| 内存 | 90 GB |
| 系统盘 | 30 GB |
| 数据盘 | 100 GB(50 free + 50 扩容) |
| 镜像 | `PyTorch / 2.5.1 / 3.12(ubuntu22.04) / 12.4` |
| 计费 | 按量计费 ¥1.78/时;**不用时关机**(数据盘 ¥0.33/天) |

### 软件栈(lab box 上)

| 软件 | 版本 | 安装方式 |
|---|---|---|
| Python | 3.12 | 镜像自带 + miniconda |
| PyTorch | 2.5.1 + CUDA 12.4 | 镜像自带 |
| Node.js | 20.x | `nodesource setup_20.x` + `apt install -y nodejs` |
| npm | 10.x | 随 Node 装 |
| git-lfs | 系统包 | `apt install -y git-lfs`(给 PRM800K 用) |
| Canvas 系统库 | system | `apt install -y libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev libpixman-1-dev`(pi 的 native dep) |
| tmux | system | 长跑任务必备 |

### 关键 Python 依赖(`pyproject.toml`)

```
torch>=2.4,<2.6
transformers>=4.45,<4.50
peft>=0.13
anthropic>=0.39
pydantic>=2.9
datasets>=3.0
pytest>=8.0
tenacity>=9.0   # API 重试
fastapi>=0.115  # PRM service (Phase 3)
rich>=13.0      # 进度条
```

### pi(TS agent)

- Fork:`git@github.com:yuzheng310/pi.git` clone 到 `~/pi`
- Build:`npm ci && npm run build`(canvas 编译比较慢,~5 min)
- CLI 入口:`~/pi/packages/coding-agent/dist/cli.js`
- Extension 软链:`~/.pi/agent/extensions/trajectory_logger.ts` → `~/code-prm/src/collector/trajectory_logger.ts`

### 环境变量(脚本默认值,可 override)

| Env var | 默认值 | 用途 |
|---|---|---|
| `ANTHROPIC_API_KEY` | **必须自己设** | DeepSeek API key(也用作 Anthropic 兼容请求的 key) |
| `ANTHROPIC_BASE_URL` | `https://api.deepseek.com/anthropic` | 中转 URL,直接 Anthropic 国内 403 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 国内不通,走镜像 |
| `TS_REPO_PATH` | **必须自己设** = `$HOME/pi/packages/coding-agent` | swebench_runner 找 `dist/cli.js` 用 |
| `CODE_PRM_LOG_DIR` | 脚本自动设 | trajectory_logger 写 jsonl 的目录 |
| `CODE_PRM_ROLLOUT_ID` / `CODE_PRM_RUN_ID` | 脚本自动设 | extension stamp 用 |
| `CODE_PRM_TASK_JSON` / `CODE_PRM_TASK_TYPE` | 脚本自动设 | extension 解析任务用 |

写到 `~/.bashrc` 永久化:

```bash
export ANTHROPIC_API_KEY=sk-...
export TS_REPO_PATH=$HOME/pi/packages/coding-agent
# 其他两个有默认值,但也可以显式 export
```

### 网络代理

- **GitHub** 国内拉慢 → 用 `source /etc/network_turbo`(AutoDL 学术加速)或 `gh-proxy.com` / `ghfast.top` 前缀
- **HuggingFace** 直连不通 → 走 `hf-mirror.com`(脚本默认)
- **Anthropic API** 直连 403 → 走 DeepSeek 兼容端点(脚本默认)
- **npm** 慢 → `npm config set registry https://registry.npmmirror.com`

### 文件系统布局(lab box `/root` 下)

```
~/code-prm/                        # 本项目(git@github.com:yuzheng310/code-prm.git)
├── src/, scripts/, tests/, docs/  # 工程代码
├── data/                          # gitignored
│   ├── raw/pilot/                 # 10-task pilot raw jsonl
│   ├── raw/swebench-lite/         # ~~已废弃(SWE-bench 跳过)~~
│   ├── raw/bigcodebench-hard/     # 全量采集落地
│   ├── labeled/                   # label_all 输出
│   ├── code-trajectory-2.4k/      # 最终 train/val/test 划分
│   └── prm800k/                   # 数学 PRM 数据(给 Phase 2 ablation 用,已 LFS pull)
├── third_party/openr/             # submodule,仅作 reference
└── ...

~/pi/                              # pi monorepo,build 完
└── packages/coding-agent/dist/cli.js

~/.pi/agent/extensions/
└── trajectory_logger.ts           # → ~/code-prm/src/collector/trajectory_logger.ts (symlink)

~/miniconda3/                      # 镜像自带 conda(系统 Python 也是 3.12)
```

### 验证环境完好(任何时候可跑)

```bash
node --version              # v20.x
python --version            # 3.12.x
nvidia-smi | head           # 看到 48GB GPU
ls ~/pi/packages/coding-agent/dist/cli.js   # pi 已 build
ls ~/.pi/agent/extensions/trajectory_logger.ts  # extension 已软链
echo "$ANTHROPIC_BASE_URL"  # 应该有值
cd ~/code-prm && pytest tests/ -q   # 110+ tests 全过
```

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
| `scripts/05_collect_pilot.sh` | 10-task pilot(**现在含 BigCodeBench grader**) |
| `scripts/06_label_pilot.sh` | labeling pilot |
| `scripts/07_label_pilot_judge.sh` | force-pass pilot(已废弃 — grader 上线后 force-pass 测试无意义,代码留作参考) |
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

## 下一步只看这三条

1. **`bash scripts/05_collect_pilot.sh`** — 看 outcome 分布是不是 30-60% pass(不再全 0)
2. **`bash scripts/06_label_pilot.sh`** — 真实 outcome 触发 judge,看 step_label 分布健康
3. **OK 后 `bash scripts/11_collect_bigcodebench.sh`(tmux 后台)** — 全量 1200 trajectory

之后顺着 "全部跑通后的剩余命令" 章节走。

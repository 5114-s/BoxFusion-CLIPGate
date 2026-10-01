# TriFusion：BoxFusion 四模块联合路线

## 目标与基线

冻结的 100 场 B6 基线为：

| 指标 | AP | APrec | ARecall |
|---|---:|---:|---:|
| IoU 0.15 | 40.0434 | 47.5954 | 60.0837 |
| IoU 0.25 | 33.5492 | 42.6755 | 53.8730 |
| IoU 0.50 | 12.1613 | 23.2172 | 29.3091 |

论文结果为 `37.46 / 31.36 / 13.41`。若目标是论文三档 AP
分别增加 10 个绝对点，则目标为 `47.46 / 41.36 / 23.41`；当前 B6
仍分别相差 `7.4166 / 7.8108 / 11.2487`。

本路线不预先承诺可以达到该目标。先用 oracle 上限回答候选集合是否具备
足够的召回与定位上限，再训练安全门控，最后才允许修改正式输出。

## 三项以上独立改动

### M1：轻量缺失实例补充

SAM3 只作为离线教师/冻结 cache。在线目标是由轻量 YOLOE mask 分支提供
候选；SAM3 cache 用于教师诊断与蒸馏，不计入在线时延。

### M2：Zoo3D 式增量 Mask Graph

仅处理未匹配到 B6 全局框的候选。语义兼容和三维几何必须同时成立，至少
两个可靠视角确认，provider-call TTL 管理生命周期；单视角候选永不输出。

### M3：SGCDet 式 occupancy/MSR OBB Refiner

在原 OBB 局部坐标系构建粗细多尺度 occupancy，使用真实 mask-depth 点、
26 邻域连通组件、空空间和面可见性，只更新有支持的边界，并保留原 yaw。

### M4：B6-v2 AP50-aware 安全门控

使用 ScanNet train-only 监督学习候选相对原框的 `ΔIoU`、不确定性、伤害
概率和 IoU 0.25/0.50 向上跨越概率。只有几何硬门控与学习式置信下界同时
通过才允许替换。验证场景不得出现在训练 archive 中。

## 固定执行顺序

```text
Frozen B6
→ M1 missing proposals
→ M2 incremental Mask Graph confirmation
→ M3 occupancy/MSR OBB refinement
→ M4 ΔIoU/uncertainty gate
→ B6 candidate quality score
```

M1/M2 主要提高召回，M3 主要提高 AP25/AP50 定位，M4 主要控制精度损失。
这些模块的增益不能直接相加。

## 实验阶段

1. `observer`：所有模块记录候选，固定 `applied=0`。
2. `oracle`：报告 proposal-union、best-box 和 oracle-score 三个上限。
3. `train-only gate`：仅使用 ScanNet train 场景训练 M4。
4. `fixed10 observer`：加载冻结阈值后在固定 10 场仅记录门控预测。
5. 只有未来另行实现并审计 active profile 后，才可讨论 full100 active。

如果 proposal-union 或 best-box oracle 本身达不到目标，停止“+10”主张，
优先改进候选召回/几何，而不是继续调整验证集阈值。

## 隔离约束

本目录为 `/data/ZhaoX/OVM3D-Dett/boxfusion_trifusion_dev`。它不修改
`boxfusion_maskgraph_dev`，独立使用 `results/`、`diagnostics/` 和 `logs/`。
冻结 SAM3 cache 以只读符号链接引用，禁止覆盖原 cache。

## 防泄漏协议入口

唯一编排入口为：

```bash
bash scripts/run_scannet_trifusion_protocol.sh check
```

所有模式默认只校验并打印命令，不创建输出、不启动 GPU。只有对单个阶段显式
设置 `EXECUTE=1`（或 `BOXFUSION_TRIFUSION_PROTOCOL_EXECUTE=1`）才会执行。
协议故意不提供一条命令串行跑完所有阶段；每一步都是人工 review boundary。

| 模式 | 数据 | 行为 |
|---|---|---|
| `check` | train / full-val / fixed10 | 校验 split、cache provenance，打印配置 |
| `train-observer` | train-only | 逐场调用 `demo.py`；不调用任何验证 evaluator |
| `train-gate` | train-only + train GT | CPU 构建几何、`--verified-only` archive、训练 gate |
| `fixed10-observer` | 固定 10 个 val 场景 | 加载冻结 gate，仅 observer；标准 AP 仍等于 B6 |
| `fixed10-report` | 固定 10 个 val 场景 | CPU 导出候选并生成 GT-conditioned oracle |

### 0. 必需的 train-only teacher provenance

train cache 没有任何默认值，也绝不回退到当前 full100 val cache。必须同时提供
cache、builder 写出的 `metadata/shard*.json` 目录和包含 `train` 的 immutable
namespace：

```bash
export BOXFUSION_TRIFUSION_TRAIN_TEACHER_CACHE=/path/in/boxfusion_trifusion_dev/cache/sam3_teacher/train-v1
export BOXFUSION_TRIFUSION_TRAIN_TEACHER_METADATA_ROOT=/path/in/boxfusion_trifusion_dev/logs/train-v1/metadata
export BOXFUSION_TRIFUSION_TRAIN_TEACHER_NAMESPACE=sam3-scannet18-train-b6-100-v1
```

若 train100 teacher cache 尚未生成，从本隔离目录执行以下 builder。该步骤会
占用 GPU；它不是本次 CPU/编排任务的一部分，必须等当前任务结束并单独审阅
后再运行：

```bash
export TRIFUSION_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_trifusion_dev

BOXFUSION_SAM3_SCENE_LIST="$TRIFUSION_ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt" \
BOXFUSION_SAM3_FRAMES_ROOT="$TRIFUSION_ROOT/data/scannet_train" \
BOXFUSION_SAM3_TEACHER_RUN_TAG=sam3_teacher_train_b6_100_c050_frozen_v1 \
BOXFUSION_SAM3_TEACHER_CACHE_ROOT="$TRIFUSION_ROOT/cache/sam3_teacher/sam3_teacher_train_b6_100_c050_frozen_v1" \
BOXFUSION_SAM3_TEACHER_LOG_ROOT="$TRIFUSION_ROOT/logs/sam3_teacher_train_b6_100_c050_frozen_v1" \
BOXFUSION_SAM3_TEACHER_METADATA_ROOT="$TRIFUSION_ROOT/logs/sam3_teacher_train_b6_100_c050_frozen_v1/metadata" \
BOXFUSION_SAM3_TEACHER_NAMESPACE=sam3-scannet18-train-b6-100-c050-frozen-v1 \
  bash "$TRIFUSION_ROOT/scripts/build_scannet_sam3_teacher_cache.sh" 0,1
```

完成后，把上面 cache、metadata 和 namespace 的 exact 值导出为三个
`BOXFUSION_TRIFUSION_TRAIN_TEACHER_*` 变量，再运行协议 `check`。

collector 对 `shard*.json` fail closed 校验 schema、complete 状态、namespace、
train scene-list SHA/scene union、frames root、output directory、完整 shard index
和每个 cache artifact。cache 或 metadata 若解析到 `boxfusion_maskgraph_dev`
会被拒绝。

默认训练列表为
`evaluation/data_util/meta_data/scannetv2_train_b6_100.txt`；完整验证禁用列表
始终为 `evaluation/data_util/meta_data/scannetv2_val.txt`。两者任何交集都会在
写文件前终止。

### 1. Train-only / no-eval observer

先只检查：

```bash
bash scripts/run_scannet_trifusion_protocol.sh train-observer 0,1
```

审阅路径后再单独执行：

```bash
EXECUTE=1 bash scripts/run_scannet_trifusion_protocol.sh train-observer 0,1
```

底层入口是
`scripts/collect_scannet_trifusion_train_observer.sh`。它直接逐场调用
`demo.py`，从不调用 `evaluation/eval_scannet.py`；所有
`c4/trifusion/trifusion_missing applied` 和 mutation flag 必须为 false，
且此阶段禁止加载 AP50 gate checkpoint。

### 2. CPU 几何、训练 archive 与 gate

检查并打印三条 CPU 命令：

```bash
bash scripts/run_scannet_trifusion_protocol.sh train-gate
```

执行：

```bash
EXECUTE=1 bash scripts/run_scannet_trifusion_protocol.sh train-gate
```

顺序固定为：

```text
build_trifusion_geometry_candidates.py
→ build_ap50_gate_training_from_trifusion.py --verified-only
→ train_ap50_safety_gate.py
```

archive builder 和 trainer 都必须收到同一个完整
`scannetv2_val.txt` 作为 `--forbidden-scene-list`。默认产物为：

```text
datasets/trifusion_gate_trainonly_geometry_v1/
datasets/scannet_trifusion_ap50_gate_trainonly_v1.npz
models/scannet_trifusion_ap50_gate_trainonly_v1.npz
reports/trifusion_gate_trainonly_v1/geometry_summary.json
```

协议拒绝覆盖任一既有产物。checkpoint 使用前还会再次核对 training archive
与 checkpoint 内部 train/validation split 的场景均为配置的 train 子集，且
与完整 val 集合零交集。

### 3. Fixed10 gate observer

当前 full100 cache 只允许在此验证阶段只读使用。协议验证它的 namespace、
完整 val scene-list SHA/scene union、frames root，并确认 fixed10 是其子集；
它绝不会作为 gate 训练输入。
其 builder metadata 默认只读指向
`/data/ZhaoX/OVM3D-Dett/boxfusion_maskgraph_dev/logs/sam3_teacher_full100_c050_frozen_v1/metadata`；
若迁移 cache，必须同时显式设置
`BOXFUSION_TRIFUSION_FIXED10_TEACHER_METADATA_ROOT`，不能省略 provenance。

```bash
# 只检查并打印 GPU 命令
bash scripts/run_scannet_trifusion_protocol.sh fixed10-observer 0,1

# 审阅后执行固定 10 场
EXECUTE=1 bash scripts/run_scannet_trifusion_protocol.sh fixed10-observer 0,1
```

默认使用
`evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt`，必须恰好
10 场且是完整 val 的子集。runner 仍使用
`trifusion_plus10_observer`：gate 仅记录预测，不能改变 boxes、scores、数量
或顺序。因此 evaluator 的标准 AP 是冻结 B6 identity，不能当作 M4 active
增益。

### 4. Fixed10 oracle/export

```bash
# 只检查诊断 provenance 并打印 CPU 命令
bash scripts/run_scannet_trifusion_protocol.sh fixed10-report

# 执行候选导出与 oracle
EXECUTE=1 bash scripts/run_scannet_trifusion_protocol.sh fixed10-report
```

该阶段依次生成 geometry candidates、supplemental candidates、
`oracle_report.json` 和 `gate_counterfactual_report.json`。前者使用 GT
选择候选，只表示离线上限；后者严格按已经冻结的
`geometry_verified AND gate_accepted` 决策替换 M3 框，保留原 B6 分数与
行顺序，用于估计这个冻结门控若激活后的 AP（仍是离线评估，不是在线输出）。
协议先核对 driver log 中的 exact cache/namespace/checkpoint，并要求 gate
enabled、全部 mutation/applied 为 false。两份报告都不能用于反复调
fixed10 阈值后再宣称泛化结果。

## 当前能力边界

本分支尚无可改变正式预测的 TriFusion active profile。因此此协议完成的是
train-only 门控训练、fixed10 observer 诊断和 oracle 上限闭环；它不产生
TriFusion active AP，也不支持“+10”结论。若 oracle 上限不足，应回到候选
召回/几何设计；若上限足够，仍需另行实现、审计并在未见验证数据上评估
active runtime。

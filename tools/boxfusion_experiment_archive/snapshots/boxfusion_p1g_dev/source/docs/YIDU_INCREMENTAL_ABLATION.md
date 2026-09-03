# YiDu 局部几何路线：严格增量消融协议

本文档定义 `boxfusion_yidu_dev` 中的隔离实验路线。目标是逐项判断
SAM3 Mask-RGBD 局部几何模块是否值得进入 BoxFusion 的正式输出，而不是把
多个模块一次性打开后仅比较最终 AP。

当前实现遵守以下硬约束：

- `B0` 冻结当前 B6 输出；`A1` 到 `A6` 都是它的只读 observer 子阶段。
- 每次相邻实验只增加一个模块，顺序固定为
  `B0 → A1 → A2 → A3 → A4 → A5 → A6`。
- `A1` 到 `A6` 只记录候选和诊断信息，不修改最终框、score、顺序或 stable ID。
- 所有预测、日志、诊断和评估结果都写入当前隔离目录
  `/data/ZhaoX/OVM3D-Dett/boxfusion_yidu_dev`，不会写入正在运行的
  BoxFusion 目录。
- 先跑固定 10 场并完成恒等性和候选 oracle 审计；通过后才能跑 100 场。
- 这里不承诺精度提升，更不承诺 `+10 AP`。是否有效必须由固定协议的实验结果决定。

## 1. 严格的一次一模块矩阵

`✓` 表示该阶段已累计启用相应 observer。每一行相对上一行只能多一个 `✓`。

| 阶段 | 本阶段唯一新增模块 | 自适应腐蚀 | DFU 点过滤 | 几何体素组件 | Occupancy/MSR | Raw/Fused Query | AP50 安全门控 |
|---|---|---:|---:|---:|---:|---:|---:|
| B0 | 无，冻结 B6 |  |  |  |  |  |  |
| A1 | 自适应 mask 边缘腐蚀 | ✓ |  |  |  |  |  |
| A2 | DFU 局部半径/全局统计过滤 | ✓ | ✓ |  |  |  |  |
| A3 | 几何体素连通组件 | ✓ | ✓ | ✓ |  |  |  |
| A4 | Occupancy/MSR 局部框候选 | ✓ | ✓ | ✓ | ✓ |  |  |
| A5 | Raw/Fused 多候选查询 | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| A6 | train-only AP50 安全/质量门控 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

矩阵由
[`boxfusion/yidu_ablation.py`](../boxfusion/yidu_ablation.py)
固定，并在导入时检查每个相邻阶段恰好增加一个模块。所有阶段的
`observer_only=True`、`mutate=False`，不能通过旧 YAML 残留配置绕过。

## 2. 每个阶段的真实含义

### B0：冻结 B6

B0 是这条路线的唯一输出锚点。运行器固定：

- B6 IoU MLP score；
- detector/quality blend 为 `0.40`；
- ScanNet minimum extent 为 `0.40`；
- proposal interval 为 `5`；
- provider-step TTL 为 `3`；
- inference/evaluation seed 均为 `0`。

以前保存的 B6 数字只能作为历史参考，不能代替这次隔离实验重新生成的 B0。
所有后续阶段必须与本次 B0 使用相同场景列表、输入、权重、seed 和代码状态。

### A1：自适应腐蚀

在 SAM3 mask 内提取真实 depth 点之前，根据 mask 尺寸自适应去掉边缘像素，
降低轮廓处背景深度和邻接物体深度的污染。A1 只生成清理后的局部点诊断，
不改最终框。

### A2：DFU 局部半径与全局统计过滤

在 A1 后累计加入两类确定性点过滤：

- 局部半径邻居过滤：去掉缺少局部支持的孤立点；
- 全局统计近邻过滤：去掉相对主体分布显著异常的点。

这里的 “DFU” 指借鉴其深度点清理思想，并不声称完整复现原论文网络。

### A3：几何体素连通组件

把清理后的 XYZ 点体素化，构建确定性的空间连通组件，并根据锚框内支持、
组件规模和密度选择局部目标组件。

**A3 不是学习式语义 superpoint，也不是 SPGroup3D/Group3D 的完整
superpoint 模块。** 当前输入只有几何 XYZ，没有学习式点特征或语义图，
因此本文档和实验报告中应称其为“几何体素连通组件”或
“superpoint-inspired 几何组件”，不能写成已复现语义 superpoint。

### A4：Occupancy/MSR 局部框候选

在 A3 的目标组件上使用已有的局部 occupancy/MSR 几何描述与框细化逻辑，
生成 SGCDet 风格的局部候选框及固定特征。该阶段仍是确定性 observer：
即使候选看起来更好，也不会写回 BoxFusion 输出。

### A5：Raw/Fused 多候选查询

同时比较以下候选：

1. 原始 B6 框；
2. raw SAM3 mask-depth 框；
3. A3 几何组件框；
4. A4 occupancy/MSR 框。

A5 记录候选间共识、质量特征和诊断选择。没有经过 train-only 训练并冻结的
scorer 时，只能使用确定性启发式作 oracle 前筛选，不能把验证集结果反向用于
调权。

### A6：train-only AP50 安全/质量门控

A6 在 A5 的固定特征上加载 AP50 safety gate，预测候选是否可能改善几何，
同时估计不确定性、伤害概率和跨越 IoU 阈值的概率。运行时固定输入 schema
为 91 维。

A6 checkpoint 必须只用 ScanNet **train 场景**诊断训练，且在任何固定 10
场/100 场 validation 评估之前冻结。禁止：

- 用 `scannetv2_val.txt` 或固定 10 个验证场景训练；
- 根据验证集 AP 反复选择 checkpoint 或阈值；
- 将验证集 GT 写入运行时特征；
- 给 A1 至 A5 传入 A6 checkpoint。

当前 A6 仍是 observer。门控“接受”只会写入诊断，不会修改正式预测。

## 3. 隔离路径与不可混用规则

代码根目录：

```text
/data/ZhaoX/OVM3D-Dett/boxfusion_yidu_dev
```

默认每个阶段的输出位置：

```text
results/yidu_ablation/<run_tag>
logs/yidu_ablation/<run_tag>
diagnostics/yidu_ablation/<run_tag>
evaluation/yidu_ablation/<run_tag>
```

默认固定 10 场列表：

```text
evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt
```

默认 100 场列表：

```text
evaluation/data_util/meta_data/scannetv2_val.txt
```

运行器会写入 `logs/yidu_ablation/<run_tag>/run_manifest.json`，记录代码、
配置、场景列表、checkpoint、SAM3 cache namespace 和 seed 的哈希。
默认拒绝写入已有的非空实验目录。需要续跑时只能对同一个 manifest 使用
`BOXFUSION_YIDU_ALLOW_RESUME=1`；代码或输入不一致时不得续跑。

不要把不同阶段指向同一 `RUN_TAG`，不要复用旧 BoxFusion 的 prediction root，
也不要一边修改同一隔离目录的代码一边续跑。

## 4. 固定 10 场：逐阶段运行

进入隔离目录：

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_yidu_dev
```

正式运行前可给任一阶段加
`BOXFUSION_YIDU_DRY_RUN=1`。它只完成输入、来源和 manifest 检查，不会启动
GPU，也不会保留输出或占用正式 run tag。

一次只运行下面一个命令。上一阶段尚未完成恒等性和候选审计时，不要启动下一阶段。

### 4.1 B0 冻结锚点

```bash
BOXFUSION_YIDU_RUN_TAG=yidu_b0_ablation10_frozen_v1 \
bash scripts/run_scannet_yidu_ablation.sh B0 0,1
```

### 4.2 A1：仅新增自适应腐蚀

```bash
BOXFUSION_YIDU_RUN_TAG=yidu_a1_ablation10_observer_v1 \
bash scripts/run_scannet_yidu_ablation.sh A1 0,1
```

### 4.3 A2：仅新增 DFU 点过滤

```bash
BOXFUSION_YIDU_RUN_TAG=yidu_a2_ablation10_observer_v1 \
bash scripts/run_scannet_yidu_ablation.sh A2 0,1
```

### 4.4 A3：仅新增几何体素连通组件

```bash
BOXFUSION_YIDU_RUN_TAG=yidu_a3_ablation10_observer_v1 \
bash scripts/run_scannet_yidu_ablation.sh A3 0,1
```

### 4.5 A4：仅新增 Occupancy/MSR

```bash
BOXFUSION_YIDU_RUN_TAG=yidu_a4_ablation10_observer_v1 \
bash scripts/run_scannet_yidu_ablation.sh A4 0,1
```

### 4.6 A5：仅新增 Raw/Fused Query

```bash
BOXFUSION_YIDU_RUN_TAG=yidu_a5_ablation10_observer_v1 \
bash scripts/run_scannet_yidu_ablation.sh A5 0,1
```

### 4.7 A6：仅新增 train-only AP50 gate

先准备并冻结 train-only checkpoint。第一步只检查 train-only A5 采集路径：

```bash
BOXFUSION_YIDU_TRAIN_TEACHER_CACHE=/path/to/train_only_sam3_cache \
BOXFUSION_YIDU_TRAIN_TEACHER_METADATA_ROOT=/path/to/train_only_cache_metadata \
BOXFUSION_YIDU_TRAIN_TEACHER_NAMESPACE=sam3-scannet18-train-v1 \
bash scripts/collect_scannet_yidu_a5_train_observer.sh 0,1
```

确认输出中的 scene list、frames root、cache metadata SHA 和 namespace 都属于
train 后，加 `BOXFUSION_YIDU_TRAIN_EXECUTE=1` 重新运行同一命令。采集完成后，先对训练协议做
check-only：

```bash
BOXFUSION_YIDU_TRAIN_PRED_ROOT="$PWD/results/yidu_a5_train_observer_v1" \
BOXFUSION_YIDU_TRAIN_DIAGNOSTICS_ROOT="$PWD/diagnostics/yidu_a5_train_observer_v1" \
bash scripts/train_scannet_yidu_a6_gate.sh
```

确认路径后加 `BOXFUSION_YIDU_GATE_TRAIN_EXECUTE=1`。脚本在 CPU 上依次完成 A5 diagnostics adapter、
固定原目标的 IoU 监督构建、91 维 gate 训练和 train-only provenance 校验。
随后运行 A6：

```bash
BOXFUSION_YIDU_GATE_CHECKPOINT="$PWD/models/scannet_yidu_ap50_gate_trainonly_v1.npz" \
BOXFUSION_YIDU_GATE_TRAINING_ARCHIVE="$PWD/datasets/scannet_yidu_ap50_gate_trainonly_v1.npz" \
BOXFUSION_YIDU_GATE_TRAIN_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt" \
BOXFUSION_YIDU_GATE_FORBIDDEN_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
BOXFUSION_YIDU_RUN_TAG=yidu_a6_ablation10_observer_v1 \
bash scripts/run_scannet_yidu_ablation.sh A6 0,1
```

若 checkpoint、训练 archive、场景 provenance、91 维 feature schema 或 A6
profile 任一不一致，运行器会拒绝启动。训练 diagnostics 必须另设独立 run tag
和目录，不能使用上述 validation 输出目录。

## 5. 每个阶段必须通过的两道审计

### 5.1 输出恒等性

A1 到 A6 是 observer，因此与 B0 相比，以下内容必须逐场完全相同：

- 框数量和顺序；
- box corners 的 shape、dtype 和数值；
- score 的 shape、dtype 和数值；
- stable ID；
- 最终 prediction 文件集合。

可以直接使用统一 CPU 审计脚本；以 A1 为例：

```bash
BOXFUSION_YIDU_STAGE_TAG=yidu_a1_ablation10_observer_v1 \
bash scripts/audit_scannet_yidu_stage.sh A1
```

该脚本先调用 `tools/verify_yidu_identity.py`，通过后才导出候选并生成 oracle
报告。将 `A1` 和对应 tag 依次替换为 `A2` 至 `A6`。工具必须同时确认诊断中的
`yidu_mutation_enabled=false`、`yidu_applied_count=0` 和阶段模块矩阵一致。
任一场不相同都视为实现错误，先修复，不能用 AP 变化解释，也不能继续 100 场。

### 5.2 候选 oracle 与覆盖率

恒等性通过后，再从当前阶段 diagnostics 做离线候选 oracle。至少报告：

- 可观察的 B6 track 数、模块 attempted/valid 数和覆盖率；
- 相比原框，候选 `ΔIoU` 的 q10/q50/q90；
- `IoU@0.15/0.25/0.50` 的原始召回和 oracle 召回；
- 新增 IoU@0.25、IoU@0.50 真阳性数；
- 从 `<0.25 → ≥0.25`、`<0.50 → ≥0.50` 的 crossing 数；
- 候选改善率、伤害率、无效框率；
- 每个场景和每个被观察 track 的附加耗时。

oracle 只能回答“候选池是否包含更好的几何”，不等于可部署 AP。禁止选择 GT
最优候选作为正式输出。若一个阶段几乎没有 AP50 crossing，即使 10 场 AP15
看起来较高，也没有依据继续把该模块激活。

统一审计脚本使用 `tools/export_yidu_geometry_candidates.py` 把每个结果行转换
为严格的 0/1 ragged candidate，再调用同一
`tools/report_trifusion_oracles.py`；随后
`tools/report_yidu_candidate_deltas.py` 统计固定原目标的 ΔIoU
q10/q50/q90、改善/伤害比例及 IoU@0.25/0.50 crossing。A6 同时保留
all-valid 候选与
verified-only（gate 接受）候选，便于区分“候选上限”和“门控能否安全选中”。
对所有阶段必须使用相同 evaluator、IoU 阈值和场景列表，禁止手工挑例子。

## 6. 从固定 10 场进入 100 场的门槛

某阶段只有同时满足以下条件，才允许进入 100 场 observer：

1. 与隔离 B0 的输出逐场 bit-exact；
2. 相比上一阶段只新增矩阵中声明的一个模块；
3. diagnostics 完整，`yidu_applied=0`；
4. 候选 oracle 有非偶然的覆盖率和 AP25/AP50 crossing；
5. 没有依赖 validation GT、validation 训练或按 validation AP 调参；
6. 运行时/显存增量仍符合在线目标；
7. 代码、cache、checkpoint 和阈值已冻结并写入 manifest。

不满足门槛时，应记录为负结果并停止该分支，而不是把下一模块叠上去掩盖失败。

## 7. 100 场 observer 命令

只有固定 10 场门槛通过的阶段才设置 `BOXFUSION_YIDU_FULL100=1`。例如 A4：

```bash
BOXFUSION_YIDU_FULL100=1 \
BOXFUSION_YIDU_RUN_TAG=yidu_a4_full100_observer_v1 \
bash scripts/run_scannet_yidu_ablation.sh A4 0,1
```

其他阶段只替换阶段名和 run tag。A6 还必须显式提供已经冻结的 train-only
checkpoint：

```bash
BOXFUSION_YIDU_FULL100=1 \
BOXFUSION_YIDU_GATE_CHECKPOINT="$PWD/models/scannet_yidu_ap50_gate_trainonly_v1.npz" \
BOXFUSION_YIDU_GATE_TRAINING_ARCHIVE="$PWD/datasets/scannet_yidu_ap50_gate_trainonly_v1.npz" \
BOXFUSION_YIDU_GATE_TRAIN_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt" \
BOXFUSION_YIDU_GATE_FORBIDDEN_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
BOXFUSION_YIDU_RUN_TAG=yidu_a6_full100_observer_v1 \
bash scripts/run_scannet_yidu_ablation.sh A6 0,1
```

100 场仍是 observer。它用于验证固定 10 场候选规律能否泛化，不授权修改最终
输出。只有 held-out 报告确认候选精度、AP50 crossing 和在线开销稳定后，才应
单独设计、单独命名并重新审查 active profile；不能把本协议的 observer profile
原地改成 active。

## 8. 结果解释原则

- A1 到 A6 的最终 AP 理论上应与 B0 完全一致；不同说明输出被意外修改或实验输入
  不一致，而不是模块“提升了 AP”。
- 判断 observer 模块是否值得进入下一轮，首先看候选 oracle、覆盖率、伤害率和
  AP50 crossing，不看挑选后的个别可视化。
- 固定 10 场只用于快速淘汰和发现实现问题，不能用来证明泛化提升。
- 100 场 validation 也不能用于训练；它只能作为冻结方案的最终验证。
- 三个以上模块的组合不自动等于创新，也不自动带来累计增益。每个增量都应有独立
  的正向证据。
- 即使全部 observer 候选都显示正向 oracle，也不能据此承诺相对论文或 B6
  `+10 AP`；还需要无 GT 的安全选择器、active 消融和完整 100 场评估。

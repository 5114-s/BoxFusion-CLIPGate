# P1G：P1S 后的 Geometry-only Refiner 冻结协议

## 1. 目的与边界

P1S 在固定 10 场上已经能提供较多类别无关残差候选，但候选几何仍不足以跨越
IoU 0.50。P1G 的唯一目标是回答：

> 在不增加候选、不改变 P1S 排序、也不引入语义信息的情况下，局部深度几何
> 能否把已有 P1S 候选精修到更高 IoU？

本协议只允许开发和验证一个 **geometry-only 局部框精修器**。它不是新增
proposal 模块，也不是 score 校准、分组、Soft-NMS 或 supplemental-output
模块。

P1S 固定 10 场结果已经被查看过，因此这些场景必须视为
`touched/quarantined`：

- 不能用它们选择网络结构、损失、epoch、阈值或安全裁剪范围；
- 不能根据其 AP、Recall 或单场可视化重新训练 P1G-v1；
- 全部配置在 train-only 阶段冻结后，最多进行一次 observer 安全/迁移检查；
- 固定 10 场通过也不直接授权运行 100 场或启用正式输出。

这里的 “geometry-only” 只描述新增的 P1G refiner。P1S 上游候选生成器仍含有
其原有输入特征，因此不能把整个 P1S→P1G 系统表述为纯几何模型。

## 2. 阶段定义

| 阶段 | Profile | 相比上一阶段唯一变化 | 正式输出 |
|---|---|---|---|
| P1S | `p1s_native_sparse_context_observer` | 冻结锚点 | B6 原输出 |
| P1G0 | `p1s_geometry_identity_observer` | 采集候选局部几何，refined=raw | 与 P1S 完全相同 |
| P1G1 | `p1s_geometry_refiner_observer` | 新增 `local_geometry_refiner_v1` | 与 P1G0 完全相同 |

P1G0 用于验证候选索引、坐标变换、局部几何采集和额外耗时。P1G1 对每个
P1S 候选一进一出地预测：

```text
delta = [Δcx, Δcy, Δcz, Δlog_sx, Δlog_sy, Δlog_sz]
```

第一版保持 ScanNet axis-aligned 约束；如果输入框携带 basis/yaw，也必须保持
原 basis/yaw，不在 P1G1 中学习旋转。yaw residual 必须是未来单独消融。

P1G1 只把 refined box 写入诊断文件。`P1G-active`、质量选择器以及 full100
均不属于本协议。

## 3. Observer 硬契约

P1G0 和 P1G1 必须同时满足：

```text
observer_only=true
mutation_enabled=false
applied_count=0
uses_ground_truth=false
class_agnostic=true
semantic_features=false
regression_dim=6
```

具体要求如下：

1. 正式 BoxFusion boxes、labels、scores、数量和顺序在同一次运行内完全不变。
2. P1S raw candidate 的 ID、frame/provider-step、数量、顺序、box、score、
   objectness 和 NMS 结果完全不变。
3. `refined_candidate_ids` 必须与 `raw_candidate_ids` 一一对应，不能新增、
   删除、合并或重新排序候选。
4. NaN、Inf、非正尺寸、坐标系错误或超出安全范围的预测必须 fail-closed 为
   identity box，并记录明确的拒绝原因。
5. P2 occupancy proposal、P3 grouping、B6 score blend、CLIP gate、Soft-NMS、
   supplemental output、SAM/MLLM teacher 和其他旧实验分支全部关闭。
6. GT 只能由离线 train-only dataset builder 和离线 evaluator 读取；在线
   observer 不得接受 GT 路径、GT box 或匹配索引。
7. 诊断必须保存 raw/refined corners、候选稳定 ID、输入特征 schema、
   checkpoint SHA、P1S/B6 SHA、拒绝原因以及逐步耗时。

跨运行 pickle 的 bit identity 可能受上游 CUDA 非确定性影响，所以安全审计以
同次运行中“进入 P1G 前的正式输出快照”和“P1G 返回后的正式输出快照”为
硬判据。跨运行比较仍需报告，但不能用“非确定性”原谅同次运行内的 mutation。

## 4. Geometry-only 输入白名单

P1G1 可以使用：

- 真实 depth 和相机 pose 反投影得到的局部 xyz；
- 候选局部坐标中的稀疏 occupancy、点数、密度和多尺度邻域支持；
- 多视角可见次数、深度一致性和几何支持率；
- raw box 的中心、尺寸和固定 basis；
- 与最近 B6 box 的纯几何距离或 IoU，但不能使用其类别或语义分数。

P1G1 禁止使用：

- RGB、任何颜色统计量；
- CLIP/image embedding、文本、类别 ID 或类别尺寸先验；
- SAM/YOLOE/MLLM 的 mask score 或 teacher prediction；
- P1S objectness、detector score 或 B6 学习式 quality score；
- validation GT、validation evaluator 结果或场景特定手工参数。

为了避免把候选绝对位置学成场景先验，点和框必须转换到 raw candidate 的局部
坐标系，并按 raw size 归一化。训练和在线 observer 必须使用同一个坐标变换
实现及其版本/hash。

## 5. 两级 train-only 数据方案

### 5.1 当前可执行的快速模块级审计：60/20/20

当前已有的 train100 可先固定拆为：

| 子集 | 场景数 | 用途 |
|---|---:|---|
| `P1G-fit60` | 60 | 唯一允许反向传播 |
| `P1G-cal20` | 20 | early stopping、epoch 选择和安全裁剪冻结 |
| `P1G-audit20` | 20 | checkpoint/config 冻结后只运行一次 |

拆分必须在查看 P1G 指标前根据 scene ID 的固定 SHA 排序生成，并写出三个
scene-list 及 SHA。三者必须两两不交，且与完整 `scannetv2_val.txt` 的交集为
空。

这个 20 场 audit **只能叫模块级审计，不能叫独立端到端泛化测试**。限制是：

- 当前 P1S 的梯度训练曾接触 train100 中的 80 场；
- P1S checkpoint 的内部选择还可能读取其余 train-only development 场景；
- 因而即使 P1G 自身没有在 `P1G-audit20` 上反向传播，上游 P1S 也不能保证
  对这 20 场完全未见。

建议优先把 P1S 原 fit80 中的 60 场分给 `P1G-fit60`、20 场分给
`P1G-cal20`，把 P1S 原 train-only development20 分给
`P1G-audit20`。manifest 必须如实记录 P1S 对各子集的既往使用方式，不能只用
“P1G 没训练过”来掩盖上游接触。

`P1G-audit20` 通过后的决定只能是：

```text
GO_FRESH50_AUDIT
```

不能是 `GO_VAL10`、`GO_FULL100` 或 `GO_ACTIVE`。

### 5.2 正式选择：额外 50 个从未使用的 ScanNet train 场景

更严格、推荐用于论文方法选择的方案是，在 canonical ScanNet train 中预先
选择额外 50 场：

```text
P1G-audit50-fresh
```

这 50 场必须同时满足：

- 不在 P1/P1R/P1S checkpoint 的任何 fit、cal、development 或 provenance
  scene IDs 中；
- 不在 P1G-fit60、P1G-cal20 或任何既往 P1/P2/B3/B5/B6 训练诊断中；
- 不在完整 ScanNet validation scene list 中；
- 选择规则只依赖 scene ID 的预注册 hash，不依赖场景 GT 数量、难度或模型
  输出。

P1G1 的结构、损失权重、checkpoint epoch、输入 schema、安全范围和全部阈值
在运行这 50 场前冻结。`P1G-audit50-fresh` 只能运行一次；失败后不得在同一
50 场上修改 P1G-v1。若未来设计 P1G-v2，应更换一组新的预注册 train-only
audit 场景，或把既有结果明确标注为 exploratory。

## 6. 离线目标分配与训练目标

所有匹配均为 class-agnostic，且只在 train-only 离线 builder 中进行。

对 raw candidate `b` 和 GT `g`，在候选局部归一化坐标中定义：

```text
target_center = (center(g) - center(b)) / size(b)
target_size   = log(size(g) / size(b))
```

一个候选仅在以下条件全部满足时获得几何回归监督：

1. raw candidate 中心位于轻度扩张后的 GT 内，或者 raw 3D IoU 至少为
   `0.05`；
2. 局部有效深度点至少为 32；
3. GT 在预注册安全裁剪范围内可达；
4. 裁剪后的 GT 目标相比 raw box 理论上能提高至少 `0.05` IoU。

同一 GT 对应多个候选时必须使用固定的 one-to-one 匹配，或按每个 GT 的候选数
归一化权重，避免重复 proposal 支配损失。无法匹配或不可安全精修的候选只训练
identity residual，不能用随意的最近 GT 做回归。

建议的 P1G-v1 目标为：

```text
L = L_huber(center/log-size)
  + λ_iou * L_axis_aligned_3d_iou
  + λ_identity * L_identity(unmatched/unrefinable)
```

损失必须按 scene 做 macro average，不能让长场景或候选密集场景支配训练。
raw IoU 低于 0.50、且安全裁剪 oracle 可以跨越 0.50 的样本可使用预注册的
2 倍权重，以直接对应 P1S 的 AP50 几何短板。这个权重只能在
`P1G-fit60/P1G-cal20` 内冻结。

建议首版安全范围为：

```text
每轴中心改变量 <= 0.20 * raw_extent
每轴尺寸比例     in [0.67, 1.50]
```

最终采用值必须写入 checkpoint 和 manifest。禁止根据 fixed val10 修改。

P1G1 不同时增加学习式 selector 或 quality gate。train-only audit 必须分别
报告：

- `refined-only`：每个 raw candidate 被对应 refined candidate 替换，属于
  可实现的确定性候选池；
- `raw ∪ refined`：仅作为候选几何 oracle 上界。

GO/STOP 必须以 `refined-only` 为主，不能靠把候选数翻倍获得通过。若之后需要
选择 raw/refined，应另立 `P1G2 geometry-quality gate` 消融。

离线候选审计器必须显式指定 `--stage module20` 或 `--stage fresh50`，不得从
scene-list 文件名猜测阶段。两阶段统一采用以下冻结统计定义：

- AP 上穿/下穿先由 B6 的 score-ordered one-to-one matching 确定 novel GT，
  再分别对 raw/refined 做 score-ordered one-to-one matching；上穿和下穿是
  两个 matched-GT 集合之差。禁止用“每个 GT 的最大 candidate IoU”计数，
  因为一个 candidate 可能同时覆盖多个 GT 并虚增 crossing。
- `可精修匹配样本` 固定为 raw candidate 按冻结 score/ID 顺序、以
  `raw IoU >= 0.05` 对 GT 做 one-to-one matching 后得到的 candidate/GT 对；
  refined IoU 必须读取同一 candidate ID 和同一 GT，不能重新匹配。全局
  median 和 harm rate 在这些场景级配对样本合并后计算；零匹配必须 fail
  closed。
- fresh50 的 confidence interval 固定为按 scene 有放回 bootstrap 10,000
  次、seed=0；每次以抽样场景的 `Σ(delta novel TP) / Σ(N_GT)` 计算 micro
  Recall 增量，报告 2.5%/97.5% 分位点。

## 7. 快速 audit20 的 GO/STOP

`P1G-audit20` 必须同时满足：

1. observer identity、candidate one-to-one、provenance 和 20 场完整性通过；
2. `B6 ∪ refined-only` 相比 `B6 ∪ P1S-raw` 至少新增 2 个 novel
   AP50 true positive；
3. novel Recall@0.25 不下降；
4. AP50 上穿数严格大于下穿数；
5. 可精修匹配样本的 median `ΔIoU >= +0.02`；
6. `ΔIoU <= -0.05` 的伤害率不高于 12%；
7. 满足第 9 节的候选和速度约束。

全部通过时只返回 `GO_FRESH50_AUDIT`。任一失败返回
`STOP_P1G1_MODULE_AUDIT`，不运行 fixed val10 或 full100，也不在同一个
audit20 上调参。

## 8. 正式 fresh-audit50 的 GO/STOP

正式 50 场以 `B6 ∪ P1S-raw` 为冻结参照，全部条件必须同时满足：

1. observer safety、身份、输入白名单、provenance 和 50 场完整性通过；
2. `B6 ∪ refined-only` 的 novel Recall@0.50 增量至少 `+1.0 pp`；
3. 上述 Recall@0.50 增量的 scene bootstrap 95% confidence interval 下界
   大于 0；
4. novel Recall@0.25 不低于 raw P1S；
5. AP50 上穿减下穿至少为
   `max(5, ceil(0.01 * N_GT))`，且上穿数至少为下穿数的 2 倍；
6. 可精修匹配样本的 median `ΔIoU >= +0.03`；
7. `ΔIoU <= -0.05` 的伤害率不高于 10%；
8. 满足第 9 节的候选和速度约束。

全部通过返回：

```text
GO_ONE_SHOT_VAL10_OBSERVER
```

任一条件失败返回：

```text
STOP_P1G1
```

STOP 后不得运行固定 10 场或 100 场，也不得使用同一 audit50 重训 P1G-v1。

### 8.1 两阶段 scene-list provenance 绑定

`module20` 必须与 P1G checkpoint 中的 `audit_scene_ids` 和
`audit_scene_list_sha256` 完全一致。`fresh50` 则不能继续与这两个字段比较：
它按定义是 checkpoint 冻结后、且不同于 module20 的额外场景；把 fresh50
强行与 checkpoint 的 audit20 字段比较会使正式审计必然崩溃。

fresh50 审计器必须改为：

1. 验证 50 个 scene ID 与 P1S checkpoint 中全部枚举的
   fit/cal/development/source-summary 场景不相交；
2. 验证其与 P1G checkpoint 的 fit/cal/audit20 场景不相交；
3. 在审计报告中保存 fresh50 scene-list 的精确 SHA 和逐场景源文件 SHA。

scene-list 的“预注册时间”不能由模型 checkpoint 自证，仍必须由独立只读
manifest/实验日志保存；审计报告中的 SHA 应与该外部记录核对。不得为了写入
fresh50 SHA 而重写已经冻结的 P1G checkpoint。

## 9. 候选数量与在线速度约束

P1G 不得增加 P1S 候选，因此：

```text
raw_candidate_count == refined_candidate_count
candidate_count <= 256 / scene
```

速度必须测量在线 live geometry path，局部点提取、坐标变换和 head forward
都计入；训练缓存或离线预计算不能隐藏在线成本。

冻结预算为：

| 指标 | 上限 |
|---|---:|
| P1G 增量 mean runtime | `0.15 s/scene` |
| P1G 增量 p95 runtime | `0.30 s/scene` |
| P1S + P1G mean runtime | `0.80 s/scene` |
| 额外 peak GPU memory | `512 MiB` |

报告至少拆分 `geometry_extract_s`、`coordinate_transform_s`、
`refiner_forward_s` 和 `total_p1g_s`。只有平均值而没有 p95，或只报告
checkpoint forward 而忽略几何提取，都视为速度门槛失败。

## 10. 固定 10 场的唯一允许用途

只有 `P1G-audit50-fresh` 通过且以下内容全部冻结后，才允许运行一次原固定
10 场 observer：

- checkpoint 及其 SHA；
- 输入特征 schema 和局部坐标实现；
- 安全裁剪范围；
- 所有训练和推理配置；
- B6/P1S checkpoint；
- 代码树、数据 root 和 scene-list hashes。

固定 10 场仍保持 `applied_count=0`。预注册检查为：

1. 正式输出身份通过；
2. refined-only 相比 raw P1S 至少新增 2 个 novel AP50 TP；
3. novel AP25 TP 不低于 raw P1S；
4. P1S+P1G mean runtime 不超过 `0.80 s/scene`。

由于该 fixed10 已经被既往 P1S 实验查看过，它只能提供工程安全和迁移证据，
不能重新充当完全未见的模型选择集。失败即 `STOP_P1G1_VAL10`，不得修改配置后
重跑。通过只能返回：

```text
GO_DESIGN_P1G2_OR_ACTIVE_PROTOCOL
```

它仍不授权直接运行 full100。下一阶段若要实际替换候选，必须另外设计
geometry-quality selector、安全门控和 active-output 审计。

## 11. 必需 provenance

训练 archive、checkpoint、observer manifest 和 audit report 至少绑定：

- `P1G-fit60`、`P1G-cal20`、`P1G-audit20`、可选 fresh-audit50 的
  scene-list 内容和 SHA；
- canonical ScanNet train/val scene-list SHA 及显式交集结果；
- P1S 既往 fit/cal/development scene IDs；
- B6 和 P1S checkpoint SHA；
- 每场 train GT、axisAlignment、P1S diagnostics 和 raw prediction SHA；
- 输入 feature names、坐标系、回归编码、安全范围和 loss 配置；
- 源代码树 hash、随机种子和 deterministic-algorithm 状态；
- 所有 observer identity、候选一一对应、失败场景及计时结果。

checkpoint loader 必须重新计算 scene 交集，不能只相信 checkpoint 内保存的
`forbidden_overlap=[]`。任何 hash 漂移、scene overlap、缺失场景或 schema
不一致都必须 fail closed。

## 12. P1G-v2 冻结实验结果（2026-07-30）

P1G-v2 使用 function-preserving adapter：零 correction 可复现冻结 P1S
的 `clip(center_delta) + exp(clip(log_extent))` 解码；训练损失直接在
ScanNet aligned frame 中计算。完整代码回归测试为 `1111 passed`。

train-only `60/20/20` 训练得到的冻结 checkpoint 为：

```text
models/scannet_p1g_aligned_geometry.pt
sha256 = 7e1197772639cefef497a3dde5da39ca68e3bccc11f9c3a83473e409dedee6c8
```

一次性 `P1G-audit20` 结果为：

| 指标 | P1S raw | P1G refined | 变化 |
|---|---:|---:|---:|
| union TP @0.15 | 174 | 151 | -23 |
| union TP @0.25 | 135 | 130 | -5 |
| union TP @0.50 | 61 | 62 | +1 |
| novel TP @0.25 | 17 | 11 | -6 |
| novel TP @0.50 | 1 | 2 | +1 |

候选一一对应、分数/ID/顺序不变，且 correction forward+decode 的 mean/p95
耗时为 `0.0042/0.0081 s/scene`。但是 205 个可比较匹配中的 median
`ΔIoU=-0.0557`，`56.59%` 的匹配下降至少 `0.05 IoU`。因此预注册决定为：

```text
STOP_P1G1_MODULE_AUDIT
```

完整报告：

```text
reports/p1g_audit/p1g_audit20_v1.json
sha256 = 46bab05695a6af2a891c10d7db86d7006b5dad0a4625ef709b6af9ca49caad22
```

按本协议不得继续运行 fresh50、fixed val10、full100 或 active output，也不得
在同一个 audit20 上重新调参。该结果说明“共享 P1S 隐特征 + 单线性几何
correction”缺少实例级局部点支持，不能作为后续主提升路线。

## 12. 结果解释

- P1G0/P1G1 的标准 AP 理论上应与同次运行的冻结 B6 输出完全一致；差异意味着
  observer 违规，而不是精度提升。
- P1G1 首先验证的是“已有 P1S 候选能否被纯几何推过 IoU 0.50”，不是最终
  detector AP。
- `raw ∪ refined` 只能表示上界；只有 refined-only 或未来无 GT selector
  的结果才接近可部署路径。
- train100 的 60/20/20 可以快速淘汰实现，但由于 P1S 已接触其中 80 场，
  不能替代额外 50 个未用 train 场的正式审计。
- fixed val10 不得用于调参；full100 不得在本协议中自动启动。
- 即使 P1G1 通过，也不能承诺相对 B6 或论文提升 10 AP；它只证明新增几何
  模块具有可泛化的 AP50 候选改善能力。

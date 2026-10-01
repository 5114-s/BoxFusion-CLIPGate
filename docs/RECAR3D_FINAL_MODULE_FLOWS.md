# ReCaR-3D 最终冻结版本模块流程核对

## 1. 文档范围

本文档依据最终锁定目录 `reports/recar3d_final_lock_20260926/` 中的源码快照、配置和实验选择记录整理。对应代码版本为：

- Repository HEAD：`816c796bbafc5b510df84f0f133557698ff72a8d`
- 最终方法：冻结 BoxFusion + PLR-v1 + CALR-v1 + MVSR-v2 Composite-Max
- 候选来源：post-NMS proposals 与 pre-NMS anchors
- NMS-suppressed children：不使用
- 运行方式：严格因果、关键帧增量更新
- 场景末操作：只读取已经在线维护的状态，不重新提取候选，不遍历历史帧
- 跨恢复分支去重：关闭
- 最终记录顺序：Native → PLR → CALR

需要特别区分两件事：

1. official100 筛选程序在同一次在线运行中同时维护 CALR-v2、CALR-v1 控制分支以及多种 MVSR 聚合状态。
2. 最终论文组合从该次因果在线状态中选择 `CALR-v1` 和 `MVSR Composite-Max`。配置文件中的默认 `support_mode: reliability` 不是最终论文采用的 MVSR 模式。

最终选择依据见：

- `reports/recar3d_final_lock_20260926/LOCK.json`
- `reports/recar3d_final_lock_20260926/evidence/SELECTION.md`
- `reports/recar3d_final_lock_20260926/source_snapshot/`

---

## 2. 总体在线流程

原始 RGB-D 序列按照时间顺序读取。ScanNet 实验采用 `gap=25`，主要状态只在关键帧上更新。

每个关键帧的处理流程为：

```text
Posed RGB-D Keyframe
├── Frozen BoxFusion Native Mapper
│   └── Native Map N(t)
└── Frozen WeDetect-Uni
    ├── Post-NMS Proposals
    │   └── Shared Boxer Lifting
    │       └── Proposal Records
    │           ├── PLR
    │           └── MVSR
    └── Top-M Pre-NMS Anchors
        └── Shared Boxer Lifting
            └── Lifted Top-M Anchors
                └── CALR

Native Map + PLR Boxes + CALR Boxes
└── Online 3D Detections
```

辅助证据提取在原生建图推理开始时异步启动。当前关键帧的原生地图准备完成后，系统将 Native Map 与同一关键帧的辅助证据提交给在线状态。

所有模块只使用当前与历史关键帧。序列结束时的 `materialize_terminal` 只将最终有效 Native rows 与已经积累的PLR、CALR和MVSR状态对齐，不执行新的候选推理。

---

## 3. 冻结原生在线建图器

### 3.1 输入与关键帧

原生BoxFusion接收RGB图像、深度、相机内参和camera-to-world位姿。普通原始帧只推进输入流；关键帧进入检测、三维候选生成、关联与融合。

ScanNet最终配置：

- 检测阈值：0.5
- 关键帧间隔：25帧
- Boxer lifting：启用
- 原生3D NMS阈值：0.1
- Reliable-View Top-K：3

### 3.2 原生候选和对象地图

真实代码中的核心过程可概括为：

```text
CuTR 2D detection
→ score / image-boundary / floor filtering
→ frozen Boxer 3D refinement
→ camera-to-world transformation
→ native association and 3D NMS
→ reliable-view selection
→ PFO geometric fusion
→ Native Map N(t)
```

CLIP语义分支保持冻结，用于读取类别语义，不参与PLR、CALR或MVSR的几何决策。

ReCaR-3D不修改原生BoxFusion的候选生成、原生关联、PFO融合或语义分支。原生建图器在关键帧 $t$ 输出：

- 稳定Native ID；
- 世界坐标三维框；
- 原生置信度；
- 当前相机参数。

这些数据构成后续PLR去重和MVSR重排序的原生地图接口。

---

## 4. 辅助候选提取与Boxer提升

### 4.1 WeDetect-Uni输出

冻结WeDetect-Uni每个关键帧只执行一次前向推理，并提供两种二维候选：

- `raw["boxes"] / raw["scores"]`：pre-NMS anchors；
- `raw["post_boxes"] / raw["post_scores"]`：经过二维NMS的proposals。

最终配置中：

- proposal最低分数：0.05；
- 每帧最多保留150个post-NMS proposals；
- 每帧选择分数最高的300个pre-NMS anchors。

### 4.2 共享Boxer Lifting

代码将proposal二维框和Top-M anchor二维框拼成一个batch，调用同一个冻结Boxer lifting过程，然后按照原始长度拆回两个逻辑分支。

因此应区分：

- 逻辑上：Proposal lane与Anchor lane互不混合；
- 实现上：二者共享一次批量Boxer调用和完全相同的冻结参数。

Boxer输出无效、非有限或零尺度三维框后，相应记录会被删除。

每条Proposal Record保存：

- proposal ID；
- 二维区域；
- 世界坐标三维八角点；
- 原始proposal分数；
- 关键帧编号。

其中：

- PLR使用三维框、分数和关键帧编号；
- MVSR同时使用二维区域、三维框和proposal分数。

每条Anchor Record保存：

- anchor ID；
- 世界坐标三维八角点；
- anchor分数；
- 关键帧编号。

CALR只使用Top-M anchor记录。

---

## 5. PLR：Proposal-Level Recovery

### 5.1 输入

最终PLR只接收post-NMS proposals：

```text
use_children = false
```

每帧proposal首先按原始分数降序排列。最终辅助提供器最多产生150个proposal，PLR内部每帧容量上限为192。

### 5.2 在线轨迹关联

每条PLR轨迹按关键帧编号保存观测，因此同一关键帧最多贡献一次跨帧支持。

当前proposal与轨迹的最近观测同时满足以下条件时才允许关联：

- 三维AABB IoU不小于0.10；
- 三维中心距离不大于0.50 m。

若多个轨迹满足门控，依次选择：

1. AABB IoU最大的轨迹；
2. 中心距离最小的轨迹；
3. track ID最小的轨迹。

这是顺序贪心关联，不是全局匈牙利匹配。

同一关键帧中的多个proposal可以依次关联到同一轨迹，但轨迹只保留其中原始分数最高的观测。因此帧内重复响应不会被计为多个关键帧支持。

### 5.3 轨迹状态与过期

每条轨迹最多保存12个不同关键帧观测。超过容量时删除最旧观测。

活动轨迹若连续10个关键帧未获得更新，则从活动轨迹集合中删除。活动轨迹总数上限为1024。

### 5.4 三关键帧确认

当轨迹包含至少3个不同关键帧的观测时，轨迹进入确认候选池。

最终PLR没有额外使用以下确认条件：

- 成对IoU中位数阈值；
- 中心RMS阈值；
- 最小medoid尺寸阈值。

这些规则属于其他历史观察器，不应写入最终PLR方法。

### 5.5 AABB-IoU Medoid

对轨迹当前保留的观测，代码计算两两三维AABB IoU，并选择累计IoU最大的实际观测框作为medoid。

并列时依次选择：

1. proposal分数更高的观测；
2. 时间更早的观测；
3. proposal ID更小的观测。

轨迹确认后不会被冻结。后续兼容proposal仍可进入轨迹，medoid和轨迹平均分数随当前历史状态因果更新。

### 5.6 Native Map覆盖检查

确认候选即使当前与Native Map重叠，仍会保留在有界确认池中。Native Map覆盖只控制候选是否进入当前输出：

- 与任一Native框的三维AABB IoU不小于0.25：当前隐藏；
- 后续Native框消失：候选可以重新进入PLR输出。

因此，Native去重是可逆的当前输出过滤，不是永久删除轨迹。

### 5.7 PLR内部选择与去重

每个关键帧重新构造PLR输出：

1. 删除当前被Native Map覆盖的候选；
2. 按轨迹中proposal的平均原始分数降序排列；
3. 以三维AABB IoU 0.50执行PLR分支内部NMS；
4. 最多保留12个PLR恢复框。

这不是“与永久接纳的PLR框比较”，而是对当前确认候选池重新排序和抑制。后到达的强候选可以替换较早的弱候选。

### 5.8 输出评分

轨迹平均proposal分数只用于有限预算下的候选选择。最终写入检测结果的PLR分数由medoid三维AABB的最大边长决定：

- 小于0.30 m：0.05；
- 小于0.50 m：0.10；
- 小于0.70 m：0.25；
- 小于1.00 m：0.40；
- 其他：0.50。

PLR输出分数不直接复用WeDetect-Uni的原始分数。

---

## 6. CALR：Causal Anchor-Level Recovery

### 6.1 最终版本边界

最终论文采用CALR-v1，即 `OnlineAnchorRecovery`。official100筛选运行中的CALR-v2被实验否决，不属于最终方法。

CALR-v1输入为每个关键帧经过Boxer提升的Top-300 pre-NMS anchors。

### 6.2 世界坐标体素投票

对每个anchor三维框，代码取八个角点的均值作为框中心，并按照0.3 m体素边长计算体素索引：

```text
voxel_key = floor(center / 0.3)
```

落入同一体素的anchor共享一个因果状态。该状态记录：

- 提供支持的不同关键帧集合；
- 累计anchor观测数；
- 最近更新时间；
- 当前代表anchor；
- 当前代表anchor的原始分数。

### 6.3 跨关键帧确认

同一关键帧可以有多个anchor落入同一体素，但关键帧编号通过集合记录，因此一个关键帧只增加一次帧级支持。

当某体素累计获得至少3个不同关键帧支持时，状态从活动集合进入CALR恢复集合。

### 6.4 代表Anchor选择

CALR-v1不计算medoid。代表框是该体素全部历史观测中原始anchor分数最高的实际框。

并列时优先：

1. 更早的关键帧；
2. 更小的anchor ID。

状态进入恢复集合后，后续落入同一体素的更高分anchor仍可修订代表框。

因此图中“Best Anchor”表示最高分实际anchor，不表示均值框或medoid。

### 6.5 有界状态

CALR-v1包含：

- 活动体素上限：4096；
- 恢复体素上限：640。

活动体素超出容量时，优先保留：

1. 不同关键帧支持更多的状态；
2. 最近更新的状态；
3. 代表anchor分数更高的状态。

最终CALR-v1没有轨迹TTL；其有界性由活动体素和恢复体素容量保证。

### 6.6 尾部分数

CALR根据代表anchor的原始分数和稳定身份生成位于以下区间的确定性尾部分数：

```text
0.040001 ≤ score ≤ 0.049999
```

该映射保持原始anchor分数顺序，并使用场景ID、关键帧ID和anchor ID的哈希打破精确分数并列。

### 6.7 最终版本没有执行的操作

最终CALR-v1：

- 不读取Native Map；
- 不与PLR恢复框去重；
- 不执行CALR分支内部NMS；
- 不改变原生BoxFusion状态；
- 不使用未来帧；
- 不使用CALR-v2的可靠性门控、视角槽或负证据。

因此论文不能把Native去重或PLR-CALR跨分支去重写成CALR-v1的组成部分。

---

## 7. MVSR：Geometry-Aware Multi-View Support Reranking

### 7.1 作用范围

MVSR只处理Native Map中的原生框。它不处理PLR或CALR恢复框。

MVSR只更新：

- Native score；
- Native box之间的排序。

MVSR不改变：

- Native三维几何；
- Native ID；
- Native语义标签；
- Native框数量。

### 7.2 原生框投影与可见比例

在每个关键帧，MVSR将当前Native三维框投影到当前二维图像。

投影失败、穿过相机平面或裁剪后区域小于4像素的框不参与当前视图匹配。

可见比例定义为：

```text
visibility = clipped_projected_area / raw_projected_area
```

### 7.3 Greedy 1-to-1 Matching

系统计算投影Native框与当前post-NMS proposals之间的二维IoU。

候选匹配要求二维IoU不小于0.10。所有合法匹配对按照以下顺序排序：

1. 二维IoU降序；
2. Native行号升序；
3. proposal行号升序。

随后执行贪心一对一分配，使每个Native框和每个proposal在当前关键帧最多匹配一次。

### 7.4 单视图复合支持

对匹配的Native框 $i$ 与proposal $p$，单视图支持由四项组成：

- proposal原始分数 $q$；
- Native投影可见比例 $v$；
- 二维匹配IoU $u$；
- 三维几何一致性 $g$。

代码采用四项几何平均：

$$
h_{i,t} = (q_{i,t} v_{i,t} u_{i,t} g_{i,t})^{1/4}.
$$

三维几何一致性由三项算术平均构成：

1. 按Native框对角线归一化的中心距离相似度；
2. 三维边长对数比例相似度；
3. Native框与lifted proposal框之间的三维AABB IoU。

因此MVSR使用的Proposal Record必须同时保留二维区域、lifted三维框和原始分数。

### 7.5 Composite-Max聚合

筛选实验比较了first、mean、max、EMA、diverse-max和可靠性下界。最终选择Composite-Max。

每个Native ID的累计支持为：

```text
S_i(t) = max(S_i(t-1), h_i(t))
```

该状态在每个关键帧增量更新，不需要保存完整场景历史，也不需要场景末重放proposal。

如果同一Native ID的当前几何与上次几何之间三维AABB IoU低于0.20，代码清空该ID已有的MVSR证据，避免将重关联后的不同几何混入同一支持状态。

### 7.6 Native-Logit更新

最终Native分数更新为：

```text
reranked_score = sigmoid(
    logit(native_score) + 2 * max(0, support - 0.5)
)
```

其中：

- 支持阈值为0.50；
- logit更新权重为2.0；
- 支持不超过0.50时，Native分数保持不变；
- 支持超过阈值时，只提高Native分数。

最终Composite-Max不使用beta可靠性下界作为重排序量。可见但未匹配的负证据会被筛选程序记录，但不会降低Max聚合值。

---

## 8. 最终输出组装

最终论文方法的输出由三组记录直接拼接：

```text
Reranked Native Boxes
+ PLR Recovered Proposal Boxes
+ CALR Recovered Anchor Boxes
```

实际记录顺序为：

```text
Native → PLR → CALR
```

最终配置：

- MVSR只重排序Native；
- PLR候选执行Native覆盖过滤和PLR内部NMS；
- CALR-v1不执行Native或PLR去重；
- PLR与CALR之间不执行跨分支去重；
- 恢复框不会反馈到BoxFusion原生关联与PFO融合；
- 恢复框不参与MVSR。

因此最终输出是直接组装，不是再次进行统一三维NMS或几何融合。

---

## 9. 严格在线与场景末读取

### 9.1 在线更新

每个关键帧到达后：

1. 冻结WeDetect-Uni和Boxer提取当前证据；
2. BoxFusion更新当前Native Map；
3. PLR更新proposal轨迹和当前输出；
4. CALR更新anchor体素状态；
5. MVSR更新Native ID的累计最大支持；
6. 系统组装当前时刻输出。

这些步骤只访问当前与历史状态。

### 9.2 异步实现

official100配置使用异步辅助分支：

- Native与辅助候选提取可以并行执行；
- 辅助任务队列容量为2；
- 当前关键帧的Native结果准备完成后才能提交完整状态；
- 状态仍按严格递增的关键帧顺序更新。

异步执行改变计算调度，不改变方法的因果关系。

### 9.3 序列结束

序列结束时只执行：

- 对齐最终有效Native rows；
- 从已积累的MVSR状态读取Composite-Max支持；
- 使用最终Native Map读取PLR当前可见候选；
- 读取CALR-v1恢复集合；
- 拼接最终输出。

不会执行：

- 新的WeDetect-Uni前向；
- 新的Boxer lifting；
- 历史proposal重放；
- 场景级重新关联；
- 未来帧访问。

---

## 10. 论文写作必须保持一致的技术点

论文中可以写：

- 所有感知和几何提升模型保持冻结；
- PLR采用三关键帧proposal轨迹确认和可修订AABB-IoU medoid；
- CALR-v1采用世界坐标体素投票和最高分代表anchor；
- MVSR采用proposal score、visibility、2D IoU和3D consistency的复合支持；
- MVSR使用跨已观测视图最大支持；
- 完整系统严格因果、关键帧在线更新；
- 恢复框与Native结果直接组装。

论文中不能写：

- PLR确认后medoid永久冻结；
- PLR与Native重叠会永久删除轨迹；
- CALR使用medoid；
- CALR-v1执行Native Map去重；
- CALR-v1与PLR执行跨分支去重；
- 最终MVSR使用beta可靠性下界；
- MVSR只使用二维proposal；
- PLR或CALR恢复框再次进入MVSR；
- 两类候选分别运行两套不同Boxer参数；
- 最终系统在场景末重新遍历历史候选；
- 辅助状态在每个原始帧更新。

---

## 11. 最终参数摘要

### Auxiliary Evidence Provider

- Proposal score threshold：0.05
- Top proposals per keyframe：150
- Top pre-NMS anchors per keyframe：300
- Boxer：冻结、共享批量调用

### PLR-v1

- Use children：false
- Minimum distinct keyframes：3
- Match AABB IoU：0.10
- Match center distance：0.50 m
- Track TTL：10 keyframes
- Native dedup IoU：0.25
- PLR self-NMS IoU：0.50
- Maximum active tracks：1024
- Maximum observations per track：12
- Maximum PLR output boxes：12

### CALR-v1

- Voxel size：0.30 m
- Minimum distinct keyframes：3
- Maximum active voxels：4096
- Maximum recovered voxels：640
- Output score range：0.040001–0.049999
- Native dedup：none
- Cross-branch dedup：none

### MVSR-v2 Composite-Max

- 2D matching IoU：0.10
- Exclusive greedy matching：true
- Geometry reset IoU：0.20
- Aggregation：maximum composite support
- Support threshold：0.50
- Native-logit weight：2.0
- State TTL：10 keyframes
- Maximum Native states：4096

---

## 12. 建议用于方法图的短标签

### Auxiliary Extraction

```text
WeDetect-Uni
Proposal Lane: 2D NMS → Proposals → Boxer Lifting
Anchor Lane: Top-M Anchors → Boxer Lifting
```

### PLR

```text
Track Association
→ 3-Keyframe Verify
→ IoU Medoid
→ Map Dedup
→ Recovered Proposal Boxes
```

图注需要说明medoid可被后续观测在线修订，Map Dedup是可逆的当前输出过滤。

### CALR

```text
Voxel Voting
→ 3-Keyframe Consensus
→ Best Anchor
→ Recovered Anchor Boxes
```

图注需要说明Best Anchor指体素状态中原始分数最高的实际anchor。

### MVSR

```text
Projection
→ Greedy 1-to-1 Match
→ Composite Support
→ View-wise Max
→ Native-Logit Reranking
```

### Output

```text
Reranked Native Boxes
+ Recovered Proposal Boxes
+ Recovered Anchor Boxes
→ Online 3D Detections
```


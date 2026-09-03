# P1G：多视角 Occupancy/MSR 几何精修消融协议

## 目标与边界

P1G 的唯一新增模块是：

```text
冻结 B6 输出
→ 冻结 P1S 类别无关残差候选
→ 多视角真实 depth 几何关联
→ Top-K 可靠视角
→ occupancy/MSR 六面局部框精修
→ 仅写 diagnostics
```

它用于判断“P1S 已经找到了目标，但框几何不准”是否是 AP50 的主要瓶颈。
P1G 不新增 proposal，不改 score，不读 RGB/类别/CLIP，也不允许把 refined box
写入正式检测结果。因此 P1G observer 的标准 AP 必须与冻结 B6 输出相同；
observer 阶段出现 AP 变化是安全契约失败，而不是提升。

硬契约如下：

```text
observer_only=true
mutation_enabled=false
applied_count=0
uses_ground_truth=false
class_agnostic=true
raw_candidate_count == refined_candidate_count
```

脚本不会提供 full100 或 active 的自动入口。`fixed_val10` 已被既往实验查看，
仅能在 train-only 配置冻结并通过后做一次迁移检查。

## 数据隔离与运行顺序

默认命令只运行两个 train-only smoke 场景：

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_p1g_dev
bash scripts/run_scannet_p1g_ablation.sh 0,1
```

正式顺序必须是：

| Scope | Scene list | 用途 | 是否允许调参 |
|---|---|---|---|
| `train_smoke2` | `scannetv2_train_p1_smoke2.txt` | 接线和 schema 检查 | 是 |
| `train_fit60` | `scannetv2_train_p1g_fit60.txt` | 粗搜索/失败分析 | 是 |
| `train_cal20` | `scannetv2_train_p1g_cal20.txt` | 选择唯一冻结配置 | 是，之后冻结 |
| `train_audit20` | `scannetv2_train_p1g_audit20.txt` | 一次性模块审计 | 否 |
| `train_fresh50` | `scannetv2_train_p1g_audit50_fresh_v1.txt` | 更严格的一次性审计 | 否 |
| `fixed_val10` | `scannetv2_val_ablation10_even.txt` | 冻结后的迁移检查 | 否 |

所有 train-only scene list 会在运行前与完整 ScanNet validation list 求交；存在
任何交集、重复 ID 或空列表时立即停止。`fixed_val10` 必须显式设置
`BOXFUSION_P1G_ALLOW_TOUCHED_VAL10=1`。不允许根据 audit20、fresh50 或
fixed val10 的结果返回调参；新版本必须换新的预注册 audit 集。

参数开发示例：

```bash
BOXFUSION_P1G_SCOPE=train_fit60 \
BOXFUSION_P1G_RUN_TAG=p1g_fit60_cfg01 \
BOXFUSION_P_CONFIG="$PWD/config/p1g_cfg01.yaml" \
bash scripts/run_scannet_p1g_ablation.sh 0,1
```

配置文件必须由基线配置复制并只修改
`online_refinement.p1_multiview_geometry` 及其局部 MSR 子配置。一次配置对应唯一
run tag；不能覆盖旧 diagnostics。

## 参数冻结规则

第一轮只允许在 `train_fit60` 搜索以下预注册参数：

- `association_iou`：P1S anchor 与多视角观测的几何关联下限；
- `crop_scale`：局部深度裁剪范围；
- `top_k_views` 与 view diversity：可靠视角数量和视角多样性；
- `max_points_per_view`：每视角确定性采样上限；
- MSR 的最小点数、occupancy voxel size、连通域阈值；
- 六面最大位移和尺寸比例安全范围。

推荐先使用少量离散配置，而不是连续扫参：

```text
C0: association=0.10, crop=1.35, K=5, points=768, face_limit=0.18
C1: association=0.05, crop=1.50, K=5, points=768, face_limit=0.25
C2: association=0.05, crop=1.50, K=7, points=1024, face_limit=0.50
```

这只是预注册搜索模板，不代表 C2 一定更好。`train_fit60` 淘汰明显失败的配置，
`train_cal20` 只在剩余配置中选一个，并冻结配置文件 SHA、代码 commit、B6/P1S
checkpoint SHA、scene-list SHA 和全部审计阈值。audit 阶段不允许临时扩大
face limit、降低关联阈值或按场景使用不同参数。

## 离线审计

完成 observer 后运行：

```bash
BOXFUSION_P1G_SCOPE=train_audit20 \
BOXFUSION_P1G_RUN_TAG=p1g_train_audit20_b6_p1s_frozen_v1 \
bash scripts/audit_scannet_p1g.sh
```

审计调用 `tools/report_p1g_geometry.py`。GT 只在此离线步骤读取，用于比较：

- 冻结 B6；
- P1S raw candidates；
- P1G refined-only candidates；
- 每个 parent 在 raw/refined 中使用 GT 选优的不可部署 oracle。

所有门槛都可通过环境变量显式冻结：

| 环境变量 | 默认值 |
|---|---:|
| `BOXFUSION_P1G_THRESHOLDS` | `0.15 0.25 0.50` |
| `BOXFUSION_P1G_MIN_NOVEL_TP50` | audit20 为 `2`，fresh50 为 `5` |
| `BOXFUSION_P1G_MIN_PARENT_TP25_DELTA` | `0` |
| `BOXFUSION_P1G_MAX_SECONDS_PER_SCENE` | `0.18` |
| `BOXFUSION_P1G_MAX_TOTAL_SECONDS_PER_SCENE` | `0.80` |
| `BOXFUSION_P1G_MAX_CANDIDATES_PER_SCENE` | `256` |
| `BOXFUSION_P1G_REQUIRE_GO` | `0` |

`REQUIRE_GO=0` 时 STOP 是正常实验结论，脚本仍返回成功；设为 `1` 时 STOP
返回状态码 3，便于 CI。报告必须以 `refined-only` 为主，GT oracle 只用于定位
失败原因，不能作为可部署精度。

## STOP/GO

`report_p1g_geometry.py` 的冻结生产门槛要求：

1. 所有 diagnostics 通过 observer、schema、provenance、一一对应和有限数检查；
2. refined-only 的 novel TP@0.50 达到冻结下限；
3. refined-only 的 novel TP@0.25 不低于 P1S parent（默认 delta `0`）；
4. P1G mean runtime、P1S+P1G mean runtime和每场候选数不超预算。

scope 级决定为：

- fit/cal/smoke 通过：`EXPLORATORY_PASS_ONLY`；
- audit20 通过：`GO_FRESH50_AUDIT`；
- fresh50 通过：`GO_ONE_SHOT_VAL10_OBSERVER`；
- fixed val10 通过：`GO_DESIGN_P1Q_OBSERVER_ONLY`；
- 任一门槛失败：`STOP_P1G`。

即使 fixed val10 通过，也只允许设计后续无 GT 的 raw/refined 选择器 P1Q；它
不授权 active 或 full100，更不能承诺相对论文提升 10 AP。

## 参数问题还是方法问题

审计应按以下顺序判断，不能看到低 AP 就直接继续调阈值：

1. **数据/实现问题**：schema、scene set、hash、一一对应、有限数或
   `mutation=false/applied_count=0` 失败。先修复实现，当前结果无效。
2. **上游 proposal 上限**：P1S raw 在 B6 未覆盖 GT 上几乎没有
   IoU@0.15/0.25 匹配。局部 refiner 无法创造缺失 proposal，应停止 P1G，
   回到残差 proposal 召回。
3. **参数或内部安全门控问题**：
   `identity-vs-refined oracle` 达到冻结 geometry gate，但 refined-only 未达到；
   说明候选/证据中存在可用几何，当前关联阈值、Top-K、crop、MSR 参数或
   fail-closed gate 没有稳定选出它。只允许在 fit/cal 上做预注册离散消融。
4. **关联/证据/方法问题**：
   oracle 也达不到 novel TP50 和 TP25 非劣门槛。此时即使 GT 选 raw/refined
   都无效，继续微调 score 或小范围阈值没有依据；应停止该方法，检查多视角
   关联、深度证据质量，或更换几何建模方式。
5. **后续选择问题**：refined geometry 的 IoU 上穿明显、oracle 有效，但
   将来无 GT 时无法判断 raw/refined。它属于未来 P1Q 质量选择器问题，不应把
   GT oracle 增益记作 P1G 实际精度。
6. **速度问题**：几何有效但超过预算，是工程/复杂度失败；不能通过离线缓存
   隐藏在线点提取和 MSR 耗时。

可进一步用 face-limit feasibility（例如 `0.18/0.25/0.50/0.75`）做
train-only 离线诊断：

- 小范围失败、放宽范围成功：安全范围参数可能是瓶颈；
- 放宽到 `0.50/0.75` 仍失败：候选关联或几何证据的方法瓶颈；
- feasibility 成功而实际 MSR 失败：MSR/连通域估计方法瓶颈；
- actual refined 跨过 0.50 但最终部署无增益：无 GT 选择/排序瓶颈。

feasibility 只用于解释和冻结 train-only 配置，不能在线读 GT，也不能在
validation 上据此调参。

### 成对因果诊断

下面的命令在同一组纯训练场景、同一 B6/P1S checkpoint 和同一候选顺序上，
依次运行保守配置与宽松配置，再自动区分限制范围、内部门控、父 proposal 上限
和关联/边界估计方法问题：

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_p1g_dev
bash scripts/diagnose_scannet_p1g_msr_train20.sh
```

该脚本只写独立 diagnostics/reports，不写正式检测结果。诊断阈值默认要求至少
5 个 raw-to-refined IoU@0.50 上穿，可通过
`BOXFUSION_P1G_MINIMUM_CROSS_IOU50` 在预注册实验前显式设置。

当前冻结 audit20 的结果为：3961 个相同父候选中，保守/宽松配置分别修改
371/667 个框，但 IoU@0.50 上穿均为 0；宽松配置的退化数由 32 增至 65。
同一候选集合的 GT 六面可达性在 0.18/0.50 位移范围下分别为 28/51 个上穿。
因此自动诊断为
`association_or_boundary_estimation_method_problem`，不支持继续仅调阈值，
也不授权 active 或 validation-100 实验。

## 结果报告

最终至少同时报告：

- B6 正式 AP 与 P1G observer 正式 AP（两者应一致）；
- B6、P1S raw、P1G refined-only、GT oracle 的 novel TP/Recall@0.15/0.25/0.50；
- raw→refined 的 IoU delta 分布、0.50 上穿/下穿、严重伤害率；
- 无点、视角不足、关联失败、MSR fail-closed 等 reason histogram；
- P1G mean/p95、P1S+P1G mean/p95、候选数、GPU 显存；
- 所有 scene/config/checkpoint/code hashes。

若报告结论为 `association_or_evidence_method_problem`，不要继续用同一 audit
集调参；若为 `parameter_or_internal_gate_problem`，也只能回到尚未封存的
fit/cal 集开发新版本。

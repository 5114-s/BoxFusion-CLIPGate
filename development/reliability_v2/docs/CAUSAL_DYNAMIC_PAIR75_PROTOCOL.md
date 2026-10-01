# 新版在线动态分支：冻结三臂配对测试

本轮只测试，不根据测试集结果修正算法或选择阈值。已有 active persistent
也修改了几何和去重，不能当作动态分支 disabled 的输出。旧24场/63场的后处理
M5亦不能作为本次75场的配对基线。

## 三臂

1. `native_off`：现有 CuTR + Boxer active + Top-K3 + real-score 底座，动态分支 disabled。
2. `full_on`：当前完整动态分支，使用其 current 输出作为主结果，persistent仅为辅助诊断。
3. `no_miss_retirement`：相同完整分支，只有 `dynamic_objects.miss_lifecycle_updates=false`。
   关闭漏观测导致的score衰减、休眠与退役，包括遮挡/coast/age路径；不是终端恢复分数。
   保留关联、命中更新、几何覆盖、PFO绕过和容量限制。生命周期变化会连带影响后续再激活和容量占用，
   因此这是整个漏观测更新机制的消融，不是假设其它内部轨迹完全不变的单步反事实。

默认 `miss_lifecycle_updates=true`，旧配置行为不变。本轮无训练、无新模型、无参数扫描。

## 数据与计分

- 同 `data_dyn/manifest.json` 全75个合成移除场景，gap=25、score_thresh=0.5、seed=0。
- 三臂同一场景在同一GPU顺序运行，按场景轮换三臂顺序；两GPU仅并行不同场景。
- 所有模型、几何/NMS/融合阈值和原始输入路径一致，输出路径隔离。
- 75场数据文件覆盖和K_depth/K_rgb兼容矩阵在启动前检查。
- GT仅用于离线评估；manifest中的 removed_ids 不是GT索引，继续使用已核对的
  最大AABB IoU>=0.5规则恢复移除身份。三臂共享同一终态GT。
- 主指标：官方ScanNet坐标转换/IoU/AP，class-agnostic，AP15/25/50，保留全部输出真实score。
- 辅助指标：移除GT旧位残留率，固定score>0.30、IoU>0.25；存活GT相对原生输出的丢失/新增覆盖。
  此覆盖非一对一AP，也不能直接当作身份保持率或误退役因果归因。
- FPS同时记录loop时间和含模型初始化的进程wall时间，按输入帧归一化，并另报实际关键帧Hz。
  不把gap25的输入FPS称为每帧检测速度，不把两个GPU的总吞吐充当单GPU速度。

## 验收与边界

原生off、完整on和消融均完整75场之后自动评估。任何场景失败即停止继续排新任务，
保留失败日志，禁止拿成功子集代替全75场。源码、配置与产物hash记录，拒绝混合已变更源码的续跑。

必须联合阅读AP、Recall、旧位残留与存活GT损失。只减少旧框而大幅降低存活对象召回，
不支持“动态分支整体有效”。本轮能检验合成移除任务上的整体收益，不能单独证明真实持续运动、
移动人群身份保持、逐帧动态AP或开放词汇语义效果。

运行：`bash scripts/run_scannet_causal_dynamic_pair75.sh`。
准备与冻结但不推理：在命令末尾加 `--prepare-only`。
默认目录：`reports/causal_dynamic_pair75_20260908/`；主结果自动写入其 `evaluation/REPORT.md`。

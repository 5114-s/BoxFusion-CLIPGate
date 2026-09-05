# M1/M2/M5 运行时集成设计（demo_tr3d_terminal_active.py 在线化）

## 现状与目标
现状：底座管线在线运行（demo.py），M1/M2/M5 以因果后处理形式消费同一次运行的观测日志
（NMS 事件、WeDetect 缓存、场景末地图）——证据链完整（v9 语义，44.85/40.17/19.89，封存确认）。
目标：单进程端到端演示 + 干净的全系统 FPS 稳态基准。

## 方案 A（推荐）：流式后处理守护进程（不动 demo.py 主管线）
管线保持现状 + 观测器已在产 NMS/kfmap 日志。新增 `tools/m1m5_online.py`：
1. 逐行尾随（tail -f 语义）`*_pvq_nms.jsonl` → 喂 M1 漏斗（receipt 流内维护，TTL/关联/min-views）；
2. WeDetect 每 2 关键帧前向改由该进程调度（或 demo.py 内已有钩子触发）→ 候选提升（Boxer 引擎共享特征缓存）；
3. 场景末（序列 EOF）：v9 终稿（对最终地图去重/medoid/cap/定价）→ M2 支持度（逐框投影，缓存关键帧提案）→ M5 双通道裁决 → 输出。
   - M5 通道② 的深度反投影复用管线已加载的深度帧（每 4 关键帧、8 像素步长子采样，实测开销 <1ms/关键帧）。
优点：零主管线手术、风险隔离、逐场景输出可对照离线版逐位验证。缺点：双进程（演示形态稍弱）。

## 方案 B：demo.py 内联（论文演示形态）
钩子点（已勘察）：
- instances.py nms_3d：现有 nms_observer 位置旁加 child 候选出口（M1b）；
- WeDetect 前向：every-2-keyframes 调度（复用 third_party/WeDetect/wedetect_uni_infer.py，617MB 显存）；
- 关键帧循环末：receipt 更新（M1c）+ 尺寸定价（M1d，查表）；
- 序列末（demo.py 已有 end_time 计时点前）：v9 终稿 + M2 投影 + M5 裁决，写输出 pkl。
工作量约 +200 行、跨 2 文件；验证标准：单场景输出与方案 A 逐位一致。
FPS 影响：WeDetect 摊销 +2.2ms/帧、其余 <1ms → livebench 40.3 基础上预计 ≥36。

## 验证与基准
1. 一致性门：任一方案 5 场景输出 vs 因果重放版逐位一致（corner ≤1e-6，score ≤1e-9）。
2. 稳态 FPS：方案 B 单卡 10 场景（复用 livebench 流程），报中位/p25/min。
3. 显存：nvidia-smi 记录峰值（预期 +~700MB）。

## CA-1M 前置（M5 主场）
- 数据：CA-1M 序列 RGB-D+pose（确认格式与 ScanNet 读法差异：内参文件、深度尺度、pose 约定）；
- 管线：config 新数据集入口（demo.py 的 dataset 路径已参数化）；评测器需 CA-1M GT 适配（或沿用
  类无关贪心 IoU 内部评测 + 官方评测器待定）；
- 预期：M5 动态退役（物体移走=旧位置持续空证）首次有数据可测；M1a 域迁移（WeDetect 室内→CA-1M 场景）；
  常数零调参主张的跨域检验。

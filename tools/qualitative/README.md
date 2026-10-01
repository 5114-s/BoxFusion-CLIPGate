# 第4.5节定性比较图

本流程只使用锁定的原始运行：

`reports/strict_causal_nochild_official100_v2_20260922/`

对应结果为 Base `34.89 / 31.46 / 15.74`、Full
`44.23 / 39.66 / 20.00`。流程拒绝读取早期 scene-end、含 child、当前
PLR-v4 或 reliability-v2 的预测目录。

## 1. 只读资产审计

```bash
cd /data/ZhaoX/BoxFusion
/home/admin1/miniconda3/envs/boxfusion-online/bin/python \
  tools/qualitative/select_qualitative_cases.py \
  --mode audit \
  --output reports/qualitative_strict_online_nochild_original
```

审计将验证：

- official100的八个factorial臂是否齐全；
- Base/M2的native框几何和顺序是否完全相同；
- Full是否能按manifest中的native、PLR和CALR数量无歧义拆分；
- manifest记录的300个历史输入哈希是否仍然一致；
- ScanNet RGB、depth、pose、intrinsics、GT和axis alignment是否存在；
- 原运行指纹是否能够绑定当前源码。

审计只写入新的报告目录，不修改历史预测、诊断、配置或运行日志。

## 2. 终态候选筛选

```bash
/home/admin1/miniconda3/envs/boxfusion-online/bin/python \
  tools/qualitative/select_qualitative_cases.py \
  --mode select \
  --output reports/qualitative_strict_online_nochild_original
```

该步骤根据锁定的终态框选择PLR、CALR、MVSR和真实失败案例。此时
`selected_cases.json`中的案例仍标记为`evidence_verified=false`，不能用于最终绘图。

筛选使用如下互斥条件：

- PLR：native IoU小于0.15、PLR IoU不小于0.25，且同一GT未被CALR覆盖；
- CALR：native和PLR IoU均小于0.15、CALR IoU不小于0.25；
- MVSR：仅比较同一native框，要求几何和数量固定、正确框rank至少前移一位；
- Failure：优先选择MVSR提升的假阳性，其次选择IoU位于`[0.25, 0.50)`的PLR框，
  再次选择与所有GT的IoU均小于0.05的CALR框。

这些阈值用于选择清晰案例，不参与AP计算。

## 3. 缺失历史与最小导出协议

原official100只保存终态预测和聚合诊断，未保存以下内容：

- PLR轨迹中的proposal、二维区域和关键帧编号；
- CALR体素key、anchor成员和支持关键帧；
- MVSR逐帧一对一匹配及历史最大IoU支持。

因此，只允许重跑`selected_cases.json`列出的场景。导出器必须位于独立源码目录和独立
输出目录，不能写入原运行或当前新实验目录。每个导出JSON必须满足：

```json
{
  "schema": "boxfusion.qualitative_selected_scene_evidence.v1",
  "locked_run": "strict_causal_nochild_official100_v2_20260922",
  "scene": "sceneXXXX_XX",
  "future_frames_used": false,
  "terminal_parity": {"base": true, "full": true},
  "frames": [
    {
      "frame_id": 0,
      "rgb_path": "/extra/ZhaoX/scannet_data/scans/.../color/0.jpg",
      "source": "plr",
      "box_xyxy": [0, 0, 10, 10],
      "gt_xyxy": null
    }
  ],
  "voxel_key": null,
  "max_geometric_support": null
}
```

PLR和CALR导出必须包含至少三个不同关键帧。MVSR导出必须记录逐帧贪心一对一匹配，
并给出`max_geometric_support`。`terminal_parity.base/full=true`只有在重新运行的终态框与
历史Base/Full在数量、顺序、几何和分数上完全一致后才能写入。

当前源码聚合指纹与历史run fingerprint不一致，因此导出器还必须：

1. 使用隔离源码副本；
2. 记录副本内所有修改文件的SHA-256；
3. 将导出开关实现为只读observer，禁止改变关联、候选池、评分和输出；
4. 对每个场景进行终态哈希/数值双重校验；
5. 任一场景无法复现时立即停止，不得将近似轨迹冒充原运行证据。

导出完成后，将每个案例的JSON路径和SHA-256写入`selected_cases.json`，并把
`evidence_verified`改为`true`。

## 4. 绘图

```bash
MPLCONFIGDIR=/tmp/boxfusion_qualitative_mpl \
/home/admin1/miniconda3/envs/boxfusion-online/bin/python \
  tools/qualitative/render_qualitative_figure.py \
  --cases reports/qualitative_strict_online_nochild_original/selected_cases.json \
  --output reports/qualitative_strict_online_nochild_original
```

渲染器采用fail-closed策略。只要缺少证据JSON、SHA-256、三关键帧、因果标志或
Base/Full终态一致性证明，就拒绝绘图。

成功后生成：

- `cases/plr_recovery.png`；
- `cases/calr_recovery.png`；
- `cases/mvsr_reranking.png`；
- `cases/failure.png`；
- `qualitative_comparison.{png,pdf,svg,tiff}`；
- `caption_zh.txt`和`caption_en.txt`。

颜色固定为：native蓝色`#1677FF`、PLR橙色`#FF7A1A`、CALR黄色
`#F2C300`、GT绿色虚线`#14B866`、失败红色`#D62728`。点云只用于可视化，
不作为方法输入。

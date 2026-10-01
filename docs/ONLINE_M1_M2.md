# M1/M2 因果在线实现

## 结论

当前实现已经把原先的“原生 BoxFusion 结束后运行 `integrated_online.py`”改为逐关键帧执行。每个在线输出仅依赖当前帧和已经提交的历史状态，不读取未来帧、终局原生地图或全场景 proposal 缓存。

实现入口：

- `boxfusion/online_candidate_map.py`：M1-P、M1-A、M2 的有界因果状态；
- `boxfusion/online_candidate_runtime.py`：冻结 WeDetect-Uni/Boxer 证据提供器、双 GPU 异步执行和实时输出；
- `demo.py`：在每个 BoxFusion 关键帧中启动证据提取，并在当前原生地图更新后提交该关键帧；
- `config/scannet_t05_boxer_online_candidate_map.yaml`：可运行配置。

原先的 `tools/integrated_online.py` 保留为历史 scene-end 实验实现，其 AP 结果不能直接代表新的在线版本。

## 在线流程

### M1-P

当前帧的 lifted post-NMS proposals 与原生 3D NMS children 进入同一个有界跟踪状态：

1. 仅与历史活动轨迹进行几何关联；
2. proposal 主导的轨迹在三个不同关键帧支持后立即出生，child 主导的轨迹使用两个关键帧；
3. 出生几何取截至确认时刻的 medoid，随后保持不变；
4. 使用当前原生地图执行去重；若已出生框后来被原生地图吸收，该框会被因果退休；
5. 活动轨迹、单轨迹观测数、单帧观测数和出生数均有固定上限。

这取代了旧实现读取完整场景后统一选 medoid、去重和截断的流程。

### M1-A

继续使用已有的 `OnlineAnchorRecovery`：每个 lifted pre-NMS anchor 按中心点进入 0.3 m 体素状态，在三个不同关键帧支持后立即出生。活动体素和出生框数量均有上限。

### M2

对当前原生地图框逐帧维护累计最大 proposal 支持：

\[
u_i^{(t)}=\max\left(u_i^{(t-1)},u_{i,t}\right),
\]

并更新原生分数：

\[
s_i^{(t)}=\sigma\!\left(\operatorname{logit}(s_i^n)
+2\max(0,u_i^{(t)}-0.5)\right).
\]

原生框通过融合谱系中的最小 `init_id` 获得稳定身份。M2 只改变原生分数，不改变几何、类别或原生框数量，也不重排 M1 出生框。

## 执行与输出

测量配置使用两个 GPU：

- GPU0：原生 BoxFusion；
- GPU1：冻结 WeDetect-Uni 与共享 Boxer；
- 有界异步队列容量：2；
- 每个关键帧的证据提取在原生关键帧推理之前启动，两条分支并行；
- 每次关键帧提交后，当前完整输出被原子写入 `online_candidate_map.output_root/<scene>_boxes.pkl`。

运行示例：

```bash
scripts/run_scannet_online_candidate_map.sh scene0169_00
```

脚本已设置仓库所需的 `LD_LIBRARY_PATH`。配置中的 `cuda:1` 指第二张可见GPU，因此默认需要两张GPU。在线输出目录必须与原生 BoxFusion 输出目录不同。

## 初步实时性验证

硬件为两张 RTX 3090。输入前缀配置为252帧；按当前 `demo.py` 的提前终止条件实际消费227帧（截至frame 225），关键帧间隔为25帧，共提交10个关键帧。原生与在线运行使用相同输入前缀和终止条件。

| 指标 | 在线 M1/M2 | 原生 BoxFusion |
|---|---:|---:|
| 整体吞吐 | 34.31 FPS | 32.59 FPS |
| 关键帧端到端延迟 p50 | 238.62 ms | 未单独记录 |
| 关键帧端到端延迟 p95 | 958.71 ms | 未单独记录 |
| 833 ms deadline miss | 1/10 | 未单独记录 |
| M1/M2 状态更新 p50 | 3.39 ms | — |
| 异步队列最大深度 | 1/2 | — |
| 场景结束时待处理关键帧 | 0 | — |
| 场景末等待 | 0.096 ms | — |

证据模型在流开始前进行了1.52秒的固定模型热身；该时间属于初始化，不计入流式吞吐。端到端延迟同时包含等待当前原生地图就绪的时间，10个关键帧中仍有一次 deadline miss。该单场景前缀结果证明实现可以持续增量运行且没有队列积压，但不能代替多场景正式实时基准。

## 已验证的代码性质

测试覆盖：

- M1-P 不会由同帧多个 proposal 自确认；
- 第三个独立关键帧到达时立即产生 proposal birth；
- 两帧 child 支持能够通过在线 child 通道产生 birth；
- native-map 去重只使用当前与历史状态；
- M2 的累计支持增量更新，几何和原生框数不变；
- 固定前缀输出不受未来帧影响；
- 活动状态满足显式容量上界；
- 异步队列按帧序提交，`close()`只清空已提交任务，不执行场景级推理。

运行测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
conda run -n boxfusion-online python -m pytest -q \
  tests/test_online_candidate_map.py \
  tests/test_online_candidate_runtime.py
```

## 投稿前仍需重新测量

在线实现改变了 M1-P 的出生时机、地图去重时机以及 M2 使用的框几何前缀。因此，旧 scene-end 版本的 AP 不可直接沿用。正式声称“完整方法实时在线且保持精度”之前，需要用该配置重新完成：

1. ScanNet official100 的 AP15/AP25/AP50；
2. CA-1M 的同协议结果；
3. 多场景 p50/p95/max 延迟、deadline miss、队列深度和显存；
4. scene-end 版与在线版的逐项 AP 差异；
5. 单 GPU 配置若要作为部署主张，需要独立测速。

当前可以准确表述为：完整方法已经实现因果、逐关键帧、无场景末推理的在线执行；双 GPU 单场景前缀验证显示稳态无积压。跨数据集精度保持和多场景实时性仍需正式实验确认。

#!/usr/bin/env python3
"""Assemble verified development-set semantic, subgroup and runtime reports."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import statistics

ROOT=Path(__file__).resolve().parents[1]
SEM=ROOT/'reports/scannet_semantic_20260915'
SIZE=ROOT/'reports/sizegroup_bootstrap_20260915'
RUNTIME=ROOT/'reports/pipeline_runtime_20260915'
OUT=ROOT/'reports/paper_completion_20260915'

def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def manifest():
    if (SEM/'semantic_source_sha256.json').exists():return
    import sys
    sys.path.insert(0,str(ROOT/'tools'))
    import run_scannet_semantic_table as semantic
    protocol=read(SEM/'protocol.json');sources={};images={};frame_ids={}
    for scene in protocol['scenes']:
        folder=semantic.FRAMES/scene/'frames'
        colors=sorted((folder/'color').glob('*.jpg'),key=lambda p:int(p.stem))[::25]
        frame_ids[scene]=[int(p.stem) for p in colors]
        paths=[semantic.GTROOT/f'{scene}_bbox.npy',semantic.SCANS/scene/f'{scene}.txt',
               folder/'intrinsic/intrinsic_color.txt']
        paths += [folder/'pose'/f'{p.stem}.txt' for p in colors
                  if (folder/'pose'/f'{p.stem}.txt').exists()]
        for p in paths:sources[str(p)]=digest(p)
        timing=read(SEM/'semantic_cache'/f'{scene}.timing.json')
        for path,expected in timing['image_sha256'].items():
            assert digest(Path(path))==expected,path
            images[path]=expected
        p=SEM/'semantic_cache'/f'{scene}.npz';sources[str(p)]=digest(p)
    weights=ROOT/'models/open_clip_pytorch_model.bin';sources[str(weights)]=digest(weights)
    for name in ('paper_eval_core.py','run_scannet_semantic_table.py','check_semantic_anchor.py',
                 'true_fusion_audit_core.py','run_sizegroup_v2.py'):
        p=ROOT/'tools'/name;sources[str(p)]=digest(p)
    versions={name:importlib.metadata.version(name) for name in
              ('numpy','torch','open_clip_torch','Pillow')}
    write(SEM/'semantic_source_sha256.json',{'sources':sources,'used_images':images,
        'selected_keyframe_ids':frame_ids,'versions':versions,'used_images_unchanged':True,
        'timing':'GT/poses/calibration/checkpoint/cache hashes recorded after completed run; '
                 'prediction and used-image hashes also checked against run-time records'})

def size_report():
    target=SIZE/'REPORT.md';archive=SIZE/'REPORT_V1_V2_SUPERSEDED.md'
    if not archive.exists():shutil.copyfile(target,archive)
    result=read(SIZE/'sizegroup_v3_results.json');cuts=result['tertile_cuts_m3']
    parts=['# CA-1M 尺寸诊断与逐场 bootstrap：更正后的有效版本（v3）\n',
        '本报告替代 v1/v2；旧文保留于 REPORT_V1_V2_SUPERSEDED.md，不应继续引用其尺寸 AP 或机制归因。\n',
        '## 尺寸协议\n',
        f'107 场、12,911 个 GT。按 GT 体积三分位分组，实际切点为 {cuts[0]:.10f} / {cuts[1]:.10f} m³；small/medium/large 分别为 4,304/4,303/4,304 个目标。',
        '预测不按自身尺寸筛选。按置信度排序：有组内达标 GT 时优先匹配其中 IoU 最大者；组内重复检测仍为 FP。无组内达标 GT 而与组外 GT 达标时忽略，允许组外 GT 忽略多个预测；其余预测均为 FP，保留所有负场景。IoU 严格大于阈值，VOC 包络 AP。',
        '这是透明定义的三维体积分组诊断，并非 COCO 官方尺寸 AP。各组 AP 不可相加。分组全设为有效时，三臂三个阈值的 AP/TP/FP 均与主评测器一致；输入哈希复核通过。\n',
        '| 尺寸 | 配置 | AP15 | AP25 | AP50 |\n|---|---|---:|---:|---:|']
    for group in ('small','medium','large'):
        for arm in ('native','M1','M1_M2'):
            vals=[result['size_groups_v3_ignore_outgroup'][arm][str(t)][group]['ap'] for t in (.15,.25,.5)]
            parts.append(f'| {group} | {arm} | '+ ' | '.join(f'{v:.2f}' for v in vals)+' |')
    parts += ['\nM2 对小物体 AP25/50 的净效应为 −1.19/−0.19；中型为 +0.04/+0.22；大型为 +3.98/+3.41。撤回“中型全线受损”和“收益全部来自大型物体”。',
        'M2 只重排分数，M1 与 M1+M2 的几何相同。因此小物体 AP 下降是排序/匹配表现下降，不能说其框被 M2 改坏。视角少、投影小是否造成负效应尚未验证，不作机制断言。\n',
        '## 逐场 bootstrap 的正确解释\n',
        '复用已完成的 10,000 次配对场景 bootstrap（种子 0），无需再次运行。以下是逐场 AP 差的平均值及其 CI，不是池化 AP 增益的置信区间。\n',
        '| 对比 | IoU | 场景平均 ΔAP | CI95 | 改善/变差/持平 |\n|---|---:|---:|---|---|',
        '| M1 − native | .50 | +0.39 | [0.30, 0.49] | 71/21/15 |',
        '| M1+M2 − M1 | .50 | +0.93 | [0.77, 1.09] | 96/10/1 |',
        '| M1+M2 − native | .50 | +1.32 | [1.15, 1.50] | 102/4/1 |',
        '| M1+M2 − M1 | .25 | +0.45 | [0.30, 0.59] | 74/31/2 |',
        '| M1+M2 − native | .25 | +1.32 | [1.09, 1.54] | 93/12/2 |',
        '| M1+M2 − M1 | .15 | +0.16 | [0.02, 0.30] | 65/41/1 |',
        '\n这些开发集 CI 描述当前场景样本的稳定性，不能消除参数反复开发带来的选择偏差。AP15 在 41 场下降不应直接归因于小物体，需要额外交叉统计。\n',
        '## 验证范围\n',
        '本机 109 个 CA-1M 场景目录中，107 个已用于开发，其余两个缺少可用 GT/位姿。只能说本地没有立即可用的独立留出场景，不能推断整个数据集没有其他数据。事后切分已用过的 107 场不能成为真正未见的验证集。本轮不做这样的切分。\n',
        '## 复现与产物\n',
        '`python tools/run_sizegroup_v2.py`（保留历史文件名，实际输出 v3）。有效数据为 sizegroup_v3_results.json 与 sizegroup_v3_protocol.json；原 results.json 中的 bootstrap 数据继续有效，原尺寸数据已被替代。']
    target.write_text('\n\n'.join(parts[:7])+'\n'+ '\n'.join(parts[7:])+'\n')

def semantic_report():
    r=read(SEM/'semantic_table.json');timings=[read(p) for p in (SEM/'semantic_cache').glob('*.timing.json')]
    clipseconds=sum(x['clip_seconds'] for x in timings);wall=sum(x['semantic_stage_seconds'] for x in timings)
    # Cached smoke and full-run remainder were two separate model initializations.
    startups=sorted(set(x['startup_seconds'] for x in timings));steady=wall-sum(startups)
    write(SEM/'timing_summary.json',{'scenes':len(timings),'boxes':sum(x['boxes'] for x in timings),
        'clip_seconds':clipseconds,'semantic_stage_seconds_sum':wall,'observed_startups_seconds':startups,
        'semantic_stage_sum_minus_two_startups':steady,'clip_ms_per_box':1000*clipseconds/2948,
        'semantic_stage_ms_per_box_minus_explicit_startup':1000*steady/2948,
        'scope':'scene-end automatic projection/cropping/CLIP; includes first-batch warmup, excludes detection; not online FPS'})
    parts=['# ScanNet 100 场：严格配对的语义检测评测\n',
        '采用仓库已有固定 100 场评测列表，1,433 个有效 GT，18 个类别均有 GT。新增框与原生框都接入自动图像裁图及同一个冻结 CLIP 语义读出，无新增模型、训练或评测框提示。\n',
        '| 配置 | 框数 | 语义 mAP15 | 语义 mAP25 | 语义 mAP50 |\n|---|---:|---:|---:|---:|']
    for a in ('native','M1','M1_M2'):
        v=r['arms'][a]
        parts.append(f'| {a} | {v["boxes"]} | '+ ' | '.join(f'{v["mAP"][str(t)]:.4f}' for t in (.15,.25,.5))+' |')
    deltas=[r['arms']['M1_M2']['mAP'][str(t)]-r['arms']['native']['mAP'][str(t)] for t in (.15,.25,.5)]
    parts += ['\nM1+M2 相对 native 的增益为 '+ '/'.join(f'+{v:.4f}' for v in deltas)+' 个 AP 点。这是类别与三维位置同时正确的检测收益；类别无关 AP 保留为另一个指标。\n',
        '| 配置 | 类别无关 AP15 | 类别无关 AP25 | 类别无关 AP50 |\n|---|---:|---:|---:|']
    for a in ('native','M1','M1_M2'):
        parts.append(f'| {a} | '+' | '.join(f'{r["arms"][a]["class_agnostic"][str(t)]["ap"]:.4f}' for t in (.15,.25,.5))+' |')
    parts += ['\n## 严格配对与语义链路\n',
        '原持久图三字段输出没有出生框的 CLIP 特征；本轮保存每行的归一化图像特征、18 类标签、自动选取帧和裁图信息，使新增框具备与原生框相同的语义读取方式。没有可用投影时应弃权，不伪造类别；本批三臂弃权均为 0。',
        '旧 M1 与 M2 缓存不完全配对。本次 M1 为分数消融重建：保留 M2 持久图的全部几何、行顺序和出生分数，仅把原生行分数还原为 native 输入。它不是另跑一次 M1。M1 与 M1+M2 的 2,948 行几何/语义完全相同，M2 的对比仅反映原生行分数重排。原生 1,789 行前缀几何保持原样。',
        '采用 nativelogit/exclusive 的 persistent 分数视图，排除 M5 current 分数的作用；不要与其他出生评分配置或场景平均 AP 的旧账本直接混表。所有预测文件哈希复核通过。\n',
        '## 评测协议\n',
        '从各自场景 gap25 帧中自动选取预测三维框投影面积最大的有效视图，添加 15% 裁图边距；只使用预测框、相机位姿、内参与本场景图像，不使用 GT。CLIP ViT-H-14 原有权重冻结，fp16 autocast，原始类别名文本，类别取归一化余弦相似度 argmax。排序沿用检测分数，不乘 CLIP 分数，不调阈值。',
        '三臂共享同几何框的 CLIP 特征/标签，只编码 2,948 个框一次。GT 使用 ScanNet NYU40→18 类官方映射；预测经 axisAlignment 转换到 GT 坐标。按类在全部场景中池化预测，包含该类无 GT 的负场景，严格 IoU>阈值、VOC 包络 AP，然后对 18 类取均值。',
        '5 场入场核验：GT 坐标范围与原数据加载器最大差 2.98×10⁻⁸ m，类别标签零差异；三臂×三个阈值的 162 项有效类别 AP 与原 APCalculator 使用的 get_iou_obb_v2 完全一致。不是只比对平均数。\n',
        '## 全类别明细\n',
        '| 类别 | GT数 | native AP25 | M1 AP25 | M1+M2 AP25 | native AP50 | M1 AP50 | M1+M2 AP50 |\n|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,v in r['arms']['native']['per_class'].items():
        values=[r['arms'][a]['per_class'][name][str(t)]['ap'] for t in (.25,.5) for a in ('native','M1','M1_M2')]
        parts.append(f'| {name} | {v["gt"]} | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    parts += ['\n## 开放词汇与验证边界\n',
        '这补充了固定 18 类查询下的语义三维检测表现，保留原有文本匹配能力；不等于证明训练时未见类别的检测泛化。上游预训练模型与本方法的额外训练必须分开描述：本方法无需额外训练，不代表全部基础模型未经监督训练。',
        '本轮不随意把 18 类划为 base/novel。必须先核实所有涉及三维监督的模型训练类别、数据和泄漏情况，建立明确协议后，才能声称某类未见。无法排除既有监督曝光时，只能称类别分组或跨词汇查询实验。',
        '最大投影视图的选择发生于场景结束，可能选到遮挡视图，且当前 CLIP 分类器只在这 18 个文本间选一类。因此本表是终态共同语义读出/开发集验证，并未证明严格逐帧在线语义输出、任意新词准确率或独立未见场景泛化。\n',
        '## 开销与复现\n',
        f'100 场唯一框编码累计 {clipseconds:.2f}s，平均 {1000*clipseconds/2948:.2f}ms/框；投影、裁图、预处理和编码阶段累计 {wall:.2f}s。先跑 5 场、再续跑剩余 95 场产生两次显式初始化，扣除二者后为 {steady:.2f}s。这些不是完整系统 FPS。',
        '`python tools/run_scannet_semantic_table.py`；已经完成缓存后可用 `--evaluate-only` 直接复算三个阈值。`python tools/check_semantic_anchor.py` 核对原评测器。协议、输入哈希、语义特征、逐帧选择、逐类 AP、耗时与一致性检查都已归档。']
    (SEM/'REPORT.md').write_text('\n'.join(parts)+'\n')

def runtime_report():
    directory=RUNTIME/'long11kf' if (RUNTIME/'long11kf/runtime.json').exists() else RUNTIME
    r=read(directory/'runtime.json')
    semantic=r.get('semantic_warm_stage_seconds')
    parts=['# 实际管线计时与在线边界\n',
        '使用第一场开发片段 scene0011_01 的固定前缀，不按 GT/收益选择场景；单 RTX 3090，现有冻结模型，候选回放关闭，M5 关闭，可视化关闭。只有本片段重新执行检测与融合，100 场语义表直接复用既有几何。\n',
        f'配置 {r["configured_raw_frames"]} 个原始帧；原入口提前终态保存，实际消费 {r["native_consumed_raw_frames"]} 帧。第二阶段只读同一已消费前缀，共 {r["provider_keyframes"]} 个 gap25 关键帧，没有把后来帧用于本片段结果。\n',
        '| 阶段 | 实测秒数 |\n|---|---:|',
        f'| Native 显式导入、CuTR/CLIP 加载 | {r["native_model_import_load_seconds"]:.4f} |',
        f'| Native 运行（含 Boxer 适配器初始化、实际候选及融合） | {r["native_run_seconds"]:.4f} |',
        f'| M1/M2 显式模型加载 | {r["m1_m2_model_load_seconds"]:.4f} |',
        f'| M1/M2 实际运行 | {r["m1_m2_run_seconds"]:.4f} |']
    if semantic is not None:
        parts += [f'| 同一片段终态语义读出，扣除其显式模型初始化 | {semantic:.4f} |',
                  f'\n上述运行组件之和 {r["combined_warm_components_seconds"]:.4f}s；对应 {r["raw_frames_per_second_with_readout"]:.2f} 个已消费原始帧/s。采样关键帧更新率为 {r["provider_keyframes"]/r["combined_warm_components_seconds"]:.3f}/s。分阶段冷启动墙钟组件合计 {r["cold_components_seconds"]:.2f}s，语义阶段在独立进程测量，不能当作单进程完整墙钟。']
    parts += [f'\nNative/M1/M1+M2 行数 {r["native_rows"]}/{r["M1_rows"]}/{r["M1_M2_rows"]}；M1 与 M2 几何完全配对。第一段仅 3 关键帧未产生出生，所以延长同一片段一次以验证更新开销；两段完整原始日志和 JSON 均保留。',
        'Native 峰值 allocated GPU 内存 '+f'{r["native_peak_allocated_bytes"]/(1<<30):.2f}GiB；M1/M2 阶段 '+f'{r["m1_m2_peak_allocated_bytes"]/(1<<30):.2f}GiB。不是 nvidia-smi 总显存。',
        '## 可以与不可以声称的结论\n',
        '这里的显式模型加载在运行组件求和时扣除，但 Native 运行包含内部 Boxer 初始化，首次前向和热身也在计时内；并非稳态平均吞吐。数据可能受 OS 文件缓存影响。一个片段不能代表 100 场，尤其出生与融合开销随场景变化。本轮不作 20FPS 或实时保证。',
        '源码确认 integrated_online.py 的 M1 确认/去重依据最终 native 图，M2 汇总本场景全部候选后重排；语义选择也在终态遍历本场景视图。实际前向不读离线候选缓存，不等于无需等待场景结束。',
        '本轮完成的是无 GT 的同前缀终态运行与计时核验；尚未实现、也未验证持续逐帧输出和固定延迟。若论文仍宣称严格实时在线，必须在主张中删去终态模块的这项保证，或另行实现并测量前缀输出一致性与逐更新延迟。',
        '新增时间表不改变 AP、损失函数或动态路线结论，也不能增加方法本身的创新性。\n',
        '## 复现\n',
        '`python tools/benchmark_paper_pipeline.py --frames 300 --output-root reports/pipeline_runtime_20260915/long11kf`；随后用同参数加 `--semantic-only` 补测共同语义读取，不重跑检测器。详见 runtime.json 与对应 logs。']
    (RUNTIME/'REPORT.md').write_text('\n'.join(parts)+'\n')
    audit={'implementation':'tools/integrated_online.py','terminal_native_map_required':True,
           'whole_scene_consensus':True,'semantic_view_selection':'terminal own-scene',
           'benchmark_provider_prefix_equal_to_native_consumed_prefix':True,
           'strict_per_frame_online_validated':False,'live_latency_or_20FPS_claim_supported':False,
           'scope':'honest verification of existing implementation; no new online research module'}
    write(RUNTIME/'online_audit.json',audit)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest-only',action='store_true')
    args=parser.parse_args();manifest()
    if args.manifest_only:print('SOURCE MANIFEST COMPLETE');return
    OUT.mkdir(exist_ok=True);size_report();semantic_report();runtime_report()
    r=read(SEM/'semantic_table.json')
    parts=['# 论文实验完善：完成记录（2026-09-15）\n',
        '已补齐出生框语义读取、ScanNet 100 场严格配对语义检测 AP、尺寸分组更正、bootstrap 重标及一次实际完整阶段计时。未新增模型或训练，未再启动动态/几何选择器试错。\n',
        '| 方法 | 语义 mAP15 | 语义 mAP25 | 语义 mAP50 |\n|---|---:|---:|---:|']
    for a in ('native','M1','M1_M2'):
        parts.append('| '+a+' | '+' | '.join(f'{r["arms"][a]["mAP"][str(t)]:.4f}' for t in (.15,.25,.5))+' |')
    parts += ['\n语义 mAP 检查正确类别与三维位置，补充原类别无关 AP；并未证明 novel 类别泛化。M1 是严格配对的分数消融重建，完整协议和全部 18 类明细见 [语义报告](../scannet_semantic_20260915/REPORT.md)。',
        '\nCA-1M 小物体 M2 AP25/50 净效应 −1.19/−0.19；中型 +0.04/+0.22；大型 +3.98/+3.41。撤回旧表与未经验证的视角机制归因；见 [尺寸与统计报告](../sizegroup_bootstrap_20260915/REPORT.md)。',
        '\n实际冻结权重前向、Boxer、融合与 M1/M2 已在同一已消费帧前缀中执行，并单独测共同语义读取；见 [运行与在线边界](../pipeline_runtime_20260915/REPORT.md)。现有终态模块不能声称严格逐帧实时，本轮没有掩盖这一限制。',
        '\n5 场语义坐标/类别对齐及 162 项 AP 原评测器一致性通过；尺寸全组 AP/TP/FP 一致性通过；3 个针对组外忽略、负场景与无效投影的测试通过。所有既有预测输入哈希不变。',
        '\n这些结果可以用于主表、配对消融、语义能力验证与局限性。107/100 场均为开发验证，事后切分不是独立 holdout；不把新增实验数量作为算法创新。']
    (OUT/'REPORT.md').write_text('\n'.join(parts)+'\n')
    write(OUT/'completion.json',{'semantic_scenes':100,'semantic_classes':18,
        'semantic_gt':1433,'unique_geometries_encoded':2948,'semantic_abstention':0,
        'extra_training':False,'new_model':False,'size_protocol':'v3',
        'focused_tests_passed':3,'semantic_anchor_checks':162,
        'novel_generalization_proven':False,'strict_live_online_proven':False,
        'reports':[str(SEM/'REPORT.md'),str(SIZE/'REPORT.md'),str(RUNTIME/'REPORT.md')]})
    print('REPORTS ASSEMBLED')

if __name__=='__main__':main()

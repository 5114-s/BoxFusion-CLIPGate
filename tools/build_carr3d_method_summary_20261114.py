#!/usr/bin/env python3
"""Create a methods-only PDF for the selected CaRR-3D paper configuration."""
from __future__ import annotations

import json
from pathlib import Path

import build_carr3d_method_drawing_brief_20261114 as render

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/carr3d_method_summary_20261114'
OUT.mkdir(parents=True, exist_ok=True)
PDF = OUT / 'CaRR-3D_方法总结.pdf'
PAGES = [
    {
        'title': 'CaRR-3D 方法总结',
        'subtitle': '01 / 模型定位、输入与基础组件',
        'blocks': [
            ('p', 'CaRR-3D 面向有位姿 RGB-D 序列的无重建在线三维目标检测。模型在原生对象建图之外，利用独立冻结辅助检测器产生的候选历史完成两类操作：把候选转化为补充检测，以及利用历史支持调整已有对象的置信度排序。模型不需要构建稠密场景模型，也不训练新的恢复或重排序网络。'),
            ('small', '本总结对应论文选定版本：PTR-v1、VAR-v1，以及不包含 visibility 因子、采用历史最大值聚合的 HSR。代码旧称 PLR、CALR、MVSR/HCSR 分别对应 PTR、VAR、HSR。'),
            ('h', '1. 输入与时间组织', render.GREEN),
            ('p', '输入为 RGB 图像、深度图、相机内参和相机位姿。系统按照观测的原始时间顺序处理，每 25 帧选取一个关键帧。当前关键帧同时提供给原生路径和辅助候选路径，提升后的三维框统一表示在世界坐标系中。历史状态仅使用已经到达的关键帧证据。'),
            ('h', '2. 原生建图：检测、关联与多视图几何优化', render.BLUE),
            ('steps', [
                '候选生成与几何提升：CuTR 产生当前帧候选；论文选定的受控原生路线启用 Boxer，对选定候选执行三维几何提升。原生路径还可维护 CLIP 视觉特征，用于相应的语义读出。',
                '对象关联：通过三维 NMS 与二维观测对应，将当前候选与已有对象记录关联，维护对象身份及其历史观测。',
                '可靠视图与 PFO：综合观测分数、投影面积、二维投影一致性和三维几何一致性，选取最多 3 条可靠历史观测。PFO 采用粒子滤波式随机优化，通过多视图投影 IoU 评价几何扰动并更新三维框；已核对配置使用 20 次迭代。',
            ]),
            ('h', '3. 共享辅助前端：WeDetect-Uni 与 Boxer', render.MUTED),
            ('p', '冻结 WeDetect-Uni 对当前关键帧前向一次，保留同一候选源的两层响应。Proposal 路使用 NMS 后分数不低于 0.05 的候选，最多 150 条；anchor 路按 NMS 前原始分数选择 Top-300。两组二维框组成一批，经共享的辅助 Boxer 提升后按来源拆分，并剔除无效三维几何。'),
            ('p', '提升后的 proposal 保留二维区域、世界坐标三维框、原始分数和来源身份，供 PTR 与 HSR 共同使用；anchor 的三维几何、分数与来源身份供 VAR 使用。这里的辅助候选来自独立的 WeDetect-Uni 路径，并不是直接取自原生 CuTR 被筛除的候选。'),
            ('table', ['模块', '处理对象', '主要作用'], [
                ['原生建图', '已经接纳的对象及其观测', '维护对象身份与三维几何'],
                ['PTR', 'Post-NMS proposal 历史', '轨迹确认后读出补充检测'],
                ['VAR', 'Pre-NMS anchor 历史', '体素确认后扩展低分覆盖'],
                ['HSR', '原生身份与匹配的 proposal', '保留历史支持，仅更新原生分数'],
            ], [87, 200, 224]),
        ],
    },
    {
        'title': '候选恢复：PTR 与 VAR',
        'subtitle': '02 / 历史候选的组织、确认、代表框选择与读出',
        'blocks': [
            ('h', '4. PTR：Proposal Track Recovery', render.ORANGE),
            ('p', 'PTR 把不同关键帧中空间一致的 proposal 组织成轨迹，在获得足够支持后生成恢复记录，并排除当前原生输出已经覆盖的候选。'),
            ('steps', [
                '轨迹关联：将新 proposal 与各轨迹最近观测比较，要求三维 AABB IoU 至少为 0.10，且中心距离不超过 0.50 m。在满足条件的轨迹中优先选择重叠较大、中心较近者；若没有合适轨迹，则在容量允许时建立新轨迹。',
                '三关键帧确认：一条轨迹至少获得 3 个不同关键帧支持才被确认。同一关键帧在同一轨迹中只保留一条优先观测，不会因同帧有多个候选而反复增加支持次数。',
                'Medoid 代表：选择与轨迹内其他观测的三维 AABB 重叠总和最大的一条观测作为代表框。该框是一个真实候选观测，不是所有框的参数均值，也不必是最高分观测。',
                '读出筛选：移除被当前原生读出覆盖的候选，按轨迹平均 proposal 分数排序，执行 PTR 分支内部三维 AABB NMS，最多输出 12 条记录。覆盖阈值为 0.25，内部 NMS 阈值为 0.50。确认后的代表记录仍可随后续观测更新。',
            ]),
            ('p', 'PTR 的选择分数与最终输出分数不同：轨迹平均分决定候选优先级，最终置信度按代表框的最大轴向尺寸分桶。尺寸区间 [0, 0.30)、[0.30, 0.50)、[0.50, 0.70)、[0.70, 1.00)、[1.00, +∞) m 分别对应 0.05、0.10、0.25、0.40、0.50。该评分是固定读出规则，不是经过概率校准的正确率。', 9.3),
            ('h', '5. VAR：Voxelized Anchor Recovery', render.GOLD),
            ('p', 'VAR 不逐个维护对象轨迹，而是利用世界坐标体素组织弱 anchor 响应。其作用是把经时序确认的 anchor 转化为低分补充检测。'),
            ('steps', [
                '体素组织：按中心位置将 anchor 分配到边长为 0.30 m 的世界坐标体素。体素化只用于聚合候选中心，不改变原始三维框的尺寸或方向。',
                '跨关键帧确认：同一体素得到至少 3 个不同关键帧支持后，产生一条输出记录。同一帧的多个 anchor 至多贡献一次帧级支持；三个关键帧不要求连续。',
                '最高分代表：维护该体素内原始分数最高的 anchor 作为代表。后续出现更高分观测时更新代表几何与分数，不对多个 anchor 做几何平均或融合。',
                '尾部分数：通过确定性保序映射，把代表 anchor 的分数压缩到 [0.040001, 0.049999]。映射保留不同原始分数之间的顺序，同分时使用确定性规则破同。',
            ]),
            ('note', '两类恢复的区别', 'PTR 使用对象轨迹、medoid、原生覆盖检查和分支内 NMS。当前 VAR-v1 使用体素共识与最高分代表，不读取原生地图，不执行自身 NMS 或与 PTR 的去重。因此，VAR 是低分候选覆盖扩展，而不是显式判断每条输出是否为原生漏检。'),
        ],
    },
    {
        'title': '历史支持重排序：HSR',
        'subtitle': '03 / 对已有原生对象保留历史证据，非递归地更新分数',
        'blocks': [
            ('p', 'HSR（Historical Support Reranking）处理已经存在的原生对象记录。它将当前辅助 proposal 与原生对象对应起来，并沿持续身份维护历史支持。HSR 不新增三维框，也不改变原生记录的身份或几何。'),
            ('h', '6. 当前关键帧的匹配与支持计算', render.PURPLE),
            ('steps', [
                '原生框投影：使用当前相机位姿与内参，把原生三维框投影到图像平面，保留具有有效投影的记录。',
                '一对一匹配：计算原生投影矩形与 post-NMS proposal 二维框之间的 IoU，将满足 0.10 阈值的对应按 IoU 从高到低处理，贪心建立一对一匹配。',
                '复合支持：对于匹配成功的记录，综合 proposal 原始分数、二维投影 IoU 与三维几何一致性，以三类信号的几何平均形成视图级支持。最终选定版本不包含投影可见比例 visibility 因子。',
            ]),
            ('table', ['支持信号', '具体衡量什么'], [
                ['Proposal 原始分数', '当前辅助候选自身的质量信号'],
                ['二维投影 IoU', '原生框投影与 proposal 二维区域是否一致'],
                ['三维几何一致性', '原生框与提升后 proposal 的中心、尺度及 AABB 重叠是否一致；三个几何子项取算术平均'],
            ], [151, 360]),
            ('h', '7. 身份条件化的历史最大支持', render.PURPLE),
            ('p', '每个持续原生身份保存一个历史最大支持值。新关键帧产生更强支持时才提高该值；当前没有匹配时，当前支持为零，但已经保存的最大值不会仅因此降低。历史最大值可能由单个历史关键帧决定，因此它是历史最佳支持的保留，而不是至少两个视图共同确认。'),
            ('p', '为避免旧支持被错误沿用，同一身份相邻状态之间的三维 AABB IoU 低于 0.20 时，会先清除旧证据，再处理当前观测。历史最大值仅在未失效、未发生重置的身份状态内延续。'),
            ('h', '8. 原生分数的非递归修正', render.PURPLE),
            ('p', 'HSR 先将当前原生分数裁剪到 [0.0001, 0.9999]，再转换到 logit 空间。历史支持超过 0.50 时，按系数 2.0 加入正向修正，随后映回分数。每次使用当前原生分数，而不是上一关键帧已经被 HSR 提升过的分数，因此不会仅因为重复读出而不断累加提升。'),
            ('note', 'HSR 的输出与作用边界', '输出是同一批原生框及更新后的分数；排名随分数改变，几何保持不变。HSR 只使用 post-NMS proposal 证据，不使用 anchor，也不对 PTR/VAR 恢复框执行同样的修正。高历史支持仍可能提升假阳性，因此支持强度不等于检测正确性。'),
            ('small', '原生 PFO 与 HSR 的职责不同：PFO利用多视图观测更新三维几何；HSR在原生几何既定的条件下更新置信度。HSR也不会将更新后的分数反馈为下一次原生建图的评分依据。'),
        ],
    },
    {
        'title': '输出、状态管理与语义读出',
        'subtitle': '04 / 完整方法的组合方式与实现范围',
        'blocks': [
            ('h', '9. 输出组装与公共读出', render.GREEN),
            ('p', '最终检测由三部分直接拼接：HSR 重评分的原生记录、PTR 恢复记录、VAR 补充记录。输出组装不再执行跨分支关联、几何融合、统一三维 NMS 或去重；恢复框不写回原生建图器，也不进入原生 PFO。'),
            ('p', '内部原生地图与用于评测的原生读出需要区分。已审计的 ScanNet 工作点要求原生 AABB 三轴尺寸均不小于 0.30 m。PTR 在状态更新时维护覆盖状态，在公共读出时针对过滤后的原生集合重新检查覆盖、排序和 NMS；HSR按对应身份读取分数；VAR独立读出。这是已有状态上的读取，不是重新推理或回放历史。'),
            ('h', '10. 因果更新与状态容量', render.GREEN),
            ('p', '原生与辅助证据按同一关键帧编号对应，只有两者均可用后才提交该关键帧的状态更新。异步任务依次处理，序列结束时排空待处理任务。相邻关键帧之间保留最近已提交状态的输出；各前缀与终态比较采用一致的公共读出规则。'),
            ('table', ['状态', '容量或时间规则', '含义'], [
                ['PTR 活动轨迹', '最多 1024 条；每轨迹最多 12 条不同帧观测', '轨迹超过 10 个关键帧未更新则失效'],
                ['PTR 确认与读出', '确认候选容量 1024；当前读出最多 12 条', '确认候选不等同于当前可见输出'],
                ['VAR 活动体素', '最多 4096 个', '超限按支持数、近期性、代表分数等确定性规则淘汰'],
                ['VAR 确认记录', '最多 640 条', '满额后拒绝新确认记录；已有记录仍可更新'],
                ['HSR 身份状态', '容量 4096；缺席身份 TTL 为 10 个关键帧', '几何变化可提前触发证据重置'],
            ], [114, 213, 184]),
            ('small', '这里给出的是已选定配置中的状态条目容量，不等于证明所有日志、帧索引集合或调试缓存都具有固定字节内存上界。'),
            ('h', '11. 类别语义读出', render.GREEN),
            ('p', 'PTR、VAR 与 HSR 的更新本身是类别无关的，不使用文本查询或 CLIP 类别预测决定候选确认。原生路径可以维护 CLIP 视觉特征；现有统一终态语义评测则对输出三维框选择有效投影最大的关键帧区域，每侧增加 15% 裁剪边距，再用冻结 CLIP 视觉与文本编码器计算相似度并获得标签。该语义读出不反向改变恢复和重排序状态。'),
            ('p', '因此，当前实现可以将类别无关检测结果接入冻结的开放词汇语义读出，但不能把现有终态读出自动解释为所有恢复框逐关键帧完成在线语义分类，也不能假定恢复框已直接继承代表候选的 CLIP 特征。'),
            ('note', '方法的一句话概括', '原生建图负责持续维护正式对象；PTR 与 VAR把辅助候选历史读成补充检测；HSR把与已有对象关联的历史候选支持用于分数修正。三者联合扩展检测覆盖并调整原生排序，但不改变各自的职责。'),
            ('small', '说明依据：原生/辅助运行接口、OnlineProposalRecovery、OnlineAnchorRecovery、OnlineNativeReliabilityReranker、公共输出组装与终态语义评测代码，以及已保存的最终路线有效配置。本文档只总结方法，不新增或修改算法、实验结果。'),
        ],
    },
]


def header(c, page, index):
    c.setFillColor(render.colors.HexColor(render.PURPLE))
    c.rect(render.MARGIN, render.H-51, 32, 3, stroke=0, fill=1)
    render.put(c, page['title'], render.MARGIN, render.H-63, render.CW,
               20, 28, bold=True)
    render.put(c, page['subtitle'], render.MARGIN, render.H-100,
               render.CW, 9, color=render.MUTED)
    c.setStrokeColor(render.colors.HexColor(render.GRID)); c.setLineWidth(.6)
    c.line(render.MARGIN, 37, render.W-render.MARGIN, 37)
    render.put(c, 'CaRR-3D · 方法总结 · PTR / VAR / HSR', render.MARGIN,
               28, render.CW-50, 7.6, color=render.MUTED)
    render.put(c, f'{index:02d} / {len(PAGES):02d}', render.W-render.MARGIN-45,
               28, 45, 7.8, color=render.MUTED, align=render.TA_CENTER)
    return render.H-126


def main():
    c=render.canvas.Canvas(str(PDF), pagesize=(render.W,render.H), pageCompression=1)
    c.setTitle('CaRR-3D 方法总结')
    c.setAuthor('CaRR-3D Research Notes')
    c.setSubject('Methods only: native mapping, frozen candidates, PTR, VAR, HSR, readout')
    bottoms=[]
    for index,page in enumerate(PAGES,1):
        y=header(c,page,index)
        for b in page['blocks']:
            kind=b[0]
            if kind=='p':
                size=b[2] if len(b)>2 else 10
                y-=render.put(c,b[1],render.MARGIN,y,render.CW,size,size*1.52)+9
            elif kind=='small':
                y-=render.put(c,b[1],render.MARGIN,y,render.CW,8.3,12.5,render.MUTED)+9
            elif kind=='h':
                y-=3
                y-=render.put(c,b[1],render.MARGIN,y,render.CW,11.5,16.5,b[2],True)+7
            elif kind=='steps':
                for n,text in enumerate(b[1],1):
                    render.put(c,f'{n:02d}',render.MARGIN,y,25,8.5,color=render.PURPLE,bold=True)
                    y-=render.put(c,text,render.MARGIN+27,y,render.CW-27,9.7,14.6)+7
            elif kind=='table':
                y=render.table_draw(c,y,*b[1:])
            elif kind=='note':
                p=render.paragraph(b[2],9.5,14.4)
                _,ph=p.wrap(render.CW-22,render.H)
                height=ph+34
                c.setFillColor(render.colors.HexColor('#F1F5FA'))
                c.roundRect(render.MARGIN,y-height,render.CW,height,5,fill=1,stroke=0)
                render.put(c,b[1],render.MARGIN+11,y-8,render.CW-22,9.8,14,render.PURPLE,True)
                p.drawOn(c,render.MARGIN+11,y-height+8)
                y-=height+11
            if y<49:
                raise RuntimeError(f'Page {index} overflows: bottom={y:.2f}, block={kind}')
        bottoms.append(round(y,2))
        c.showPage()
    c.save()
    if render.MISSING_GLYPHS:
        raise RuntimeError(f'Missing glyphs: {render.MISSING_GLYPHS}')
    manifest={'pdf':str(PDF),'pages':len(PAGES),'page_bottoms_pt':bottoms,
              'scope':'PTR-v1 / VAR-v1 / HSR max + no_visibility',
              'methods_only':True,'new_inference':False,'gpu_used':False}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

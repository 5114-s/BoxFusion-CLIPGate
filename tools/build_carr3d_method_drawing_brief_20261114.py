#!/usr/bin/env python3
"""Build a code-grounded Chinese method/drawing brief. CPU-only document work."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
from xml.sax.saxutils import escape

sys.path.append('/usr/lib/python3/dist-packages')
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from fontTools.ttLib import TTFont as FontReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/carr3d_method_drawing_brief_20261114'
OUT.mkdir(parents=True, exist_ok=True)
FONT_CN = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
FONT_EN = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
for name, path in [('CN', FONT_CN), ('EN', FONT_EN), ('ENB', FONT_BOLD)]:
    pdfmetrics.registerFont(TTFont(name, path))
CMAP_CN = FontReader(FONT_CN).getBestCmap()
CMAP_EN = FontReader(FONT_EN).getBestCmap()
W, H = 595.276, 841.890
MARGIN = 42
CW = W - 2*MARGIN
INK = '#172B45'
MUTED = '#586A7D'
BLUE = '#2879B9'
ORANGE = '#E77C2A'
GOLD = '#B88608'
PURPLE = '#8051BC'
GREEN = '#258B68'
GRID = '#D7E0E9'
DATE = '2026-11-14'
OVERFLOWS = []
MISSING_GLYPHS = set()


def markup(text: str, bold: bool = False) -> str:
    result = []
    group = ''
    selected = None
    for ch in text:
        if ch == '\n':
            if group:
                result.append(f'<font name="{selected}">{escape(group)}</font>')
            group, selected = '', None
            result.append('<br/>')
            continue
        if ord(ch) in CMAP_EN:
            name = 'ENB' if bold else 'EN'
        elif ord(ch) in CMAP_CN:
            name = 'CN'
        else:
            MISSING_GLYPHS.add(ch)
            name = 'CN'
        if selected != name and group:
            result.append(f'<font name="{selected}">{escape(group)}</font>')
            group = ''
        selected = name
        group += ch
    if group:
        result.append(f'<font name="{selected}">{escape(group)}</font>')
    return ''.join(result)


def paragraph(text, size=10, leading=None, color=INK, bold=False, align=TA_LEFT):
    style = ParagraphStyle('p', fontName='CN', fontSize=size,
                           leading=leading or size*1.55,
                           textColor=colors.HexColor(color),
                           alignment=align, wordWrap='CJK',
                           spaceBefore=0, spaceAfter=0)
    return Paragraph(markup(text, bold), style)


def put(c, text, x, top, width, size=10, leading=None, color=INK,
        bold=False, align=TA_LEFT):
    p = paragraph(text, size, leading, color, bold, align)
    _, height = p.wrap(width, H)
    p.drawOn(c, x, top-height)
    return height


def arrow(c, points, color, dashed=False, line=1.3):
    c.saveState()
    c.setStrokeColor(colors.HexColor(color))
    c.setFillColor(colors.HexColor(color))
    c.setLineWidth(line)
    if dashed:
        c.setDash(3.3, 2.3)
    p = c.beginPath()
    p.moveTo(*points[0])
    for point in points[1:]:
        p.lineTo(*point)
    c.drawPath(p)
    c.setDash()
    x, y = points[-1]
    px, py = points[-2]
    import math
    a = math.atan2(y-py, x-px)
    length, half = 5.2, 2.6
    p = c.beginPath()
    p.moveTo(x, y)
    p.lineTo(x-length*math.cos(a)+half*math.sin(a),
             y-length*math.sin(a)-half*math.cos(a))
    p.lineTo(x-length*math.cos(a)-half*math.sin(a),
             y-length*math.sin(a)+half*math.cos(a))
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    c.restoreState()


def node(c, x, y, w, h, title, sub, color, fill='#FFFFFF', title_size=9.5, sub_size=7.4):
    c.setFillColor(colors.HexColor(fill))
    c.setStrokeColor(colors.HexColor(color))
    c.setLineWidth(1.0)
    c.roundRect(x, y, w, h, 5, fill=1, stroke=1)
    put(c, title, x+5, y+h-7, w-10, title_size, color=color, bold=True, align=TA_CENTER)
    if sub:
        put(c, sub, x+5, y+h-23, w-10, sub_size, leading=sub_size*1.35,
            color=INK, align=TA_CENTER)


def diagram(c, x, y, width=CW):
    """Vector-only wiring diagram; no experimental images are synthesized."""
    c.saveState()
    c.translate(x, y)
    c.scale(width/511.0, width/511.0)
    node(c, 171, 276, 169, 31, 'Posed RGB-D stream', '', GREEN, '#F1FAF6', 10)
    node(c, 11, 199, 190, 53, 'Native Mapper',
         'CuTR / Boxer → association → PFO\nMap state + native readout', BLUE, '#F2F8FE')
    node(c, 278, 199, 222, 53, 'Frozen Auxiliary Front-end',
         'WeDetect-Uni → pre/post-NMS candidates\nShared Boxer lifting', MUTED, '#F6F8FA')
    node(c, 280, 133, 102, 43, 'Proposals', '2D + 3D + score', ORANGE, '#FFF8F2', 9)
    node(c, 403, 133, 97, 43, 'Anchors', '3D + raw score', GOLD, '#FFFBEE', 9)
    node(c, 11, 55, 145, 56, 'HSR', 'Match → historical max\nNative scores only', PURPLE, '#F8F4FD', 11, 8)
    node(c, 191, 55, 145, 56, 'PTR', 'Track → confirm → medoid\nCoverage check + NMS', ORANGE, '#FFF8F2', 11, 8)
    node(c, 365, 55, 135, 56, 'VAR', 'Voxel consensus\nHighest-score exemplar', GOLD, '#FFFBEE', 11, 8)
    node(c, 191, 0, 145, 32, 'Concatenate outputs', '', GREEN, '#F1FAF6', 10)
    arrow(c, [(204,276),(204,265),(106,265),(106,252)], MUTED)
    arrow(c, [(308,276),(308,265),(389,265),(389,252)], MUTED)
    arrow(c, [(331,199),(331,176)], ORANGE)
    arrow(c, [(451,199),(451,176)], GOLD)
    arrow(c, [(83,199),(83,111)], BLUE)
    put(c, 'IDs / boxes / scores', 18, 189, 122, 6.7, color=BLUE)
    # Native coverage check: leave a bridge gap at the purple crossing.
    c.setStrokeColor(colors.HexColor(BLUE)); c.setLineWidth(1.15); c.setDash(3.3, 2.3)
    c.line(186,199,186,159); c.setDash()
    arrow(c, [(186,151),(186,121),(228,121),(228,111)], BLUE, True, 1.15)
    put(c, 'coverage', 189, 194, 59, 6.7, color=BLUE)
    arrow(c, [(280,155),(133,155),(133,111)], PURPLE)
    put(c, '2D + 3D support', 195, 170, 83, 6.7, color=PURPLE)
    arrow(c, [(330,133),(330,121),(294,121),(294,111)], ORANGE)
    arrow(c, [(450,133),(450,111)], GOLD)
    arrow(c, [(83,55),(83,16),(191,16)], PURPLE)
    arrow(c, [(263,55),(263,32)], ORANGE)
    arrow(c, [(433,55),(433,16),(336,16)], GOLD)
    c.restoreState()


PAGES = [
    {
        'title': 'CaRR-3D 方法与流程图绘制说明',
        'subtitle': '01 / 一页总览 · 先确定数据依赖，再安排图形和场景素材',
        'blocks': [
            ('p', '用途：用于绘制论文总流程图和模块细节图，不是新增实验报告。本版对应已选定的 PTR-v1、VAR-v1 与无 visibility 的历史最大支持 HSR；不把仓库中其他探索分支或消融变体混入主模型。'),
            ('diagram',),
            ('small', '上图为数据依赖草图。蓝色虚线表示原生覆盖检查；紫色箭头来自提升后的 proposal 记录。两线交叉处无连接。状态历史回路可在正式细节图中展开。'),
            ('table', ['模块', '通俗作用', '可直接放到图中的关键词'], [
                ['Native Mapper', '维护正式对象及其几何', 'Association · Reliable views · PFO'],
                ['Auxiliary\nFront-end', '产生并提升两层辅助候选', 'WeDetect-Uni · Shared Boxer'],
                ['PTR', '先把 proposal 连成轨迹，再确认和读出', 'Track & confirm · Medoid · Filter'],
                ['VAR', '把 anchor 放入空间格子，确认后取代表', 'Voxel consensus · Top-score anchor'],
                ['HSR', '记住已有关联对象的历史支持，只改分数', 'Proposal match · Running max · Rerank'],
            ], [93, 170, 248]),
            ('note', '命名统一', '论文用 CaRR-3D / PTR / VAR / HSR。代码仍常用 ReCaR-3D、PLR（m1p）、CALR（m1a）、MVSR/HCSR（m2）；这里是名称对应，不代表更换算法。'),
        ],
    },
    {
        'title': '输入、原生建图与共享辅助前端',
        'subtitle': '02 / 哪些是基础组件，哪些数据会进入你的三个模块',
        'blocks': [
            ('h', 'A. 输入与关键帧', GREEN),
            ('p', '输入包括 RGB、深度、相机内参和相机位姿。按原始时间顺序处理，已核对配置每 25 帧取一个关键帧。原生路径和辅助路径使用同一关键帧，三维候选统一到世界坐标。主图输入建议写 Posed RGB-D stream；不要画成先重建稠密场景再做检测。'),
            ('h', 'B. Native Mapper：维护原生对象', BLUE),
            ('steps', [
                'CuTR 产生当前帧候选；论文选定的受控原生路线启用 Boxer，对选定候选执行三维几何提升。原版 BoxFusion 的配置不能与该受控路线混为一谈。',
                '通过三维 NMS 与二维观测对应，将当前检测同已有对象记录关联，维护对象身份与历史观测。',
                '综合观测分数、投影面积、二维投影一致性和三维几何一致性，选择最多 3 条可靠历史观测。PFO（粒子滤波式随机优化）在当前几何附近提出扰动，用多视图投影 IoU 评价并迭代优化三维框；本配置为 20 次迭代。这是几何优化，不是 HSR 的分数重排序。',
                '输出内部原生地图及读出记录。原生路径可以维护 CLIP 视觉特征，但 CLIP 不是 PTR、VAR 或 HSR 的状态更新依据。',
            ]),
            ('h', 'C. Auxiliary Front-end：一次候选提取，共享几何提升', MUTED),
            ('steps', [
                '冻结 WeDetect-Uni 在当前 RGB 上前向一次，获取同一候选源的 NMS 前与 NMS 后响应，不是从 CuTR 被筛掉的候选中直接取数。',
                'Proposal 路：保留 post-NMS 分数不低于 0.05 的候选，最多 150 条。Anchor 路：按 pre-NMS 原始分数选 Top-300。',
                '两组二维框拼成一批，送入共享的辅助 Boxer 提升，再按来源拆回 proposal 与 anchor。利用深度、内参和位姿获得世界坐标三维框，剔除无效几何。',
            ]),
            ('table', ['记录', '保留的关键字段', '下游'], [
                ['Native record', '持续身份、三维框、原生分数；相机参数随当前帧提供', 'HSR；PTR 覆盖检查'],
                ['Lifted proposal', '二维区域、世界三维框、原始分数、候选 ID；关键帧 ID 随批次提供', 'PTR 与 HSR'],
                ['Lifted anchor', '世界三维框、原始分数、anchor ID；关键帧 ID 随批次提供', 'VAR'],
            ], [96, 305, 110]),
            ('note', '绘图建议', '把 WeDetect-Uni 与 Boxer 放在共享前端大框内，不在 PTR/VAR 内各画一个独立检测器。主图可画一个 Shared Boxer；两个 Boxer 小图标仅能表示两路逻辑提升，不能暗示两套独立辅助模型。'),
        ],
    },
    {
        'title': '候选恢复：PTR 与 VAR',
        'subtitle': '03 / 两种组织方式、两种代表框规则、两套输出评分',
        'blocks': [
            ('h', 'PTR — Proposal Track Recovery', ORANGE),
            ('p', '输入：提升后的 post-NMS proposal 历史，以及用于当前覆盖检查的原生几何。输出：当前读出的恢复三维框；不写回原生建图器。'),
            ('steps', [
                '轨迹关联：把新 proposal 与轨迹最近观测比较，要求三维 AABB IoU 至少 0.10，且中心距离不超过 0.50 m；优先选择重叠较大、距离较近的轨迹。没有合适轨迹时建立新轨迹。',
                '跨关键帧确认：同一轨迹至少由 3 个不同关键帧支持。每帧每条轨迹只保留一个优先观测，同一帧的密集候选不能反复增加支持次数。',
                '代表框：选取与轨迹内其他观测总体 AABB 重叠最大的一条观测，即 IoU medoid；不是平均三维框，也不是必然选择分数最高框。',
                '当前读出：排除被当前原生读出覆盖的代表框，按轨迹平均 proposal 分数排序，执行 PTR 内部 AABB NMS，最多读出 12 条。覆盖状态会随原生输出变化重新检查。',
            ]),
            ('small', '对应阈值：原生覆盖 IoU 0.25；PTR 内部 NMS IoU 0.50。轨迹平均分用于选择优先级，最终置信度另按代表框最大轴向尺寸分桶。'),
            ('small', 'PTR 分桶规则：最大轴向尺寸依次位于 [0, 0.30)、[0.30, 0.50)、[0.50, 0.70)、[0.70, 1.00)、[1.00, +∞) m 时，输出分数依次为 0.05、0.10、0.25、0.40、0.50。这不是平均 proposal 分数，也不是校准概率。'),
            ('h', 'VAR — Voxelized Anchor Recovery', GOLD),
            ('p', '输入：提升后的 pre-NMS Top-300 anchors。输出：确认体素的当前代表框，以低分尾部评分加入输出。当前选定的 VAR-v1 不读取原生地图，也不做自身 NMS 或与 PTR 的去重。'),
            ('steps', [
                '世界体素：把 anchor 中心分配到边长 0.30 m 的固定世界坐标格子；体素只组织中心位置，不改变候选框尺寸或方向。',
                '体素确认：同一体素至少获得 3 个不同关键帧支持；同帧多个 anchor 仅贡献一次帧级支持。不要求三个关键帧连续。',
                '代表更新：在该体素维护历史原始分数最高的 anchor。后续出现更高分观测时，更新代表几何和分数；不是对所有框做几何融合。',
                '尾部输出：代表分数经确定性保序映射压缩至 [0.040001, 0.049999]，使 VAR 位于低分尾部。确认输出容量最多 640；达到上限后新体素确认被拒绝，已接纳体素仍可更新。',
            ]),
            ('table', ['需要区分', 'PTR', 'VAR'], [
                ['组织 / 代表', '对象轨迹 / IoU medoid', '世界体素 / 最高原始分 anchor'],
                ['选择与输出分数', '按轨迹平均分选择；按尺度分桶赋分', '帧内 Top-300 与代表按原始分；输出尾部映射'],
                ['原生覆盖 / 去重', '有覆盖检查；有分支内 NMS', '不查原生覆盖；无分支内/跨分支去重'],
            ], [97, 207, 207]),
            ('small', 'Recovery 表示把候选转成补充输出；不能保证每条记录都对应新目标或真实漏检。三关键帧确认是筛选规则，不等于消除全部误检与重复。'),
        ],
    },
    {
        'title': '历史支持重排序：HSR',
        'subtitle': '04 / 用历史证据改变原生分数，不改变原生框集合和几何',
        'blocks': [
            ('note', '最短解释', '当前视角不好时，HSR仍可利用这个原生对象曾获得的较强辅助支持。它不新增目标，不移动三维框，也不替 PTR/VAR 的恢复框重排序。'),
            ('h', '完整处理顺序', PURPLE),
            ('steps', [
                '读取原生记录：获得当前原生身份、三维几何与未被 HSR 修正的原生分数；同时读取当前提升后的 proposal 记录。',
                '二维投影与匹配：将原生三维框投影到当前相机，在有效投影与 post-NMS proposal 二维框之间，按二维 IoU 从高到低做贪心一对一匹配，阈值为 0.10。',
                '形成当前支持：综合 proposal 原始分数、二维投影 IoU、三维几何一致性。几何一致性包含中心、尺度与 AABB 重叠；三类支持信号做几何平均。最终版本不乘 visibility 因子。',
                '沿身份更新历史：对每个持续原生身份保留历史最大支持。当前没有匹配时，当前支持为零，但已经保存的最大值不会仅因此降低。它不要求两个或三个视图共同确认。',
                '几何变化重置：同一身份相邻状态的三维 AABB IoU 低于 0.20 时，清除旧支持，避免把旧几何的支持直接转移给发生明显变化的新几何。',
                '分数读出：支持超过 0.50 才对原生 logit 加正向修正，系数为 2.0，随后映回分数。每次以当前原生分数为基准，不把上一时刻已提升分数继续累计。',
            ]),
            ('table', ['HSR 需要的数据', '原因', '主图箭头标签'], [
                ['原生 ID、框、分数', '按身份保存支持；投影、重置、重评分', 'Native records'],
                ['Proposal 二维区域', '计算二维 IoU 与一对一对应', '2D boxes'],
                ['Proposal 世界三维框与分数', '计算几何一致性及证据质量', 'Lifted proposal records'],
            ], [153, 228, 130]),
            ('h', '为什么必须从“提升后的 proposal”引出箭头？', PURPLE),
            ('p', '只有二维框不能计算三维几何一致性，所以 HSR 的证据箭头应来自已包含二维区域、三维框与分数的 proposal 记录。它不应来自 PTR 的最终恢复框，也不能仅画成 WeDetect 的二维分数输出。'),
            ('h', '为什么没有 anchor → HSR？', PURPLE),
            ('p', '当前实现用 post-NMS proposal 做匹配与支持计算，没有读取 anchor 作为 HSR 证据。这是选定接口，不是“anchor 在理论上不能重排序”。图中不要为了对称性补上不存在的连接。'),
            ('small', '推荐细节图：Project & match → Composite support → Running maximum → Native-score update。旁注：Fixed native identities and geometry；No visibility factor；No recursive boosting。'),
        ],
    },
    {
        'title': '可直接交给绘图者的箭头与标签',
        'subtitle': '05 / 总流程图保留依赖关系；阈值和公式移到细节图或正文',
        'blocks': [
            ('table', ['从哪里', '到哪里', '箭头代表什么'], [
                ['Posed RGB-D', 'Native Mapper；Auxiliary Front-end', '同一个当前关键帧观测'],
                ['Native Mapper', 'HSR', '持续 ID、原生框、原生分数'],
                ['WeDetect-Uni', '共享辅助 Boxer', 'Post-NMS proposals 与 pre-NMS Top-300'],
                ['Shared Boxer', 'Lifted proposals；Lifted anchors', '世界坐标三维几何；保留来源与分数'],
                ['Lifted proposals', 'PTR', '跨关键帧 proposal 观测'],
                ['Lifted proposals', 'HSR', '二维区域 + 三维几何 + proposal 分数'],
                ['Lifted anchors', 'VAR', 'Anchor 中心、几何与原始分数'],
                ['原生地图 / 读出', 'PTR', '当前覆盖检查；可用蓝色虚线'],
                ['HSR；PTR；VAR', '输出组装', '重评分原生记录；两类恢复记录'],
                ['各模块上一状态', '该模块当前状态', '因果历史；用局部虚线回路表示'],
            ], [140, 151, 220]),
            ('h', '主图推荐：输入加五个功能区域', INK),
            ('p', '左侧输入；上方 Native Mapper；下方 Frozen Auxiliary Candidates；中部 Candidate Recovery（内含 PTR 与 VAR）；右上 HSR、右下 3D Detection Output。共享前端必须位于 PTR、VAR 和 HSR 的依赖上游，不属于恢复模块开关本身。'),
            ('table', ['模块', '主图中保留', '留到细节图 / 正文'], [
                ['Native Mapper', 'Candidate → Associate → PFO → Map', '可靠视图权重、PFO迭代、过滤阈值'],
                ['PTR', 'Track & confirm → Medoid → Filter', '轨迹距离、计数、分桶评分、容量规则'],
                ['VAR', 'Voxel consensus → Top-score anchor', '体素尺寸、容量、确认数、尾部映射'],
                ['HSR', 'Proposal match → Running max → Rerank', '三项支持定义、重置、logit 参数'],
                ['Output', 'Concatenate → 3D detections', '公共读出协议、类别语义读出'],
            ], [94, 233, 184]),
            ('note', '不要画出的箭头', '恢复框 → PFO / 原生地图；VAR → 原生覆盖检查；anchor → HSR；PTR 的恢复框 → HSR；最终统一三维 NMS。上述连接均不属于本版选定路线。'),
            ('small', '颜色建议：原生蓝、PTR橙、VAR金、HSR紫、输入/输出绿。历史状态回路用虚线；数据流用实线。框的颜色表示来源，不自动表示正确性。真实场景缩略图只作可视化，不能让读者误以为网格是模型输入。'),
        ],
    },
    {
        'title': '输出协议、消融解释与代码依据',
        'subtitle': '06 / 画图时必须保留的边界，以及本版说明的核验来源',
        'blocks': [
            ('h', '输出不是融合；内部地图不等于最终读出', GREEN),
            ('p', '内部原生地图与评测读出应区分。已审计的 ScanNet 协议要求原生 AABB 三轴尺寸均不小于 0.30 m。PTR 在状态更新时维护覆盖状态，在公共读出时针对经过过滤的原生集合重新检查覆盖；HSR按对应身份读出分数。VAR独立读出。最后直接拼接三类记录，不再统一几何融合、NMS或跨分支去重。'),
            ('p', '历史状态按关键帧顺序提交，处理仅依赖截至当前关键帧的观测。异步运行须等待该任务的原生与辅助证据均可用，结束时排空任务后再读出。各前缀与终态必须调用一致的公共读出规则；不能直接把内部快照与经过过滤、重新读出的终态当作同一集合。'),
            ('h', '语义读出建议另画虚线可选框', GREEN),
            ('p', '主流程输出是三维框及置信度。已有统一语义评测脚本在终态预测上选择有效投影最大的观测，扩大二维裁剪边界后用冻结 CLIP 与文本查询计算相似度。该语义读出不反向更新 PTR、VAR 或 HSR。不能从“原生路径有 CLIP”推断恢复记录自动继承代表候选的 CLIP 特征，更不能据此标成所有分支逐关键帧在线分类。'),
            ('table', ['消融设置', '共享辅助前端是否仍需要', '应如何解释'], [
                ['仅关闭 PTR 或 VAR', '需要，其他消费者仍在使用', '移除该恢复规则，不等于移除候选源'],
                ['只保留 HSR', '需要 proposal 与 Boxer 三维几何', 'HSR 不是仅依赖原生框的独立后处理'],
                ['三模块均关闭，纯 Ctrl', '若无其他用途，可不执行', '恢复与重排序均不参与输出'],
                ['三模块关闭，但直接读出辅助框', '需要', '这是辅助流即时/直接读出基线，不是纯 Ctrl'],
            ], [158, 157, 196]),
            ('small', '接口依赖可共享，模块作用仍可消融。必须在表注明确“关闭后的候选是否直接输出”；不要用同一个“无模块”标签混合纯原生基线与辅助直接读出基线。'),
            ('h', '实现参数速查（已选定路线，不是所有实验通用值）', INK),
            ('p', 'PTR：轨迹容量 1024，每轨迹最多 12 条不同帧观测，活动轨迹超过 10 个关键帧未更新则失效；确认候选另有容量，当前输出上限 12。VAR：活动体素 4096、确认输出 640。HSR：身份状态容量 4096、缺席身份 TTL 为 10 个关键帧。不要把“条目数设上限”直接等同于所有日志和历史字段的严格字节内存上界。', 9.2),
            ('h', '源码索引（项目根目录下的相对路径）', INK),
            ('sources', [
                'development/final_controls/boxfusion/online_candidate_runtime.py — FrozenWeDetectBoxerProvider.process：共享前端、两类候选及提升。',
                'development/final_controls/boxfusion/online_candidate_map.py — OnlineProposalRecovery / _size_price：PTR 关联、medoid、筛选和评分。',
                'development/final_controls/boxfusion/m1_anchor_online.py — OnlineAnchorRecovery / deterministic_tail_score：VAR-v1 与尾部分数。',
                'development/final_controls/boxfusion/native_reliability_reranker.py — _evidence_strength / update / _score：选 max + no_visibility。',
                'development/final_controls/boxfusion/reliability_candidate_map.py — selected_calr_v1 / _assemble / materialize_terminal：组件选择与输出。',
                'development/final_controls/boxfusion/reliable_views.py；box_fusion.py — 原生可靠视图选择与 PFO。',
                'tools/replay_scannet_online_prefix.py；tools/run_scannet_base_novel_eval.py — 一致前缀读出与终态语义评测。',
            ]),
            ('small', '配置依据：20261009 效率运行 scene0426_03 的 effective_config.yaml，完整路径与 SHA-256 见 manifest.json。v2 是容器名；本路线 selected_calr_v1=true，HSR 选择 max / no_visibility。'),
        ],
    },
]


def table_draw(c, y, headers, rows, widths):
    values = [[paragraph(t, 8.8, 12.0, '#FFFFFF', True) for t in headers]]
    values += [[paragraph(t, 8.8, 13.3) for t in row] for row in rows]
    t = Table(values, colWidths=widths, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor(INK)),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LINEBELOW', (0,0), (-1,0), .5, colors.HexColor(INK)),
        ('LINEBELOW', (0,1), (-1,-1), .4, colors.HexColor(GRID)),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F4F7FA'), colors.white]),
    ]))
    _, height = t.wrap(CW, H)
    t.drawOn(c, MARGIN, y-height)
    return y-height-12


def page_header(c, title, subtitle, index):
    c.setFillColor(colors.HexColor(PURPLE))
    c.rect(MARGIN, H-51, 32, 3, stroke=0, fill=1)
    put(c, title, MARGIN, H-63, CW, 19.5, 27.5, bold=True)
    put(c, subtitle, MARGIN, H-100, CW, 9, color=MUTED)
    c.setStrokeColor(colors.HexColor(GRID)); c.setLineWidth(.6)
    c.line(MARGIN, 37, W-MARGIN, 37)
    put(c, f'CaRR-3D · 方法 / 绘图工作稿 · {DATE}', MARGIN, 28, CW-45, 7.6, color=MUTED)
    put(c, f'{index:02d} / 06', W-MARGIN-45, 28, 45, 7.8, color=MUTED, align=TA_CENTER)
    return H-126


def build_pdf():
    target = OUT / 'CaRR-3D_方法与流程图绘制说明_20261114.pdf'
    c = canvas.Canvas(str(target), pagesize=(W,H), pageCompression=1)
    c.setTitle('CaRR-3D 方法与流程图绘制说明')
    c.setAuthor('CaRR-3D Research Notes')
    c.setSubject('Code-grounded method summary and diagram specification; PTR / VAR / HSR')
    bottoms = []
    for index, page in enumerate(PAGES, 1):
        y = page_header(c, page['title'], page['subtitle'], index)
        for block in page['blocks']:
            typ = block[0]
            if typ == 'p':
                size = block[2] if len(block)>2 else 9.8
                y -= put(c, block[1], MARGIN, y, CW, size, size*1.52)+9
            elif typ == 'small':
                y -= put(c, block[1], MARGIN, y, CW, 8.2, 12.3, MUTED)+9
            elif typ == 'h':
                y -= 3
                y -= put(c, block[1], MARGIN, y, CW, 11.5, 16.5, block[2], True)+7
            elif typ == 'steps':
                for n, text in enumerate(block[1], 1):
                    put(c, f'{n:02d}', MARGIN, y, 25, 8.5, color=PURPLE, bold=True)
                    y -= put(c, text, MARGIN+27, y, CW-27, 9.5, 14.3)+7
            elif typ == 'table':
                y = table_draw(c, y, *block[1:])
            elif typ == 'note':
                note = paragraph(block[2], 9.3, 14.2)
                _, nh = note.wrap(CW-22, H)
                height = nh+34
                c.setFillColor(colors.HexColor('#F1F5FA'))
                c.roundRect(MARGIN, y-height, CW, height, 5, fill=1, stroke=0)
                put(c, block[1], MARGIN+11, y-8, CW-22, 9.7, 14, PURPLE, True)
                note.drawOn(c, MARGIN+11, y-height+8)
                y -= height+11
            elif typ == 'diagram':
                diagram(c, MARGIN+7.7, y-298, CW*.97)
                y -= 309
            elif typ == 'sources':
                for n, text in enumerate(block[1], 1):
                    y -= put(c, f'[{n}] {text}', MARGIN, y, CW, 7.5, 10.5, MUTED)+4
            if y < 49:
                OVERFLOWS.append((index, typ, round(y,2)))
        bottoms.append(round(y,2))
        c.showPage()
    c.save()
    diagram_path = OUT / 'CaRR-3D_流程依赖草图_矢量.pdf'
    g = canvas.Canvas(str(diagram_path), pagesize=(551, 360))
    g.setTitle('CaRR-3D data dependencies — vector drawing draft')
    put(g, 'CaRR-3D — data dependencies', 20, 348, 511, 12, bold=True)
    diagram(g, 20, 20, 511)
    g.save()
    return target, bottoms


def docx_run_font(run, size, bold=False, color=INK):
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor
    run.font.name = 'DejaVu Sans'
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string(color.lstrip('#'))
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), 'Noto Sans CJK SC')


def build_docx(diagram_png):
    from docx import Document
    from docx.shared import Cm, Pt
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(1.5)
    sec.left_margin = sec.right_margin = Cm(1.48)
    normal = doc.styles['Normal']
    normal.font.name = 'DejaVu Sans'; normal.font.size = Pt(9)
    normal._element.rPr.rFonts.set(qn('w:eastAsia'), 'Noto Sans CJK SC')
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.1
    footer = sec.footer.paragraphs[0]
    docx_run_font(footer.add_run(f'CaRR-3D · 方法与流程图绘制说明 · {DATE}'), 8, color=MUTED)
    def para(text, size=9, bold=False, color=INK):
        p = doc.add_paragraph()
        docx_run_font(p.add_run(text), size, bold, color)
        return p
    for idx, page in enumerate(PAGES):
        if idx: doc.add_page_break()
        para(page['title'], 18, True)
        para(page['subtitle'], 9, color=MUTED)
        for b in page['blocks']:
            if b[0] == 'diagram':
                if diagram_png.exists(): doc.add_picture(str(diagram_png), width=Cm(17.8))
            elif b[0] == 'p': para(b[1])
            elif b[0] == 'small': para(b[1], 8, color=MUTED)
            elif b[0] == 'h': para(b[1], 11, True, b[2])
            elif b[0] == 'note':
                para(b[1], 10, True, PURPLE); para(b[2])
            elif b[0] == 'steps':
                for n,t in enumerate(b[1],1): para(f'{n}. {t}')
            elif b[0] == 'sources':
                for n,t in enumerate(b[1],1): para(f'[{n}] {t}',7.5,color=MUTED)
            elif b[0] == 'table':
                headers, rows, widths = b[1:]
                t = doc.add_table(rows=1, cols=len(headers))
                t.style='Table Grid'
                for col, label in enumerate(headers):
                    t.rows[0].cells[col].text=label
                    sh=OxmlElement('w:shd'); sh.set(qn('w:fill'),'E8EEF5')
                    t.rows[0].cells[col]._tc.get_or_add_tcPr().append(sh)
                for row in rows:
                    cells=t.add_row().cells
                    for col,text in enumerate(row): cells[col].text=text
                for r_idx,row in enumerate(t.rows):
                    for cell in row.cells:
                        for p in cell.paragraphs:
                            p.paragraph_format.space_after=Pt(3)
                            for run in p.runs: docx_run_font(run,8,r_idx==0)
                doc.add_paragraph().paragraph_format.space_after=Pt(0)
    target=OUT/'CaRR-3D_方法与流程图绘制说明_可编辑.docx'
    doc.save(target)
    return target


def manifest(bottoms):
    config = ROOT/'reports/official_github_boxfusion_vs_final_recar_public_cli_coldcache_nowarmup100_20261009/receipts/recar/scene0426_03/effective_config.yaml'
    names = [
        'online_candidate_runtime.py', 'online_candidate_map.py',
        'm1_anchor_online.py', 'native_reliability_reranker.py',
        'reliability_candidate_map.py', 'reliable_views.py', 'box_fusion.py',
    ]
    paths = [ROOT/'development/final_controls/boxfusion'/n for n in names]
    paths += [config, ROOT/'tools/replay_scannet_online_prefix.py',
              ROOT/'tools/run_scannet_base_novel_eval.py']
    payload = {
        'date': DATE, 'scope': 'Selected paper route: PTR-v1 + VAR-v1 + HSR max/no_visibility',
        'no_model_inference': True, 'gpu_used': False,
        'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'pdf_page_bottoms_pt': bottoms, 'layout_overflows': OVERFLOWS,
        'missing_glyphs': sorted(MISSING_GLYPHS),
    }
    (OUT/'manifest.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    import subprocess
    target,bottoms=build_pdf()
    subprocess.run(['pdftoppm','-png','-r','180','-singlefile',
                    str(OUT/'CaRR-3D_流程依赖草图_矢量.pdf'),str(OUT/'diagram_preview')],check=True)
    docx=build_docx(OUT/'diagram_preview.png')
    manifest(bottoms)
    print(json.dumps({'pdf':str(target),'docx':str(docx),'bottoms':bottoms,
                      'overflows':OVERFLOWS,'missing_glyphs':sorted(MISSING_GLYPHS)},ensure_ascii=False,indent=2))
    if OVERFLOWS or MISSING_GLYPHS:
        raise RuntimeError('Document quality check failed; see manifest.json')


if __name__=='__main__':
    main()

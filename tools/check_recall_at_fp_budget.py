#!/usr/bin/env python3
"""Supplementary whole-score-threshold recall, using fixed FP/100 GT budgets."""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from true_fusion_audit_core import aabb_iou, class_agnostic_ap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/recall_fp_budget_20260915'
THRESHOLDS = (.15, .25, .50)
BUDGETS = (1, 5, 10, 20)
SCENES = ROOT / 'reports/ca1m_lifting_factors_full107_20260908/capture/protocol.json'
METHODS = {'native': 'ca1m_thr15', 'M1': 'ca1m_thr15_m1_full107',
           'M1_M2': 'ca1m_thr15_m1_m2nl_full107'}
EXPECTED = {'native': (44.8526, 36.7678, 15.3999),
            'M1': (45.7277, 37.6384, 15.7733),
            'M1_M2': (45.9762, 38.2087, 16.7660)}


def ranked_flags(predictions, gts, threshold):
    scores, locations, matrices = [], [], {}
    matched = {s: np.zeros(len(gt), bool) for s, gt in gts.items()}
    for scene, (boxes, confidence) in predictions.items():
        matrices[scene] = aabb_iou(boxes, gts[scene])
        scores.extend(confidence.tolist())
        locations.extend((scene, i) for i in range(len(boxes)))
    scores = np.asarray(scores)
    order = np.argsort(-scores)
    flags = np.zeros(len(order), dtype=int)
    for rank, index in enumerate(order):
        scene, row = locations[index]
        values = matrices[scene][row]
        if len(values) and values.max() > threshold:
            target = int(values.argmax())
            if not matched[scene][target]:
                matched[scene][target] = True
                flags[rank] = 1
    sorted_scores = scores[order]
    # Evaluate realizable confidence thresholds; never cut inside a score tie.
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    tp, fp = flags.cumsum(), (1 - flags).cumsum()
    return sorted_scores, ends, tp, fp


def main():
    scenes = json.loads(SCENES.read_text())['scenes']
    assert len(scenes) == len(set(scenes)) == 107
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        methods=METHODS, IoU_thresholds=THRESHOLDS, FP_per_100_GT_budgets=BUDGETS,
        metric='Maximum recall among whole confidence-threshold prefixes satisfying FP budget',
        evaluation='All107, AABB strict >, same best-GT greedy matching as historical AP',
        role='Supplementary development analysis; standard AP remains primary',
        caution='Reported test-set confidence cutoffs are diagnostic and not deployable calibrated thresholds',
    )
    # Fixed budget grid saved before reading or scoring the predictions.
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    gts, hashes = {}, {}
    for scene in scenes:
        path = Path('/extra/ZhaoX/boxfusion_ca1m') / scene / 'after_filter_boxes.npy'
        gts[scene] = np.load(path, allow_pickle=False)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    total = sum(len(g) for g in gts.values())
    assert total == 12911
    results = {}
    for method, directory in METHODS.items():
        predictions = {}
        for scene in scenes:
            path = ROOT / 'results' / directory / f'{scene}_boxes.pkl'
            with path.open('rb') as stream:
                rows = pickle.load(stream)[0]
            predictions[scene] = (
                np.asarray([r[1] for r in rows], float).reshape(-1, 8, 3),
                np.asarray([float(r[2]) for r in rows]))
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        results[method] = {}
        for threshold, expected in zip(THRESHOLDS, EXPECTED[method]):
            metric = class_agnostic_ap(predictions, gts, threshold)
            assert abs(metric['ap'] - expected) < .0001, (method, threshold, metric)
            scores, ends, tp, fp = ranked_flags(predictions, gts, threshold)
            assert tp[-1] == metric['tp'] and fp[-1] == metric['fp']
            budgets = {}
            for budget in BUDGETS:
                allowed = int(np.floor(budget * total / 100))
                valid = ends[fp[ends] <= allowed]
                last = int(valid[-1]) if len(valid) else None
                budgets[str(budget)] = dict(
                    recall=100 * int(tp[last]) / total if last is not None else 0.,
                    tp=int(tp[last]) if last is not None else 0,
                    fp=int(fp[last]) if last is not None else 0,
                    fp_budget=allowed,
                    confidence_cutoff=float(scores[last]) if last is not None else None)
            results[method][str(threshold)] = dict(
                standard_AP=metric['ap'], all_output_recall=100 * metric['tp'] / total,
                recall_at_fp_budget=budgets)
        print(method, json.dumps(results[method]), flush=True)
    for path, sha in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
    (OUT / 'results.json').write_text(json.dumps(dict(
        scenes=107, GT=total, results=results,
        verification=dict(historical_AP_parity=True, matching_TP_FP_parity=True,
                          whole_score_groups_only=True, inputs_unchanged=True,
                          prediction_mutation=False, models_run=False)), indent=2) + '\n')
    (OUT / 'input_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    lines = ['# 固定误检预算下的召回：补充开发评价', '',
             '使用现有 full107 冻结结果，12,911 GT；保留历史标准 AP。三种配置为同一配对底座、+M1、+M1+M2。', '',
             '指标为 FP≤预算时所有完整置信度阈值前缀中的最大召回；预算取每100个GT允许1/5/10/20个FP，运行前固定。不能截断同分预测挑选有利结果。', '',
             '| IoU | 配置 | 标准AP | R@1 FP/100GT | R@5 | R@10 | R@20 | 全部输出召回 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for threshold in THRESHOLDS:
        for method in METHODS:
            r = results[method][str(threshold)]
            cells = [str(threshold), method, f"{r['standard_AP']:.4f}"]
            cells.extend(f"{r['recall_at_fp_budget'][str(b)]['recall']:.3f}" for b in BUDGETS)
            cells.append(f"{r['all_output_recall']:.3f}")
            lines.append('| ' + ' | '.join(cells) + ' |')
    lines.extend(['', '## 边界', '',
                  '- 每个预算下的置信度截点依据当前评测集选择，属于曲线诊断；部署阈值需要在其他开发数据上固定后独立验证。',
                  '- 预算按GT数归一化，不能直接解释为每平方米或每秒的误检。',
                  '- 此处复用末帧输出，不评价在线延迟、计算预算公平性、开放词汇类别能力或泛化。',
                  '- 三项标准AP与已核验历史配对结果逐项在0.0001点以内一致，TP/FP计数也一致，输入SHA256前后不变。',
                  '- 完整预定网格全部报告；不只选择其中胜出的预算点。补充指标本身不构成新的模型创新。', '',
                  '脚本：`tools/check_recall_at_fp_budget.py`；逐点结果与阈值：`results.json`；预定口径：`protocol.json`。'])
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()

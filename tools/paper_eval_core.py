"""NumPy metrics for frozen-output paper audits; no model or GT mutation."""
import numpy as np
from true_fusion_audit_core import aabb_iou


def subgroup_ap(predictions, ground_truth, masks, threshold):
    """Prefer eligible in-group GT; else ignore out-group overlap, even repeats.

    Unmatched detections remain FP regardless of predicted volume. This declared
    3D volume diagnostic is not COCO area AP. All-GT group equals the anchor.
    """
    scores, identities, overlaps = [], [], {}
    taken = {s: np.zeros(len(g), bool) for s, g in ground_truth.items()}
    for s, (boxes, confidence) in predictions.items():
        overlaps[s] = aabb_iou(boxes, ground_truth[s])
        scores.extend(confidence)
        identities.extend((s, i) for i in range(len(boxes)))
    tp, fp, ignored = [], [], 0
    for index in np.argsort(-np.asarray(scores)):
        s, row = identities[index]
        values, group = overlaps[s][row], masks[s]
        eligible = np.flatnonzero(group & (values > threshold))
        if len(eligible):
            target = eligible[np.argmax(values[eligible])]
            success = not taken[s][target]; taken[s][target] = True
            tp.append(int(success)); fp.append(int(not success))
        elif np.any((~group) & (values > threshold)):
            ignored += 1
        else:
            tp.append(0); fp.append(1)
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    count = sum(int(m.sum()) for m in masks.values())
    recall = tp / (count + 1e-6)
    precision = tp / np.maximum(tp + fp, np.finfo(float).eps)
    r, p = np.r_[0, recall, 1], np.r_[0, precision, 0]
    p = np.maximum.accumulate(p[::-1])[::-1]
    k = np.flatnonzero(r[1:] != r[:-1])
    return {'ap': float(100 * np.sum((r[k+1]-r[k])*p[k+1])), 'gt': count,
            'tp': int(tp[-1]) if len(tp) else 0, 'fp': int(fp[-1]) if len(fp) else 0,
            'ignored': ignored, 'predictions': len(scores)}


def corners_from_center_size(boxes):
    signs = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    boxes = np.asarray(boxes, float)
    return boxes[:,None,:3] + signs[None]*boxes[:,None,3:6]/2


def best_view(boxes, poses, intrinsic, width, height):
    """Automatic terminal view selection; no annotation or depth input."""
    boxes = np.asarray(boxes).reshape(-1,8,3)
    areas = np.zeros(len(boxes)); ids = np.full(len(boxes),-1,int)
    rectangles = np.zeros((len(boxes),4))
    for fi, pose in enumerate(poses):
        inv = np.linalg.inv(pose)
        camera = boxes @ inv[:3,:3].T + inv[:3,3]
        projected = camera @ intrinsic.T
        uv = projected[...,:2]/np.maximum(projected[...,2:3],1e-9)
        lo = np.maximum(uv.min(1),0); hi = np.minimum(uv.max(1),[width,height])
        delta = hi-lo
        valid = (camera[...,2].min(1)>=.1) & (delta.min(1)>=2)
        area = np.where(valid,delta.prod(1),0); better = area>areas
        areas[better]=area[better]; ids[better]=fi
        rectangles[better]=np.c_[lo,hi][better]
    return ids, rectangles

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P0-7：静态预测代理 vs 真实预测的动态分配（同一图、同一 GT、同一 assigner）。

为什么需要这个脚本
------------------
论文 §4 的核心数字（静态 12.90% vs 动态 5.13%、二值一致率 88.1%、Jaccard 0.205、
方向反转 +2.10 / −0.84 pp）此前来自 2026-11-15 的一次性分析，脚本未入库。
本次重写，目的是让这些数字可复算：同一份代码同时跑「静态代理」与「真实预测」两臂，
两臂之间**只差预测框与分数**，其余（图像、GT、候选规则、topk、alpha/beta、冲突消解）
全部共用。

静态代理的定义（与历史口径一致）
--------------------------------
对每个 anchor j（中心 c_j，所属层 stride s_j）：
    合成框 = [c_jx − s_j, c_jy − s_j, c_jx + s_j, c_jy + s_j]（边长 2·s_j），裁剪到图像内
    合成分数 = 0.99（所有类别相同）
然后用**原生** TaskAlignedAssigner 分配。因此静态臂的 align_metric 完全由几何决定，
不含任何模型预测。

一个重要推论：S1（几何候选数）只依赖 GT 框与 anchor 网格，两臂**必然完全相同**。
本脚本把它作为自检项断言，而不是当作一个"结果"。

输出
----
<out>.csv          逐 GT：bucket, d, 两臂的 s1/s3pre/s4, 是否零正, 两臂正样本集的 Jaccard
<out>_frames.csv   逐图汇总
<out>_meta.json    参数、权重哈希、耗时、恒等式违反数
<out>_summary.json 全部汇总指标（零正率、一致率、Jaccard、分层）

用法
----
    AITOD_RULE=default python p0_7_static_proxy.py \
        --ckpt .../E0_default_s3/weights/epoch60.pt --weight-type ema \
        --ann .../aitod_val_v2.json --images .../images/val \
        --map .../gtmap_val_map.csv --rule default \
        --out /root/runs/p1/p07_recompute_default
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np


# ----------------------------------------------------------------------------
# 规则必须在导入 ultralytics 之前设定（aitod_assigner 在 import 时打补丁）
# ----------------------------------------------------------------------------
def _pre_rule():
    for i, a in enumerate(sys.argv):
        if a == '--rule' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith('--rule='):
            return a.split('=', 1)[1]
    return os.environ.get('AITOD_RULE', 'default')


os.environ['AITOD_RULE'] = _pre_rule()

import torch                                                    # noqa: E402
import cv2                                                      # noqa: E402
import aitod_assigner                                           # noqa: E402,F401  打补丁

from p0_6_dynamic_trace import (bucket_of, letterbox_img, load_map,   # noqa: E402
                                file_sha)
from p0_tracing import TracingAssigner, check_invariants        # noqa: E402
from ultralytics.utils.tal import make_anchors                  # noqa: E402


# ----------------------------------------------------------------------------
# 静态代理：在 assigner 入口替换预测
# ----------------------------------------------------------------------------
class StaticTracer(TracingAssigner):
    """TracingAssigner + 可切换的静态预测替换 + 记录冲突消解后的正样本 mask。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.static = False
        self.anchor_stride = None      # (na,) 每个 anchor 所属层的 stride
        self.imgsz = 800
        self.clip = True
        self.post_mask = None          # (n_gt, na) bool，冲突消解后的归属
        self.synth_box = None          # 供事后核对

    def make_synthetic(self, anc_points):
        """按 anchor 中心与所属 stride 构造合成框与分数。"""
        s = self.anchor_stride.to(anc_points.device).unsqueeze(1)   # (na,1)
        # 合成框边长 = 2*stride（即半边长 = stride），与论文 §4.1 的表述一致
        half = 1.0 * s
        x1 = anc_points[:, 0:1] - half
        y1 = anc_points[:, 1:2] - half
        x2 = anc_points[:, 0:1] + half
        y2 = anc_points[:, 1:2] + half
        box = torch.cat([x1, y1, x2, y2], dim=1)
        if self.clip:
            box = box.clamp(min=0.0, max=float(self.imgsz))
        return box

    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes,
                mask_gt, **extra):
        if self.static:
            box = self.make_synthetic(anc_points)                   # (na,4)
            self.synth_box = box
            pd_bboxes = box.unsqueeze(0)                            # (1,na,4)
            pd_scores = torch.full_like(pd_scores, 0.99)
        return super().forward(pd_scores, pd_bboxes, anc_points, gt_labels,
                               gt_bboxes, mask_gt, **extra)

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        out = super().select_highest_overlaps(mask_pos, overlaps, n_max_boxes,
                                              align_metric)
        self.post_mask = out[2].bool()
        return out


class Dispatcher(torch.nn.Module):
    """把一次 assigner 调用分派给两臂，返回动态臂结果（loss 数值不使用）。"""

    def __init__(self, dyn, sta):
        super().__init__()
        self.dyn, self.sta = dyn, sta

    def __getattr__(self, name):
        # crit 可能读取 topk / stride 等属性
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.dyn, name)

    def forward(self, *a, **kw):
        self.dyn.static = False
        out = self.dyn(*a, **kw)
        self.sta.static = True
        self.sta(*a, **kw)
        return out


# ----------------------------------------------------------------------------
def jaccard(a, b):
    """两个布尔向量的 Jaccard；两侧都为空时返回 1.0。"""
    inter = int((a & b).sum())
    union = int((a | b).sum())
    if union == 0:
        return 1.0
    return inter / union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--weight-type', default='auto', choices=['auto', 'ema', 'raw'])
    ap.add_argument('--ann', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--map', required=True)
    ap.add_argument('--ids-file', default=None)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--rule', default=None,
                    help='候选规则，缺省取 AITOD_RULE 环境变量')
    ap.add_argument('--imgsz', type=int, default=800)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--no-clip', action='store_true',
                    help='合成框不裁剪到图像内（默认裁剪）')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if args.rule:
        if args.rule != aitod_assigner.MODE:
            print('[p07] --rule 与 AITOD_RULE 不一致（%s vs %s）'
                  % (args.rule, aitod_assigner.MODE), file=sys.stderr)
            return 2
    dev = torch.device(args.device)

    # ---- 权重 ----
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    has_m, has_e = ck.get('model') is not None, ck.get('ema') is not None
    wt = args.weight_type
    if wt == 'auto':
        wt = 'raw' if has_m else ('ema' if has_e else None)
    obj = ck.get('model') if wt == 'raw' else ck.get('ema')
    if obj is None:
        print('[p07] 请求 %s 权重但不存在（model=%s ema=%s）'
              % (wt, has_m, has_e), file=sys.stderr)
        return 2
    model = obj.float().to(dev)
    model.eval()
    model.model[-1].training = True        # 返回原始特征
    crit = model.init_criterion()

    a0 = crit.assigner
    tracer_d = StaticTracer(topk=int(a0.topk), num_classes=int(a0.num_classes),
                            alpha=float(a0.alpha), beta=float(a0.beta),
                            stride=[float(x) for x in a0.stride], eps=float(a0.eps))
    tracer_s = StaticTracer(topk=int(a0.topk), num_classes=int(a0.num_classes),
                            alpha=float(a0.alpha), beta=float(a0.beta),
                            stride=[float(x) for x in a0.stride], eps=float(a0.eps))
    crit.assigner = Dispatcher(tracer_d, tracer_s)

    stride = [float(x) for x in tracer_d.stride]
    feats = [torch.zeros(1, 1, args.imgsz // int(s), args.imgsz // int(s)) for s in stride]
    pts, _ = make_anchors(feats, stride, 0.5)
    n_anchor = int(pts.shape[0])
    # 每个 anchor 的所属层 stride，与 make_anchors 的层级顺序一致
    counts = [(args.imgsz // int(s)) ** 2 for s in stride]
    per_stride = torch.cat([torch.full((c,), s, dtype=torch.float32)
                            for c, s in zip(counts, stride)])
    assert int(per_stride.numel()) == n_anchor, (per_stride.numel(), n_anchor)
    for t in (tracer_d, tracer_s):
        t.anchor_stride = per_stride
        t.imgsz = args.imgsz
        t.clip = not args.no_clip

    # ---- 数据 ----
    ann = json.load(open(args.ann, encoding='utf-8'))
    imgs_meta = {i['id']: i for i in ann['images']}
    gtmap = load_map(args.map)
    if args.ids_file:
        ids = [int(x) for x in open(args.ids_file) if x.strip()]
    else:
        ids = sorted(imgs_meta.keys())
        if args.limit:
            ids = ids[:args.limit]
    ids = [i for i in ids if i in imgs_meta]

    rows, frames = [], []
    viol_d = viol_s = n_fail = 0
    t_assign = 0.0
    s1_mismatch = 0

    for iid in ids:
        im = imgs_meta[iid]
        p = os.path.join(args.images, im['file_name'])
        raw = cv2.imread(p, cv2.IMREAD_COLOR)
        if raw is None:
            print('[p07] 读图失败: %s' % p, file=sys.stderr)
            n_fail += 1
            continue
        chw, sc, px, py = letterbox_img(raw, args.imgsz)

        gts = sorted([(int(r['gt_idx']), r) for (mi, _), r in gtmap.items() if mi == iid],
                     key=lambda t: t[0])
        if not gts:
            frames.append({'image_id': iid, 'n_gt': 0})
            continue

        boxes = torch.tensor([[float(r['x']), float(r['y']),
                               float(r['x']) + float(r['w']),
                               float(r['y']) + float(r['h'])] for _, r in gts],
                             dtype=torch.float32)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * sc + px
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * sc + py
        boxes = boxes.clamp(min=0.0, max=float(args.imgsz)) / float(args.imgsz)
        xywh = torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                            (boxes[:, 1] + boxes[:, 3]) / 2,
                            boxes[:, 2] - boxes[:, 0],
                            boxes[:, 3] - boxes[:, 1]], dim=1)
        cls = torch.tensor([[float(r['category_id']) - 1] for _, r in gts],
                           dtype=torch.float32)
        batch = {'batch_idx': torch.zeros(len(gts), dtype=torch.long, device=dev),
                 'cls': cls.to(dev), 'bboxes': xywh.to(dev)}

        x = torch.from_numpy(chw).float().unsqueeze(0).to(dev) / 255.0
        with torch.no_grad():
            preds = model.predict(x)
        tracer_d.trace.clear(); tracer_d.post_mask = None
        tracer_s.trace.clear(); tracer_s.post_mask = None
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                crit(preds, batch)
        except Exception as e:                                  # noqa: BLE001
            print('[p07] img %d assigner 失败: %s' % (iid, e), file=sys.stderr)
            n_fail += 1
            continue
        t_assign += time.perf_counter() - t0

        trd, trs = tracer_d.trace, tracer_s.trace
        if 'fg' not in trd or 'fg' not in trs or tracer_s.post_mask is None:
            n_fail += 1
            continue

        ng = len(gts)
        fg_d, tgt_d = trd['fg'][0], trd['tgt'][0]
        s4_d = (torch.bincount(tgt_d[fg_d], minlength=ng).tolist()
                if fg_d.any() else [0] * ng)
        # 静态臂的 S4：直接用冲突消解后的 mask，避免与 fg/tgt 口径混淆
        post_s = tracer_s.post_mask[0, :ng]
        s4_s = post_s.sum(-1).tolist()
        post_d = tracer_d.post_mask[0, :ng]

        col_d = {k: trd[k][0].tolist() for k in ('s1', 's3pre', 's3post') if k in trd}
        col_s = {k: trs[k][0].tolist() for k in ('s1', 's3pre', 's3post') if k in trs}

        bad_d = check_invariants(trd, s4_d)
        bad_s = check_invariants(trs, s4_s)
        if bad_d:
            viol_d += 1
        if bad_s:
            viol_s += 1

        for k, (_, r) in enumerate(gts):
            s1d, s1s = col_d['s1'][k], col_s['s1'][k]
            if s1d != s1s:
                s1_mismatch += 1
            rows.append({
                'image_id': iid, 'gt_idx': int(r['gt_idx']),
                'bucket': r['bucket'], 'd': float(r['d']), 'area': float(r['area']),
                's1_dyn': s1d, 's1_sta': s1s,
                's3pre_dyn': col_d['s3pre'][k], 's3pre_sta': col_s['s3pre'][k],
                's4_dyn': s4_d[k], 's4_sta': s4_s[k],
                'zero_dyn': int(s4_d[k] == 0), 'zero_sta': int(s4_s[k] == 0),
                'jaccard': jaccard(post_d[k], post_s[k]),
            })
        frames.append({'image_id': iid, 'n_gt': ng})

    # ---- 写逐 GT ----
    if not rows:
        print('[p07] 无有效样本', file=sys.stderr)
        return 2
    fields = list(rows[0].keys())
    with open(args.out + '.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(args.out + '_frames.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['image_id', 'n_gt'])
        w.writeheader()
        w.writerows(frames)

    # ---- 汇总 ----
    def rate(sel, key):
        n = sum(1 for r in rows if sel(r))
        z = sum(r[key] for r in rows if sel(r))
        return (100.0 * z / n) if n else float('nan'), n, z

    buckets = ['verytiny', 'tiny', 'small', 'medium']
    bins = [(0, 4), (4, 5), (5, 6), (6, 7), (7, 8.001)]

    sumry = {'rule': aitod_assigner.MODE, 'n_gt': len(rows), 'n_img': len(frames),
             'n_anchor': n_anchor, 'imgsz': args.imgsz,
             'clip': not args.no_clip, 'sim': 'ciou', 'weight_type': wt,
             'ckpt_sha256_16': file_sha(args.ckpt),
             'viol_dyn': viol_d, 'viol_sta': viol_s, 'n_fail': n_fail,
             's1_mismatch_rows': s1_mismatch,
             'seconds_per_image': t_assign / max(len(ids), 1)}

    for key in ('zero_dyn', 'zero_sta'):
        sumry[key] = {'all': dict(zip(('rate', 'n', 'z'), rate(lambda r: True, key)))}
        for b in buckets:
            sumry[key][b] = dict(zip(('rate', 'n', 'z'),
                                     rate(lambda r, b=b: r['bucket'] == b, key)))
        sumry[key]['bins'] = {}
        for lo, hi in bins:
            sel = (lambda r, lo=lo, hi=hi: lo <= r['d'] < hi)
            rr, n, z = rate(sel, key)
            sumry[key]['bins']['%g-%g' % (lo, min(hi, 8))] = {'rate': rr, 'n': n, 'z': z}

    n = len(rows)
    sumry['agreement'] = 100.0 * sum(1 for r in rows if r['zero_dyn'] == r['zero_sta']) / n
    sumry['jaccard_mean'] = float(np.mean([r['jaccard'] for r in rows]))
    sumry['jaccard_micro'] = None      # 需要按集合合并不便，保留均值口径
    for arm in ('dyn', 'sta'):
        for col in ('s1', 's3pre', 's4'):
            sumry['mean_%s_%s' % (col, arm)] = float(np.mean([r['%s_%s' % (col, arm)]
                                                              for r in rows]))
    sumry['conflict_loss_dyn'] = sumry['mean_s3pre_dyn'] - sumry['mean_s4_dyn']
    sumry['conflict_loss_sta'] = sumry['mean_s3pre_sta'] - sumry['mean_s4_sta']

    with open(args.out + '_summary.json', 'w', encoding='utf-8') as f:
        json.dump(sumry, f, ensure_ascii=False, indent=2)
    with open(args.out + '_meta.json', 'w', encoding='utf-8') as f:
        json.dump(sumry, f, ensure_ascii=False, indent=2)

    print('[p07] rule=%s  n_img=%d  n_gt=%d  clip=%s' %
          (aitod_assigner.MODE, len(frames), len(rows), not args.no_clip))
    print('[p07] 零正样本率  静态 %.2f%%  动态 %.2f%%  （全体）'
          % (sumry['zero_sta']['all']['rate'], sumry['zero_dyn']['all']['rate']))
    print('[p07] verytiny    静态 %.2f%%  动态 %.2f%%'
          % (sumry['zero_sta']['verytiny']['rate'], sumry['zero_dyn']['verytiny']['rate']))
    print('[p07] 二值一致率 %.1f%%   Jaccard 均值 %.3f' %
          (sumry['agreement'], sumry['jaccard_mean']))
    print('[p07] s1 静态/动态 %.3f / %.3f（应为同一值，不一致行数 %d）'
          % (sumry['mean_s1_sta'], sumry['mean_s1_dyn'], s1_mismatch))
    print('[p07] s3pre %.3f / %.3f   s4 %.3f / %.3f' %
          (sumry['mean_s3pre_sta'], sumry['mean_s3pre_dyn'],
           sumry['mean_s4_sta'], sumry['mean_s4_dyn']))
    if s1_mismatch:
        print('[p07] !! S1 两臂不一致，说明 GT 或 anchor 处理有差异', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练效应下的 anchor 级归属追踪（申请 2 第 7 项）。

与 p0_9 的区别
    p0_9：同一份预测，用 CIoU 与 NWD 两个 assigner 各跑一次 → 接口效应
    本脚本：两个**不同训练出来的模型**，各自用自己的相似度分配 → 训练效应
            E0 的 epoch60（CIoU）与 NWD 的 epoch60（NWD）

为什么可以逐 anchor 对比
    两臂输入尺寸、stride、anchor 网格完全相同（800px, [8,16,32], 13125 个点），
    同一张图的 GT 顺序也相同（同一份 gtmap）。所以 owner 数组可以直接对齐。

分类
    不变 / 易主(a→b) / 仅 E0 有归属 / 仅 NWD 有归属 / 两臂都无

用法
    python anchor_transfer_trained.py --ckpt-a <E0 epoch60> --ckpt-b <NWD epoch60> \
        --ann ... --images ... --map ... --ids-file ... --out ...
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import torch

import p0_6_dynamic_trace as P6
import nwd_tal
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors


class Grab(TaskAlignedAssigner):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.post = None
        self.n_cand = None       # 每 anchor 在 S3pre 的候选 GT 数

    def get_pos_mask(self, *a, **kw):
        out = super().get_pos_mask(*a, **kw)
        # out[0] 是 mask_pos（S3pre）：(b, n_max, na)。求和即每 anchor 的候选 GT 数
        self.n_cand = out[0].sum(1).int().cpu().numpy()      # (b, na)
        return out

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        out = super().select_highest_overlaps(mask_pos, overlaps, n_max_boxes, align_metric)
        self.post = out[2].bool().clone()
        return out


def owner_of(post):
    na = post.shape[1]
    own = np.full(na, -1, dtype=np.int64)
    idx = post.nonzero(as_tuple=False)
    if idx.numel():
        own[idx[:, 1].cpu().numpy()] = idx[:, 0].cpu().numpy()
    return own


def load_model(path, dev, imgsz, ann_path):
    from ultralytics.cfg import get_cfg
    ck = torch.load(path, map_location='cpu', weights_only=False)
    obj = ck.get('ema') or ck.get('model')
    m = obj.float().to(dev)
    m.args = get_cfg(overrides={'imgsz': imgsz, 'data': ann_path, 'batch': 1})
    m.eval()
    m.model[-1].training = True
    crit = m.init_criterion()
    a0 = crit.assigner
    g = Grab(topk=int(a0.topk), num_classes=int(a0.num_classes),
             alpha=float(a0.alpha), beta=float(a0.beta),
             stride=[float(x) for x in a0.stride],
             topk2=getattr(a0, 'topk2', None), eps=float(a0.eps))
    crit.assigner = g
    return m, crit, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-a', required=True, help='E0 epoch60（CIoU 训练）')
    ap.add_argument('--ckpt-b', required=True, help='NWD epoch60（NWD 训练）')
    ap.add_argument('--ann', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--map', required=True)
    ap.add_argument('--ids-file', required=True)
    ap.add_argument('--imgsz', type=int, default=800)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--nwd-c', type=float, default=12.8)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)

    mE, cE, gE = load_model(a.ckpt_a, dev, a.imgsz, a.ann)
    nwd_tal.uninstall()                       # A 臂：原生 CIoU
    mN, cN, gN = load_model(a.ckpt_b, dev, a.imgsz, a.ann)
    nwd_tal.install(C=a.nwd_c)                # B 臂：NWD

    stride = [float(x) for x in gE.stride]
    feats = [torch.zeros(1, 1, a.imgsz // int(s), a.imgsz // int(s)) for s in stride]
    pts, _ = make_anchors(feats, stride, 0.5)

    ann = json.load(open(a.ann, encoding='utf-8'))
    meta = {i['id']: i for i in ann['images']}
    gtmap = P6.load_map(a.map)
    by_img = collections.defaultdict(list)
    for (iid, gi), r in gtmap.items():          # key 是 (image_id, gt_idx) 元组
        by_img[iid].append((gi, r))
    ids = [int(x) for x in open(a.ids_file) if x.strip()]
    ids = [i for i in ids if i in meta and i in by_img]

    import cv2
    stat = collections.Counter()
    cross = collections.defaultdict(int)
    per_bucket = collections.defaultdict(collections.Counter)
    n_img = 0

    for si, iid in enumerate(ids):
        rows = sorted(by_img[iid], key=lambda t: t[0])   # 按 gt_idx 排序，两臂一致
        raw = cv2.imread(os.path.join(a.images, meta[iid]['file_name']), cv2.IMREAD_COLOR)
        if raw is None or not rows:
            continue
        chw, sc, px, py = P6.letterbox_img(raw, a.imgsz)
        x = torch.from_numpy(chw).float().unsqueeze(0).to(dev) / 255.0

        gts, bks = [], []
        for _, r in rows:
            x1 = float(r['x']) * sc + px          # gtmap 存的是左上角 x,y + 宽高
            y1 = float(r['y']) * sc + py
            w = float(r['w']) * sc
            h = float(r['h']) * sc
            gts.append([x1, y1, x1 + w, y1 + h])
            bks.append(r['bucket'])
        gts = np.array(gts, dtype=np.float32) / a.imgsz
        gts = np.clip(gts, 0, 1)
        bx = torch.tensor(gts, dtype=torch.float32, device=dev)
        xywh = torch.empty_like(bx)
        xywh[:, 0] = (bx[:, 0] + bx[:, 2]) / 2
        xywh[:, 1] = (bx[:, 1] + bx[:, 3]) / 2
        xywh[:, 2] = bx[:, 2] - bx[:, 0]
        xywh[:, 3] = bx[:, 3] - bx[:, 1]
        batch = {'batch_idx': torch.zeros(len(gts), dtype=torch.long, device=dev),
                 'cls': torch.tensor([[0.0]] * len(gts), dtype=torch.float32, device=dev),
                 'bboxes': xywh}

        own = {}
        cand = {}
        for tag, (m, crit, g, sim) in (('A', (mE, cE, gE, 'ciou')), ('B', (mN, cN, gN, 'nwd'))):
            if sim == 'ciou':
                nwd_tal.uninstall()
            else:
                nwd_tal.install(C=a.nwd_c)
            g.post = None
            g.n_cand = None
            with torch.no_grad():
                out = m.predict(x)
                y = out[0] if isinstance(out, tuple) else out
                try:
                    crit(y, batch)
                except Exception:                          # noqa: BLE001
                    own[tag] = None
                    continue
            own[tag] = owner_of(g.post[0]) if g.post is not None else None
            cand[tag] = g.n_cand[0] if g.n_cand is not None else None

        if own.get('A') is None or own.get('B') is None:
            continue
        A, B = own['A'], own['B']
        CA, CB = cand['A'], cand['B']
        n_img += 1
        for i in range(len(A)):
            a1, b1 = int(A[i]), int(B[i])
            ca = int(CA[i]) if CA is not None else 0
            cb = int(CB[i]) if CB is not None else 0
            # 竞争结构：S3pre 阶段该 anchor 被几个 GT 争
            if ca > 1:
                stat['multi_cand_A'] += 1
            if cb > 1:
                stat['multi_cand_B'] += 1
            if ca > 1 and cb <= 1:
                stat['comp_narrowed'] += 1      # 多 GT 竞争 → 单/无（竞争被收窄）
            if ca <= 1 and cb > 1:
                stat['comp_widened'] += 1

            if a1 < 0 and b1 < 0:
                continue
            if a1 == b1:
                stat['unchanged'] += 1
            elif a1 >= 0 and b1 < 0:
                stat['only_A'] += 1
                per_bucket[bks[a1] if a1 < len(bks) else 'na']['only_A'] += 1
            elif a1 < 0 and b1 >= 0:
                stat['only_B'] += 1
                per_bucket[bks[b1] if b1 < len(bks) else 'na']['only_B'] += 1
            else:
                stat['switched'] += 1
                sa = bks[a1] if a1 < len(bks) else 'na'
                sb = bks[b1] if b1 < len(bks) else 'na'
                cross[(sa, sb)] += 1
                per_bucket[sa]['switched_out'] += 1

    tot = sum(stat.values())
    print("[anchor] 有效图 %d，参与统计的 anchor %d" % (n_img, tot))
    if not tot:
        return 2
    print("%-12s %10s %8s" % ("类别", "数量", "占比"))
    for k, lab in (('unchanged', '不变'), ('switched', '易主'), ('only_A', '仅E0有'), ('only_B', '仅NWD有')):
        print("%-12s %10d %7.2f%%" % (lab, stat[k], 100.0 * stat[k] / tot))

    print("\n竞争结构（S3pre 阶段被几个 GT 争）")
    print("  %-22s %10d" % ("被多个 GT 争 (E0)", stat['multi_cand_A']))
    print("  %-22s %10d" % ("被多个 GT 争 (NWD)", stat['multi_cand_B']))
    print("  %-22s %10d   ← 多GT竞争 → 单/无" % ("竞争收窄", stat['comp_narrowed']))
    print("  %-22s %10d" % ("竞争放宽", stat['comp_widened']))
    if stat['multi_cand_A']:
        print("  ※ 两种'损失'需分开看：anchor 易主（转给别的 GT）≠ 监督丢失（该 GT 少一个 anchor）")

    print("\n易主的跨桶分布（源 → 目标，前 10）")
    for (sa, sb), c in sorted(cross.items(), key=lambda x: -x[1])[:10]:
        print("  %-10s → %-10s %6d" % (sa, sb, c))
    same = sum(c for (sa, sb), c in cross.items() if sa == sb)
    if stat['switched']:
        print("  同桶内易主占 %.1f%%" % (100.0 * same / stat['switched']))

    print("\n按源桶统计")
    for b in ("verytiny", "tiny", "small", "medium"):
        if b in per_bucket:
            d = per_bucket[b]
            print("  %-10s 仅E0有 %6d  仅NWD有 %6d  易主流出 %6d" % (b, d['only_A'], d['only_B'], d['switched_out']))

    res = {"n_img": n_img, "total": tot, "stat": dict(stat),
           "cross": {"%s->%s" % k: v for k, v in cross.items()},
           "per_bucket": {k: dict(v) for k, v in per_bucket.items()}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    json.dump(res, open(a.out + '.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print("\n写出 %s.json" % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())

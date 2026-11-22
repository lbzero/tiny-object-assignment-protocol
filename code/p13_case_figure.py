#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p13_case_figure.py —— 为 P1-3 的三类案例出对照图（CIoU 臂 vs NWD 臂）

图里画的是**分配结果**，不是预测框：两臂用同一套权重，预测框完全相同，只有
anchor 的最终归属不同。每幅子图叠加：
  - 目标框（蓝，粗线）与同图其他 GT（灰，细线）；C2 另标"得主"（紫）
  - 冲突前的候选 anchor（橙色空心圈，来自 S3pre 掩码）
  - 最终正样本 anchor（橙红实心点，来自前景掩码）
  标题给出 S3pre → S4，即"候选 → 最终正样本"。

两种输出模式
------------
  默认        每个案例一张 1×2 图（PDF+PNG），供补充材料或单独引用
  --composite 每类取 rank=1 的案例，拼成 3×2 图，供正文使用

选例来自 p13_case_selection.py（规则在看数据前固定）；本脚本只画已选定的
(image_id, gt_idx)，不参与选样。

用法
  python p13_case_figure.py --cases ... --dump ... --images ... --ann ... \
      --trace-ciou ... --trace-nwd ... --out DIR [--composite FILE.pdf]
"""
import argparse
import csv
import json
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

GT_COLOR = '#0072B2'
POS_COLOR = '#D55E00'
CAND_COLOR = '#E69F00'
OTHER_COLOR = '#999999'
NEIGH_COLOR = '#CC79A7'

CLASS_LABEL = {
    'C1_candidates_up_no_coverage': 'Candidates up, coverage unchanged',
    'C2_ownership_transfer': 'Ownership transferred to a neighbour',
    'C3_more_positives_on_covered': 'More positives on a covered box',
}
ROW_ORDER = ['C1_candidates_up_no_coverage', 'C2_ownership_transfer',
             'C3_more_positives_on_covered']


def load_trace(path):
    out = {}
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            out[(int(r['image_id']), int(r['gt_idx']))] = r
    return out


def img_box_from_norm(b, imgsz, scale, pad):
    x1, y1, x2, y2 = b[0] * imgsz, b[1] * imgsz, b[2] * imgsz, b[3] * imgsz
    return ((x1 - pad[0]) / scale, (y1 - pad[1]) / scale,
            (x2 - pad[0]) / scale, (y2 - pad[1]) / scale)


def prepare(case, args, meta, tc, tn):
    """读图、定窗口、取坐标，返回后续画图需要的一切。"""
    iid, gt = int(case['image_id']), int(case['gt_idx'])
    zc = np.load(os.path.join(args.dump, '%d_ciou.npz' % iid))
    zn = np.load(os.path.join(args.dump, '%d_nwd.npz' % iid))
    imgsz = float(zc['imgsz'][0]); sc = float(zc['scale'][0]); pad = zc['pad']
    gtb = zc['gt_boxes']; gt_ids = zc['gt_idx'].tolist()
    k = gt_ids.index(gt)
    x1, y1, x2, y2 = img_box_from_norm(gtb[k], imgsz, sc, pad)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    diag = float(np.hypot(x2 - x1, y2 - y1))
    # 固定视场：所有案例同一像素窗口，六幅图才是同一尺度、可横向比较
    half = max(args.min_window / 2.0, diag)

    src = cv2.imread(os.path.join(args.images, meta[iid]['file_name']))
    if src is None:
        raise SystemExit('读图失败: %s' % meta[iid]['file_name'])
    rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    X1, Y1 = int(max(0, cx - half)), int(max(0, cy - half))
    X2, Y2 = int(min(W, cx + half)), int(min(H, cy + half))

    nb = case.get('neighbour_gt_idx') or ''
    nb = int(nb) if str(nb).strip().isdigit() else None

    # 预先算出窗口内所有 GT 的像素框，避免重复换算
    rects = []
    for j, b in enumerate(gtb):
        bx1, by1, bx2, by2 = img_box_from_norm(b, imgsz, sc, pad)
        if bx2 < X1 or bx1 > X2 or by2 < Y1 or by1 > Y2:
            continue
        rects.append((j, bx1 - X1, by1 - Y1, bx2 - bx1, max(by2 - by1, 0.8)))

    return {'case': case, 'iid': iid, 'gt': gt, 'k': k, 'gt_ids': gt_ids,
            'imgsz': imgsz, 'sc': sc, 'pad': pad, 'zc': zc, 'zn': zn,
            'crop': rgb[Y1:Y2, X1:X2], 'X1': X1, 'Y1': Y1,
            'rects': rects, 'neigh': nb,
            's3pre_c': tc[(iid, gt)]['s3pre'], 's4_c': tc[(iid, gt)]['s4'],
            's3pre_n': tn[(iid, gt)]['s3pre'], 's4_n': tn[(iid, gt)]['s4']}


def draw_panel(ax, p, arm, fontsize=5.0, dot=9, ring=11, boxlw=1.1, label=None):
    z = p['zc'] if arm == 'CIoU' else p['zn']
    k, sc, pad = p['k'], p['sc'], p['pad']
    ax.imshow(p['crop'], interpolation='nearest', aspect='auto')

    for j, rx, ry, rw, rh in p['rects']:
        if j == k:
            lw, ec, al = boxlw, GT_COLOR, 1.0
        elif p['neigh'] is not None and p['gt_ids'][j] == p['neigh']:
            lw, ec, al = boxlw, NEIGH_COLOR, 1.0
        else:
            lw, ec, al = 0.45, OTHER_COLOR, 0.7
        ax.add_patch(Rectangle((rx, ry), rw, rh, fill=False, lw=lw, ec=ec,
                               alpha=al, zorder=3 if j == k else 2))

    pts = z['points']
    if 'cand' in z.files and k < z['cand'].shape[0]:
        cm = z['cand'][k]
        if cm.any():
            ax.scatter((pts[cm, 0] - pad[0]) / sc - p['X1'],
                       (pts[cm, 1] - pad[1]) / sc - p['Y1'],
                       s=ring, facecolors='none', edgecolors=CAND_COLOR,
                       linewidths=0.45, zorder=3.5)
    sel = z['fg'] & (z['tgt'] == k)
    if sel.any():
        ax.scatter((pts[sel, 0] - pad[0]) / sc - p['X1'],
                   (pts[sel, 1] - pad[1]) / sc - p['Y1'],
                   s=dot, c=POS_COLOR, marker='o', linewidths=0, zorder=4)
    if p['neigh'] is not None and p['neigh'] in p['gt_ids']:
        kn = p['gt_ids'].index(p['neigh'])
        seln = z['fg'] & (z['tgt'] == kn)
        if seln.any():
            ax.scatter((pts[seln, 0] - pad[0]) / sc - p['X1'],
                       (pts[seln, 1] - pad[1]) / sc - p['Y1'],
                       s=dot - 1, c=NEIGH_COLOR, marker='s', linewidths=0, zorder=4)

    nc, nn = (p['s3pre_c'], p['s4_c']) if arm == 'CIoU' else (p['s3pre_n'], p['s4_n'])
    prefix = ('%s\u00b7' % label) if label else ''
    ax.set_title('%s%s   %s$\\rightarrow$%s' % (prefix, arm, nc, nn),
                 fontsize=fontsize, pad=1.6)
    ax.set_xlim(0, p['crop'].shape[1]); ax.set_ylim(p['crop'].shape[0], 0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.45)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True)
    ap.add_argument('--dump', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--ann', required=True)
    ap.add_argument('--trace-ciou', required=True)
    ap.add_argument('--trace-nwd', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pad-factor', type=float, default=8.0)
    ap.add_argument('--min-window', type=int, default=48)
    ap.add_argument('--dpi', type=int, default=400)
    ap.add_argument('--composite', default=None, help='拼图输出路径（PDF）')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ann = json.load(open(args.ann, encoding='utf-8'))
    meta = {int(i['id']): i for i in ann['images']}
    tc, tn = load_trace(args.trace_ciou), load_trace(args.trace_nwd)
    cases = list(csv.DictReader(open(args.cases, encoding='utf-8')))

    # ---- 单案例图 ----
    stems = []
    for c in cases:
        p = prepare(c, args, meta, tc, tn)
        fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.9), dpi=args.dpi)
        draw_panel(axes[0], p, 'CIoU')
        draw_panel(axes[1], p, 'NWD')
        fig.text(0.5, 0.012,
                 '%s   image %d, gt %d, $d$ = %.1f px'
                 % (CLASS_LABEL.get(c['class'], c['class']), p['iid'], p['gt'],
                    float(c['d'])), ha='center', va='bottom', fontsize=5.4)
        fig.tight_layout(pad=0.3, rect=(0, 0.05, 1, 1))
        stem = '%s_img%d_gt%d' % (c['class'], p['iid'], p['gt'])
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(args.out, stem + '.' + ext), bbox_inches='tight')
        plt.close(fig)
        stems.append(stem)

    # ---- 拼图：每类 rank=1 ----
    if args.composite:
        chosen = {}
        for c in cases:
            if c['rank'] == '1':
                chosen[c['class']] = c
        rows = [chosen[k] for k in ROW_ORDER if k in chosen]
        fig, axes = plt.subplots(2, len(rows), figsize=(2.15 * len(rows), 2.05 * 2),
                                 dpi=args.dpi, squeeze=False)
        for j, c in enumerate(rows):
            p = prepare(c, args, meta, tc, tn)
            code = c['class'].split('_')[0]
            draw_panel(axes[0][j], p, 'CIoU', fontsize=4.8, dot=5, ring=7,
                       boxlw=0.8, label=code)
            draw_panel(axes[1][j], p, 'NWD', fontsize=4.8, dot=5, ring=7,
                       boxlw=0.8, label=code)
        fig.tight_layout(pad=0.2, h_pad=0.5, w_pad=0.35)
        fig.savefig(args.composite, bbox_inches='tight')
        plt.close(fig)
        print('[p13fig] composite -> %s (%d rows)' % (args.composite, len(rows)))

    print('[p13fig] wrote %d case panels -> %s' % (len(stems), args.out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

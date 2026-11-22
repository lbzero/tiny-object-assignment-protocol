#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bootstrap_zpr.py —— 以图像为重采样单位的 bootstrap 区间（P1-2）

为什么必须按图像重采样
----------------------
诊断 CSVs 是逐 GT 一行。若把 12,184 个 GT 当作独立观测去算区间，会低估不确定性：
同一张图内的 GT 共享场景、光照、尺度分布，彼此不独立。重采样单位必须是图像，
图内实例结构整体进出。

方法
----
1. 把逐 GT 表按 image_id 聚合为每图计数：n_bucket、zero_bucket（S4=0 的个数）。
2. 在 B 次重复中，对图像集合做有放回抽样（n_img 次），得到该重复的 ZPR。
3. 四个单元共享同一批图像，因此**同一次重复使用同一组重采样图像索引**，
   这保留了单元之间的配对结构（这一点对"差值"的区间至关重要）。
4. 报告：各单元 ZPR 的百分位区间；四个对比（界面效应、训练权重效应×2、
   交互项）的区间；以及逐 seed 的点估计。

区间是百分位 bootstrap（2.5/97.5），不做偏差校正，不做显著性检验。
报告措辞应为"图像抽样层面的不确定区间"，而非"置信区间证明差异显著"。

用法
----
  python bootstrap_zpr.py --diag-dir /root/runs/p1/diag --suffix A \
      --seeds 0 1 2 3 --bucket verytiny --B 2000 --out /root/runs/p1/boot_A.json
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

# 单元名模板：{s} 为 seed
CELLS = {
    'E0_ciou': 'E0_default_s{s}',
    'E0_nwd': 'E0_s{s}_nwd',
    'NWD_ciou': 'NWD_s{s}_ciou',
    'NWD_nwd': 'NWD_nwd_s{s}',
}
# 对比：(名称, 被减单元, 减单元)
CONTRASTS = [
    ('interface_nwd_minus_ciou_at_E0', 'E0_nwd', 'E0_ciou'),
    ('weight_effect_ciou_NWD_minus_E0', 'NWD_ciou', 'E0_ciou'),
    ('weight_effect_nwd_NWD_minus_E0', 'NWD_nwd', 'E0_nwd'),
    ('interaction', 'NWD_nwd', 'E0_nwd'),
]


def load_counts(path, bucket):
    """image_id -> (n_bucket, n_zero)"""
    out = {}
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['bucket'] != bucket:
                continue
            iid = int(r['image_id'])
            n, z = out.get(iid, (0, 0))
            out[iid] = (n + 1, z + (1 if int(r['s4']) == 0 else 0))
    return out


def zpr(counts, idx):
    n = z = 0
    for i in idx:
        c = counts.get(i)
        if c:
            n += c[0]
            z += c[1]
    return (100.0 * z / n) if n else float('nan')


def pct(a, q=2.5):
    return float(np.percentile(a, q)), float(np.percentile(a, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag-dir', required=True)
    ap.add_argument('--suffix', default='A', help='只用于输出标注')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--bucket', default='verytiny')
    ap.add_argument('--B', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261121)
    ap.add_argument('--ids-file', default=None,
                    help='诊断子集的图像清单（500 行）。给出时以此为重采样总体，'
                         '使区间反映"换一批同规模图像"的抽样不确定性，'
                         '而不是只反映"含该尺寸桶的图像"内部的重采样。')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    # 图像全集：任一单元出现过的 image_id（四个单元共享同一批图）
    universe = set()
    per_cell_seed = {}
    for cell, pat in CELLS.items():
        for s in args.seeds:
            p = os.path.join(args.diag_dir, pat.format(s=s) + '.csv')
            if not os.path.isfile(p):
                raise SystemExit('缺文件: %s' % p)
            c = load_counts(p, args.bucket)
            per_cell_seed[(cell, s)] = c
            universe |= set(c.keys())
    images = np.array(sorted(universe))
    if args.ids_file:
        ids = [int(x) for x in open(args.ids_file) if x.strip()]
        images = np.array(sorted(ids))
    n_img = len(images)

    # 点估计
    point = {}
    for cell in CELLS:
        vals = [zpr(per_cell_seed[(cell, s)], images) for s in args.seeds]
        point[cell] = {'per_seed': [round(v, 3) for v in vals],
                       'mean': round(float(np.mean(vals)), 3)}
    contrast_point = {}
    for name, a, b in CONTRASTS[:3]:
        vals = [zpr(per_cell_seed[(a, s)], images) - zpr(per_cell_seed[(b, s)], images)
                for s in args.seeds]
        contrast_point[name] = {'per_seed': [round(v, 3) for v in vals],
                                'mean': round(float(np.mean(vals)), 3)}
    # 交互项按 2×2 定义逐 seed 计算
    inter = []
    for s in args.seeds:
        d = (zpr(per_cell_seed[('NWD_nwd', s)], images) - zpr(per_cell_seed[('E0_nwd', s)], images)) \
            - (zpr(per_cell_seed[('NWD_ciou', s)], images) - zpr(per_cell_seed[('E0_ciou', s)], images))
        inter.append(d)
    contrast_point['interaction'] = {'per_seed': [round(v, 3) for v in inter],
                                     'mean': round(float(np.mean(inter)), 3)}

    # 图像级 bootstrap：同一次重复用同一组图像索引（保留单元间配对）
    rng = np.random.default_rng(args.seed)
    boot_cell = defaultdict(list)
    boot_con = defaultdict(list)
    for _ in range(args.B):
        idx = images[rng.integers(0, n_img, size=n_img)]
        v = {}
        for cell in CELLS:
            v[cell] = np.array([zpr(per_cell_seed[(cell, s)], idx) for s in args.seeds])
            boot_cell[cell].append(float(np.mean(v[cell])))
        for name, a, b in CONTRASTS[:3]:
            boot_con[name].append(float(np.mean(v[a] - v[b])))
        boot_con['interaction'].append(
            float(np.mean((v['NWD_nwd'] - v['E0_nwd']) - (v['NWD_ciou'] - v['E0_ciou']))))

    out = {
        'meta': {'diag_dir': args.diag_dir, 'subset': args.suffix, 'bucket': args.bucket,
                 'n_images': int(n_img), 'B': args.B, 'seed': args.seed,
                 'unit': 'image (resampled with replacement, within-image structure kept)',
                 'seeds': args.seeds,
                 'note': ('Percentile bootstrap over images, common resampling across cells '
                          'and seeds to preserve pairing. Intervals describe image-sampling '
                          'uncertainty only.')},
        'cells': {}, 'contrasts': {},
    }
    for cell in CELLS:
        lo, hi = pct(boot_cell[cell])
        out['cells'][cell] = {**point[cell], 'ci95': [round(lo, 3), round(hi, 3)]}
    for name, _, _ in CONTRASTS:
        lo, hi = pct(boot_con[name])
        out['contrasts'][name] = {**contrast_point[name],
                                  'ci95': [round(lo, 3), round(hi, 3)]}

    json.dump(out, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    print('[boot] subset %s  bucket=%s  n_img=%d  B=%d' %
          (args.suffix, args.bucket, n_img, args.B))
    for cell in CELLS:
        d = out['cells'][cell]
        print('[boot] %-9s mean %7.3f%%  per-seed %s  CI [%7.3f, %7.3f]'
              % (cell, d['mean'], d['per_seed'], d['ci95'][0], d['ci95'][1]))
    for name, _, _ in CONTRASTS:
        d = out['contrasts'][name]
        print('[boot] %-34s %+7.3f pp  CI [%+7.3f, %+7.3f]  per-seed %s'
              % (name, d['mean'], d['ci95'][0], d['ci95'][1], d['per_seed']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

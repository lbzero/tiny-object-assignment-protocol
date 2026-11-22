#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""subset_robustness.py —— P0-1：诊断子集 A/B 的稳健性对照

输入
----
两个目录，各自含 16 个逐 GT 诊断 CSV（4 单元 × 4 seed）：
    diag/   Subset A（验证集列表前 500 张）
    diagB/  Subset B（分层系统抽样 500 张，make_subset_b.py）
单元命名沿用既有约定：
    E0_default_s{s}   E0 权重 + CIoU
    E0_s{s}_nwd       E0 权重 + NWD
    NWD_s{s}_ciou     NWD 权重 + CIoU
    NWD_nwd_s{s}      NWD 权重 + NWD

输出什么
--------
1. 每个子集、每个尺寸桶的四个单元 ZPR（均值与逐 seed）。
2. 四个对比（界面效应、训练权重效应×2、交互项）在两个子集上的逐 seed 值与均值。
3. 稳健性判据（预先声明，避免事后挑选）：
   R1 方向一致性：两子集均值的符号相同。
   R2 逐 seed 方向：两子集内 4/4 seed 符号相同。
   R3 幅度稳定性：|A−B| 是否落在两子集自身 seed 离散度之内（用 max-min 范围做参照）。
   R4 结论依赖：主要效应在两子集上的区间是否重叠（区间取 seed 范围）。

判据在脚本里固定，不看结果再定阈值；R3 的参照量是 seed 范围而非 SD，因为 n=4。

用法
----
  python subset_robustness.py --dirA /root/runs/p1/diag --dirB /root/runs/p1/diagB \
      --out-json /root/runs/p1/subset_robustness.json \
      --out-tex /root/runs/p1/subset_robustness.tex
"""
import argparse
import csv
import json
import os

import numpy as np

CELLS = {
    'E0_ciou': 'E0_default_s{s}',
    'E0_nwd': 'E0_s{s}_nwd',
    'NWD_ciou': 'NWD_s{s}_ciou',
    'NWD_nwd': 'NWD_nwd_s{s}',
}
BUCKETS = ['verytiny', 'tiny', 'small', 'medium', 'all']
CONTRASTS = [
    ('interface', 'E0_nwd', 'E0_ciou'),
    ('weight_ciou', 'NWD_ciou', 'E0_ciou'),
    ('weight_nwd', 'NWD_nwd', 'E0_nwd'),
]


def zpr(path, bucket):
    n = z = 0
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if bucket != 'all' and r['bucket'] != bucket:
                continue
            n += 1
            z += (int(r['s4']) == 0)
    return (100.0 * z / n) if n else float('nan'), n, z


def load_dir(d, seeds):
    out = {}
    for cell, pat in CELLS.items():
        for s in seeds:
            p = os.path.join(d, pat.format(s=s) + '.csv')
            if not os.path.isfile(p):
                raise SystemExit('缺文件: %s' % p)
            out[(cell, s)] = p
    return out


def summarize(paths, seeds):
    res = {}
    for bucket in BUCKETS:
        cell_val = {}
        for cell in CELLS:
            vals = []
            det = []
            for s in seeds:
                r, n, z = zpr(paths[(cell, s)], bucket)
                vals.append(r)
                det.append({'seed': s, 'rate': round(r, 4), 'n': n, 'z': z})
            cell_val[cell] = {'mean': round(float(np.mean(vals)), 4),
                              'per_seed': [round(v, 4) for v in vals],
                              'detail': det}
        con = {}
        for name, a, b in CONTRASTS:
            d = [cell_val[a]['per_seed'][i] - cell_val[b]['per_seed'][i] for i in range(len(seeds))]
            con[name] = {'mean': round(float(np.mean(d)), 4),
                         'per_seed': [round(v, 4) for v in d]}
        inter = [(cell_val['NWD_nwd']['per_seed'][i] - cell_val['E0_nwd']['per_seed'][i])
                 - (cell_val['NWD_ciou']['per_seed'][i] - cell_val['E0_ciou']['per_seed'][i])
                 for i in range(len(seeds))]
        con['interaction'] = {'mean': round(float(np.mean(inter)), 4),
                              'per_seed': [round(v, 4) for v in inter]}
        res[bucket] = {'cells': cell_val, 'contrasts': con}
    return res


def judge(a, b, seeds):
    """对每个桶、每个对比给出 R1–R4。"""
    out = {}
    for bucket in BUCKETS:
        out[bucket] = {}
        for name in ('interface', 'weight_ciou', 'weight_nwd', 'interaction'):
            ca, cb = a[bucket]['contrasts'][name], b[bucket]['contrasts'][name]
            sa, sb = np.sign(ca['mean']), np.sign(cb['mean'])
            same_mean = bool(sa == sb and sa != 0)
            per_seed_same = all(np.sign(x) == np.sign(y) for x, y in
                                zip(ca['per_seed'], cb['per_seed'])) if same_mean else False
            ra = max(ca['per_seed']) - min(ca['per_seed'])
            rb = max(cb['per_seed']) - min(cb['per_seed'])
            delta = abs(ca['mean'] - cb['mean'])
            scale = max(ra, rb, 1e-9)
            out[bucket][name] = {
                'A_mean': ca['mean'], 'B_mean': cb['mean'],
                'abs_delta': round(delta, 4),
                'seed_range_A': round(ra, 4), 'seed_range_B': round(rb, 4),
                'R1_same_sign': same_mean,
                'R2_per_seed_same_sign': per_seed_same,
                'R3_delta_within_seed_range': bool(delta <= scale),
                'R4_seed_ranges_overlap': bool(
                    not (max(ca['per_seed']) < min(cb['per_seed'])
                         or max(cb['per_seed']) < min(ca['per_seed']))),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirA', required=True)
    ap.add_argument('--dirB', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-tex', default=None)
    args = ap.parse_args()

    pA, pB = load_dir(args.dirA, args.seeds), load_dir(args.dirB, args.seeds)
    A, B = summarize(pA, args.seeds), summarize(pB, args.seeds)
    rep = {'seeds': args.seeds, 'subsetA': A, 'subsetB': B,
           'judgement': judge(A, B, args.seeds),
           'criteria': {
               'R1': 'mean effect has the same sign in both subsets',
               'R2': 'per-seed effect has the same sign in both subsets (4/4)',
               'R3': '|A-B| does not exceed the larger within-subset seed range',
               'R4': 'the two subsets\' per-seed ranges overlap',
               'fixed_before_running': True}}
    json.dump(rep, open(args.out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    print('[robust] very-tiny ZPR (mean over %d seeds)' % len(args.seeds))
    for cell in CELLS:
        print('  %-9s A %6.3f%%   B %6.3f%%' % (cell, A['verytiny']['cells'][cell]['mean'],
                                                B['verytiny']['cells'][cell]['mean']))
    print('[robust] very-tiny contrasts (pp)')
    for name in ('interface', 'weight_ciou', 'weight_nwd', 'interaction'):
        j = rep['judgement']['verytiny'][name]
        print('  %-12s A %+6.3f  B %+6.3f  |d| %5.3f  R1 %s R2 %s R3 %s R4 %s'
              % (name, j['A_mean'], j['B_mean'], j['abs_delta'],
                 j['R1_same_sign'], j['R2_per_seed_same_sign'],
                 j['R3_delta_within_seed_range'], j['R4_seed_ranges_overlap']))

    if args.out_tex:
        rows = []
        for bucket, label in (('verytiny', 'Very tiny'), ('tiny', 'Tiny'),
                              ('small', 'Small'), ('medium', 'Medium'), ('all', 'All')):
            for name, nlabel in (('interface', 'Interface (NWD$-$CIoU at E0)'),
                                 ('weight_ciou', 'Weight (NWD$-$E0, CIoU)'),
                                 ('weight_nwd', 'Weight (NWD$-$E0, NWD)'),
                                 ('interaction', 'Interaction')):
                j = rep['judgement'][bucket][name]
                rows.append('%s & %s & %+.2f & %+.2f & %.2f & %s & %s \\\\'
                            % (label, nlabel, j['A_mean'], j['B_mean'], j['abs_delta'],
                               'yes' if j['R1_same_sign'] and j['R2_per_seed_same_sign'] else 'no',
                               'yes' if j['R3_delta_within_seed_range'] else 'no'))
        open(args.out_tex, 'w').write('\n'.join(rows) + '\n')
        print('[robust] wrote %s (%d rows)' % (args.out_tex, len(rows)))


if __name__ == '__main__':
    raise SystemExit(main())

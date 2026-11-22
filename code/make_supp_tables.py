#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成补充材料所需的逐 seed 表格（LaTeX 片段）。

数据来源全部是已归档的产物，不重新训练、不重新推理：
  诊断逐 GT     /root/runs/p1/diag/*.csv          (p0_6_dynamic_trace.py)
  软标签        /root/runs/p1/soft2/*.json        (soft_label_analysis.py)
  anchor 竞争   /root/runs/p1/anchor_comp_s*.json (anchor_transfer_trained.py)
  逐 seed PR    /root/runs/p1/score_seed/*.json   (score_analysis_v2.py)
  训练配置      /root/runs/p1/<run>/args.yaml + results.csv

输出：tables/*.tex，由 supplementary.tex 以 \input 引入。
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np

RUNS = [
    ('E0', 0, 'E0_default_s0'), ('E0', 1, 'E0_default_s1'),
    ('E0', 2, 'E0_default_s2'), ('E0', 3, 'E0_default_s3'),
    ('NWD', 0, 'NWD_nwd_s0'), ('NWD', 1, 'NWD_nwd_s1'),
    ('NWD', 2, 'NWD_nwd_s2'), ('NWD', 3, 'NWD_nwd_s3'),
]

# 2x2 四单元格：权重来源 × 相似度
CELLS = {
    ('E0', 'ciou'): 'E0_default_s%d.csv',
    ('E0', 'nwd'): 'E0_s%d_nwd.csv',
    ('NWD', 'ciou'): 'NWD_s%d_ciou.csv',
    ('NWD', 'nwd'): 'NWD_nwd_s%d.csv',
}


def vt_zero_rate(path):
    """very tiny 桶的零正样本率（%），分母为该桶 GT 数。"""
    n = z = 0
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['bucket'] != 'verytiny':
                continue
            n += 1
            if int(r['s4']) == 0:
                z += 1
    return 100.0 * z / n if n else float('nan')


def identity_residuals(path):
    """返回 (I1, I2, I3) 的最大绝对残差与违反行数。"""
    r1 = r2 = r3 = 0
    v1 = v2 = v3 = 0
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            s2, spre, spost, s4 = (int(r['s2_in_s1']), int(r['s3pre']),
                                   int(r['s3post']), int(r['s4']))
            lost, gain = int(r['n_lost']), int(r['n_gained'])
            d1 = abs(s2 - spre)
            d2 = abs(spost - s4)
            d3 = abs(spre - lost + gain - spost)
            r1, r2, r3 = max(r1, d1), max(r2, d2), max(r3, d3)
            v1 += d1 != 0
            v2 += d2 != 0
            v3 += d3 != 0
    return (r1, v1), (r2, v2), (r3, v3)


def read_best_epoch(run_dir):
    """从 results.csv 取 mAP50-95 最大的 epoch（strip 后 best.pt 的 epoch 字段已置 -1）。"""
    p = os.path.join(run_dir, 'results.csv')
    if not os.path.isfile(p):
        return None
    best_ep, best_v = None, -1
    with open(p, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            k = [c for c in row if 'mAP50-95' in c]
            if not k:
                continue
            try:
                v = float(row[k[0]])
            except (TypeError, ValueError):
                continue
            if v > best_v:
                best_v, best_ep = v, int(float(row['epoch']))
    return best_ep


def read_args(run_dir):
    """读 args.yaml 的少量标量字段（不依赖 pyyaml）。"""
    out = {}
    p = os.path.join(run_dir, 'args.yaml')
    if not os.path.isfile(p):
        return out
    for ln in open(p, encoding='utf-8'):
        m = re.match(r'^(\w+):\s*(.*)$', ln.strip())
        if m and m.group(1) in ('model', 'optimizer', 'batch', 'imgsz', 'seed',
                                'epochs', 'lr0', 'lrf', 'close_mosaic'):
            out[m.group(1)] = m.group(2).strip().strip('"')
    return out



def fmt_int(x):
    """整数带 LaTeX 千位分隔：54997 -> 54{,}997。"""
    return ('%d' % round(x)).replace(',', '') if False else ('{:,}'.format(int(round(x)))).replace(',', '{,}')


def write_rows(path, rows):
    """行片段：末行不带 \\\\，由宿主 tabular 补，避免 Misplaced \\noalign。"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows) + '\n')


def fmt(x, nd=2, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '--'
    s = ('%+.' + str(nd) + 'f') % x if sign else ('%.' + str(nd) + 'f') % x
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag', default='/root/runs/p1/diag')
    ap.add_argument('--soft', default='/root/runs/p1/soft2')
    ap.add_argument('--anchor', default='/root/runs/p1')
    ap.add_argument('--score', default='/root/runs/p1/score_seed')
    ap.add_argument('--runs', default='/root/runs/p1')
    ap.add_argument('--out', default='/root/runs/p1/supp_tables')
    a = ap.parse_args()
    T = os.path.join(a.out, 'tables')
    os.makedirs(T, exist_ok=True)

    # ---------- T1 运行清单 ----------
    rows = []
    for arm, sd, run in RUNS:
        d = os.path.join(a.runs, run)
        ar = read_args(d)
        resumed = 'yes' if 'last.pt' in ar.get('model', '') else 'no'
        be = read_best_epoch(d)
        meta = os.path.join(a.diag, run + '_meta.json')
        sha = json.load(open(meta))['ckpt_sha256_16'] if os.path.isfile(meta) else '--'
        rows.append(r'%s & %d & %s & %s & \texttt{%s} \\' %
                    (run.replace('_', r'\_'), sd, resumed,
                     be if be is not None else '--', sha))
    write_rows(os.path.join(T, 't1_runs.tex'), rows)

    # ---------- T2 四单元格逐 seed ----------
    cells = {}
    for key, pat in CELLS.items():
        vals = []
        for sd in range(4):
            p = os.path.join(a.diag, pat % sd)
            vals.append(vt_zero_rate(p) if os.path.isfile(p) else float('nan'))
        cells[key] = vals
    order = [('E0', 'ciou'), ('E0', 'nwd'), ('NWD', 'ciou'), ('NWD', 'nwd')]
    label = {('E0', 'ciou'): r'E0 weights, CIoU', ('E0', 'nwd'): r'E0 weights, NWD',
             ('NWD', 'ciou'): r'NWD weights, CIoU', ('NWD', 'nwd'): r'NWD weights, NWD'}
    rows = []
    for k in order:
        v = np.array(cells[k], dtype=float)
        rows.append(r'%s & %s & %s & %s \\' % (
            label[k], ' & '.join(fmt(x, 2) for x in v),
            fmt(float(np.nanmean(v)), 2), fmt(float(np.nanstd(v, ddof=1)), 2)))
    write_rows(os.path.join(T, 't2_cells.tex'), rows)

    # ---------- T3 因子效应与交互 ----------
    def eff(a_key, b_key):
        return np.array(cells[b_key], dtype=float) - np.array(cells[a_key], dtype=float)

    sim_e0 = eff(('E0', 'ciou'), ('E0', 'nwd'))
    sim_nw = eff(('NWD', 'ciou'), ('NWD', 'nwd'))
    w_ci = eff(('E0', 'ciou'), ('NWD', 'ciou'))
    w_nw = eff(('E0', 'nwd'), ('NWD', 'nwd'))
    inter = sim_nw - sim_e0
    rows = []
    for lab, v in ((r'Similarity $|$ E0 weights', sim_e0),
                   (r'Similarity $|$ NWD weights', sim_nw),
                   (r'Weights $|$ CIoU', w_ci),
                   (r'Weights $|$ NWD', w_nw),
                   (r'Interaction $\Delta_{\mathrm{int}}$', inter)):
        rows.append(r'%s & %s & %s & %s & [%s, %s] \\' % (
            lab, ' & '.join(fmt(x, 2, True) for x in v),
            fmt(float(np.nanmean(v)), 2, True),
            fmt(float(np.nanstd(v, ddof=1)), 2),
            fmt(float(np.nanmin(v)), 2, True),
            fmt(float(np.nanmax(v)), 2, True)))
    write_rows(os.path.join(T, 't3_contrasts.tex'), rows)

    # ---------- T4 软标签与耦合 ----------
    rows = []
    for arm, sd, run in RUNS:
        p = os.path.join(a.soft, run + '.json')
        if not os.path.isfile(p):
            continue
        d = json.load(open(p))
        rows.append(r'%s & %d & %s & %s & %s & %s & %s \\' % (
            arm, sd,
            fmt(d['final_ts']['mean'], 4),
            fmt(d['q1_norm']['ratio']['p50'], 3),
            fmt(d['q2_ciou']['dist']['mean'], 4),
            fmt(d['q2_ciou']['corr_ts']['pearson'], 4),
            fmt(d['q2_ciou']['high']['corr']['pearson'], 4)))
    write_rows(os.path.join(T, 't4_soft.tex'), rows)

    # ---------- T5 anchor 竞争 ----------
    # 该文件本身就是 E0/NWD 的配对比较，每个 seed 一行（不分臂）
    rows = []
    for sd in range(4):
        p = os.path.join(a.anchor, 'anchor_comp_s%d.json' % sd)
        if not os.path.isfile(p):
            continue
        d = json.load(open(p))['stat']
        ma, mb = d.get('multi_cand_A', 0), d.get('multi_cand_B', 0)
        net = mb - ma
        rows.append(r'%d & %s & %s & %s & %s & %s%s \\' % (
            sd, fmt_int(ma), fmt_int(mb), fmt_int(d.get('comp_narrowed', 0)),
            fmt_int(d.get('comp_widened', 0)),
            '$+$' if net >= 0 else '$-$', fmt_int(abs(net))))
    write_rows(os.path.join(T, 't5_anchor.tex'), rows)

    # ---------- T6 逐 seed 固定阈值 PR ----------
    rows = []
    for arm, sd, run in RUNS:
        p = os.path.join(a.score, run + '.json')
        if not os.path.isfile(p):
            continue
        d = json.load(open(p))[arm]
        out = [arm, str(sd)]
        for k in ('iou50', 'iou75'):
            c = [x for x in d['pr'][k] if abs(x['thr'] - 0.5) < 1e-9][0]
            out += [fmt_int(c['tp'] + c['fp']), fmt_int(c['tp']), fmt_int(c['fp']),
                    '%s / %s' % (fmt(c['prec'], 3), fmt(c['rec'], 3))]
        rows.append(' & '.join(out) + r' \\')
    write_rows(os.path.join(T, 't6_pr.tex'), rows)

    # ---------- T7 恒等式残差 ----------
    rows = []
    for arm, sd, run in RUNS:
        p = os.path.join(a.diag, run + '.csv')
        if not os.path.isfile(p):
            continue
        (r1, v1), (r2, v2), (r3, v3) = identity_residuals(p)
        rows.append(r'%s & %d & %d / %d & %d / %d & %d / %d \\' %
                    (arm, sd, r1, v1, r2, v2, r3, v3))
    write_rows(os.path.join(T, 't7_identity.tex'), rows)

    # ---------- 汇总打印 ----------
    print('=== T2 四单元格（vt 零正样本率 %%）===')
    for k in order:
        v = np.array(cells[k], float)
        print('  %-30s %s  mean %.2f  sd %.2f' % (label[k],
              ' '.join('%.2f' % x for x in v), np.nanmean(v), np.nanstd(v, ddof=1)))
    print('\n=== T3 因子效应 ===')
    for lab, v in (('sim|E0', sim_e0), ('sim|NWD', sim_nw), ('w|CIoU', w_ci),
                   ('w|NWD', w_nw), ('inter', inter)):
        print('  %-10s %s  mean %+.2f  sd %.2f  range [%+.2f, %+.2f]' % (
            lab, ' '.join('%+.2f' % x for x in v), np.nanmean(v),
            np.nanstd(v, ddof=1), np.nanmin(v), np.nanmax(v)))
    print('\n表格已写入:', T)


if __name__ == '__main__':
    main()

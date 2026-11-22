#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_eval_ema60_table.py —— 把 8 个 run 的 epoch-60 EMA 官方评测汇总成补充材料表

背景
----
正文的分配统计取 epoch-60 的 EMA 分支，而 Table（检测性能）的 AP 取各 run 的
best.pt —— 是跨节点对应。本脚本汇总"同一批 epoch-60 EMA 权重"的官方评测，
用于把那一处对应升级为同节点对应（或证伪）。

输入
----
<root>/<run>/metrics.json，run ∈ {E0_default_s0..3, NWD_nwd_s0..3}

输出
----
--out-tex   补充材料表行（6 个指标 × 2 臂）
--out-json  逐 seed 数值、两臂均值/SD、与 best.pt 节点的差值

约定：SD 以第 4 位小数为单位写在括号里（与正文一致，如 (37) 表示 0.0037）。
"""
import argparse
import json
import os
import statistics as st

METRICS = ['AP', 'AP50', 'AP75', 'APvt', 'APs', 'APm']
ARMS = [('E0', 'E0_default'), ('NWD', 'NWD_nwd')]
# 正文 Table（best.pt 节点）的均值，用于算差值；来自 §VI-E
BEST_PT = {
    'E0':  {'AP': 0.2493, 'AP50': 0.5600, 'AP75': 0.1824, 'APvt': 0.1005, 'APs': 0.3269, 'APm': 0.4167},
    'NWD': {'AP': 0.2324, 'AP50': 0.5606, 'AP75': 0.1424, 'APvt': 0.1000, 'APs': 0.3072, 'APm': 0.3844},
}


def load(root, arm_prefix, seeds=(0, 1, 2, 3)):
    out = {}
    for s in seeds:
        p = os.path.join(root, '%s_s%d' % (arm_prefix, s), 'metrics.json')
        if not os.path.exists(p):
            continue
        out[s] = json.load(open(p, encoding='utf-8'))
    return out


def fmt(mean, sd):
    return '%.4f (%d)' % (mean, round(sd * 1e4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out-tex', required=True)
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()

    summary, rows, notes = {}, [], []
    for label, prefix in ARMS:
        runs = load(args.root, prefix)
        if not runs:
            raise SystemExit('没有找到 %s 的结果' % prefix)
        if len(runs) < 4:
            notes.append('%s 只有 %d 个 seed' % (label, len(runs)))
        cell = {}
        for m in METRICS:
            vals = [r[m] for r in runs.values()]
            cell[m] = {'mean': st.mean(vals), 'sd': st.stdev(vals) if len(vals) > 1 else 0.0,
                       'per_seed': [round(v, 5) for v in vals]}
        summary[label] = cell
        rows.append('%s (epoch-60 EMA) & %s' % (
            label, ' & '.join(fmt(cell[m]['mean'], cell[m]['sd']) for m in METRICS)))

    # 同节点下的 AP75 差（NWD − E0），单位 pp
    d_ap75 = (summary['NWD']['AP75']['mean'] - summary['E0']['AP75']['mean']) * 100
    d_ap = (summary['NWD']['AP']['mean'] - summary['E0']['AP']['mean']) * 100
    d_vt = (summary['NWD']['APvt']['mean'] - summary['E0']['APvt']['mean']) * 100
    d_50 = (summary['NWD']['AP50']['mean'] - summary['E0']['AP50']['mean']) * 100

    rep = {'node': 'epoch-60 EMA', 'n_per_arm': {k: len(load(args.root, p)) for k, p in ARMS},
           'cells': summary,
           'contrast_pp': {'AP': d_ap, 'AP50': d_50, 'AP75': d_ap75, 'APvt': d_vt},
           'best_pt_reference': BEST_PT,
           'best_pt_contrast_pp': {'AP': -1.69, 'AP50': 0.06, 'AP75': -4.00, 'APvt': -0.05},
           'notes': notes}
    json.dump(rep, open(args.out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    with open(args.out_tex, 'w', encoding='utf-8', newline='\n') as f:
        # 行间用 \\，末行不加（由宿主 tabular 补），与 t2..t7 的惯例一致
        f.write(' \\\\\n'.join(rows) + '\n')

    print('[ema60] 每臂 %d 个 seed' % len(next(iter(summary.values()))['AP']['per_seed']))
    for label in ('E0', 'NWD'):
        print('  %-4s AP %.4f  AP50 %.4f  AP75 %.4f  APvt %.4f'
              % (label, summary[label]['AP']['mean'], summary[label]['AP50']['mean'],
                 summary[label]['AP75']['mean'], summary[label]['APvt']['mean']))
    print('[ema60] 同节点对比 NWD−E0:  AP %+.2f  AP50 %+.2f  AP75 %+.2f  APvt %+.2f pp'
          % (d_ap, d_50, d_ap75, d_vt))
    print('[ema60] 参考（best.pt 节点）: AP −1.69  AP50 +0.06  AP75 −4.00  APvt −0.05 pp')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

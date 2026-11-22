#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_subset_b.py —— P0-1 诊断子集稳健性：构造 Subset B（分层抽样）

背景
----
论文 2×2 诊断目前使用验证集列表前 500 张（Subset A，image_id 0..499）。
A 不是随机抽样，只是列表顺序截断。虽然四个尺寸区间的比例与全量接近，
但"比例接近"只约束尺寸构成，不约束场景、类别与目标密度。

本脚本从**剩余**验证图像中抽取 Size=500 的 Subset B：
  1. 分层变量：主导尺寸桶（4 层，按图内 GT 数最多的桶；并列取更小的桶）
  2. 桶内按 (GT 数, 主导类别) 排序后做**系统抽样**（随机起点、固定步长）
  3. 各桶配额按该桶在剩余池中的图像数比例分配（最大余数法保证总数=500）
  4. 输出 ids 清单 + 逐图 manifest + 平衡性报告（A / B / 全集 的 SMD）

为什么用"桶内排序 + 系统抽样"而不是"桶 × 密度四分位交叉后的层内随机"
--------------------------------------------------------------------------
GT 数在 AI-TODv2 里是重尾的（单图 1 → 数百）。按四分位交叉分层后，层内
随机抽样的抽样方差仍很大：首版实现抽出的 B 平均 30.6 GT/图，比全集的 26.8
高 14%，远超随机误差。系统抽样等价于隐式分层——样本的经验分位数与总体
分位数之差为 O(1/n) 量级，因此密度分布、尺寸构成与类别构成都会紧贴全集。

注意：分层只对齐**可观测**协变量。脚本不声称 B 在场景语义上代表全集，
论文里应写作"在尺寸构成与目标密度上均衡"，而不是"具有代表性"。

用法
----
  python make_subset_b.py \
      --map /root/runs/p1/gtmap_val_map.csv \
      --ann /root/data/AI-TOD/annotations_v2/aitod_val_v2.json \
      --exclude /root/runs/p1/diag_ids500.txt \
      --n 500 --seed 20261121 \
      --out-ids /root/runs/p1/diag_subsetB_ids500.txt \
      --out-manifest /root/runs/p1/diag_subsetB_manifest.csv \
      --out-report /root/runs/p1/diag_subsetB_balance.json
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

BUCKETS = ['verytiny', 'tiny', 'small', 'medium']
BUCKET_EDGES = [(0.0, 64.0), (64.0, 256.0), (256.0, 1024.0), (1024.0, float('inf'))]


def bucket_of_area(area):
    for name, (lo, hi) in zip(BUCKETS, BUCKET_EDGES):
        if lo <= area < hi:
            return name
    return BUCKETS[-1]


def dominant_bucket(counts):
    """GT 数最多的桶；并列时取更小的桶（更保守）。"""
    best, best_n = None, -1
    for b in BUCKETS:                      # 顺序即从小到大
        n = counts.get(b, 0)
        if n > best_n:
            best, best_n = b, n
    return best if best_n > 0 else None


def load_images(map_csv, ann_json):
    """逐图协变量：GT 数、分桶计数、类别计数、面积均值。"""
    gt = defaultdict(list)
    with open(map_csv, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            gt[int(r['image_id'])].append(r)

    ann = json.load(open(ann_json, encoding='utf-8'))
    meta = {int(i['id']): i for i in ann['images']}
    # 官方标注里的 GT 总数（用于声明 75,091 vs 75,065）
    off_counts = Counter(int(a['image_id']) for a in ann['annotations'])

    rows = []
    for iid in sorted(meta):
        recs = gt.get(iid, [])
        counts = Counter(r['bucket'] for r in recs)
        cats = Counter(int(r['category_id']) for r in recs)
        areas = [float(r['area']) for r in recs]
        rows.append({
            'image_id': iid,
            'n_gt_visible': len(recs),
            'n_gt_official': off_counts.get(iid, 0),
            'n_vt': counts.get('verytiny', 0),
            'n_t': counts.get('tiny', 0),
            'n_s': counts.get('small', 0),
            'n_m': counts.get('medium', 0),
            'gt_area_mean': float(np.mean(areas)) if areas else 0.0,
            'gt_diameter_mean': float(np.mean([float(r['d']) for r in recs])) if recs else 0.0,
            'dominant': dominant_bucket(counts),
            'cats': cats,
            'w': int(meta[iid]['width']), 'h': int(meta[iid]['height']),
        })
    return rows


def density_quartile_edges(pool):
    v = np.array([r['n_gt_visible'] for r in pool], dtype=float)
    return list(np.percentile(v, [25, 50, 75]))


def assign_stratum(r, edges):
    q = 0
    for e in edges:
        if r['n_gt_visible'] > e:
            q += 1
    return '%s|Q%d' % (r['dominant'], q)


def allocate(sizes, total):
    """最大余数法：按规模比例分配配额，保证总和 = total。"""
    n = sum(sizes.values())
    exact = {k: v * total / n for k, v in sizes.items()}
    base = {k: int(np.floor(v)) for k, v in exact.items()}
    rem = total - sum(base.values())
    order = sorted(exact, key=lambda k: (exact[k] - base[k]), reverse=True)
    for k in order[:rem]:
        base[k] += 1
    return base


def smd(a, b):
    """标准化均值差（合并 SD）；两侧任一 SD=0 时返回 0 或 inf。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    sp = np.sqrt((sa ** 2 + sb ** 2) / 2)
    if sp == 0:
        return 0.0 if ma == mb else float('inf')
    return float((ma - mb) / sp)


def describe(rows, label):
    n = len(rows)
    ngt = sum(r['n_gt_visible'] for r in rows)
    per_bucket = {
        'verytiny': sum(r['n_vt'] for r in rows),
        'tiny': sum(r['n_t'] for r in rows),
        'small': sum(r['n_s'] for r in rows),
        'medium': sum(r['n_m'] for r in rows),
    }
    cats = Counter()
    for r in rows:
        cats.update(r['cats'])
    ncat = sum(cats.values()) or 1
    out = {
        'label': label, 'n_images': n, 'n_gt_visible': ngt,
        'gt_per_image_mean': float(np.mean([r['n_gt_visible'] for r in rows])) if n else 0.0,
        'gt_per_image_sd': float(np.std([r['n_gt_visible'] for r in rows], ddof=1)) if n > 1 else 0.0,
        'gt_per_image_p50': float(np.median([r['n_gt_visible'] for r in rows])) if n else 0.0,
        'gt_diameter_mean': float(np.mean([r['gt_diameter_mean'] for r in rows if r['n_gt_visible']])) if n else 0.0,
        'bucket_gt_share_pct': {b: round(100.0 * v / max(ngt, 1), 2) for b, v in per_bucket.items()},
        'bucket_gt_count': per_bucket,
        'empty_image_share_pct': round(100.0 * sum(1 for r in rows if r['n_gt_visible'] == 0) / max(n, 1), 2),
        'category_gt_share_pct': {str(k): round(100.0 * v / ncat, 2) for k, v in sorted(cats.items())},
        'image_wh_unique': sorted({(r['w'], r['h']) for r in rows}),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', required=True)
    ap.add_argument('--ann', required=True)
    ap.add_argument('--exclude', required=True, help='Subset A 的 id 清单')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--seed', type=int, default=20261121)
    ap.add_argument('--out-ids', required=True)
    ap.add_argument('--out-manifest', required=True)
    ap.add_argument('--out-report', required=True)
    args = ap.parse_args()

    rows = load_images(args.map, args.ann)
    by_id = {r['image_id']: r for r in rows}
    excl = {int(x) for x in open(args.exclude) if x.strip()}

    A = [by_id[i] for i in sorted(excl) if i in by_id]
    pool = [r for r in rows if r['image_id'] not in excl and r['n_gt_visible'] > 0]
    empty_pool = [r for r in rows if r['image_id'] not in excl and r['n_gt_visible'] == 0]

    pool_by_bucket = {b: [r for r in pool if r['dominant'] == b] for b in BUCKETS}
    sizes = {b: len(v) for b, v in pool_by_bucket.items() if v}
    quota = allocate(sizes, args.n)

    def sort_key(r):
        cats = r['cats']
        dom_cat = min(cats, key=lambda c: (-cats[c], c)) if cats else 0
        return (r['n_gt_visible'], dom_cat, r['image_id'])

    rng = np.random.default_rng(args.seed)
    chosen, picks, unmet = [], {}, {}
    for b in BUCKETS:
        members = sorted(pool_by_bucket[b], key=sort_key)
        n_b, q = len(members), quota.get(b, 0)
        if q == 0:
            continue
        step = n_b / q
        off = float(rng.uniform(0.0, step))
        idx = [min(int(off + j * step), n_b - 1) for j in range(q)]
        # 浮点舍入可能产生重复位置；用去重后按需补位
        if len(set(idx)) < q:
            seen, fixed = set(), []
            for i in idx:
                while i in seen:
                    i += 1
                seen.add(i)
                fixed.append(min(i, n_b - 1))
            idx = fixed
        if q > n_b:
            unmet[b] = q - n_b
        picks[b] = [members[i] for i in idx]
        chosen.extend(picks[b])

    chosen.sort(key=lambda r: r['image_id'])

    full_gt = sum(r['n_gt_visible'] for r in rows)
    full_off = sum(r['n_gt_official'] for r in rows)
    A_desc = describe(A, 'subset_A')
    B_desc = describe(chosen, 'subset_B')
    F_desc = describe(rows, 'full_val')
    P_desc = describe([r for r in rows if r['image_id'] not in excl], 'pool_minus_A')

    # 协变量平衡：A vs B、B vs 全集、A vs 全集
    cov = ['n_gt_visible', 'gt_area_mean', 'gt_diameter_mean']
    balance = {}
    for name, (x, y) in {
        'A_vs_B': (A, chosen), 'B_vs_full': (chosen, rows), 'A_vs_full': (A, rows),
    }.items():
        balance[name] = {c: round(smd([r[c] for r in x], [r[c] for r in y]), 4) for c in cov}
    # 尺寸构成（以 GT 级占比做 SMD 的近似：按桶占比差，单位 pp）
    balance['bucket_share_diff_pp'] = {}
    for name, D in (('A_vs_full', A_desc), ('B_vs_full', B_desc), ('A_vs_B', None)):
        if D is None:
            balance['bucket_share_diff_pp'][name] = {
                b: round(A_desc['bucket_gt_share_pct'][b] - B_desc['bucket_gt_share_pct'][b], 2)
                for b in BUCKETS}
        else:
            balance['bucket_share_diff_pp'][name] = {
                b: round(D['bucket_gt_share_pct'][b] - F_desc['bucket_gt_share_pct'][b], 2)
                for b in BUCKETS}

    report = {
        'design': {
            'method': ('dominant-size-bucket strata + systematic sampling within stratum, '
                       'members sorted by (visible GT count, dominant category, image id)'),
            'stratify_var': 'dominant_bucket',
            'within_stratum_order': ['n_gt_visible', 'dominant_category_id', 'image_id'],
            'seed': args.seed, 'n_requested': args.n, 'n_drawn': len(chosen),
            'excluded_subset_A': len(excl), 'n_strata': len(picks),
            'quota_by_bucket': quota, 'stratum_sizes': sizes, 'unmet_quota': unmet,
            'pool_size': len(pool), 'pool_empty_images_excluded': len(empty_pool),
            'note': ('Systematic sampling inside size strata makes the sample empirical quantiles '
                     'track the population closely (implicit stratification). It balances size '
                     'composition and target density; it does not guarantee scene-level '
                     'representativeness.'),
        },
        'gt_totals': {
            'full_val_visible_gt': full_gt, 'full_val_official_gt': full_off,
            'subset_A_visible_gt': A_desc['n_gt_visible'],
            'subset_B_visible_gt': B_desc['n_gt_visible'],
        },
        'descriptions': [A_desc, B_desc, P_desc, F_desc],
        'balance': balance,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_ids)) or '.', exist_ok=True)
    with open(args.out_ids, 'w') as f:
        for r in chosen:
            f.write('%d\n' % r['image_id'])

    with open(args.out_manifest, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image_id', 'n_gt_visible', 'n_gt_official', 'n_vt', 'n_t', 'n_s', 'n_m',
                    'n_categories', 'gt_area_mean', 'gt_diameter_mean', 'dominant',
                    'dominant_category_id', 'rank_in_stratum', 'stratum_size'])
        for b in BUCKETS:
            members = sorted(pool_by_bucket[b], key=sort_key)
            picked_ids = {r['image_id']: i for i, r in enumerate(picks.get(b, []))}
            rank = {r['image_id']: i for i, r in enumerate(members)}
            for r in picks.get(b, []):
                cats = r['cats']
                dom_cat = min(cats, key=lambda c: (-cats[c], c)) if cats else 0
                w.writerow([r['image_id'], r['n_gt_visible'], r['n_gt_official'],
                            r['n_vt'], r['n_t'], r['n_s'], r['n_m'], len(r['cats']),
                            round(r['gt_area_mean'], 3), round(r['gt_diameter_mean'], 4),
                            r['dominant'], dom_cat, rank[r['image_id']], len(members)])

    json.dump(report, open(args.out_report, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    print('[subsetB] pool=%d (empty images excluded %d)  drawn=%d  seed=%d'
          % (len(pool), len(empty_pool), len(chosen), args.seed))
    print('[subsetB] GT  visible: A=%d  B=%d  full=%d  official=%d'
          % (A_desc['n_gt_visible'], B_desc['n_gt_visible'], full_gt, full_off))
    for b in BUCKETS:
        print('[subsetB] %-9s  A %5.2f%%  B %5.2f%%  full %5.2f%%'
              % (b, A_desc['bucket_gt_share_pct'][b], B_desc['bucket_gt_share_pct'][b],
                 F_desc['bucket_gt_share_pct'][b]))
    print('[subsetB] SMD  A_vs_B %s' % balance['A_vs_B'])
    print('[subsetB] SMD  B_vs_full %s' % balance['B_vs_full'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

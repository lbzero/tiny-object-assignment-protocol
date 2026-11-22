#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p13_case_selection.py —— P1-3 案例筛选（规则在看数据之前固定）

三类案例对应论文里的三条机制：

  C1  候选出现、覆盖不变：界面替换使 S3pre 上升，而两臂的 S4 都为 0。
  C2  归属转移：very-tiny GT 在 NWD 臂上丢失 ≥2 个候选，且同图中存在
      一个邻近 GT（中心距离 < 2×该 GT 对角线）在 NWD 臂上仍有正样本。
  C3  已覆盖目标获得更多正样本：S4 上升且替换前 S4 > 0。

**一条必须写明的更正**：首版把 C1 定义为 "S1 上升"，跑了 0 例。原因是 S1 是几何量，
两臂按构造完全相同（论文 §4.2 自己就把它当作两臂一致的自检项）。界面替换改变的是
对齐有效性之后的量（S2/S3pre），不是 S1。判据据此改为 S3pre，且在看到任何案例之前
改定。

固定选样规则（写死在代码里，不接受运行时修改）
--------------------------------------------------
R1  只取 very-tiny 桶（area < 64）。
R2  排序量写死：C1 按 (s3pre_nwd − s3pre_ciou) 降序；C2 按 n_lost 降序；
    C3 按 (s4_nwd − s4_ciou) 降序。
R3  每张图最多取 1 例，避免密集图像主导。
R4  同分时按 (image_id, gt_idx) 升序。
R5  每类取前 3 例；不足则全取并如实标注 short_of_quota。
R6  同一个 (image_id, gt_idx) 不跨类复用，按 C1→C2→C3 的顺序先到先得。
    原因：同一个框可以同时满足两类判据（候选上升且丢候选），在图上会造成
    两行看起来是同一件事。

输出
----
<out>.csv / <out>.json；stdout 打印每类候选数、涉及图像数与选中项。

本脚本只做筛选，不画图：案例图需要 anchor 级坐标（哪些 anchor 属于谁），
当前 trace 只记录计数，先补一个坐标转储才能画出有信息量的对照图。

用法
  python p13_case_selection.py --diag /root/runs/p1/diag \
      --map /root/runs/p1/gtmap_val_map.csv --out /root/runs/p1/p13_cases
"""
import argparse
import csv
import json
import math
import os

BUCKET = 'verytiny'
PER_IMAGE = 1
TOP_N = 3
NEIGH_FACTOR = 2.0     # 邻域半径 = 2 × 该 GT 的对角线
MIN_LOST = 2


def load_rows(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_boxes(map_csv):
    """(image_id, gt_idx) -> (cx, cy, diag)"""
    out = {}
    with open(map_csv, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            x, y, w, h = float(r['x']), float(r['y']), float(r['w']), float(r['h'])
            out[(int(r['image_id']), int(r['gt_idx']))] = (
                x + w / 2.0, y + h / 2.0, math.hypot(w, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag', required=True)
    ap.add_argument('--map', required=True, help='gtmap_val_map.csv（含 GT 坐标）')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ciou = load_rows(os.path.join(args.diag, 'E0_default_s%d.csv' % args.seed))
    nwd = load_rows(os.path.join(args.diag, 'E0_s%d_nwd.csv' % args.seed))
    if len(ciou) != len(nwd):
        raise SystemExit('两臂行数不一致')
    boxes = load_boxes(args.map)

    rows = []
    for a, b in zip(ciou, nwd):
        if (a['image_id'], a['gt_idx']) != (b['image_id'], b['gt_idx']):
            raise SystemExit('两臂行序不一致，不能逐行配对')
        if a['bucket'] != BUCKET:
            continue
        rows.append({
            'image_id': int(a['image_id']), 'gt_idx': int(a['gt_idx']),
            'd': float(a['d']), 'area': float(a['area']),
            's1': int(a['s1']),
            's3pre_ciou': int(a['s3pre']), 's3pre_nwd': int(b['s3pre']),
            's4_ciou': int(a['s4']), 's4_nwd': int(b['s4']),
            'n_lost': int(b['n_lost']), 'n_gained': int(b['n_gained']),
        })

    # C1 / C3 不需要邻域信息
    c1 = [r for r in rows
          if r['s3pre_nwd'] > r['s3pre_ciou'] and r['s4_ciou'] == 0 and r['s4_nwd'] == 0]
    c3 = [r for r in rows if r['s4_nwd'] > r['s4_ciou'] and r['s4_ciou'] > 0]

    # C2 需要邻域：丢候选的 very-tiny GT + 同图邻近的有覆盖 GT
    covered_by_img = {}
    for r in rows:
        if r['s4_nwd'] > 0:
            covered_by_img.setdefault(r['image_id'], []).append(r)
    c2 = []
    for r in rows:
        if r['n_lost'] < MIN_LOST:
            continue
        key = (r['image_id'], r['gt_idx'])
        if key not in boxes:
            continue
        cx, cy, diag = boxes[key]
        for o in covered_by_img.get(r['image_id'], []):
            if o['gt_idx'] == r['gt_idx']:
                continue
            ok = (r['image_id'], o['gt_idx'])
            if ok not in boxes:
                continue
            ox, oy, _ = boxes[ok]
            if math.hypot(ox - cx, oy - cy) <= NEIGH_FACTOR * diag:
                c2.append({**r, 'neighbour_gt_idx': o['gt_idx'],
                           'neighbour_s4': o['s4_nwd'],
                           'neighbour_dist': round(math.hypot(ox - cx, oy - cy), 2)})
                break

    groups = [('C1_candidates_up_no_coverage', c1, lambda r: r['s3pre_nwd'] - r['s3pre_ciou']),
              ('C2_ownership_transfer', c2, lambda r: r['n_lost']),
              ('C3_more_positives_on_covered', c3, lambda r: r['s4_nwd'] - r['s4_ciou'])]

    picked, summary = [], {}
    used_box = set()
    for name, cand, key in groups:
        ordered = sorted(cand, key=lambda r: (-key(r), r['image_id'], r['gt_idx']))
        chosen, used_img = [], set()
        for r in ordered:
            if r['image_id'] in used_img or (r['image_id'], r['gt_idx']) in used_box:
                continue
            used_img.add(r['image_id'])
            used_box.add((r['image_id'], r['gt_idx']))
            chosen.append(r)
            if len(chosen) >= TOP_N:
                break
        summary[name] = {
            'candidates': len(cand),
            'images_with_candidates': len({r['image_id'] for r in cand}),
            'selected': len(chosen),
            'short_of_quota': len(chosen) < TOP_N,
        }
        for rank, r in enumerate(chosen, 1):
            picked.append({'class': name, 'rank': rank, **r, 'rank_value': key(r)})

    fields = ['class', 'rank', 'image_id', 'gt_idx', 'd', 'area', 's1',
              's3pre_ciou', 's3pre_nwd', 's4_ciou', 's4_nwd', 'n_lost', 'n_gained',
              'neighbour_gt_idx', 'neighbour_s4', 'neighbour_dist', 'rank_value']
    with open(args.out + '.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(picked)
    json.dump({'seed': args.seed, 'bucket': BUCKET, 'per_image': PER_IMAGE,
               'top_n': TOP_N, 'min_lost': MIN_LOST, 'neighbour_factor': NEIGH_FACTOR,
               'summary': summary,
               'note': 'selection only; figures need an anchor-coordinate dump'},
              open(args.out + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    print('[p13] very-tiny boxes: %d' % len(rows))
    for name, _c, _k in groups:
        s = summary[name]
        print('[p13] %-32s candidates %5d (images %4d)  selected %d%s'
              % (name, s['candidates'], s['images_with_candidates'], s['selected'],
                 '  <-- short of quota' if s['short_of_quota'] else ''))
    for p in picked:
        print('  %-32s img %5d gt %2d  d %4.1f  rank %.0f'
              % (p['class'], p['image_id'], p['gt_idx'], p['d'], p['rank_value']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

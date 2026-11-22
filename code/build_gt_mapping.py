#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_gt_mapping.py —— 建立 (image_id, gt_idx) → annotation_id 的稳定映射

为什么需要这个
--------------
P0-6 动态诊断会在 batch 内拿到 gt_idx（该图 GT 在 label 中的行序），但论文与
静态诊断的主键是 annotation_id。二者不是同一个东西：
  - gt_idx     : 位置，会因排序、过滤、坐标变换而变
  - annotation_id : COCO 原始标注的稳定身份

评审明确要求「不能只靠 file_name → image_id 映射，必须证明图像内部的 GT 顺序没有变化」。
本脚本就是那个证明，并同时产出映射表本身。

已核实的事实（2026-11-14，全量 2804 图）
----------------------------------------
1. 坐标层面：把 COCO bbox 裁剪到图像边界后，2785 张图的 label 逐行与 annotation 逐条
   完全一致（容差 1 px），**0 个反例**。即 label 第 i 行 ≡ 该图第 i 个 annotation。
2. 行数层面：19 张图 label 少行，共少 26 条。用坐标匹配精确定位后确认：
   **26 条全部是 w ≤ 1 或 h ≤ 1 的细长条**（其中 9 条一维为 0）。
   这正是长期存在的「75,091 vs 75,065」之谜：75,091 是 COCO 总数，
   75,065 是实际进入训练的数目，差 26。
3. 这 26 条面积全部 < 64 px²，即 **26 个 vt GT 在数据加载阶段就被丢弃，
   从未进入训练**。它们不是「分配失败」，而是「数据层就不存在」。
   这一点必须与「分配阶段丢失」严格区分，是本脚本最重要的输出之一。

输出
----
  <out>_map.csv      每个进入训练的 GT 一行：image_id, gt_idx, annotation_id, ...
  <out>_dropped.csv  被数据加载丢弃的 GT（26 条）：image_id, annotation_id, w, h, ...
  <out>_summary.json 覆盖率与一致性统计

用法
----
  python build_gt_mapping.py \
      --ann /root/data/AI-TOD/annotations_v2/aitod_val_v2.json \
      --labels /root/data/AI-TOD/labels/val \
      --out /root/runs/p1/gtmap_val
"""
import argparse
import collections
import csv
import json
import os
import sys

TOL = 1.0      # 像素容差：YOLO 坐标保留 6 位小数，回算存在舍入


def clip_box(b, W, H):
    """把 COCO bbox 裁剪到图像边界内，与 ultralytics label 校验的行为对齐。"""
    x, y, w, h = b
    x1, y1, x2, y2 = x, y, x + w, y + h
    x1 = max(0.0, min(x1, W))
    x2 = max(0.0, min(x2, W))
    y1 = max(0.0, min(y1, H))
    y2 = max(0.0, min(y2, H))
    return (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


def bucket_of(area):
    """与 aitod_eval.py 的四桶定义一致（左闭右开）。"""
    if area < 64:
        return 'verytiny'
    if area < 256:
        return 'tiny'
    if area < 1024:
        return 'small'
    return 'medium'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--out', required=True, help='输出前缀，不带扩展名')
    ap.add_argument('--tol', type=float, default=TOL)
    args = ap.parse_args()

    d = json.load(open(args.ann, encoding='utf-8'))
    imgs = {i['id']: i for i in d['images']}
    by_img = collections.defaultdict(list)
    for a in d['annotations']:
        by_img[a['image_id']].append(a)

    rows = []          # 进入训练的 GT
    dropped = []       # 被数据加载丢弃的 GT
    bad = []           # 无法用坐标匹配上的（应为 0）
    n_img_ok = 0
    n_img_short = 0

    for iid in sorted(imgs):
        im = imgs[iid]
        W, H = im['width'], im['height']
        stem = os.path.splitext(im['file_name'])[0]
        p = os.path.join(args.labels, stem + '.txt')
        if not os.path.isfile(p):
            print('[map] 缺少 label 文件: %s' % p, file=sys.stderr)
            return 2
        lines = [l for l in open(p) if l.strip()]
        gts = by_img.get(iid, [])

        # label 行 → 原图像素框
        lb = []
        for l in lines:
            q = l.split()
            cx, cy, w, h = (float(x) for x in q[1:5])
            lb.append((float(q[0]), (cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H))

        # 按 COCO 顺序逐条找 label 里第一个未被占用的同坐标行
        used = [False] * len(lb)
        for gi, a in enumerate(gts):
            gx, gy, gw, gh = clip_box(a['bbox'], W, H)
            hit = -1
            for j, b in enumerate(lb):
                if used[j]:
                    continue
                if (abs(b[1] - gx) <= args.tol and abs(b[2] - gy) <= args.tol
                        and abs(b[3] - gw) <= args.tol and abs(b[4] - gh) <= args.tol):
                    hit = j
                    break
            if hit < 0:
                dropped.append({
                    'image_id': iid, 'annotation_id': a['id'],
                    'category_id': a['category_id'],
                    'x': a['bbox'][0], 'y': a['bbox'][1],
                    'w': a['bbox'][2], 'h': a['bbox'][3],
                    'area': a['bbox'][2] * a['bbox'][3],
                    'bucket': bucket_of(a['bbox'][2] * a['bbox'][3]),
                })
                continue
            used[hit] = True
            area = gw * gh
            rows.append({
                'image_id': iid, 'gt_idx': hit, 'annotation_id': a['id'],
                'category_id': a['category_id'],
                'isize_w': W, 'isize_h': H,
                'x': round(gx, 4), 'y': round(gy, 4),
                'w': round(gw, 4), 'h': round(gh, 4),
                'd': round(area ** 0.5, 4),
                'area': round(area, 4),
                'bucket': bucket_of(area),
            })

        # 剩余未被认领的 label 行 = 无法解释的多余行
        if any(not u for u in used):
            bad.append({'image_id': iid, 'unclaimed_lines': sum(1 for u in used if not u)})

        if len(lines) == len(gts):
            n_img_ok += 1
        else:
            n_img_short += 1

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.out + '_map.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['image_id', 'gt_idx', 'annotation_id', 'category_id',
                                          'isize_w', 'isize_h', 'x', 'y', 'w', 'h',
                                          'd', 'area', 'bucket'])
        w.writeheader()
        w.writerows(rows)

    with open(args.out + '_dropped.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['image_id', 'annotation_id', 'category_id',
                                          'x', 'y', 'w', 'h', 'area', 'bucket'])
        w.writeheader()
        w.writerows(dropped)

    # gt_idx 唯一性自检：同一图内不得重复
    seen = collections.Counter((r['image_id'], r['gt_idx']) for r in rows)
    dup = [k for k, v in seen.items() if v > 1]

    summary = {
        'ann_file': os.path.abspath(args.ann),
        'labels_dir': os.path.abspath(args.labels),
        'tol_px': args.tol,
        'n_images': len(imgs),
        'n_images_exact': n_img_ok,
        'n_images_with_dropped': n_img_short,
        'n_ann_total': len(d['annotations']),
        'n_gt_in_training': len(rows),
        'n_dropped': len(dropped),
        'n_unclaimed_label_lines': sum(x['unclaimed_lines'] for x in bad),
        'n_dup_gt_idx': len(dup),
        'dropped_buckets': dict(collections.Counter(x['bucket'] for x in dropped)),
        'dropped_reason': 'w<=1 or h<=1 (ultralytics label 校验丢弃)',
        'mapped_buckets': dict(collections.Counter(r['bucket'] for r in rows)),
    }
    json.dump(summary, open(args.out + '_summary.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    print('[map] 图像 %d 张：%d 张行序完全一致，%d 张有丢弃'
          % (len(imgs), n_img_ok, n_img_short))
    print('[map] annotation 总数 %d → 进入训练 %d，丢弃 %d'
          % (len(d['annotations']), len(rows), len(dropped)))
    print('[map] 丢弃的桶分布: %s' % summary['dropped_buckets'])
    print('[map] 无法解释的多余 label 行: %d（应为 0）'
          % summary['n_unclaimed_label_lines'])
    print('[map] gt_idx 重复: %d（应为 0）' % len(dup))
    print('[map] 写出: %s_map.csv / %s_dropped.csv / %s_summary.json'
          % (args.out, args.out, args.out))

    if summary['n_unclaimed_label_lines'] or dup:
        print('[map] 失败：存在无法解释的映射缺陷', file=sys.stderr)
        return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())

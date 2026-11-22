#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NWD vs CIoU 的定位质量分析（离线，不重训）。

回答 2026-11-19 意见 §3.4 的五个问题：
    1. matched detection 的 IoU 分布
    2. 按目标尺寸划分的 IoU 分布
    3. 固定置信度阈值下的召回率
    4. 固定召回水平下的定位误差（中心偏移、宽高误差）
    5. 分类分数与定位质量的相关性

目的
    AP75 掉 4.00 pp 是事实，但"框回归不准"只是候选解释之一。另一可能是
    "分类分数排序不适合高 IoU 匹配"（高分框集中在宽 IoU 但中心偏移的位置）。
    本脚本用匹配后的 IoU 分布 + 分数-IoU 相关性把两者分开。

方法
    逐图按分数降序做贪心匹配（同类、IoU≥0.5、每个 GT 至多匹配一次），
    记录每对 (pred, gt) 的 IoU、分数、GT 尺寸、中心偏移、宽高误差。
    GT 尺寸桶沿用 AI-TOD 官方定义（按 area）。
    另算"每 GT 的最佳同类别预测 IoU"（不看分数），作为定位上限的参考。

用法
    python loc_error_analysis.py --gt ~/data/AI-TOD/annotations_v2/aitod_val_v2.json \
        --runs "E0:/root/runs/p1/E0_default_s0:E0_default_s1:E0_default_s2:E0_default_s3" \
               "NWD:/root/runs/p1/NWD_nwd_s0:NWD_nwd_s1:NWD_nwd_s2:NWD_nwd_s3" \
        --out /root/runs/p1/loc_err
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np


BUCKETS = [("verytiny", 0, 64), ("tiny", 64, 256), ("small", 256, 1024), ("medium", 1024, 1 << 30)]


def bucket_of(w, h):
    a = w * h
    for name, lo, hi in BUCKETS:
        if lo <= a < hi:
            return name
    return "medium"


def iou_matrix(a, b):
    """a:(N,4) b:(M,4) xywh → (N,M) IoU"""
    ax1, ay1 = a[:, 0], a[:, 1]
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    union = aa[:, None] + bb[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def analyze_run(gt_by_img, preds_by_img, iou_thr=0.5, confs=(0.25, 0.5)):
    """返回该 run 的汇总统计。"""
    matched = []          # (iou, score, bucket, dx_rel, dy_rel, dw_rel, dh_rel)
    best_iou = []         # (best_iou, bucket)
    n_gt = 0
    recall_at = {c: [0, 0] for c in confs}   # conf -> [hit, total]

    for iid, gts in gt_by_img.items():
        n_gt += len(gts)
        for c in confs:
            recall_at[c][1] += len(gts)
        pr = preds_by_img.get(iid)
        if not pr:
            continue

        gbox = np.array([g["bbox"] for g in gts], dtype=np.float64)
        gcat = np.array([g["category_id"] for g in gts], dtype=np.int64)
        gb = [bucket_of(g["bbox"][2], g["bbox"][3]) for g in gts]

        pbox = np.array([p["bbox"] for p in pr], dtype=np.float64)
        pcat = np.array([p["category_id"] for p in pr], dtype=np.int64)
        psc = np.array([p["score"] for p in pr], dtype=np.float64)

        order = np.argsort(-psc)
        pbox, pcat, psc = pbox[order], pcat[order], psc[order]

        # 同类 IoU 矩阵（不同类置 0，不参与匹配）
        iou = iou_matrix(pbox, gbox)
        same = pcat[:, None] == gcat[None, :]
        iou = np.where(same, iou, 0.0)

        # 每 GT 的最佳同类别预测（不限阈值，看定位上限）
        if iou.size:
            bi = iou.max(axis=0)
            for k in range(len(gts)):
                best_iou.append((float(bi[k]), gb[k]))

        # 贪心匹配
        used_gt = np.zeros(len(gts), dtype=bool)
        for pi in range(len(pbox)):
            row = iou[pi].copy()
            row[used_gt] = -1
            gi = int(row.argmax())
            if row[gi] < iou_thr:
                continue
            used_gt[gi] = True
            v = float(row[gi])
            g = gts[gi]
            gw, gh = g["bbox"][2], g["bbox"][3]
            pcx, pcy = pbox[pi][0] + pbox[pi][2] / 2, pbox[pi][1] + pbox[pi][3] / 2
            gcx, gcy = g["bbox"][0] + gw / 2, g["bbox"][1] + gh / 2
            matched.append((v, float(psc[pi]), gb[gi],
                            abs(pcx - gcx) / max(gw, 1e-6),
                            abs(pcy - gcy) / max(gh, 1e-6),
                            abs(pbox[pi][2] - gw) / max(gw, 1e-6),
                            abs(pbox[pi][3] - gh) / max(gh, 1e-6)))
            for c in confs:
                if psc[pi] >= c:
                    recall_at[c][0] += 1

    return {"matched": matched, "best_iou": best_iou, "n_gt": n_gt, "recall_at": recall_at}


def summarize(name, r):
    m = r["matched"]
    n_gt = r["n_gt"]
    print("\n" + "=" * 74)
    print("%s    GT=%d   匹配对=%d  (匹配率 %.2f%%)" % (name, n_gt, len(m), 100.0 * len(m) / n_gt))
    print("=" * 74)
    if not m:
        return None

    arr = np.array([[x[0], x[1], x[3], x[4], x[5], x[6]] for x in m])
    bks = [x[2] for x in m]
    ious, scores = arr[:, 0], arr[:, 1]

    print("匹配 IoU   : mean %.4f  P25 %.4f  P50 %.4f  P75 %.4f  >=0.75 占 %.1f%%"
          % (ious.mean(), np.percentile(ious, 25), np.percentile(ious, 50),
             np.percentile(ious, 75), 100.0 * (ious >= 0.75).mean()))
    print("分数-IoU   : Pearson %.3f   Spearman %.3f"
          % (np.corrcoef(scores, ious)[0, 1],
             float(np.corrcoef(np.argsort(np.argsort(scores)), np.argsort(np.argsort(ious)))[0, 1])))
    print("中心偏移(相对GT宽高): dx %.4f  dy %.4f   宽高误差: dw %.4f  dh %.4f"
          % (arr[:, 2].mean(), arr[:, 3].mean(), arr[:, 4].mean(), arr[:, 5].mean()))
    print("固定 conf 召回: " + "  ".join("%.2f→%.2f%%" % (c, 100.0 * h / max(t, 1))
                                        for c, (h, t) in sorted(r["recall_at"].items())))

    print("\n%-10s %7s %8s %8s %8s %9s %9s" % ("桶", "n", "IoU均值", "P50", "≥0.75占", "dx", "dw"))
    per = {}
    for bk, _, _ in BUCKETS:
        idx = [i for i, b in enumerate(bks) if b == bk]
        if not idx:
            continue
        sub = arr[idx]
        per[bk] = {"n": len(idx), "iou": float(sub[:, 0].mean()),
                   "p50": float(np.percentile(sub[:, 0], 50)),
                   "ge75": float((sub[:, 0] >= 0.75).mean()),
                   "dx": float(sub[:, 2].mean()), "dw": float(sub[:, 4].mean())}
        print("%-10s %7d %8.4f %8.4f %7.1f%% %9.4f %9.4f"
              % (bk, len(idx), per[bk]["iou"], per[bk]["p50"], 100 * per[bk]["ge75"],
                 per[bk]["dx"], per[bk]["dw"]))

    bi = np.array([x[0] for x in r["best_iou"]])
    print("\n每 GT 最佳同类别预测 IoU: mean %.4f  P50 %.4f  >=0.75 占 %.1f%%"
          % (bi.mean(), np.percentile(bi, 50), 100.0 * (bi >= 0.75).mean()))
    return {"n_gt": n_gt, "n_match": len(m), "iou_mean": float(ious.mean()),
            "iou_p50": float(np.percentile(ious, 50)),
            "ge75": float((ious >= 0.75).mean()),
            "pearson": float(np.corrcoef(scores, ious)[0, 1]),
            "dx": float(arr[:, 2].mean()), "dy": float(arr[:, 3].mean()),
            "dw": float(arr[:, 4].mean()), "dh": float(arr[:, 5].mean()),
            "best_iou_mean": float(bi.mean()), "per_bucket": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--runs', nargs='+', required=True, help='标签:run1:run2... 形式')
    ap.add_argument('--runs-root', default='/root/runs/p1')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    ann = json.load(open(a.gt, encoding='utf-8'))
    gt_by_img = defaultdict(list)
    for g in ann["annotations"]:
        gt_by_img[g["image_id"]].append(g)
    print("[gt] %d 图 / %d 框" % (len(gt_by_img), len(ann["annotations"])))

    groups = {}
    for spec in a.runs:
        label, *dirs = spec.split(":")
        res = {}
        for d in dirs:
            p = os.path.join(d, "eval_official", "predictions.json")
            if not os.path.isfile(p):
                p = os.path.join(a.runs_root, d, "eval_official", "predictions.json")
            if not os.path.isfile(p):
                print("[skip] %s 无 predictions.json" % d, file=sys.stderr)
                continue
            preds = json.load(open(p, encoding='utf-8'))
            pb = defaultdict(list)
            for q in preds:
                pb[q["image_id"]].append(q)
            print("[run] %-18s %d 条预测" % (d, len(preds)))
            res[d] = summarize(d, analyze_run(gt_by_img, pb))
            del preds, pb
        groups[label] = res

    # 组间对比
    print("\n" + "#" * 74)
    print("组间对比（各 run 先平均，再比较）")
    print("#" * 74)
    agg = {}
    for label, res in groups.items():
        vals = [v for v in res.values() if v]
        if not vals:
            continue
        agg[label] = {k: float(np.mean([v[k] for v in vals]))
                      for k in ("iou_mean", "iou_p50", "ge75", "pearson", "dx", "dy", "dw", "dh", "best_iou_mean")}
        agg[label]["n_match"] = float(np.mean([v["n_match"] for v in vals]))
        agg[label]["per_bucket"] = {
            bk[0]: {kk: float(np.mean([v["per_bucket"][bk[0]][kk] for v in vals if bk[0] in v["per_bucket"]]))
                    for kk in ("n", "iou", "p50", "ge75", "dx", "dw")}
            for bk in BUCKETS if any(bk[0] in v["per_bucket"] for v in vals)}

    keys = ("n_match", "iou_mean", "iou_p50", "ge75", "best_iou_mean", "pearson", "dx", "dy", "dw", "dh")
    labels = list(agg.keys())
    print("%-16s " % "量" + " ".join("%10s" % l for l in labels))
    for k in keys:
        print("%-16s " % k + " ".join("%10.4f" % agg[l][k] for l in labels))
    print()
    print("%-12s %-10s " % ("桶", "量") + " ".join("%10s" % l for l in labels))
    for bk, _, _ in BUCKETS:
        for kk in ("iou", "p50", "ge75", "dx", "dw"):
            row = []
            for l in labels:
                pb = agg[l]["per_bucket"].get(bk)
                row.append("%10.4f" % pb[kk] if pb else "%10s" % "-")
            print("%-12s %-10s " % (bk, kk) + " ".join(row))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    json.dump({"gt": a.gt, "groups": agg}, open(a.out + ".json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("\n写出 %s.json" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

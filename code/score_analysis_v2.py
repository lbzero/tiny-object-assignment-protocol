#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充 A：置信度分布、分档精度与 PR 曲线（按 2026-11-19 裁定重做）。

裁定要求修正的三点
    1. 「匹配预测中的高 IoU 比例」不等于 precision —— 标准 precision 的分母含未匹配预测。
       本脚本同时给出 n_pred / 匹配 / 未匹配，并单独计算 precision。
    2. 必须区分两种分档：
         · 固定阈值档（绝对置信度分布）
         · 固定数量分位档（排序质量，更接近 AP 的排序语义）
    3. 按类别分别统计，再给宏观汇总，避免类别分布掩盖变化。

输出
    A1 每档：预测数 / IoU≥0.5 匹配 / IoU≥0.75 匹配 / 未匹配
    A2 累积阈值下的 precision / recall（IoU=0.5 与 0.75 两种）
    A3 固定数量分位档（前 10%、10–25%、25–50%、50–100%）
    A4 按类别（8 类）

用法
    python score_analysis_v2.py --gt <aitod_val_v2.json> \
        --runs "E0:E0_default_s0:..." "NWD:NWD_nwd_s0:..." --out <前缀>
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

THRESH_BINS = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.01)]
THRESH_LABELS = ["<0.05", "0.05-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", ">=0.5"]
QUANT_LABELS = ["top10%", "10-25%", "25-50%", "50-100%"]
PR_THRS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def iou_matrix(a, b):
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :]); iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :]); iy2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    aa = (ax2 - ax1) * (ay2 - ay1); bb = (bx2 - bx1) * (by2 - by1)
    union = aa[:, None] + bb[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def run_one(gt_by_img, preds_by_img, n_gt_total):
    """返回 (阈值档统计, 分位档统计, PR 数据, 按类别)。"""
    th = {lab: {"n_pred": 0, "m50": 0, "m75": 0} for lab in THRESH_LABELS}
    qt = {lab: {"n_pred": 0, "m50": 0, "m75": 0} for lab in QUANT_LABELS}
    cat = defaultdict(lambda: {"n_pred": 0, "m50": 0, "m75": 0})
    all_rows = []          # (score, iou_of_match or 0, matched_at_50)

    for iid, gts in gt_by_img.items():
        pr = preds_by_img.get(iid)
        if not pr:
            continue
        pbox = np.array([p["bbox"] for p in pr], dtype=np.float64)
        pcat = np.array([p["category_id"] for p in pr], dtype=np.int64)
        psc = np.array([p["score"] for p in pr], dtype=np.float64)
        gbox = np.array([g["bbox"] for g in gts], dtype=np.float64)
        gcat = np.array([g["category_id"] for g in gts], dtype=np.int64)

        order = np.argsort(-psc)
        pbox, pcat, psc = pbox[order], pcat[order], psc[order]
        iou = iou_matrix(pbox, gbox)
        iou = np.where(pcat[:, None] == gcat[None, :], iou, 0.0)

        used = np.zeros(len(gts), dtype=bool)
        n_pred = len(pbox)
        for pi in range(n_pred):
            s = psc[pi]
            lab = None
            for k, (lo, hi) in enumerate(THRESH_BINS):
                if lo <= s < hi:
                    lab = THRESH_LABELS[k]; break
            q = QUANT_LABELS[0] if pi < 0.10 * n_pred else (
                QUANT_LABELS[1] if pi < 0.25 * n_pred else (
                    QUANT_LABELS[2] if pi < 0.50 * n_pred else QUANT_LABELS[3]))
            c = int(pcat[pi])

            row = iou[pi].copy(); row[used] = -1
            gi = int(row.argmax())
            hit50 = row[gi] >= 0.5
            hit75 = row[gi] >= 0.75
            if hit50:
                used[gi] = True

            for box in (th.get(lab), qt.get(q), cat[c]):
                if box is None:
                    continue
                box["n_pred"] += 1
                if hit50:
                    box["m50"] += 1
                if hit75:
                    box["m75"] += 1
            # 注意：未匹配的预测也要进 PR 序列（它们是 FP），
            # 否则 precision 的分母只有匹配预测，就成了"匹配内的高 IoU 比例"。
            all_rows.append((float(s), bool(hit50), bool(hit75)))

    # PR：按分数降序累积（含未匹配预测作为 FP）
    all_rows.sort(key=lambda x: -x[0])
    scores = np.array([r[0] for r in all_rows])
    tp50 = np.array([r[1] for r in all_rows])
    tp75 = np.array([r[2] for r in all_rows])
    pr = {}
    for name, tp_arr in (("iou50", tp50), ("iou75", tp75)):
        curve = []
        for t in PR_THRS:
            sel = scores >= t
            n_sel = int(sel.sum())
            tp = int(tp_arr[sel].sum())
            curve.append({"thr": t, "tp": tp, "fp": n_sel - tp,
                          "prec": tp / n_sel if n_sel else 0.0,
                          "rec": tp / n_gt_total if n_gt_total else 0.0})
        pr[name] = curve
    return th, qt, pr, dict(cat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--runs-root', default='/root/runs/p1')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    ann = json.load(open(a.gt, encoding='utf-8'))
    gt_by_img = defaultdict(list)
    for g in ann["annotations"]:
        gt_by_img[g["image_id"]].append(g)
    n_gt = len(ann["annotations"])
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    print("[gt] %d 图 / %d 框 / %d 类" % (len(gt_by_img), n_gt, len(cats)))

    groups = {}
    for spec in a.runs:
        label, *dirs = spec.split(":")
        res = []
        for d in dirs:
            p = os.path.join(d, "eval_official", "predictions.json")
            if not os.path.isfile(p):
                p = os.path.join(a.runs_root, d, "eval_official", "predictions.json")
            if not os.path.isfile(p):
                print("[skip] %s" % d, file=sys.stderr); continue
            preds = json.load(open(p, encoding='utf-8'))
            pb = defaultdict(list)
            for q in preds:
                pb[q["image_id"]].append(q)
            res.append(run_one(gt_by_img, pb, n_gt))
            print("[run] %-16s 完成" % d)
            del preds, pb
        if res:
            groups[label] = res

    def agg_stat(res, idx):
        return {lab: {k: float(np.mean([r[idx][lab][k] for r in res])) for k in res[0][idx][lab]}
                for lab in res[0][idx]}

    print("\n" + "=" * 96)
    print("A1  固定阈值档：预测数 / IoU≥0.5 匹配 / IoU≥0.75 匹配 / 未匹配")
    print("=" * 96)
    for g, res in groups.items():
        th = agg_stat(res, 0)
        print("\n[%s]" % g)
        print("  %-10s %10s %10s %10s %10s %11s %11s" % ("档", "预测", "m@0.5", "m@0.75", "未匹配", "m50/m55分", "m75/匹配"))
        for lab in THRESH_LABELS:
            d = th[lab]
            un = d["n_pred"] - d["m50"]
            print("  %-10s %10.0f %10.0f %10.0f %10.0f %10.1f%% %10.1f%%"
                  % (lab, d["n_pred"], d["m50"], d["m75"], un,
                     100 * d["m50"] / d["n_pred"] if d["n_pred"] else 0,
                     100 * d["m75"] / d["m50"] if d["m50"] else 0))

    print("\n" + "=" * 96)
    print("A2  累积阈值下的 precision / recall（IoU 0.5 与 0.75，4 seed 平均）")
    print("=" * 96)
    # 关键：必须与 A1 同口径（各 seed 先算再平均）。
    # 2026-11-19 导师核对时发现 A1 是 4 seed 均值、A2 只用了 res[0]，
    # 导致"预测数 ≠ TP+FP"的差额（E0 763 / NWD 443）。
    for g, res in groups.items():
        for key in ("iou50", "iou75"):
            curves = [r[2][key] for r in res]
            avg = []
            for i, t in enumerate(PR_THRS):
                avg.append({"thr": t,
                            "prec": float(np.mean([c[i]["prec"] for c in curves])),
                            "rec": float(np.mean([c[i]["rec"] for c in curves])),
                            "tp": float(np.mean([c[i]["tp"] for c in curves])),
                            "fp": float(np.mean([c[i]["fp"] for c in curves]))})
            curve = {c["thr"]: (c["prec"], c["rec"]) for c in avg}
            print("[%s] %s  (n_seed=%d)" % (g, key, len(res)))
            print("  %-8s" % "thr" + "".join("%9.2f" % t for t in PR_THRS))
            print("  %-8s" % "prec" + "".join("%9.3f" % curve[t][0] for t in PR_THRS))
            print("  %-8s" % "rec" + "".join("%9.3f" % curve[t][1] for t in PR_THRS))

    print("\n" + "=" * 96)
    print("A3  固定数量分位档（排序质量）")
    print("=" * 96)
    for g, res in groups.items():
        qt = agg_stat(res, 1)
        print("\n[%s]" % g)
        print("  %-10s %10s %10s %10s %11s %11s" % ("档", "预测", "m@0.5", "m@0.75", "m50/预测", "m75/匹配"))
        for lab in QUANT_LABELS:
            d = qt[lab]
            print("  %-10s %10.0f %10.0f %10.0f %10.1f%% %10.1f%%"
                  % (lab, d["n_pred"], d["m50"], d["m75"],
                     100 * d["m50"] / d["n_pred"] if d["n_pred"] else 0,
                     100 * d["m75"] / d["m50"] if d["m50"] else 0))

    print("\n" + "=" * 96)
    print("A4  按类别（m75 / 匹配，越高表示该类的框越准）")
    print("=" * 96)
    allc = set()
    for res in groups.values():
        for r in res:
            allc |= set(r[3].keys())
    print("%-14s" % "类别" + "".join("%14s" % g for g in groups))
    for c in sorted(allc):
        line = "%-14s" % cats.get(c, str(c))
        for g, res in groups.items():
            vals = [r[3].get(c, {}).get("m75", 0) / max(r[3].get(c, {}).get("m50", 1), 1) for r in res]
            line += "%13.1f%%" % (100 * np.mean(vals))
        print(line)

    out = {}
    for g, res in groups.items():
        pr_avg = {}
        for key in ("iou50", "iou75"):
            curves = [r[2][key] for r in res]
            pr_avg[key] = [{"thr": t,
                            "prec": float(np.mean([c[i]["prec"] for c in curves])),
                            "rec": float(np.mean([c[i]["rec"] for c in curves])),
                            "tp": float(np.mean([c[i]["tp"] for c in curves])),
                            "fp": float(np.mean([c[i]["fp"] for c in curves])),
                            "n_pred": float(np.mean([c[i]["tp"] + c[i]["fp"] for c in curves]))}
                           for i, t in enumerate(PR_THRS)]
        out[g] = {"thresh": agg_stat(res, 0), "quant": agg_stat(res, 1),
                  "pr": pr_avg, "n_seed": len(res),
                  "per_cat": {str(c): {k: float(np.mean([r[3].get(c, {}).get(k, 0) for r in res]))
                                       for k in ("n_pred", "m50", "m75")} for c in allc}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    json.dump(out, open(a.out + '.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print("\n写出 %s.json" % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""申请 1：soft label 与"分数—定位质量"耦合的离线重放分析。

导师 2026-11-19 的要求
    不能从 cls_loss 反推 soft label，必须对 E0/NWD 权重做同协议前向重放、
    记录 loss/assigner 路径中的真实张量。

输出（四组）
    A. 正样本 target_scores 分布：均值/中位数/标准差/P10-P90，按 seed 与尺寸桶
    B. 归一化前后对照：原始 align_metric、pos_align_metrics、pos_overlaps、
       norm_align_metric、final target_scores（判断变化来自相似度还是归一化）
    C. 排序耦合：分类分数 vs align_metric / target_scores / 匹配 IoU（Pearson+Spearman），
       并单看高分段（分数前 10%）
    D. 分层：四个尺寸桶，区分正样本与其他 anchor

实现要点
    * target_scores 由 assigner 的 _forward 生成，不在权重里；这里 hook assigner。
    * norm_align_metric 不直接返回，用 final / pre_norm 反算：
        pre_norm = one_hot(target_labels) * fg_mask
        norm_align = final_target_scores / pre_norm   （仅正样本处有意义）
    * 检查点用 last.pt 的 model 字段 —— ultralytics 在 strip 时已把 ema 赋给 model。

用法
    python soft_label_analysis.py --ckpt <last.pt> --data <smoke 或全量 yaml> \
        --n-img 500 --out <前缀>
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.utils.tal import TaskAlignedAssigner


BUCKETS = [("verytiny", 0, 64), ("tiny", 64, 256), ("small", 256, 1024), ("medium", 1024, 1 << 30)]


def bucket_of(w, h):
    a = w * h
    for n, lo, hi in BUCKETS:
        if lo <= a < hi:
            return n
    return "medium"


class Probe(TaskAlignedAssigner):
    """只读取中间量，不改任何计算。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rec = {}

    def get_pos_mask(self, *a, **kw):
        out = super().get_pos_mask(*a, **kw)
        # ★ 必须 clone：原生 _forward 里对 align_metric / overlaps 做了**原地**运算
        #   （align_metric *= mask_pos; align_metric.mul_(pos_overlaps).div_(pos_align_metrics + eps)），
        #   detach() 与源张量共享存储，不加 clone 记录到的会是归一化**之后**的值。
        #   2026-11-19 实测踩过：三者统计完全相同时才发现。
        self.rec['align'] = out[1].detach().clone()        # (b, n_max, na) 原始 align_metric
        self.rec['overlaps'] = out[2].detach().clone()     # (b, n_max, na) 原始 overlap
        return out

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        out = super().select_highest_overlaps(mask_pos, overlaps, n_max_boxes, align_metric)
        self.rec['mask_pos'] = out[2].detach().bool()
        return out

    def _forward(self, *a, **kw):
        res = super()._forward(*a, **kw)
        # res = (target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx)
        self.rec['target_labels'] = res[0].detach()
        self.rec['target_scores'] = res[2].detach()
        self.rec['fg_mask'] = res[3].detach().bool()
        self.rec['target_gt_idx'] = res[4].detach()
        if a:
            # a = (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
            # pd_bboxes / gt_bboxes 已是像素坐标 xyxy，用于独立算 CIoU（Q2 统一度量）
            self.rec['pd_scores'] = a[0].detach()
            self.rec['pd_bboxes'] = a[1].detach().clone()
            if len(a) > 4:
                self.rec['gt_bboxes'] = a[4].detach().clone()
        return res


def _mini_yaml(path, keys=('path', 'train', 'val')):
    d = {}
    for line in open(path, encoding='utf-8'):
        s = line.strip()
        if not s or s.startswith('#') or ':' not in s:
            continue
        k, v = s.split(':', 1)
        if k.strip() in keys:
            d[k.strip()] = v.strip()
    return d


def letterbox(img, imgsz):
    import cv2
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    py, px = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[py:py + nh, px:px + nw] = rs
    return np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)), r, px, py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--list', default=None,
                    help='显式清单（一行一个图像路径）；默认用 data yaml 的 train 字段。'
                         '做同协议比较时应指向 val 清单，与评测口径一致。')
    ap.add_argument('--n-img', type=int, default=500)
    ap.add_argument('--imgsz', type=int, default=800)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)

    import cv2
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    obj = ck.get('ema') or ck.get('model')
    if obj is None:
        print('[soft] 权重里没有 model/ema', file=sys.stderr)
        return 2
    model = obj.float().to(dev)
    model.args = get_cfg(overrides={'imgsz': a.imgsz, 'data': a.data, 'batch': 1})
    model.eval()
    model.model[-1].training = True

    crit = model.init_criterion()
    probe = Probe(topk=int(crit.assigner.topk), num_classes=int(crit.assigner.num_classes),
                  alpha=float(crit.assigner.alpha), beta=float(crit.assigner.beta),
                  stride=[float(s) for s in crit.assigner.stride],
                  topk2=getattr(crit.assigner, 'topk2', None),
                  eps=float(crit.assigner.eps))
    crit.assigner = probe

    cfg = _mini_yaml(a.data)
    root = cfg['path']
    if a.list:
        lst = a.list
    else:
        lst = cfg['train'] if os.path.isabs(cfg['train']) else os.path.join(root, cfg['train'])
    paths = [l.strip() for l in open(lst) if l.strip()][:a.n_img]

    # 累积容器
    acc = {k: [] for k in ('ts', 'norm', 'align', 'score', 'bucket', 'overlap',
                           'ratio', 'pos_align', 'pos_overlap', 'ciou')}
    n_img_used = 0

    for p in paths:
        raw = cv2.imread(p, cv2.IMREAD_COLOR)
        if raw is None:
            continue
        chw, r, px, py = letterbox(raw, a.imgsz)
        lp = p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        if not os.path.isfile(lp):
            continue
        arr = np.loadtxt(lp, ndmin=2)
        if arr.size == 0:
            continue
        arr = np.atleast_2d(arr)
        h0, w0 = raw.shape[:2]
        cx = arr[:, 1] * w0 * r + px
        cy = arr[:, 2] * h0 * r + py
        w = arr[:, 3] * w0 * r
        h = arr[:, 4] * h0 * r
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) / a.imgsz
        boxes = np.clip(boxes, 0, 1)

        bx = torch.tensor(boxes, dtype=torch.float32, device=dev)
        xywh = torch.empty_like(bx)
        xywh[:, 0] = (bx[:, 0] + bx[:, 2]) / 2
        xywh[:, 1] = (bx[:, 1] + bx[:, 3]) / 2
        xywh[:, 2] = bx[:, 2] - bx[:, 0]
        xywh[:, 3] = bx[:, 3] - bx[:, 1]
        batch = {'batch_idx': torch.zeros(len(arr), dtype=torch.long, device=dev),
                 'cls': torch.tensor(arr[:, :1], dtype=torch.float32, device=dev),
                 'bboxes': xywh}

        x = torch.from_numpy(chw).float().unsqueeze(0).to(dev) / 255.0
        probe.rec.clear()
        with torch.no_grad():
            out = model.predict(x)
            y = out[0] if isinstance(out, tuple) else out
            try:
                crit(y, batch)
            except Exception as e:                              # noqa: BLE001
                print('[soft] 分配失败 %s: %s' % (os.path.basename(p), e), file=sys.stderr)
                continue

        r_ = probe.rec
        if 'target_scores' not in r_ or not r_['fg_mask'].any():
            continue

        fg = r_['fg_mask'][0]                     # (na,)
        ts = r_['target_scores'][0]               # (na, nc) 已是最终值（乘过 norm）
        tlab = r_['target_labels'][0]             # (na,)
        tgi = r_['target_gt_idx'][0]              # (na,)
        algo = r_['align'][0]                     # (n_max, na)
        ov = r_['overlaps'][0]
        pds = r_['pd_scores'][0]                  # (na, nc)

        if not fg.any():
            continue

        # pre-norm = one_hot(target_labels) * fg_mask；norm = final / pre
        pre = F.one_hot(tlab.long(), num_classes=probe.num_classes).float() * fg[:, None].float()
        norm = ts / pre.clamp(min=1e-9)

        # 该 anchor 对应 GT 的尺寸桶（用像素 w/h，与 AI-TOD 官方 area 定义一致）
        bk = []
        for i in range(len(fg)):
            gi = int(tgi[i].item())
            if gi < 0 or gi >= len(w):
                bk.append('na')
            else:
                bk.append(bucket_of(w[gi], h[gi]))

        score_max = pds.max(-1)[0]                # 每 anchor 的最大类分数
        idx = fg.nonzero(as_tuple=False).flatten()
        mp = r_['mask_pos'][0]                    # (n_max, na)

        # ---- Q1：归一化因子（每 GT 的 pos_align / pos_overlap 及其比值）----
        #   norm_align = align * pos_overlap / pos_align
        #   两臂的绝对量不可比（NWD vs IoU），但比值 pos_overlap/pos_align 是无量纲的，
        #   用来看"归一化把软标签压低了多少"。
        pos_align = (algo * mp).amax(dim=1)       # (n_max,)
        pos_overlap = (ov * mp).amax(dim=1)
        okg = pos_align > 1e-12
        for q in (pos_overlap[okg] / pos_align[okg]).tolist():
            acc['ratio'].append(float(q))
        for q in pos_align[okg].tolist():
            acc['pos_align'].append(float(q))
        for q in pos_overlap[okg].tolist():
            acc['pos_overlap'].append(float(q))

        # ---- Q2：独立 CIoU（两臂同度量，用于替代不可比的 assigner overlap）----
        ciou_arr = None
        if 'pd_bboxes' in r_ and 'gt_bboxes' in r_ and len(idx):
            from ultralytics.utils.metrics import bbox_iou
            gi_all = tgi[idx].long()
            ok = gi_all >= 0
            if ok.any():
                pb = r_['pd_bboxes'][0][idx[ok]]
                gb = r_['gt_bboxes'][0][gi_all[ok]]
                c = bbox_iou(pb, gb, xywh=False).squeeze(-1)
                ciou_arr = np.full(len(idx), np.nan, dtype=np.float64)
                ciou_arr[ok.cpu().numpy()] = c.cpu().numpy()

        for j, i in enumerate(idx.tolist()):
            gi = int(tgi[i].item())
            if gi < 0 or gi >= algo.shape[0]:
                continue
            acc['ts'].append(float(ts[i].max()))
            acc['norm'].append(float(norm[i].max()))
            acc['align'].append(float(algo[gi, i]))
            acc['overlap'].append(float(ov[gi, i]))
            acc['score'].append(float(score_max[i]))
            acc['bucket'].append(bk[i])
            acc['ciou'].append(float(ciou_arr[j]) if ciou_arr is not None else float('nan'))
        n_img_used += 1
        if n_img_used % 100 == 0:
            print('[soft] %d 图，累计 %d 正样本' % (n_img_used, len(acc['ts'])))

    if not acc['ts']:
        print('[soft] 无正样本，退出', file=sys.stderr)
        return 2

    ts = np.array(acc['ts']); nrm = np.array(acc['norm'])
    al = np.array(acc['align']); sc = np.array(acc['score']); ovl = np.array(acc['overlap'])
    bks = np.array(acc['bucket'])

    def stat(x):
        return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()),
                "p10": float(np.percentile(x, 10)), "p25": float(np.percentile(x, 25)),
                "p50": float(np.percentile(x, 50)), "p75": float(np.percentile(x, 75)),
                "p90": float(np.percentile(x, 90))}

    def corr(x, y):
        if len(x) < 10:
            return {"pearson": None, "spearman": None}
        pr = float(np.corrcoef(x, y)[0, 1])
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        sp = float(np.corrcoef(rx, ry)[0, 1])
        return {"pearson": pr, "spearman": sp}

    print("\n" + "=" * 70)
    print("A. 正样本 final target_scores 分布（n=%d，%d 图）" % (len(ts), n_img_used))
    print("=" * 70)
    print(json.dumps(stat(ts), indent=1, ensure_ascii=False))
    print("\n分桶：")
    per = {}
    for name, _, _ in BUCKETS:
        m = bks == name
        if m.sum() == 0:
            continue
        per[name] = stat(ts[m])
        print("  %-10s n=%-6d mean %.4f  P50 %.4f  P10 %.4f  P90 %.4f"
              % (name, m.sum(), per[name]["mean"], per[name]["p50"],
                 per[name]["p10"], per[name]["p90"]))

    print("\n" + "=" * 70)
    print("B. 归一化前后")
    print("=" * 70)
    print("  align_metric      " + json.dumps(stat(al), ensure_ascii=False))
    print("  norm_align_metric " + json.dumps(stat(nrm), ensure_ascii=False))
    print("  final target_score" + json.dumps(stat(ts), ensure_ascii=False))

    print("\n" + "=" * 70)
    print("C. 排序耦合（正样本上）")
    print("=" * 70)
    for name, y in (("align_metric", al), ("norm_align_metric", nrm), ("overlap", ovl)):
        print("  score vs %-18s %s" % (name, json.dumps(corr(sc, y), ensure_ascii=False)))
    # 高分段（分数前 10%）
    thr = np.percentile(sc, 90)
    hi = sc >= thr
    print("\n  高分段（分数 ≥ P90=%.4f，n=%d）：" % (thr, int(hi.sum())))
    print("    align 均值 %.4f  norm 均值 %.4f  ts 均值 %.4f  overlap 均值 %.4f"
          % (al[hi].mean(), nrm[hi].mean(), ts[hi].mean(), ovl[hi].mean()))
    print("    高分段 score-alignment 相关 %s" % json.dumps(corr(sc[hi], al[hi]), ensure_ascii=False))

    # ---- Q1：归一化因子 ----
    pa = np.array(acc['pos_align']); po = np.array(acc['pos_overlap'])
    ra = np.array(acc['ratio'])
    print("\n" + "=" * 70)
    print("Q1  归一化因子（每 GT，n=%d）" % len(ra))
    print("=" * 70)
    print("  pos_align   " + json.dumps(stat(pa), ensure_ascii=False))
    print("  pos_overlap " + json.dumps(stat(po), ensure_ascii=False))
    print("  ratio=po/pa " + json.dumps(stat(ra), ensure_ascii=False))
    print("  ※ 绝对量跨臂不可比；看 ratio 的分位与离散程度")

    # ---- Q2：独立 CIoU（两臂同度量）----
    ci = np.array(acc['ciou'])
    okc = np.isfinite(ci)
    print("\n" + "=" * 70)
    print("Q2  独立 CIoU（两臂同度量，有效 %d/%d）" % (int(okc.sum()), len(ci)))
    print("=" * 70)
    ci_ok = ci[okc]; sc_ok = sc[okc]; ts_ok = ts[okc]
    print("  CIoU 分布  " + json.dumps(stat(ci_ok), ensure_ascii=False))
    print("  score vs CIoU  " + json.dumps(corr(sc_ok, ci_ok), ensure_ascii=False))
    print("  ts    vs CIoU  " + json.dumps(corr(ts_ok, ci_ok), ensure_ascii=False))
    thr2 = np.percentile(sc_ok, 90)
    h2 = sc_ok >= thr2
    print("  高分段（≥%.4f, n=%d）: CIoU 均值 %.4f, score-CIoU %s"
          % (thr2, int(h2.sum()), ci_ok[h2].mean(),
             json.dumps(corr(sc_ok[h2], ci_ok[h2]), ensure_ascii=False)))

    res = {"ckpt": a.ckpt, "n_img": n_img_used, "n_pos": int(len(ts)),
           "final_ts": stat(ts), "norm_align": stat(nrm), "align": stat(al),
           "per_bucket": per,
           "corr": {"score_align": corr(sc, al), "score_norm": corr(sc, nrm),
                    "score_overlap": corr(sc, ovl)},
           "high_seg": {"thr": float(thr), "n": int(hi.sum()),
                        "align_mean": float(al[hi].mean()), "norm_mean": float(nrm[hi].mean()),
                        "ts_mean": float(ts[hi].mean()),
                        "corr_score_align": corr(sc[hi], al[hi])},
           "q1_norm": {"n_gt": int(len(ra)), "pos_align": stat(pa),
                       "pos_overlap": stat(po), "ratio": stat(ra)},
           "q2_ciou": {"n": int(okc.sum()), "dist": stat(ci_ok),
                       "corr_score": corr(sc_ok, ci_ok),
                       "corr_ts": corr(ts_ok, ci_ok),
                       "high": {"thr": float(thr2), "n": int(h2.sum()),
                                "ciou_mean": float(ci_ok[h2].mean()),
                                "corr": corr(sc_ok[h2], ci_ok[h2])}}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    json.dump(res, open(a.out + '.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print("\n写出 %s.json" % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())

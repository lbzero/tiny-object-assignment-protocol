#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p0_6_dynamic_trace.py —— P0-6 动态诊断：真实模型预测 × 真实 loss 路径的 assigner

与静态诊断（p0_stage_trace.py）的区别
-------------------------------------
静态：人工构造预测框（anchor 中心 ± 2×stride，cls 恒 0.99）→ assigner
动态：真实模型前向得到 pd_scores / pd_bboxes → **真实 criterion(preds, batch)** → assigner

后者走的是 ultralytics 训练损失里的那一条链路（v8DetectionLoss.parse_output → loss →
get_assigned_targets_and_loss → self.assigner），不是自己重写的近似。

三个必须遵守的方法学约束（评审 2026-11-14）
-------------------------------------------
1. **不是历史重放**。闭合增强、固定图像与权重、单图 batch 得到的是
   「固定模型快照下、受控输入条件下的动态分配诊断」，
   不等于 epoch N 当时训练 batch 的真实分配。
2. **BN 状态必须声明**。默认用 `model.eval()`（BN 走 running stats，可复现、不受 batch 组成影响）；
   `--train-mode` 改为训练态（BN 用当前 batch 统计），作为敏感性对照，不可混报。
3. **GT 身份不能靠位置推断**。gt_idx → annotation_id 用 build_gt_mapping.py 产出的映射表，
   映射表本身经全量坐标核对（2785/2804 逐行一致、0 反例）。

输出
----
  <out>.csv          逐 GT：image_id, gt_idx, annotation_id, bucket, s1..s4, 预测分数统计
  <out>_frames.csv   逐图：预测框数、score 分位、assigner 耗时
  <out>_meta.json    运行配置（权重类型、epoch、模式、assigner 参数、代码指纹）

用法
----
  AITOD_RULE=default python p0_6_dynamic_trace.py \
      --ckpt /root/runs/p1/E0_default_s3/weights/last.pt \
      --weight-type ema --ann ... --labels ... \
      --map /root/runs/p1/gtmap_val_map.csv \
      --ids-file /root/runs/p1/p0_invariance_ids50.txt \
      --out /root/runs/p1/p06_default_ema
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aitod_assigner                                     # noqa: E402  安装 AITOD_RULE patch
from p0_tracing import TracingAssigner, check_invariants    # noqa: E402
from ultralytics.utils import tal                         # noqa: E402
from ultralytics.utils.tal import make_anchors            # noqa: E402


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
def bucket_of(area):
    if area < 64:
        return 'verytiny'
    if area < 256:
        return 'tiny'
    if area < 1024:
        return 'small'
    return 'medium'


def file_sha(path, n=16):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:n]


def load_map(path):
    """(image_id, gt_idx) -> 映射行"""
    out = {}
    with open(path, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            out[(int(r['image_id']), int(r['gt_idx']))] = r
    return out


def letterbox_img(img, imgsz):
    """与 ultralytics 训练预处理对齐的 letterbox。

    AI-TODv2 val 里 2803/2804 是 800x800，scale=1、pad=0，等价于恒等变换；
    剩余 1 张（以及其他数据集）走标准缩放 + 居中填充。
    返回 (uint8 CHW, scale, pad_x, pad_y)
    """
    import cv2
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    px, py = (imgsz - nw) // 2, (imgsz - nh) // 2
    canvas[py:py + nh, px:px + nw] = img
    chw = canvas[:, :, ::-1].transpose(2, 0, 1)           # BGR->RGB, HWC->CHW
    return np.ascontiguousarray(chw), r, float(px), float(py)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--weight-type', default='auto', choices=['auto', 'ema', 'raw'])
    ap.add_argument('--ann', required=True)
    ap.add_argument('--images', required=True, help='图像根目录')
    ap.add_argument('--map', required=True, help='build_gt_mapping.py 产出的 _map.csv')
    ap.add_argument('--ids-file', default=None, help='固定图像 ID 清单，一行一个')
    ap.add_argument('--limit', type=int, default=0, help='0 = 全部（与 ids-file 互斥优先 ids-file）')
    ap.add_argument('--imgsz', type=int, default=800)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--train-mode', action='store_true',
                    help='BN 用训练态（batch 统计）。默认 eval 态。两种不可混报')
    ap.add_argument('--out', required=True, help='输出前缀')
    ap.add_argument('--sim', default='ciou', choices=['ciou', 'nwd'],
                    help='分配相似度：ciou=原生（默认，与历史结果一致）；nwd=NWD-TAL')
    args = ap.parse_args()

    dev = torch.device(args.device)

    # ---- 0. 相似度挂载（必须在 assigner 被调用前）----
    # TracingAssigner 继承自 TaskAlignedAssigner 且未重写 iou_calculation / pair_overlaps，
    # 所以这里改类属性即可让插桩路径也用上 NWD，插桩本身不需要改。
    nwd_c = None
    if args.sim == 'nwd':
        import nwd_tal
        nwd_c = float(os.environ.get('AITOD_NWD_C', '12.8'))
        nwd_tal.install(C=nwd_c)
        print('[p06] 相似度 = NWD   C=%.2f' % nwd_c)
    else:
        print('[p06] 相似度 = CIoU（原生）')

    # ---- 1. 权重 ----
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    has_m, has_e = ck.get('model') is not None, ck.get('ema') is not None
    wt = args.weight_type
    if wt == 'auto':
        if has_m:
            wt = 'raw'
        elif has_e:
            wt = 'ema'
        else:
            print('[p06] checkpoint 既无 model 也无 ema', file=sys.stderr)
            return 2
    obj = ck.get('model') if wt == 'raw' else ck.get('ema')
    if obj is None:
        print('[p06] 请求 %s 权重但 checkpoint 中不存在（model=%s ema=%s）'
              % (wt, has_m, has_e), file=sys.stderr)
        return 2
    model = obj.float().to(dev)
    det = model.model[-1]

    # 前向分支：Detect 在 training=True 时返回原始输出（list），否则走推理后处理。
    # 我们统一用 model.predict() 直接取原始输出，因此这里只需要控制 BN 行为。
    model.eval() if not args.train_mode else model.train()
    det.training = True                                    # 保证返回原始特征
    bn_mode = 'train' if args.train_mode else 'eval'

    # ---- 2. criterion 与 assigner ----
    crit = model.init_criterion()
    a0 = crit.assigner
    def _num(v):
        return float(v) if not isinstance(v, (int, float)) else v

    tracer = TracingAssigner(topk=int(a0.topk), num_classes=int(a0.num_classes),
                             alpha=_num(a0.alpha), beta=_num(a0.beta),
                             stride=[float(x) for x in a0.stride], eps=float(a0.eps))
    crit.assigner = tracer
    stride = [float(x) for x in tracer.stride]
    feats = [torch.zeros(1, 1, args.imgsz // int(s), args.imgsz // int(s)) for s in stride]
    pts, sts = make_anchors(feats, stride, 0.5)
    n_anchor = int(pts.shape[0])

    # ---- 3. 数据 ----
    ann = json.load(open(args.ann, encoding='utf-8'))
    imgs_meta = {i['id']: i for i in ann['images']}
    gtmap = load_map(args.map)

    if args.ids_file:
        ids = [int(x) for x in open(args.ids_file) if x.strip()]
    else:
        ids = sorted(imgs_meta.keys())
        if args.limit:
            ids = ids[:args.limit]
    ids = [i for i in ids if i in imgs_meta]

    # 预期 GT 数（用于 fail-closed 计数核对）
    expect_gt = sum(1 for k in gtmap if k[0] in set(ids))

    rows = []
    frames = []
    viol_total = 0
    n_fail = 0
    t_assign = 0.0

    for si, iid in enumerate(ids):
        im = imgs_meta[iid]
        # ---- 读图 ----
        import cv2
        p = os.path.join(args.images, im['file_name'])
        if not os.path.isfile(p):
            print('[p06] 缺图: %s' % p, file=sys.stderr)
            return 2
        raw = cv2.imread(p, cv2.IMREAD_COLOR)
        if raw is None:
            print('[p06] 读图失败: %s' % p, file=sys.stderr)
            return 2
        chw, sc, px, py = letterbox_img(raw, args.imgsz)

        # ---- GT（用映射表，保证与训练看到的裁剪后坐标一致）----
        gts = []
        for (mi, gi), r in gtmap.items():
            if mi != iid:
                continue
            gts.append((int(r['gt_idx']), r))
        gts.sort(key=lambda t: t[0])
        if not gts:
            frames.append({'image_id': iid, 'n_gt': 0, 'n_pred': 0,
                           'score_p50': '', 'score_p95': '', 'score_max': '',
                           'skip': 'no_gt_in_training'})
            continue

        boxes = torch.tensor([[float(r['x']), float(r['y']),
                               float(r['x']) + float(r['w']),
                               float(r['y']) + float(r['h'])] for _, r in gts],
                             dtype=torch.float32)
        # 变换到 letterbox 坐标系并归一化
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * sc + px
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * sc + py
        boxes = boxes.clamp(min=0.0, max=float(args.imgsz)) / float(args.imgsz)

        # 关键：v8DetectionLoss.preprocess 会对 batch['bboxes'] 执行
        #   xywh2xyxy(x.mul_(scale_tensor))
        # 即它期待的是 **归一化 xywh**，不是 xyxy。传错会让框放大约 10 倍，
        # s1 虚高到数千（2026-11-14 实测踩过：s1 均值 2222 而真值 2）。
        xywh = torch.empty_like(boxes)
        xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
        xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
        xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
        xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
        cls = torch.tensor([[float(r['category_id']) - 1] for _, r in gts], dtype=torch.float32)

        batch = {
            'batch_idx': torch.zeros(len(gts), dtype=torch.long, device=dev),
            'cls': cls.to(dev),
            'bboxes': xywh.to(dev),
        }

        # ---- 前向（真实模型预测）----
        x = torch.from_numpy(chw).float().unsqueeze(0).to(dev) / 255.0
        try:
            with torch.no_grad():
                preds = model.predict(x)
        except Exception as e:                             # noqa: BLE001
            print('[p06] img %d 前向失败: %s' % (iid, e), file=sys.stderr)
            n_fail += 1
            continue

        # ---- 真实 loss 路径 ----
        tracer.trace.clear()
        try:
            t0 = time.perf_counter()
            with torch.no_grad():
                out = crit(preds, batch)                   # 内部调用 tracer(...)
            t_assign += time.perf_counter() - t0
        except Exception as e:                             # noqa: BLE001
            print('[p06] img %d assigner 失败: %s' % (iid, e), file=sys.stderr)
            n_fail += 1
            continue

        tr = tracer.trace
        if 'fg' not in tr or 'tgt' not in tr:
            print('[p06] img %d 未捕获最终分配，trace keys=%s' % (iid, list(tr)), file=sys.stderr)
            n_fail += 1
            continue

        # ---- 解析 ----
        # S4 = 从最终 fg_mask 与 target_gt_idx 反算的正样本数（与原生 _forward 同源）
        fg, tgt = tr['fg'][0], tr['tgt'][0]
        s4_list = (torch.bincount(tgt[fg], minlength=len(gts)).tolist()
                   if fg.any() else [0] * len(gts))
        col = {k: tr[k][0].tolist() for k in
               ('s1', 'topk_kept', 's2_in_s1', 'n_shared', 's3pre', 's3post', 'n_lost', 'n_gained')
               if k in tr}
        bad = check_invariants(tr, s4_list)
        if bad:
            viol_total += 1
            print('[p06] img %d 恒等违反: %s' % (iid, bad), file=sys.stderr)

        # 预测分数统计（真实模型的 cls 分支，未经 sigmoid 之外的处理）
        po = crit.parse_output(preds)
        sc_all = po['scores'].sigmoid().max(1)[0][0].float().cpu()
        n_pred = int((sc_all > 0.25).sum())                # 仅供参考的粗计数
        frames.append({'image_id': iid, 'n_gt': len(gts), 'n_pred': n_pred,
                       'score_p50': round(float(sc_all.median()), 6),
                       'score_p95': round(float(sc_all.quantile(0.95)), 6),
                       'score_max': round(float(sc_all.max()), 6), 'skip': ''})

        for k, (gi, r) in enumerate(gts):
            if k >= len(s4_list):
                print('[p06] img %d gt_idx %d 超出 assigner 输出长度' % (iid, gi), file=sys.stderr)
                n_fail += 1
                break
            rows.append({
                'image_id': iid, 'gt_idx': gi, 'annotation_id': r['annotation_id'],
                'category_id': r['category_id'], 'bucket': r['bucket'],
                'd': r['d'], 'area': r['area'],
                's1': col['s1'][k], 'topk_kept': col['topk_kept'][k],
                's2_in_s1': col['s2_in_s1'][k], 'n_shared': col['n_shared'][k],
                's3pre': col['s3pre'][k], 's3post': col['s3post'][k],
                'n_lost': col['n_lost'][k], 'n_gained': col['n_gained'][k],
                's4': s4_list[k],
            })

        if (si + 1) % 10 == 0 or si == 0:
            print('[p06] %d/%d 图，累计 %d GT' % (si + 1, len(ids), len(rows)))

    # ---- fail-closed 总闸 ----
    if n_fail:
        print('[p06] 失败 %d 张，不写结果（fail-closed）' % n_fail, file=sys.stderr)
        return 2
    if viol_total:
        print('[p06] 恒等违反 %d 张，不写结果（fail-closed）' % viol_total, file=sys.stderr)
        return 3
    if len(rows) != expect_gt:
        print('[p06] GT 行数 %d != 预期 %d（fail-closed）' % (len(rows), expect_gt), file=sys.stderr)
        return 4

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    cols = ['image_id', 'gt_idx', 'annotation_id', 'category_id', 'bucket', 'd', 'area',
            's1', 'topk_kept', 's2_in_s1', 'n_shared', 's3pre', 's3post',
            'n_lost', 'n_gained', 's4']
    with open(args.out + '.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    with open(args.out + '_frames.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['image_id', 'n_gt', 'n_pred',
                                          'score_p50', 'score_p95', 'score_max', 'skip'])
        w.writeheader()
        w.writerows(frames)

    meta = {
        'ckpt': os.path.abspath(args.ckpt),
        'ckpt_sha256_16': file_sha(args.ckpt),
        'weight_type': wt,
        'ckpt_epoch': ck.get('epoch'),
        'bn_mode': bn_mode,
        'assigner': {'topk': tracer.topk, 'topk2': getattr(tracer, 'topk2', None),
                     'alpha': tracer.alpha, 'beta': tracer.beta, 'eps': tracer.eps,
                     'stride': stride},
        'n_anchor': n_anchor,
        'imgsz': args.imgsz,
        'n_images': len(ids),
        'n_gt_rows': len(rows),
        'rule': os.environ.get('AITOD_RULE', 'default'),
        'min_pos': os.environ.get('AITOD_MIN_POS'),
        'sim': args.sim,
        'nwd_c': nwd_c,
        'device': str(dev),
        'seconds_total': round(t_assign, 3),
        'ultralytics': __import__('ultralytics').__version__,
        'torch': torch.__version__,
        'note': ('固定权重 + 关闭增强 + 单图 batch 下的动态分配诊断；'
                 '不等于训练时历史 batch 的精确重放。'),
    }
    json.dump(meta, open(args.out + '_meta.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    vt = [r for r in rows if r['bucket'] == 'verytiny']
    z = sum(1 for r in vt if r['s4'] == 0)
    print('[p06] 写出 %s.csv  (%d GT，其中 vt %d)' % (args.out, len(rows), len(vt)))
    print('[p06] vt 零正样本率: %.2f%%  |  assigner 累计耗时 %.2fs  (%.1f ms/图)'
          % (100.0 * z / max(len(vt), 1), t_assign, 1000 * t_assign / max(len(ids), 1)))
    return 0


if __name__ == '__main__':
    sys.exit(main())

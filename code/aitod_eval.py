#!/usr/bin/env python
"""AI-TOD 官方口径评测：权重 -> COCO 预测 -> aitodpycocotools COCOeval。

为什么不用 ultralytics 自带的 mAP：
  * ultralytics 用 COCO 默认面积桶 [0,32^2],[32^2,96^2],[96^2,1e10]，
    而 AI-TOD 官方是 [[0,1e10],[0,64),[64,256),[256,1024),[1024,1e10)]
    （verytiny / tiny / small / medium），两套数字不能互换；
  * AI-TOD 的中位目标只有 11.83 px，14% 的框面积 < 64 px^2，
    用错面积桶会把 APvt 整个吞掉；
  * 官方 maxDets 是 [1,100,1500]，LTDNet 论文覆写成 [100,300,1500]，两者必须分开报。

GT 一律用官方 annotations_v2 的 json，不用 yolo labels 反推：
  实测 val json 有 75091 个框，yolo labels 只有 75065，差 26 个（19 张图），
  其中 9 个是宽或高 <= 0 的退化框在转换时被丢弃。用 labels 反推会同时
  少框并让面积分桶偏移。

用法：
  # 全量 val 评测（官方口径）
  python aitod_eval.py --weights /root/runs/p1/E0_default_s0/weights/best.pt

  # 小样本冒烟（50 张，验证脚本本身）
  python aitod_eval.py --weights ... --limit 50 --out /tmp/eval_smoke

  # 同时报 LTDNet 口径
  python aitod_eval.py --weights ... --also-ldtnet
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = '/root/data/AI-TOD'
ANN_VAL = os.path.join(ROOT, 'annotations_v2', 'aitod_val_v2.json')
IMG_VAL = os.path.join(ROOT, 'images', 'val')

AREA_LABELS = ['all', 'verytiny', 'tiny', 'small', 'medium']


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--ann', default=ANN_VAL)
    ap.add_argument('--img-dir', default=IMG_VAL)
    ap.add_argument('--imgsz', type=int, default=800)
    ap.add_argument('--conf', type=float, default=0.001,
                    help='评测要放很低，否则 AP 积分被截断，APvt 尤其敏感')
    ap.add_argument('--iou', type=float, default=0.7, help='NMS 的 IoU 阈值')
    ap.add_argument('--max-det', type=int, default=1500,
                    help='单图最多保留的预测框；官方 maxDets 上限也是 1500')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--chunk', type=int, default=256,
                    help='每多少张图重新调用一次 predict，用来压住内存峰值')
    ap.add_argument('--device', default='0')
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 张（冒烟用）')
    ap.add_argument('--out', default=None, help='输出目录，默认 <weights>/../eval_official')
    ap.add_argument('--also-ldtnet', action='store_true',
                    help='额外按 LTDNet 口径 maxDets=[100,300,1500] 再算一遍')
    ap.add_argument('--max-dets', default='1,100,1500',
                    help='主口径，逗号分隔，必须包含 1500')
    ap.add_argument('--cpus', default='',
                    help='绑定到指定核（如 0,1）。训练在用 8 个 dataloader worker，'
                         '评测如果不绑核会和它抢 CPU，实测 load 冲到 9.35 导致 WSL 的'
                         'wsl.exe 通道都超时（WSAETIMEDOUT）')
    ap.add_argument('--no-nice', action='store_true',
                    help='不自动把优先级降到 19（默认会降，保证训练优先）')
    return ap.parse_args()


# ---------------------------------------------------------------- GT

def load_gt(ann_path):
    """读官方 json，返回 (coco_dict, file_name -> image_id)。"""
    with open(ann_path) as f:
        d = json.load(f)
    name2id = {im['file_name']: im['id'] for im in d['images']}
    return d, name2id


# ---------------------------------------------------------------- 推理

def run_inference(args, name2id):
    """在 val 图上推理，返回 COCO 格式的预测列表。

    必须分批。2026-11-07 第一次全量跑被 OOM killer 杀掉（anon-rss 8.1 GB）：
    ultralytics 的每个 Results 都持有 orig_img（800x800x3 uint8 约 1.9 MB），
    2804 张就是 5 GB 以上，再叠加 108 万条预测和 COCO 索引，直接挤爆 WSL 的
    14 GB 配额（训练本身还要占 4 GB）。分批 predict + 显式释放能把峰值压到 2 GB 内。
    """
    import gc

    from ultralytics import YOLO

    model = YOLO(args.weights)
    names = model.names  # {0:'airplane', ...}

    files = sorted(os.listdir(args.img_dir))
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if args.limit:
        files = files[:args.limit]
    paths = [os.path.join(args.img_dir, f) for f in files]
    print('[infer] %d 张图  imgsz=%d conf=%.4f iou=%.2f max_det=%d device=%s chunk=%d'
          % (len(paths), args.imgsz, args.conf, args.iou, args.max_det,
             args.device, args.chunk))

    preds = []
    missing_id = 0
    n_img = 0
    t0 = time.time()

    for start in range(0, len(paths), args.chunk):
        part = paths[start:start + args.chunk]
        results = model.predict(source=part, imgsz=args.imgsz, conf=args.conf,
                                iou=args.iou, max_det=args.max_det, device=args.device,
                                batch=args.batch, stream=True, verbose=False)
        for r in results:
            n_img += 1
            fname = os.path.basename(r.path)
            iid = name2id.get(fname)
            if iid is None:
                missing_id += 1
                continue
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                for k in range(len(conf)):
                    x1, y1, x2, y2 = xyxy[k]
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    if w <= 0 or h <= 0:
                        continue
                    preds.append({
                        'image_id': int(iid),
                        'category_id': int(cls[k]),
                        'bbox': [float(x1), float(y1), w, h],
                        'score': float(conf[k]),
                    })
            # 断开对原图的引用，否则 1.9 MB/张会一路累积
            r.orig_img = None
            r.orig_shape = None
            del r
        del results
        gc.collect()
        el = time.time() - t0
        print('  %d/%d  %.1f img/s  框数 %d  内存 %.2f GB'
              % (n_img, len(paths), n_img / max(el, 1e-9), len(preds),
                 _rss_gb()))

    print('[infer] 完成 %d 张，预测 %d 框，耗时 %.1f s，找不到 image_id 的图 %d'
          % (n_img, len(preds), time.time() - t0, missing_id))
    with_pred = len(set(p['image_id'] for p in preds))
    print('[infer] 有预测的图片数 %d / %d（其余是模型确实一个框都没检出）'
          % (with_pred, n_img))
    return preds, files, missing_id, with_pred


def _rss_gb():
    try:
        with open('/proc/self/status') as f:
            for ln in f:
                if ln.startswith('VmRSS:'):
                    return int(ln.split()[1]) / 1024 / 1024
    except Exception:                                          # noqa: BLE001
        pass
    return float('nan')


# ---------------------------------------------------------------- 官方评测

def coco_eval(ann_path, preds, max_dets, tag=''):
    """走 aitodpycocotools，返回 {指标名: 值}。max_dets 必须含 1500。"""
    from aitodpycocotools.coco import COCO
    from aitodpycocotools.cocoeval import COCOeval

    cocoGt = COCO(ann_path)
    if not preds:
        raise SystemExit('[eval] 没有任何预测框，无法评测')
    cocoDt = cocoGt.loadRes(preds)

    E = COCOeval(cocoGt, cocoDt, 'bbox')
    E.params.maxDets = list(max_dets)
    E.evaluate()
    E.accumulate()
    E.summarize()

    # 自己按 areaRng 取 AP，行为对齐 cocoapi 的 _summarize
    iouThrs = np.array(E.params.iouThrs)
    areaRng = E.params.areaRng
    areaLbl = E.params.areaRngLbl
    maxDetList = E.params.maxDets
    p = E.eval['precision']          # [T, R, K, A, M]
    aind_all = areaLbl.index('all')

    def _ap(iou_thr=None, area='all', max_det=1500):
        a = areaLbl.index(area)
        if max_det not in maxDetList:
            return float('nan')
        m = maxDetList.index(max_det)
        if iou_thr is None:
            s = p[:, :, :, a, m]
            s = s[s > -1]
            return float(np.mean(s)) if s.size else float('nan')
        t = int(np.argmin(np.abs(iouThrs - iou_thr)))
        s = p[t, :, :, a, m]
        s = s[s > -1]
        return float(np.mean(s)) if s.size else float('nan')

    # 逐类 AP（all area, maxDet=1500, IoU .5:.95）
    catIds = E.params.catIds
    per_cat = {}
    for ci, cid in enumerate(catIds):
        s = p[:, :, ci, aind_all, maxDetList.index(1500)]
        s = s[s > -1]
        per_cat[str(cid)] = float(np.mean(s)) if s.size else float('nan')

    out = {
        'tag': tag,
        'maxDets': list(maxDetList),
        'n_preds': len(preds),
        'AP': _ap(None, 'all', 1500),
        'AP50': _ap(0.5, 'all', 1500),
        'AP75': _ap(0.75, 'all', 1500),
        'APvt': _ap(None, 'verytiny', 1500),
        'APt':  _ap(None, 'tiny', 1500),
        'APs':  _ap(None, 'small', 1500),
        'APm':  _ap(None, 'medium', 1500),
        'APvt50': _ap(0.5, 'verytiny', 1500),
        'APt50':  _ap(0.5, 'tiny', 1500),
        'per_cat_AP': per_cat,
    }
    # AR@1500（按面积桶），来自 recall 数组 [T,K,A,M]
    rec = E.eval['recall']
    for lbl in AREA_LABELS:
        a = areaLbl.index(lbl)
        s = rec[:, :, a, maxDetList.index(1500)]
        s = s[s > -1]
        out['AR1500_' + lbl] = float(np.mean(s)) if s.size else float('nan')
    return out


# ---------------------------------------------------------------- 逐桶召回（自定义匹配）

def _xywh_iou(A, B):
    """A: (N,4) xywh, B: (M,4) xywh -> (N,M) IoU"""
    ax1, ay1 = A[:, 0], A[:, 1]
    ax2, ay2 = A[:, 0] + A[:, 2], A[:, 1] + A[:, 3]
    bx1, by1 = B[:, 0], B[:, 1]
    bx2, by2 = B[:, 0] + B[:, 2], B[:, 1] + B[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    ua = (A[:, 2] * A[:, 3])[:, None] + (B[:, 2] * B[:, 3])[None, :] - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-12), 0.0)


def _bucket(area):
    if area < 64:
        return 'verytiny'
    if area < 256:
        return 'tiny'
    if area < 1024:
        return 'small'
    return 'medium'


def bucket_recall(gt, preds, iou_thr=0.5, max_det=1500):
    """显式匹配，算每个面积桶的召回率。

    定义：该桶内的 GT 中，被 IoU>=iou_thr 且按 score 贪心一对一匹配到的比例，
    单图预测上限 max_det。与 COCOeval 的 AR 口径不同——后者按 maxDets 切，
    这里按面积桶切，专门用来看"极小目标到底漏了多少"。
    """
    gt_by_img = {}
    for a in gt['annotations']:
        gt_by_img.setdefault(a['image_id'], []).append(a)
    dt_by_img = {}
    for d in preds:
        dt_by_img.setdefault(d['image_id'], []).append(d)

    tot = {k: 0 for k in AREA_LABELS[1:]}
    hit = {k: 0 for k in AREA_LABELS[1:]}

    for iid, gts in gt_by_img.items():
        g_boxes = np.array([g['bbox'] for g in gts], dtype=np.float64)
        g_area = g_boxes[:, 2] * g_boxes[:, 3]
        g_bucket = [_bucket(x) for x in g_area]
        for b in g_bucket:
            tot[b] += 1

        dts = dt_by_img.get(iid, [])
        if not dts:
            continue
        dts = sorted(dts, key=lambda x: -x['score'])[:max_det]
        d_boxes = np.array([d['bbox'] for d in dts], dtype=np.float64)
        iou = _xywh_iou(d_boxes, g_boxes)          # (N_dt, N_gt)

        used_g = np.zeros(len(g_boxes), dtype=bool)
        for i in range(iou.shape[0]):              # 按 score 从高到低
            # 必须在**尚未匹配**的 GT 中找 IoU 最大的那个。
            # 旧写法是「先 argmax 全局，再检查是否已被占用」——一旦最优 GT 已被
            # 前面的预测占掉，这个预测就直接放弃，本可以匹配次优 GT 的机会丢失，
            # 召回被系统性低估。2026-11-11 发现：E2 的 vt 召回因此被算成
            # 59.09%，按 COCO 口径（在未匹配 GT 中取最大）重算约 66.7%。
            row = np.where(used_g, -1.0, iou[i])
            j = int(np.argmax(row))
            if row[j] >= iou_thr:
                used_g[j] = True
                hit[g_bucket[j]] += 1

    out = {}
    for b in AREA_LABELS[1:]:
        out[b] = {
            'gt': tot[b],
            'hit': hit[b],
            'recall': (hit[b] / tot[b]) if tot[b] else float('nan'),
        }
    return out


# ---------------------------------------------------------------- 主流程

def _yield_to_training(args):
    """主动让路：把 nice 降到 19，可选绑核。

    训练用 8 个 dataloader worker + 主进程，10 个核本来就紧。评测若不限自己，
    load 会冲到 9.35，实测连 wsl.exe 的命令通道都开始超时（Wsl/Service/WSAETIMEDOUT），
    训练虽然没停但整机响应会卡。默认降优先级，需要时再配 --cpus 绑核。
    """
    if not args.no_nice:
        try:
            cur = os.nice(0)
            if cur < 19:
                os.nice(19 - cur)
            print('[cpu] nice=%d（训练优先）' % os.nice(0))
        except Exception:                                      # noqa: BLE001
            pass
    if args.cpus:
        try:
            cores = [int(x) for x in args.cpus.split(',') if x.strip()]
            os.sched_setaffinity(0, cores)
            print('[cpu] 已绑定核心 %s（核数 %d）' % (cores, len(cores)))
        except Exception as e:                                 # noqa: BLE001
            print('[cpu] 绑核失败: %s' % e)


def main():
    args = parse_args()
    _yield_to_training(args)
    out_dir = args.out or os.path.join(os.path.dirname(os.path.dirname(args.weights)), 'eval_official')
    os.makedirs(out_dir, exist_ok=True)

    gt, name2id = load_gt(args.ann)
    print('[gt] %d 图 / %d 框 / %d 类'
          % (len(gt['images']), len(gt['annotations']), len(gt['categories'])))

    preds, files, missing_id, with_pred = run_inference(args, name2id)

    # 小样本冒烟必须把 GT 一起裁到同样的图。否则 20 张图去对 2804 张的标注，
    # 剩下 2784 张全算漏检，AP 会被摊薄两个数量级（实测 0.003 vs 实际 0.25 量级），
    # 看起来像脚本坏了，其实只是口径错了。
    ann_use, gt_use = args.ann, gt
    if args.limit:
        ids = {name2id[f] for f in files if f in name2id}
        gt_use = {
            'images': [im for im in gt['images'] if im['id'] in ids],
            'annotations': [a for a in gt['annotations'] if a['image_id'] in ids],
            'categories': gt['categories'],
        }
        ann_use = os.path.join(out_dir, 'gt_subset.json')
        with open(ann_use, 'w') as f:
            json.dump(gt_use, f)
        print('[gt] 小样本模式：GT 已裁剪到 %d 图 / %d 框 -> %s'
              % (len(gt_use['images']), len(gt_use['annotations']), ann_use))

    pred_path = os.path.join(out_dir, 'predictions.json')
    with open(pred_path, 'w') as f:
        json.dump(preds, f)
    print('[save] 预测 -> %s' % pred_path)

    main_dets = [int(x) for x in args.max_dets.split(',')]
    if 1500 not in main_dets:
        raise SystemExit('--max-dets 必须包含 1500，否则 AR/AP 无法与官方对齐')

    res = coco_eval(ann_use, preds, main_dets, tag='official')
    res['weights'] = args.weights
    res['conf'] = args.conf
    res['nms_iou'] = args.iou
    res['n_images'] = len(gt_use['images'])
    res['coverage'] = {
        'gt_images': len(gt_use['images']),
        'images_with_pred': with_pred,
        'missing_file_id': missing_id,
        'limited': bool(args.limit),
    }
    if missing_id:
        print('[warn] 有 %d 张图的文件名在官方 json 里找不到，这些图被跳过' % missing_id)

    rec = bucket_recall(gt_use, preds)
    res['bucket_recall_iou50_maxdet1500'] = rec

    if args.also_ldtnet:
        res['ldtnet_scale'] = coco_eval(ann_use, preds, [100, 300, 1500], tag='ldtnet')

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    # ---- 打印 ----
    print()
    print('=' * 64)
    print('AI-TOD 官方口径 (maxDets=%s)  %s' % (main_dets, os.path.basename(args.weights)))
    print('=' * 64)
    print('  图片 %d 张，其中有预测 %d 张%s'
          % (len(gt_use['images']), with_pred,
             '   [小样本模式，结果不代表全集]' if args.limit else ''))
    print('  AP      (IoU .50:.95, all)      %.4f' % res['AP'])
    print('  AP50                            %.4f' % res['AP50'])
    print('  AP75                            %.4f' % res['AP75'])
    print('  ---- 按面积桶 ----')
    for key, lbl, rng in [('APvt', 'verytiny', '[0,64)'),
                          ('APt', 'tiny', '[64,256)'),
                          ('APs', 'small', '[256,1024)'),
                          ('APm', 'medium', '[1024,inf)')]:
        print('  %-6s %-10s %-14s  %.4f' % (key, lbl, rng, res[key]))
    print('  APvt50                          %.4f' % res['APvt50'])
    print('  APt50                           %.4f' % res['APt50'])
    print('  ---- 逐桶召回 (IoU>=0.5, maxDet=1500) ----')
    for b in AREA_LABELS[1:]:
        r = rec[b]
        print('  %-10s %5d/%5d = %6.2f%%' % (b, r['hit'], r['gt'], 100 * r['recall']))
    print('  ---- 逐类 AP ----')
    id2name = {c['id']: c['name'] for c in gt['categories']}
    for cid, v in res['per_cat_AP'].items():
        print('  %-14s %.4f' % (id2name.get(int(cid), cid), v))

    if args.also_ldtnet:
        L = res['ldtnet_scale']
        print()
        print('LTDNet 口径 (maxDets=[100,300,1500])')
        print('  AP %.4f  AP50 %.4f  APvt %.4f  APt %.4f  APs %.4f  APm %.4f'
              % (L['AP'], L['AP50'], L['APvt'], L['APt'], L['APs'], L['APm']))

    print()
    print('[done] 结果 -> %s' % os.path.join(out_dir, 'metrics.json'))


if __name__ == '__main__':
    main()

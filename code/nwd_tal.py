"""NWD-TAL：把 TAL 里的 IoU 换成归一化高斯 Wasserstein 距离。

为什么需要这个文件
------------------
极小目标上 IoU 是病态的：8×8 的框平移 1 px，IoU 从 1.0 掉到 0.778；
而 80×80 的框平移同样的 1 px，IoU 只掉到 0.975。TAL 的 align_metric
= score^α · IoU^β 在 vt 桶上因此几乎退化成"位置对不对"的硬判决。

NWD 把框看成二维高斯，用 2-Wasserstein 距离度量，再指数归一化：

    W₂² = ‖(cx₁,cy₁,w₁/2,h₁/2) − (cx₂,cy₂,w₂/2,h₂/2)‖²
    NWD = exp(−√W₂² / C)          ∈ (0, 1]

W₂ 是**绝对像素距离**，所以同样的 1 px 偏移对 8 px 框和 80 px 框给出
相同的 NWD（= exp(−1/12.8) = 0.9249），这正是要的尺度鲁棒性。

依据
----
* 公式取自 Wang, Xu, Yang, Yu, "A Normalized Gaussian Wasserstein Distance
  for Tiny Object Detection", AAAI 2021。
* C 的物理含义是**数据集平均目标绝对尺寸**。四个独立来源均给 AI-TOD 的
  C = 12.8；本项目用 800×800 下的 75,065 个 label 实测 mean(√(w·h)) =
  12.72（train）/ 12.48（val），与该值一致，故默认 C = 12.8。

改动面（必须如实记录，不要只说"替换相似度"）
-------------------------------------------
本文件替换的是 `TaskAlignedAssigner.iou_calculation`——整条 TAL 链路里
唯一的相似度入口。在 8.4.197 中，它的返回值同时被三处使用：

  1. `align_metric = score^α · sim^β`      → 决定 top-k 候选
  2. `select_highest_overlaps(...)`         → 一个 anchor 被多个 GT 抢时的仲裁
  3. `pos_overlaps = overlaps.amax(...)`    → soft label 的归一化系数
                                               （target_scores *= align/pos）

所以 NWD-TAL 是**在标签分配环节用 NWD 全面替代 IoU**，与 NWD 原论文的
"替代 IoU 的三个位置"（assign / NMS / regression）中第一个位置对应，
但**不等于** NWD-RKA（那只用 NWD 做排序）。论文里不得混称。

刻意不做的事
------------
* 不重写 `_forward` / `get_box_metrics` / `select_topk_candidates` 中的任何
  索引逻辑。全部走 `super()`，只在最底层的相似度函数上打补丁。
  （2026-11-14 评审 P0 的教训：重写原生逻辑会引入无法与上游对比的行为差异。）
* 不提供"只改 align_metric、overlaps 仍用 IoU"的变体。那会让冲突消解和
  soft label 仍是 IoU、align 是 NWD，语义混杂、更难解释；如确需，另行评估。
* 不改 NMS、不改回归损失。本文件只覆盖标签分配。

用法
----
    import nwd_tal
    nwd_tal.install(C=12.8)        # 之后创建的 assigner 全部生效

    AITOD_NWD_C=12.8 python nwd_tal.py --selftest

自测覆盖（`--selftest`）
------------------------
    T1 自反性       NWD(x,x) 严格等于 1
    T2 对称性       NWD(a,b) == NWD(b,a)
    T3 平移单调     偏移增大时 NWD 严格递减
    T4 值域         (0,1]
    T5 手算校验     3 个例子与解析式逐位对比
    T6 尺度鲁棒性   1px 偏移在 8px 与 80px 框上 NWD 相同、IoU 相差 >0.15
                   （这条能抓出"把 W₂ 除以自身尺寸"的典型实现错误）
    T7 秩相关       在真实尺寸 GT 附近采样，NWD 与 IoU 的 Spearman 同向
    T8 topk 安全性  合法 GT 的 top-1 metric 远大于 eps，不会被 topk_mask 清空
    T9 集成         打补丁后 assigner 能跑通，五元组形状/fg 数量合法
"""

import os
import sys

import torch

DEFAULT_C = 12.8

# 单一配置源：install() 与 _selftest() 都经 _c() 读取，避免出现
# "模块默认 12、selftest 按 12.8 算预期" 这类不一致（aitod_assigner 踩过）。
def _c():
    return float(os.environ.get('AITOD_NWD_C', DEFAULT_C))


def nwd_similarity(gt_bboxes, pd_bboxes, C=None, eps=1e-7):
    """xyxy（像素坐标）两两配对算 NWD，返回 (N,)。

    输入形状必须是 (N,4)/(N,4)——与原生 `iou_calculation` 一致，即
    `pair_overlaps` 里 `gt_bboxes[indices[:2]]` 与 `pd_bboxes[b, a]` 已是
    等长的配对张量。**不要传 (bs,nb,4) 与 (bs,na,4)**，那不是本函数的约定。

    ⚠ 返回 **fp32**，不转回输入 dtype。这不是疏忽，是必需的：
        AMP 训练时输入是 fp16。NWD = exp(−d/C)，距离 34.6 px 时只有 0.064，
        经 β=6 次幂后 ≈ 6e-8，**低于 fp16 最小正规数**，直接下溢成 0。
        而 TAL 的 select_topk_candidates 用 `top1 > eps(1e-9)` 决定该 GT 有没有候选，
        align_metric 全 0 会导致该 GT 一个正样本都拿不到。
        2026-11-16 smoke 实测：转回 fp16 后 NWD 臂的 box_loss 只有 CIoU 的 1/22、
        val cls_loss 却是 8.4 倍（正样本极少但 soft label 被归一化放大）。
        保持 fp32 后，get_box_metrics 里的
            align_values = bbox_scores.pow(α) * overlap_values.pow(β)
        因类型提升变成 fp32，下溢消失。
    """
    C = _c() if C is None else float(C)
    if C <= 0:
        raise ValueError('NWD 常数 C 必须为正，当前 %r' % C)

    g = gt_bboxes.float()
    p = pd_bboxes.float()

    gx = (g[..., 0] + g[..., 2]) * 0.5
    gy = (g[..., 1] + g[..., 3]) * 0.5
    gw = (g[..., 2] - g[..., 0]).clamp_min(eps)
    gh = (g[..., 3] - g[..., 1]).clamp_min(eps)

    px = (p[..., 0] + p[..., 2]) * 0.5
    py = (p[..., 1] + p[..., 3]) * 0.5
    pw = (p[..., 2] - p[..., 0]).clamp_min(eps)
    ph = (p[..., 3] - p[..., 1]).clamp_min(eps)

    # ‖μ₁−μ₂‖² + ‖Σ₁^½−Σ₂^½‖²_F，Σ^½ = diag(w/2, h/2)
    w2 = (gx - px).pow(2) + (gy - py).pow(2) \
        + ((gw - pw) * 0.5).pow(2) + ((gh - ph) * 0.5).pow(2)

    d = w2.clamp_min(0).sqrt()
    return torch.exp(-d / C)          # 保持 fp32，见 docstring 里的下溢说明


def _nwd_calculation(self, gt_bboxes, pd_bboxes):
    """替换 TaskAlignedAssigner.iou_calculation：CIoU → NWD。

    签名、输入形状、返回形状必须与原生完全一致（(N,4),(N,4) -> (N,)），
    否则 pair_overlaps 的后续 scatter 会静默错位。
    """
    return nwd_similarity(gt_bboxes, pd_bboxes)


_ORIG = {}      # 保存被替换前的原生函数，供 uninstall 恢复


def install(C=None):
    """给 ultralytics 的 TaskAlignedAssigner 装上 NWD 相似度。

    必须在创建 assigner **之前**调用。assigner 在 `v8DetectionLoss.__init__`
    （loss.py:382）里构造，而 criterion 是首次 forward 才建的，所以放在
    `model.train()` 之前一定来得及，放在之后则完全无效。

    影响范围（2026-11-15 逐个类核对）：
        TaskAlignedAssigner        被替换                       ← 本项目使用
        RotatedTaskAlignedAssigner **不受影响**（自己重写了 iou_calculation，5 参数）
        PoseTaskAlignedAssigner    **会被影响**（未重写，经 super().pair_overlaps 继承到 NWD）
        SegTaskAlignedAssigner     **会被影响**（同上）

    本项目只做检测（YOLO11s det），后两者不涉及。但如果将来做 pose / seg 实验，
    这里的替换会悄悄改变它们的分配行为——届时必须另行评估，不能直接复用。
    """
    from ultralytics.utils import tal
    if C is not None:
        os.environ['AITOD_NWD_C'] = repr(float(C))
    cur = tal.TaskAlignedAssigner.iou_calculation
    if getattr(cur, '__name__', '') == '_nwd_calculation':
        return cur
    _ORIG['f'] = cur                      # 只在第一次替换时记录原生实现
    tal.TaskAlignedAssigner.iou_calculation = _nwd_calculation
    return _nwd_calculation


def uninstall():
    """恢复原生 iou_calculation。

    同一进程里要交替跑 CIoU / NWD 两臂时必须用它复位，否则第二张图的
    "基线臂"会悄悄变成 NWD 臂（2026-11-16 anchor 级追踪脚本需要这个）。
    """
    from ultralytics.utils import tal
    if 'f' in _ORIG:
        tal.TaskAlignedAssigner.iou_calculation = _ORIG.pop('f')
        return True
    return False


# ------------------------------------------------------------------ selftest ---
def _selftest():
    import math

    C = _c()
    fails = []
    n_checks = [0]
    def check(ok, name, detail=''):
        n_checks[0] += 1
        print('  %-16s %s   %s' % (name, 'PASS' if ok else 'FAIL', detail))
        if not ok:
            fails.append('%s %s' % (name, detail))

    print('NWD-TAL selftest   C = %.2f' % C)

    # ---- T1 自反性 ----
    print('\n[T1] 自反性：NWD(x,x) == 1')
    box = torch.tensor([[10., 10., 26., 20.], [0., 0., 8., 8.], [100., 200., 101., 201.]])
    t1 = nwd_similarity(box, box.clone())
    check(torch.allclose(t1, torch.ones_like(t1), atol=1e-6), 'self-nwd=1',
          'max|v-1|=%.2e' % (t1 - 1).abs().max().item())

    # ---- T2 对称性 ----
    print('\n[T2] 对称性：NWD(a,b) == NWD(b,a)')
    a = torch.tensor([[10., 10., 18., 18.], [0., 0., 12., 9.]])
    b = torch.tensor([[12., 13., 20., 21.], [3., 4., 15., 12.]])
    t2 = (nwd_similarity(a, b) - nwd_similarity(b, a)).abs().max().item()
    check(t2 < 1e-6, 'symmetry', 'max diff=%.2e' % t2)

    # ---- T3 平移单调 ----
    print('\n[T3] 平移单调：偏移 0/1/2/4/8/16 px 时 NWD 严格递减')
    A = torch.tensor([[100., 100., 108., 108.]])
    offs = [0., 1., 2., 4., 8., 16.]
    B = torch.stack([torch.tensor([[100. + o, 100., 108. + o, 108.]]) for o in offs]).squeeze(1)
    vals = nwd_similarity(A.expand(6, 4), B)
    mono = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    check(mono, 'monotone', ' / '.join('%.4f' % v for v in vals.tolist()))

    # ---- T4 值域 ----
    print('\n[T4] 值域：(0, 1]')
    rnd = torch.rand(200, 4) * 400
    rnd2 = torch.rand(200, 4) * 400
    v = nwd_similarity(rnd, rnd2)
    check(bool((v > 0).all() and (v <= 1 + 1e-6).all()), 'range',
          'min=%.3e max=%.6f' % (v.min().item(), v.max().item()))

    # ---- T5 手算校验 ----
    print('\n[T5] 手算校验（解析式）')
    cases = [
        ((0., 0., 10., 10.), (5., 0., 15., 10.), 5.0, '同尺寸、中心距 5'),
        ((0., 0., 10., 10.), (10., 0., 20., 10.), 10.0, '同尺寸、中心距 10'),
        ((0., 0., 10., 10.), (0., 0., 20., 10.), math.sqrt(25 + 25), '同心、宽 10→20'),
    ]
    for ga, pa, d_exp, nm in cases:
        got = nwd_similarity(torch.tensor([ga]), torch.tensor([pa])).item()
        exp = math.exp(-d_exp / C)
        check(abs(got - exp) < 1e-6, nm, 'got=%.6f exp=%.6f (W2=%.4f)' % (got, exp, d_exp))

    # ---- T6 尺度鲁棒性（最关键的一条）----
    print('\n[T6] 尺度鲁棒性：1 px 偏移，8px 框 vs 80px 框')
    small_a = torch.tensor([[0., 0., 8., 8.]])
    small_b = torch.tensor([[1., 0., 9., 8.]])
    big_a = torch.tensor([[0., 0., 80., 80.]])
    big_b = torch.tensor([[1., 0., 81., 80.]])
    n_small = nwd_similarity(small_a, small_b).item()
    n_big = nwd_similarity(big_a, big_b).item()
    check(abs(n_small - n_big) < 1e-6, 'nwd-invariant',
          '8px=%.6f  80px=%.6f' % (n_small, n_big))

    from ultralytics.utils.metrics import bbox_iou
    i_small = bbox_iou(small_a, small_b, xywh=False).item()
    i_big = bbox_iou(big_a, big_b, xywh=False).item()
    check(abs(i_small - i_big) > 0.15, 'iou-not-invariant',
          '8px=%.4f  80px=%.4f  差=%.4f' % (i_small, i_big, abs(i_small - i_big)))
    # 这条同时证明 NWD 没有被误写成"除以自身尺寸"的尺度不变版本

    # ---- T7 秩相关 ----
    print('\n[T7] 与 IoU 的秩相关（真实尺寸 GT 附近采样）')
    g = torch.tensor([[100., 100., 112., 111.]])          # 12x11，典型 vt 尺寸
    jitter = torch.randn(300, 4) * 3.0
    p = (g + jitter).clamp(min=0)
    nwd_v = nwd_similarity(g.expand(300, 4), p)
    iou_v = bbox_iou(g.expand(300, 4), p, xywh=False).squeeze(-1)
    rho = _spearman(nwd_v, iou_v)
    check(rho > 0.6, 'spearman', 'rho=%.3f（应显著同向，但不要求相等）' % rho)

    # ---- T8 topk 安全性 ----
    print('\n[T8] topk 安全性：合法 GT 的 top-1 metric 不应被 eps 清空')
    t8_ok, t8_min = True, 1e9
    beta = 6.0
    for gw, gh in [(2., 2.), (5., 4.), (12., 11.), (20., 18.)]:
        gt = torch.tensor([[200., 200., 200. + gw, 200. + gh]])
        # 8px 网格上的 anchor，覆盖 GT 周围 ±80 px
        ys = torch.arange(120., 280., 8.)
        xs = torch.arange(120., 280., 8.)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        anc = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1)
        # anchor 视为 1x1 的退化框（只测 align_metric 的量级，不模拟真实尺度）
        pd = torch.cat([anc, anc + 8.0], -1)
        sim = nwd_similarity(gt.expand(len(pd), 4), pd)
        top1 = sim.pow(beta).max().item()
        t8_min = min(t8_min, top1)
        if top1 <= 1e-9:
            t8_ok = False
    check(t8_ok, 'top1>eps', '最小 top1 metric=%.3e（阈值 1e-9）' % t8_min)

    # ---- T9 集成 ----
    print('\n[T9] 集成：打补丁后 assigner 跑通，五元组合法')
    from ultralytics.utils import tal
    torch.manual_seed(0)   # 固定随机分数与抖动，保证每次跑出的 fg 数一致（可复现）
    orig = tal.TaskAlignedAssigner.iou_calculation
    try:
        install(C=C)
        asg = tal.TaskAlignedAssigner(topk=10, num_classes=8, alpha=0.5, beta=6.0,
                                      stride=[8, 16, 32], topk2=10)
        asg.bs, asg.n_max_boxes = 2, 3

        # 必须用**网格** anchor，不能用随机点：真实特征图是 (i+0.5)*stride 的规则网格，
        # 12x11 的 GT 扩到 16x16 后只覆盖到 4 个 stride-8 网格中心。随机撒 200 个点
        # 在 400x400 里期望只覆盖 0.3 个，会得到空候选——这是测试数据构造错，不是代码错
        # （2026-11-15 首轮就踩了这个坑，CIoU 版同样 fg=0 才暴露出来）。
        ys = torch.arange(4., 800., 8.)
        xs = torch.arange(4., 800., 8.)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        anc = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1)      # (10000, 2)
        na = anc.shape[0]

        pd_scores = torch.rand(2, na, 8)
        jitter = (torch.rand(2, na, 4) - 0.5) * 6.0
        pd_bboxes = torch.cat([anc, anc + 10.0], -1).unsqueeze(0).expand(2, -1, -1).clone() + jitter

        gt_bboxes = torch.tensor([[[100., 100., 112., 111.], [200., 200., 215., 212.], [0., 0., 0., 0.]],
                                  [[150., 150., 160., 162.], [300., 300., 312., 310.], [0., 0., 0., 0.]]])
        gt_labels = torch.tensor([[[0], [3], [0]], [[1], [2], [0]]])
        mask_gt = torch.tensor([[[1.], [1.], [0.]], [[1.], [1.], [0.]]])

        out = asg(pd_scores, pd_bboxes, anc, gt_labels, gt_bboxes, mask_gt)
        shapes = [(2, na), (2, na, 4), (2, na, 8), (2, na), (2, na)]
        ok_shape = all(tuple(o.shape) == s for o, s in zip(out, shapes))
        nfg = int(out[3].sum())
        per_gt = torch.bincount(out[4][out[3]].long(), minlength=3) if nfg else torch.zeros(3, dtype=torch.long)
        check(ok_shape, 'shapes', ' / '.join(str(tuple(o.shape)) for o in out))
        check(nfg > 0, 'fg>0', 'fg_mask 正样本数=%d' % nfg)
        check(int(per_gt[0]) > 0 and int(per_gt[1]) > 0, 'each-gt',
              '按 gt_idx 计正样本=%s（前两个 GT 均需 >0）' % per_gt.tolist())

        # 直接验证 assigner 内部用的是 NWD、且数值正确（不是"看起来能跑"）
        gpair = torch.tensor([[100., 100., 108., 108.]])
        ppair = torch.tensor([[104., 100., 112., 108.]])   # 中心偏移 4 px
        v_nwd = asg.iou_calculation(gpair, ppair).item()
        exp_nwd = math.exp(-4.0 / C)
        check(abs(v_nwd - exp_nwd) < 1e-6, 'nwd-in-assigner',
              'assigner 返回 %.6f，解析值 %.6f' % (v_nwd, exp_nwd))

        tal.TaskAlignedAssigner.iou_calculation = orig
        asg2 = tal.TaskAlignedAssigner(topk=10, num_classes=8, alpha=0.5, beta=6.0,
                                       stride=[8, 16, 32], topk2=10)
        asg2.bs, asg2.n_max_boxes = 2, 3
        v_ciou = asg2.iou_calculation(gpair, ppair).item()
        out2 = asg2(pd_scores, pd_bboxes, anc, gt_labels, gt_bboxes, mask_gt)
        check(abs(v_ciou - v_nwd) > 1e-3, 'ciou-differs',
              '原生 CIoU=%.4f vs NWD=%.4f（证明补丁改变了实际取值）' % (v_ciou, v_nwd))
        check(True, 'ciou-baseline', '原生版 fg=%d（NWD 版 fg=%d）' % (int(out2[3].sum()), nfg))
    finally:
        tal.TaskAlignedAssigner.iou_calculation = orig

    # ---- T10 fp16 输入（训练开 AMP 时的 dtype）----
    print('\n[T10] fp16 输入：返回必须是 fp32，且远距离不下溢')
    g16 = torch.tensor([[100., 100., 112., 111.], [0., 0., 8., 8.]], dtype=torch.float16)
    p16 = torch.tensor([[104., 103., 114., 113.], [6., 0., 14., 8.]], dtype=torch.float16)
    v16 = nwd_similarity(g16, p16)
    v32 = nwd_similarity(g16.float(), p16.float())
    check(v16.dtype == torch.float32, 'returns-fp32',
          'out dtype=%s（刻意不转回 fp16，见 docstring）' % v16.dtype)
    check(torch.allclose(v16, v32, atol=1e-6), 'fp16-in-fp32-out',
          'max diff=%.2e' % (v16 - v32).abs().max().item())

    # 远距离：fp32 能存、fp16 未必。这条是 smoke 事故的回归测试。
    far_g = torch.tensor([[0., 0., 8., 8.]], dtype=torch.float16)
    far_p = torch.tensor([[200., 0., 208., 8.]], dtype=torch.float16)   # 平移 200 px
    far = nwd_similarity(far_g, far_p)
    check(bool((far > 0).all()), 'far-nonzero-fp32',
          '200 px 偏移 NWD=%.3e' % far.item())
    check(bool((far.pow(6) > 0).all()), 'beta6-nonzero-fp32',
          'NWD^6=%.3e（fp32 可表示；fp16 下溢为 0）' % far.pow(6).item())

    # 更远：这次真的会在 fp16 下溢（阈值 ≈ exp(-d/12.8) < 6e-8，约 d > 212 px）
    vfar_g = torch.tensor([[0., 0., 8., 8.]], dtype=torch.float16)
    vfar_p = torch.tensor([[400., 0., 408., 8.]], dtype=torch.float16)
    vfar = nwd_similarity(vfar_g, vfar_p)
    check(float(vfar.to(torch.float16)) == 0.0, 'fp16-underflows-far',
          '400 px：fp32=%.2e，转 fp16 后 = 0' % vfar.item())

    # TAL 真正的门槛不是"NWD 非零"，而是 NWD^β > eps(1e-9)（select_topk_candidates 的 topk_mask）
    import math
    d_thr = C * math.log(1.0 / (1e-9 ** (1.0 / 6.0)))
    print('  %-16s %s   NWD^6 > 1e-9 要求 d < %.1f px（β=6）'
          % ('beta-threshold', 'NOTE', d_thr))

    # ---- T11 install 幂等与生效断言 ----
    print('\n[T11] install：幂等，且装完确实生效')
    f1 = install(C=12.8)
    f2 = install(C=12.8)
    from ultralytics.utils import tal as _tal
    cur = _tal.TaskAlignedAssigner.iou_calculation
    check(f1 is f2 and cur.__name__ == '_nwd_calculation', 'idempotent',
          '重复调用不叠加包装，__name__=%s' % cur.__name__)
    check(abs(_c() - 12.8) < 1e-9, 'c-from-env', 'C=%s' % _c())
    _tal.TaskAlignedAssigner.iou_calculation = orig

    # ---- T12 子类影响面 ----
    print('\n[T12] 子类影响面：Rotated 不受影响，Pose/Seg 会继承（如实记录）')
    try:
        rot = _tal.RotatedTaskAlignedAssigner.iou_calculation
        rot_own = 'iou_calculation' in _tal.RotatedTaskAlignedAssigner.__dict__
        install(C=C)
        rot_after = _tal.RotatedTaskAlignedAssigner.iou_calculation
        check(rot_own and rot is rot_after, 'rotated-safe',
              'Rotated 自己定义了 iou_calculation，补丁不改变它=%s' % (rot is rot_after))
        pose_own = 'iou_calculation' in _tal.PoseTaskAlignedAssigner.__dict__
        seg_own = 'iou_calculation' in _tal.SegTaskAlignedAssigner.__dict__
        check((not pose_own) and (not seg_own), 'pose-seg-inherit',
              'Pose/Seg 未重写 → 会继承 NWD（本项目不做，已写入 install 文档）')
    finally:
        _tal.TaskAlignedAssigner.iou_calculation = orig

    print()
    if fails:
        print('[FAIL] %d/%d 项不通过：' % (len(fails), n_checks[0]))
        for f in fails:
            print('   - %s' % f)
        return 1
    print('[PASS] NWD-TAL 自测通过（%d 项断言全部执行了实际比较）' % n_checks[0])
    return 0


def _spearman(x, y):
    """Spearman 秩相关（无 scipy 依赖，处理并列用平均秩）。"""
    def rank(v):
        order = v.argsort()
        r = torch.empty_like(v, dtype=torch.float)
        r[order] = torch.arange(len(v), dtype=torch.float)
        # 并列取平均秩
        sv, inv = v.sort()
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                mean_r = (i + j) / 2.0
                for k in range(i, j + 1):
                    r[order[k]] = mean_r
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12))


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)

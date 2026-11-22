#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
p0_tracing.py —— P0 追踪插桩的唯一实现（共享模块）

为什么单独成模块
    评审（2026-11-14 第三轮）P0-C 指出：在插桩类里重新实现 select_topk_candidates
    已经不是「只读 hook」，而是复制了一遍原生算法。一旦原生实现或版本变化，
    插桩版就会悄悄偏离，而这正是 P0-8 要排除的风险。

    现改为：S2 的实际输出来自 super()，插桩只记录返回值（见 select_topk_candidates）。

    同时，p0_stage_trace.py 与 p0_invariance_test.py 必须使用**同一个**类，
    否则「验证通过的类」与「实际产出数据的类」可能不是同一个。此前靠注释要求
    「逐字复制」，这种约束不可靠，改为单一定义、两处 import。

术语（评审 P0-B）
    topk_mask_valid_slots  topk_mask 中允许参与统计的槽位数（由 mask_gt 扩展而来）
                            —— 不是「有效 anchor 数」，也不是「正对齐指标数」
    topk_kept              S2 实际保留（已去重、已清零）的 anchor 数
    两者不是同义指标，不得混用。

字段语义（评审 P1-4）
    n_lost    当前 GT 在 S3 冲突消解中**失去的候选 anchor 数**
              （不是「该 GT 最终没有正样本」）
    s3post    冲突消解后分配给该 GT 的 anchor 数
    s4        由最终 fg_mask 与 target_gt_idx 反算的正样本数
"""
import torch
from ultralytics.utils import tal


class TracingAssigner(tal.TaskAlignedAssigner):
    """只读插桩：所有阶段结果都来自 super()，本类只读取返回值。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.trace = {}
        self.violations = []

    # ---------------- S1 ----------------
    def select_candidates_in_gts(self, *a, **kw):
        mask = super().select_candidates_in_gts(*a, **kw)     # 原生结果
        mb = mask.bool()
        self.trace['mask_s1'] = mb
        self.trace['s1'] = mb.sum(-1)
        # 该 GT 的候选点中，同时是其他 GT 候选的 anchor 数
        self.trace['n_shared'] = ((mask.sum(-2) > 1).unsqueeze(1) & mb).sum(-1)
        return mask

    # ---------------- S2 ----------------
    def select_topk_candidates(self, metrics, topk_mask=None):
        # 评审 P0-C：不得重新实现。结果完全来自原生方法。
        out = super().select_topk_candidates(metrics, topk_mask)
        self.trace['mask_s2'] = out.bool()
        self.trace['topk_kept'] = out.bool().sum(-1)          # bool.sum 为 int64，避免 int8 溢出
        # 评审 P0-B：这是「允许参与统计的槽位数」，不是有效 anchor 数
        self.trace['topk_mask_valid_slots'] = (
            topk_mask.sum(-1) if topk_mask is not None else None)
        return out

    # ---------------- S3pre ----------------
    def get_pos_mask(self, *a, **kw):
        out = super().get_pos_mask(*a, **kw)
        mask_pos = out[0]
        self.trace['s3pre'] = mask_pos.sum(-1)
        inter = (self.trace['mask_s2'] & self.trace['mask_s1']).sum(-1)
        self.trace['s2_in_s1'] = inter
        # 仅在 mask_gt 全为 True（静态诊断的典型情形）时二者应严格相等；
        # 有 padding GT 时 mask_pos 还会乘 mask_gt，此时不做断言（由 P1-5 单独测）。
        if not torch.equal(inter, self.trace['s3pre']):
            self.violations.append('s2_in_s1 != s3pre')
        return out

    # ---------------- S3post ----------------
    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        # 评审 P1-4：不能用 overlaps.argmax() 独立推断 lost，必须以消解前后实际 mask 为准。
        # 原因：原生实现取的是**全局** IoU 最大者（overlaps.max(1)），
        # 若该 GT 原本不在这个 anchor 的 mask_pos 里，anchor 会被判给一个「原本不相关」的 GT
        # —— 表现为某 GT 凭空获得 anchor。用 argmax 推断会漏掉这一项，恒等式不成立。
        pre = mask_pos.clone().bool()
        self.trace['n_multi_anchor'] = (pre.sum(-2) > 1).sum(-1)

        out = super().select_highest_overlaps(mask_pos, overlaps, n_max_boxes, align_metric)
        post = out[2].bool()

        self.trace['n_lost'] = (pre & ~post).sum(-1)       # 消解中失去的 anchor
        self.trace['n_gained'] = (post & ~pre).sum(-1)     # 消解中凭空获得的 anchor
        self.trace['s3post'] = post.sum(-1)
        self.trace['winner_gt'] = overlaps.max(1).indices   # 仅留作 anchor 级明细
        return out

    # ---------------- S4 ----------------
    def _forward(self, *a, **kw):
        res = super()._forward(*a, **kw)
        self.trace['fg'] = res[3]
        self.trace['tgt'] = res[4]
        return res


def check_invariants(trace, s4_list):
    """阶段恒等自检。返回违反描述列表（空 = 通过）。

    恒等式（v3 修正）
        s2_in_s1 == s3pre
        s3post   == s4
        s3pre − n_lost + n_gained == s3post
            ↑ 必须带 n_gained。原生冲突消解取全局 IoU 最大者，
              可能把 anchor 判给一个原本不在 mask_pos 里的 GT，
              此时该 GT 凭空获得 anchor，只用 n_lost 的恒等式不成立。
    """
    bad = []
    if trace.get('s2_in_s1') is not None:
        if not torch.equal(trace['s2_in_s1'], trace['s3pre']):
            bad.append('s2_in_s1 != s3pre')
    if trace.get('s3post') is not None and s4_list is not None:
        if trace['s3post'][0].tolist() != s4_list:
            bad.append('s3post != s4')
    if trace.get('n_lost') is not None and trace.get('s3pre') is not None:
        lhs = trace['s3pre'] - trace['n_lost'] + trace['n_gained']
        if not torch.equal(lhs, trace['s3post']):
            bad.append('s3pre - n_lost + n_gained != s3post')
    return bad

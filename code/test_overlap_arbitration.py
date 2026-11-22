#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_overlap_arbitration.py —— 冲突消解仲裁依据的单元测试（审稿意见 1）

要证明的命题
------------
TaskAlignedAssigner.select_highest_overlaps 对**被多个 GT 同时claim**的 anchor，
按定位重叠度 overlaps 裁决归属，而不是按对齐分数 t_ij = s^alpha * u^beta。

为什么必须单测
--------------
两个量通常高度相关（t 里就含 u^beta），在真实数据上很难观察出差别；论文正文却
依据这一点来解释"候选变多但归属不跟着变"。所以构造一个两者排序相反的最小例子，
直接断言归属跟随 overlaps。

做法
----
不加载模型、不跑前向，直接调用该函数，手造三个张量：
  mask_pos     (1, 2, 3)   anchor 0 被 GT0 与 GT1 同时 claim
  overlaps     (1, 2, 3)   GT1 的 overlap 更大
  align_metric (1, 2, 3)   GT0 的 t_ij 更大
断言：结果掩码里 anchor 0 归 GT1（overlap 大者），于是"t 大者未获胜"。
随后把 overlaps 反过来，断言归属翻到 GT0——证明测试确实能区分，不是因为掩码
本来就只有一列。

运行
  python test_overlap_arbitration.py
退出码 0 = 通过；非 0 = 失败（可直接接进 CI）。
"""
import sys

import torch
from ultralytics.utils.tal import TaskAlignedAssigner

CK = '\u2713'
BAD = '\u2717'


def build(assigner, gt0_overlap, gt1_overlap, gt0_align, gt1_align):
    """anchor 0 被两个 GT 同时 claim；anchor 1 只被 GT0 claim。"""
    mask_pos = torch.zeros(1, 2, 3)
    mask_pos[0, 0, 0] = 1.0
    mask_pos[0, 1, 0] = 1.0
    mask_pos[0, 0, 1] = 1.0

    overlaps = torch.zeros(1, 2, 3)
    overlaps[0, 0, 0], overlaps[0, 1, 0] = gt0_overlap, gt1_overlap
    overlaps[0, 0, 1] = 0.5

    align = torch.zeros(1, 2, 3)
    align[0, 0, 0], align[0, 1, 0] = gt0_align, gt1_align
    align[0, 0, 1] = 0.4
    return mask_pos, overlaps, align


def owners(assigner, mask_pos, overlaps, align):
    """返回 (owner_of_anchor0, fg_mask)。owner: 0/1/-1(无主)。"""
    out = assigner.select_highest_overlaps(mask_pos.clone(), overlaps, 2, align)
    m = out[2]
    col0 = m[0, :, 0]
    owner = 0 if col0[0] > 0 else (1 if col0[1] > 0 else -1)
    return owner, out[1], m


def main():
    a = TaskAlignedAssigner(topk=10, num_classes=3, alpha=0.5, beta=6.0)
    if hasattr(a, 'topk2'):
        a.topk2 = a.topk          # 关掉二次 topk2 过滤，隔离仲裁本身
    fails = 0

    # 情形 1：overlap 偏向 GT1，t_ij 偏向 GT0 → 期望归属 GT1
    mp, ov, al = build(a, gt0_overlap=0.80, gt1_overlap=0.95,
                       gt0_align=0.90, gt1_align=0.10)
    owner, fg, m = owners(a, mp, ov, al)
    ok = (owner == 1)
    print('%s case 1  overlap favours GT1 (0.95>0.80), t favours GT0 (0.90>0.10) -> owner %d'
          % (CK if ok else BAD, owner))
    fails += 0 if ok else 1
    if not ok:
        print('   mask_pos after resolve:\n%s' % m)

    # 情形 2：把 overlap 反过来，t 不变 → 期望归属 GT0
    mp, ov, al = build(a, gt0_overlap=0.95, gt1_overlap=0.80,
                       gt0_align=0.90, gt1_align=0.10)
    owner, fg, m = owners(a, mp, ov, al)
    ok = (owner == 0)
    print('%s case 2  overlap favours GT0 (0.95>0.80), t favours GT0      -> owner %d'
          % (CK if ok else BAD, owner))
    fails += 0 if ok else 1

    # 情形 3：只有 GT0 claim，不应受影响
    mp = torch.zeros(1, 2, 3); mp[0, 0, 0] = 1.0
    ov = torch.zeros(1, 2, 3); ov[0, 0, 0] = 0.3; ov[0, 1, 0] = 0.9
    al = torch.zeros(1, 2, 3); al[0, 0, 0] = 0.2; al[0, 1, 0] = 0.99
    owner, fg, m = owners(a, mp, ov, al)
    ok = (owner == 0)
    print('%s case 3  single-claim anchor untouched                        -> owner %d'
          % (CK if ok else BAD, owner))
    fails += 0 if ok else 1

    print()
    if fails:
        print('%s %d case(s) failed' % (BAD, fails))
        return 1
    print('%s all cases passed: ownership follows overlap, not t_ij' % CK)
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env bash
# run_eval_ema60.sh —— 对已有 checkpoint 的同节点对齐性复测（不训练、不改超参）
#
# 目的：正文的诊断量取 epoch-60 的 EMA 分支，而官方 AP 取各 run 的 best.pt。
# 审稿意见 R1.3 指出这是跨节点对应。本脚本用**同一批 epoch-60 EMA 权重**重跑
# Table 10 的官方评测配置，得到可比的 AP / AP50 / AP75 / APvt / APs / APm。
#
# 这不是新实验：不训练、不选超参、不产生新模型，只是对已保存权重的复测。
#
# 评测配置与 Table 10 完全一致：imgsz 800、conf 0.001、NMS IoU 0.7、
# max_det 1500、maxDets=[1,100,1500]、aitodpycocotools 12.0.3。
set -u

RUNS=/root/runs/p1
OUT=$RUNS/eval_ema60
PY=/root/venv/torch/bin/python
CODE=/mnt/d/WorkBuddy/BING/LTNet/code/yolo
LOG=$OUT/status.tsv
mkdir -p "$OUT"

printf '[eval] start %s  8 runs\n' "$(date -Iseconds)" | tee "$LOG"

for arm in E0_default NWD_nwd; do
  for s in 0 1 2 3; do
    run=${arm}_s${s}
    ck=$RUNS/$run/weights/epoch60.pt
    if [ ! -f "$ck" ]; then
      printf '%s\tMISSING\n' "$run" | tee -a "$LOG"
      continue
    fi
    t0=$(date +%s)
    if "$PY" "$CODE/aitod_eval.py" --weights "$ck" --weight-type ema --batch 8 --chunk 64 \
         --out "$OUT/$run" > "$OUT/$run.log" 2>&1; then
      st=OK
    else
      st=FAIL
    fi
    t1=$(date +%s)
    printf '%s\t%s\t%ss\n' "$run" "$st" "$((t1-t0))" | tee -a "$LOG"
  done
done

printf '[eval] ALLDONE %s\n' "$(date -Iseconds)" | tee -a "$LOG"

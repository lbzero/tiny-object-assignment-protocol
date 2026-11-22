# Reproducibility package

Companion to *An Auditable Diagnostic Framework for Tiny-Object Label
Assignment*. This package covers the **diagnostic and analysis** toolchain, which
is what the paper's claims rest on. It does not include the AI-TODv2 imagery or
the trained weights; both are obtainable from their original sources, and the
checkpoint hashes in `manifest.csv` identify exactly which weights were used.

---

## 1. What is here

```
repro/
  README.md                this file
  requirements.txt         exact versions used
  CITATION.cff             citation metadata for this release
  CHANGELOG.md             what v1.0.0 contains, and what it does not
  ZENODO.md                release and archiving procedure
  SUBMISSION_CHECKLIST.md  blocking items before the manuscript is sent
  manifest.csv             one row per run: seed, resumption, best epoch,
                           diagnostic checkpoint hash, assigner parameters
  results/                 aggregate JSONs behind the subset and bootstrap tables
  data/README.md           obtaining AI-TODv2 and rebuilding the GT map
  tools/make_archive.sh    builds the Zenodo payload, refuses images and weights
  code/
    nwd_tal.py             the NWD substitution (monkey-patch of the assigner)
    p0_tracing.py          read-only instrumentation of the native assigner
    p0_6_dynamic_trace.py  per-ground-truth S1-S4 trace + identity validation
    soft_label_analysis.py soft-label and coupling statistics
    score_analysis_v2.py   fixed-threshold precision/recall, score bins
    anchor_transfer_trained.py  anchor-level ownership and competition
    loc_error_analysis.py  matched-box IoU and localisation error
    p0_7_static_proxy.py   static proxy versus real predictions, one forward,
                           two assigner calls, per-box confusion
    aitod_eval.py          evaluation under the AI-TOD fork of the COCO API
    make_supp_tables.py    regenerates every table in the Supplementary Material
    make_subset_b.py       stratified systematic draw of diagnostic subset B
    bootstrap_zpr.py       image-level bootstrap of zero-positive rates
    subset_robustness.py   subset A/B comparison under four fixed criteria
  config/
    aitod.yaml             dataset descriptor
    args_E0_s0.yaml        fully expanded training arguments, baseline arm
    args_NWD_s0.yaml       fully expanded training arguments, substituted arm
    subsetA_ids500.txt     image identifiers of diagnostic subset A
    subsetB_ids500.txt     image identifiers of diagnostic subset B
    subsetB_manifest.csv   per-image covariates and stratum of subset B
    subsetB_balance.json   sampling design, quotas and covariate balance
    resume_counts.csv      trainer starts and resumes per run
```

All scripts take their inputs as command-line arguments. The defaults point at a
reference directory layout (`/root/runs/p1/...`) and are meant to be overridden;
nothing needs to be edited in place to run on a different machine.

---

## 2. Environment

| Component | Version used |
|---|---|
| Python | 3.10 |
| PyTorch | 2.14.1+cu126 |
| Ultralytics | 8.4.197 |
| aitodpycocotools | 12.0.3 (tag `coco-v12.0.3+lvis-v10.5.3`, commit `01f805c`) |
| CUDA / driver | 12.6 / 560.94 |
| GPU | one GTX 1080 Ti (11 GB) |

The Ultralytics version matters: the instrumentation subclasses
`TaskAlignedAssigner` and hooks three of its methods. A different minor version may
rename or reorder them, in which case the identities will fail loudly rather than
silently — which is the intended behaviour.

---

## 3. Expected directory layout

The commands below assume:

```
$DATA/ai-tod/                 images and the official annotation json
  annotations_v2/aitod_val_v2.json
  images/val/
  yolo/aitod.yaml             (config/aitod.yaml here)
  yolo/val.txt                image list, one path per line
$RUNS/<run>/weights/          epoch60.pt, best.pt, last.pt
$RUNS/<run>/eval_official/    metrics.json, predictions.json
$RUNS/<run>/results.csv       per-epoch training log
$OUT/                         scratch output
```

`config/aitod.yaml` has a `path:` field that must point at `$DATA/ai-tod/yolo`.

---

## 4. Regenerating the paper's numbers

### 4.1 Assignment trace (Sections 4, 6; Supplementary S2--S6)

```bash
python code/p0_6_dynamic_trace.py \
    --ckpt  $RUNS/E0_default_s0/weights/epoch60.pt --weight-type ema \
    --ann   $DATA/ai-tod/annotations_v2/aitod_val_v2.json \
    --images $DATA/ai-tod/images/val \
    --map   $OUT/gtmap_val_map.csv \
    --ids-file $OUT/diag_ids500.txt \
    --sim ciou --device cpu \
    --out   $OUT/diag/E0_default_s0
```

This writes `<out>.csv` (one row per ground-truth box), `<out>_meta.json`
(checkpoint hash, assigner parameters, versions) and `<out>_frames.csv`
(per-image totals). The run aborts without writing output if any of the three
identities fails on any image; see Supplementary S3--S4.

The same command with `--sim nwd` and `--ckpt .../NWD_nwd_s*/weights/epoch60.pt`
produces the corresponding cell of the 2×2 design. The four cells are:

| Weights | Similarity | Flag |
|---|---|---|
| E0 | CIoU | `--sim ciou` |
| E0 | NWD | `--sim nwd` |
| NWD | CIoU | `--sim ciou` |
| NWD | NWD | `--sim nwd` |

### 4.2 Soft labels and coupling (Section 7; Supplementary S8)

```bash
python code/soft_label_analysis.py \
    --ckpt $RUNS/E0_default_s0/weights/last.pt \
    --data $DATA/ai-tod/yolo/aitod.yaml \
    --list $DATA/ai-tod/yolo/val.txt \
    --n-img 500 --out $OUT/soft/E0_default_s0
```

Note the two quantities that are easy to confuse: the raw alignment metric is
modified **in place** by the native forward pass, so a recorder that stores
`.detach()` alone captures the normalised value. The script clones before the
in-place operation; a run whose `align` and `norm_align` statistics are identical
indicates the bug has been reintroduced.

### 4.3 Fixed-threshold precision and recall (Supplementary S10)

```bash
python code/score_analysis_v2.py \
    --gt $DATA/ai-tod/annotations_v2/aitod_val_v2.json \
    --runs "E0:$RUNS/E0_default_s0" --out $OUT/score/E0_default_s0
```

Matching is greedy, per image and per class, with no ignore region. This is a
**diagnostic** approximation, not the official evaluator; see §5.

### 4.4 Static proxy audit (Section 4)

```bash
AITOD_RULE=default python code/p0_7_static_proxy.py \
    --ckpt  $RUNS/E0_default_s3/weights/epoch60.pt --weight-type ema \
    --ann   $DATA/ai-tod/annotations_v2/aitod_val_v2.json \
    --images $DATA/ai-tod/images/val \
    --map   $OUT/gtmap_val_map.csv \
    --device cpu --rule default \
    --out   $OUT/p07/default
```

Both arms come from a single forward pass: the assigner is invoked twice, once with
the real predictions and once with synthetic boxes of side $2\times$stride and a
constant score of 0.99. The script asserts that $S_1$ is identical in the two arms,
which is what identifies the two arms as differing only in the two substituted
inputs. `--rule cover` repeats the audit under the stride-floor-free rule.

Archive: `40-experiments/static-proxy/full_{default,cover}.csv` are the per-box
records behind Section 4, so the numbers there can be checked without re-running
the model.

### 4.6 Anchor ownership and competition (Supplementary S9)

```bash
python code/anchor_transfer_trained.py \
    --ckpt-a $RUNS/E0_default_s0/weights/epoch60.pt \
    --ckpt-b $RUNS/NWD_nwd_s0/weights/epoch60.pt \
    --ann $DATA/ai-tod/annotations_v2/aitod_val_v2.json \
    --images $DATA/ai-tod/images/val --map $OUT/gtmap_val_map.csv \
    --ids-file $OUT/diag_ids500.txt --device cpu --out $OUT/anchor_s0
```

### 4.7 Supplementary tables

```bash
python code/make_supp_tables.py \
    --diag $OUT/diag --soft $OUT/soft --score $OUT/score \
    --anchor $OUT --runs $RUNS --out $OUT/supp
```

Every table in the Supplementary Material is generated by this command from the
artefacts above; no number is transcribed by hand.

### 4.8 Diagnostic subsets A and B (Supplementary S11)

Subset A is the first 500 validation images in list order, i.e. not a random
sample; its identifier list is `config/subsetA_ids500.txt`. Subset B is drawn from
the remaining 2,304 images by `code/make_subset_b.py`:

```bash
python code/make_subset_b.py \
    --map  $OUT/gtmap_val_map.csv \
    --ann  $DATA/ai-tod/annotations_v2/aitod_val_v2.json \
    --exclude config/subsetA_ids500.txt \
    --n 500 --seed 20261121 \
    --out-ids      $OUT/subsetB_ids500.txt \
    --out-manifest $OUT/subsetB_manifest.csv \
    --out-report   $OUT/subsetB_balance.json
```

The design is one stratum per dominant size bucket, with systematic sampling
inside each stratum after sorting by (visible ground-truth count, dominant
category, image id). Systematic sampling is used because a first version that
sampled uniformly inside bucket $\times$ density-quartile cells drew a subset with
30.6 ground-truth boxes per image against 26.8 in the full validation set; the
sorted variant lands at 28.2. The shipped manifest and balance report
(`config/subsetB_manifest.csv`, `config/subsetB_balance.json`) are the ones used in
the paper, so the draw can be checked without re-running it.

### 4.9 Image-level bootstrap intervals (Supplementary S7)

```bash
python code/bootstrap_zpr.py \
    --diag-dir $OUT/diag --suffix A --seeds 0 1 2 3 \
    --bucket verytiny --B 2000 --ids-file $OUT/diag_ids500.txt \
    --out $OUT/boot_A.json
```

The resampling unit is the **image**, not the ground-truth box: boxes inside one
image share scene, illumination and scale composition and are not independent
observations. The four cells and the four seeds share one resampled index set per
replicate, which preserves the pairing that the contrasts rely on.

### 4.10 Subset robustness (Supplementary S11)

```bash
python code/subset_robustness.py \
    --dirA $OUT/diag --dirB $OUT/diagB \
    --out-json $OUT/subset_robustness.json --out-tex $OUT/subset_robustness.tex
```

Four criteria are fixed in the script before any result is inspected: same sign of
the mean effect (R1), same sign in all four seeds (R2), absolute subset difference
not exceeding the larger within-subset seed range (R3), and overlap of the
per-seed ranges (R4). The table reports all four rather than a single verdict, so
that a reader can see which effect is stable and which is not.

---

## 5. Evaluation

```bash
python code/aitod_eval.py --weights $RUNS/E0_default_s0/weights/best.pt ...
```

The evaluator is **not** the standard COCO API. It wraps `aitodpycocotools`, a fork
maintained for the AI-TOD benchmark that adds the four-bin area subdivision and
raises the detection limit to 1500. `AP_vt` cannot be produced with unmodified
`pycocotools`. Reported configuration: `maxDets = [1, 100, 1500]`, score threshold
0.001, NMS IoU 0.7.

Consequently two different precision/recall conventions appear in this work and
must not be conflated:

* **Official AP** — the fork above, on all 2,804 validation images, 75,091 boxes.
* **Diagnostic precision/recall** — `score_analysis_v2.py`, greedy matching, no
  ignore region, used only to describe the composition of high-confidence
  detections. Its numbers are not AP and are never presented as such.

---

## 6. Metric definitions

* **Zero-positive rate** — fraction of ground-truth boxes in a bin with
  `S4 = 0`, i.e. no positive anchor after conflict resolution. Reported in
  percent, over the diagnostic subset, per bin.
* **Interface effect** — change in a quantity when the similarity function is
  replaced while the weights are held fixed.
* **Weight-condition effect** — change when the weights differ while the
  similarity function is held fixed. Descriptive: the two weight sets differ in
  more than localisation quality.
* **Positive IoU** — intersection over union between the predicted box of an
  assigned candidate and its ground-truth box, recomputed independently of the
  similarity used inside the assigner, so that the two arms are comparable.

---

## 7. Training runs, resumes and checkpoint semantics

`config/resume_counts.csv` lists, per run, the number of trainer starts and the
number of resumes, counted from the `Resuming training` records in each training
log. The two arms are not balanced: 14 resumes across the four baseline runs, two
across the four NWD runs.

What a resume restores, verified against `ultralytics 8.4.197` and the checkpoint
keys of these runs:

| State | Saved in checkpoint | Restored on resume |
|---|---|---|
| EMA weights (half precision) | yes | weights loaded into a fresh EMA, and the **student is re-initialised from it** |
| Optimiser state | yes (half) | yes |
| AMP gradient-scaler state | yes | yes |
| Epoch counter, best fitness, early-stopping state | yes | yes |
| EMA update counter | yes | yes |
| Training arguments | yes | yes |
| Learning-rate scheduler state | no | no — reconstructed from the epoch index |
| Python / NumPy / torch RNG states | no | no |
| Data-loader order and augmentation streams | no | no |

Two consequences for anyone reusing these runs: the batch and augmentation
sequence after a resume is not a continuation of the interrupted one, and the
weights a resumed run loads carry half-precision rounding relative to the live
EMA. Runs that share a seed index therefore share a seed, not a realised
trajectory.

---

## 7. What is not reproducible from this package

* **Training.** The eight runs used `train_p1.py` plus an early-stopping
  configuration not included here; `config/args_*.yaml` records the fully expanded
  arguments as executed, which is sufficient to re-run with a stock Ultralytics
  trainer but does not reproduce bit-identical weights.
* **The 26-box annotation difference.** The diagnostic count (75,065) derives from
  the letterbox-and-clip step of the loader, and reproducing it requires the same
  image ordering and crop behaviour. The paper never mixes the two totals.
* **Best-checkpoint epoch.** The released checkpoints are stripped, so their epoch
  field is `-1`. `manifest.csv` reconstructs the value as the epoch maximising
  validation mAP50-95 in `results.csv`; this reproduces the recorded weights in all
  eight runs but is a reconstruction.

---

## 8. Checks worth running first

1. Re-run §4.1 on one run and confirm the three identity residuals are zero.
2. Confirm `align` and `norm_align` differ in the output of §4.2; if they are
   identical, the in-place modification was captured instead of the input.
3. Confirm the very-tiny zero-positive rate for the four cells lands near
   4.15 / 2.76 / 5.97 / 4.16 %, in the order of the table in §4.1. A materially
   different value usually means a different image subset was used rather than a
   different result.

---

## 8. Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # torch cu126 wheels; see requirements.txt

export DATA=$PWD/../data                 # AI-TODv2 layout, see data/README.md
export RUNS=/path/to/runs                # one directory per run, see §3
export OUT=$PWD/out

# 1. trace one cell (fails closed if an identity breaks)
python code/p0_6_dynamic_trace.py \
    --ckpt $RUNS/E0_default_s0/weights/epoch60.pt --weight-type ema \
    --ann $DATA/ai-tod/annotations_v2/aitod_val_v2.json \
    --images $DATA/ai-tod/images/val --map $OUT/gtmap_val_map.csv \
    --ids-file config/subsetA_ids500.txt --sim ciou --device cpu \
    --out $OUT/diag/E0_default_s0

# 2. regenerate every supplementary table (byte-identical to the submitted ones)
python code/make_supp_tables.py --diag $OUT/diag --soft $OUT/soft \
    --score $OUT/score --anchor $OUT --runs $RUNS --out $OUT/supp
```

The four cells, the two diagnostic subsets and the eight seeds are enumerated in
`config/` and in the table of §4.1 of this file. Nothing needs editing in place:
every script takes its inputs as arguments.

## 9. Release and archive

`ZENODO.md` gives the order of operations. In short: freeze the content, build the
payload with `tools/make_archive.sh`, push the repository, tag `v1.0.0`, create the
GitHub release, archive that release on Zenodo, and only then insert the URL and
DOI into the manuscript. The tool refuses to package `.png`, `.jpg` or `.pt` files.

## 10. Known placeholders in this version

Two values are intentionally not filled, because inventing them would be worse than
leaving them visibly empty:

| Where | Placeholder | Filled when |
|---|---|---|
| `CITATION.cff`, `.zenodo.json` | `10.5281/zenodo.PLACEHOLDER`, `REPOSITORY_URL` | the Zenodo record is created |
| manuscript Code Availability | `[repository URL]`, `[Zenodo DOI]` | same |

`SUBMISSION_CHECKLIST.md` §1 lists these as blocking.

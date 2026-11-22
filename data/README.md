# Data preparation

The experiments use **AI-TOD-v2**. No new imagery was collected and none is
redistributed with this package; the benchmark remains subject to the terms of its
original release.

## 1. Obtain the benchmark

Download AI-TOD and the v2 annotation from the benchmark's own distribution channel
(https://github.com/jwwangchn/AI-TOD) and place it as:

```
$DATA/ai-tod/
  annotations_v2/aitod_val_v2.json
  annotations_v2/aitod_train_v2.json
  images/train/ , images/val/
  labels/train/ , labels/val/     (YOLO format, produced by the conversion below)
  yolo/aitod.yaml                 (copy of config/aitod.yaml with `path:` edited)
  yolo/val.txt                    (one image path per line)
```

Check the counts before running anything: the validation split must contain **2,804
images and 75,091 annotated boxes**. Every number in the paper depends on this
matching; if the counts differ, stop.

## 2. Convert annotations to YOLO format

```bash
# decodes the annotations, writes YOLO labels, the data.yaml, and runs a sanity check
bash code/prepare_aitod.sh
```

It writes `labels/{train,val}/*.txt` beside `images/` and a `data.yaml` whose
`path:` must point at your copy. If you prefer to convert by hand, one `.txt` per
image with lines `class cx cy w h` normalised to the image, and
`class = category_id - 1`.

The conversion used here is the one in `code/prepare_aitod.sh`, kept for
reproducibility; it drops no boxes and does not merge classes, and it is the script
that produced the 26-box difference explained in Section III-H of the paper.

## 3. Training-visible annotation map

The assignment diagnostics read a mapping from `(image_id, gt_idx)` to the box as
the trainer sees it, i.e. after clipping at the image border. That map has 75,065
rows, 26 fewer than the official annotation:

```bash
python code/build_gt_mapping.py \
    --ann $DATA/ai-tod/annotations_v2/aitod_val_v2.json \
    --labels $DATA/ai-tod/labels/val \
    --out $OUT/gtmap_val_map.csv
```

The two conventions are not interchangeable. Performance metrics use the official
75,091 boxes; assignment statistics use the 75,065 training-visible boxes. No
number in the paper compares one against the other.

## 4. Which artefacts are in git and which are in the archive

| Artefact | Size | Where |
|---|---|---|
| Code, config, image ID lists, sampling manifest | < 1 MB | this repository |
| Aggregate result JSONs (`results/`) | < 1 MB | this repository |
| Per-ground-truth traces, 32 files | ~26 MB | Zenodo archive (`tools/make_archive.sh`) |
| Trained weights (`epoch60.pt`, `best.pt`) | 57 MB each | not redistributed; hashes in `manifest.csv` |

Weights are identified by SHA-256 in `manifest.csv`. They are not redistributed
because the total is about 0.5 GB and the hash plus the training arguments are
sufficient to re-create them; the authors can supply them on request where the
benchmark licence allows.

# Direct-instance Mask R-CNN v2 lane

Pretrained `torchvision.models.detection.maskrcnn_resnet50_fpn_v2` with a
36-output detector head (background 0 + canonical classes 0..34 as labels
1..35).  All parameters are trainable; every BatchNorm module is held in eval
mode so running statistics never move while affine weight/bias still train.

This directory owns the model seam only.  It never reads a ground-truth
annotation at inference time and never emits a dense score-winner raster as the
official document.

## Files

| file | role |
| --- | --- |
| `model.py` | `DetectorConfig`, pretrained build / head replacement, partial-label `training_losses`, transform invariant |
| `partial.py` | restricted anchor/ROI labels, licensed negative boxes, mask-domain BCE |
| `emit.py` | per-independent-mask `vectorize_masks(..., emitter="runs")` merge + roundtrip diagnostics |
| `data.py` | expanded-manifest FIT source, canonical first-16 capacity selection, instance-centred crops |
| `train.py` | `capacity` / `train` / `predict` bounded CLI |
| `infer.py` | native PNG -> boxes/classes/scores/masks -> official 12-field document |
| `contract.py` | read-only import seam for `src/cadvec` and sibling `targets/` |
| `tests/test_synthetic.py` | CPU synthetic contract tests |

## Exact configuration

```text
weights                  none for synthetic / local COCO state dict for real fit
labels                   1..35 = canonical 0..34; 0 = background
train transform          min_size=384, max_size=384 (native crop, no hidden resize)
inference transform      min_size=1000, max_size=1000 (actual PNG H/W preserved)
mask ROI pool output     14
mask head output         28  (MaskRCNNPredictor ConvTranspose2d 14 -> 28)
                         planned 28 -> 56 is NOT configured or claimed
BN                       running stats frozen, affine trainable
negative supervision     only boxes whose every touched pixel is blank_negative
mask BCE domain          loss_domain cells only; own positive wins at crossings
```

`training_losses` calls `assert_training_transform` before every forward.  A
model previously switched to inference size therefore cannot silently train at
1000 px.

## Commands

Run from `outputs/recognition-direct-instance-20261114-01` so `model` is a
package.  Output directories must not already contain the target checkpoint,
log or report.

Capacity (first-16 original FIT, memorisation scope only):

```bash
python -m model.train capacity \
  --output-dir runs/direct-instance-capacity-20261114-01 \
  --crops 16 --cache-size 2
```

`--cache-size 16` is opt-in and only after a measured host-RAM preflight; the
default 2 is the measured-safe fallback (15.63 GiB total, 3.21 GiB free at
review time).

Real fit (one micro-batch per update, accumulate 2):

```bash
python -m model.train train \
  --weights-file <COCO_STATE_DICT> \
  --output-dir runs/direct-instance-fit-20261114-01 \
  --crops 16 --updates 600 --accumulate 2 --jitter 24 \
  --cache-size 2 --save-every 0 --seed 20261108 --device cuda
```

`--save-every 0` is the default and writes only the final `checkpoint.pt`.
`--crops 0` uses the full admitted manifest lazily instead of the first 16.

Predict (PNG only; no GT read):

```bash
python -m model.infer \
  --checkpoint runs/direct-instance-fit-20261114-01/checkpoint.pt \
  --png <INPUT.png> --output <OUTPUT.json> --device cuda
```

Emission is one merged 12-field document built from independent masks.  Thing
instances get globally unique ids; Stuff merges per class with instance `-1`.
Ink intersection is opt-in only (`--ink-intersection`) and reports removed
pixels.  Skeleton emission is never automatic.

## Partial-label rules

* Detector classes come from `target["keys"][:,0] + 1`; dense
  `target["labels"]` is diagnostic and ignored.
* A below-threshold anchor/ROI becomes negative only when its entire touched
  pixel rectangle is inside `blank_negative`.  Everything else unmatched stays
  `-1` and is removed by the sampler before `fastrcnn_loss`.
* `blank_negative` is `loss_domain & ~union(masks)`; if the target exposes its
  own map it is accepted only when it does not overlap any positive.
* Mask BCE uses soft ROI-projected occupancy by default.  Hard `>0.5` targets
  erase one-pixel strokes on loose proposals.
* The GT-target and licensed-domain projections share one convention.
  `mask_target_projection="legacy"` (default) with `mask_target_aligned=True`
  and `mask_target_sampling_ratio=2` reproduces the original lane exactly,
  including checkpoints written before the field existed.
  `--mask-target-projection paste_consistent` instead derives both ROIs from
  the installed paste box equation (float32 `expand_boxes`, int64 truncation,
  one-pixel pad) and forces effective `aligned=False, sampling_ratio=-1`;
  requested legacy arguments are recorded separately so no mixed convention is
  possible.  It is a serialized trial option, not the promoted default.
  Inference transforms and the mask ROI pool are unchanged.
* Empty GT is valid: RPN still trains, ROI/mask losses are zero (not NaN), and
  the zero regression row count matches the proposal count.

## Measured synthetic results

`python -m unittest model.tests.test_synthetic -v` from
`outputs/recognition-direct-instance-20261114-01`: **31 tests, OK**,
CPU base environment `torch 2.14.1+cu130 / torchvision 0.29.2+cpu`.

Covered: 36-class topology and frozen BN, class mapping from keys, two-image
two-positive backward with unchanged BN buffers, empty GT backward, fractional
/ outside negative licensing, unknown-cell zero gradient, two-image mask batch
columns, non-square official letterbox coordinates, square overlapping-instance
roundtrip IoU 1.0, shifted equal-count IoU < 1, Stuff merge, targets build/crop/
prepare, strict checkpoint load + inference + no-GT diagnostics, canonical
first-16 capacity list, output-directory refusal, transform drift rejection,
paste-consistent materialized ROIs against the independent formula, mixed
valid/narrow proposal batches with exact per-row zero gradients, and
all-invalid paste batches yielding a graph-connected zero loss with excluded
rows and zero denominator.

Soft-target probe, 1 px x 240 px line at x=192 in 384, all-domain:

```text
tight 191.5..192.5   soft_positive_mass 391.997  hard 392
loose 172..212       soft_positive_mass  20.000  hard  28
loose 152..232       soft_positive_mass  11.000  hard   0
wide 191..193        soft_positive_mass 343.000  hard 392
```

CPU synthetic inference (untrained weights, score threshold 0.0, mask
threshold 0.5, 20 detections) exported continuous probabilities: max 0.999998,
1.04 % strictly between 0 and 1, 98.96 % exactly 0, 641 pixels above 0.5.  This
is a plumbing probe only, not a quality result.

No real dataset, GPU run or success metric has been measured in this lane.
Fixed-384 crop memorisation and full native-image submission recall are always
reported separately.

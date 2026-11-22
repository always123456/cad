# Direct-instance evaluation lane

Model-independent scoring for the direct-instance lane. Input is a directory of
prediction JSONs plus the sample list; output is a pre-GT seal and, later, one
report. The lane never runs a model and never reads ground truth before every
prediction is sealed.

```
<project-root>/evaluation/
├── instance_eval/
│   ├── contract.py    paths, guards, hashing, input resolution, document/map/baseline validation
│   ├── provider.py    PNG/npz prediction masks -> official 12-field document (+ diagnostics)
│   ├── metrics.py     canonical score_documents + recall/precision/F1 + per-class/Thing/Stuff/image
│   ├── evaluator.py   prepare() (seal) and score() (GT once, on grant)
│   └── batch.py        optional PNG-only producer: one checkpoint load, one document per PNG
├── baselines/floorplancad_pancadnet_pq.json   metadata-only reference, never a gate
├── run_batch_predict.py                       CLI: PNGs -> prediction envelopes
├── run_instance_eval.py                       CLI: prepare | prepare-masks | score
└── tests/test_instance_eval.py                stdlib unittest, 53 synthetic tests
```

Requires Python 3.14 and `numpy`; `Pillow` only for mask PNGs; `torch` only when
`batch.py` is actually used to run the detector. No pytest needed.

## The three stages

1. **produce** (`batch.py`, optional) — PNG in, prediction envelope out. No SVG,
   no GT, no seal.
2. **prepare** — validate every prediction, hash it, optionally re-derive it from
   its masks, then write `pre_gt_seal.json`. Still no GT.
3. **score** — verify the seal byte-for-byte, read GT once behind an explicit
   grant, score with the canonical document scorer, write `evaluation_report.json`.

```
python run_batch_predict.py --inputs parent_capacity16_inputs.json \
    --checkpoint <ckpt.pt> --device cuda --predictions-dir runs/fit16/predictions

python run_instance_eval.py prepare --predictions runs/fit16/predictions \
    --out runs/fit16/seal --sample-set explicit --dataset fit16-diagnostic

python run_instance_eval.py score --out runs/fit16/seal --allow-gt
```

`score` exits 0 only when the success block passes; a guard failure is
`{"status": "error", "code": ..., "detail": ...}` on stderr with exit 2.

## Python API

```python
from instance_eval import evaluator as E, batch as B, contract as C

# producer (optional)
B.run_batch(predictions_dir, checkpoint=ckpt, device="cuda",
            inputs="parent_capacity16_inputs.json", sample_set="explicit")

# seal: predictions only, no GT
seal = E.prepare("runs/fit16/predictions", "runs/fit16/seal",
                 inputs="parent_capacity16_inputs.json",
                 dataset="fit16-diagnostic")

# or seal documents the model already holds in memory
E.prepare_from_documents({sample_id: document, ...}, "runs/tune/seal",
                         dataset="tune-fixed4", origin="model_json")

# or seal winner-take-all diagnostic masks (never the source of truth for overlaps)
E.prepare_from_masks("runs/tune/masks", "runs/tune/seal", mask_role="diagnostic")

# score, after the parent grant, GT read exactly once
report = E.score("runs/tune/seal", allow_gt=True,
                 reference_baseline="baselines/floorplancad_pancadnet_pq.json")
```

Signatures:

```python
prepare(predictions_dir, out_dir=None, *, inputs=None, maps_dir=None, masks_dir=None,
        mask_format="png", mask_role="authoritative", dataset="tune-fixed4",
        origin="prediction_json", manifest=MANIFEST_PATH) -> dict
prepare_from_documents(documents, out_dir, *, inputs=None, dataset="tune-fixed4",
                       manifest=MANIFEST_PATH, origin="documents") -> dict
prepare_from_masks(mask_dir, out_dir, *, format="png", inputs=None,
                   dataset="tune-fixed4", manifest=MANIFEST_PATH) -> dict
score(out_dir, *, allow_gt=False, gt_builder=None, reference_baseline=None,
      report_path=None, trace=None) -> dict

run_batch(predictions_dir, *, checkpoint=None, device="cpu", inputs=None,
          sample_set="fixed4", manifest=MANIFEST_PATH, max_detections=None,
          score_threshold=None, mask_threshold=None, dense_diagnostic=False,
          loader=None, predict=None, emit=None, png_loader=None, trace=None) -> dict

metrics.instance_report(gt_documents, pred_documents, *, sample_ids=None) -> dict
```

`gt_builder` is the only GT access: default
`cadvec.recognition_raster_data.build_sample(png_path, svg_path, sample_id)`
(imported, not reimplemented); tests inject a fake builder.

## Inputs

- Prediction file per sample: `<slug>.json`, slug = `sample_id.replace(":", "_")`.
  Content is either a bare 12-field document or an envelope
  `{"schema": "cadvec.direct_instance.prediction.v1", "sample_id": ..., "document": ...}`
  with optional `diagnostics`/`source` (batch writes this shape).
- Sample list: `{"schema": "cadvec.direct_instance.inputs.v1", "samples": [{"sample_id", "group_id", "png_path"}]}`,
  or a bare list, or none for the pinned fixed four. Explicit lists keep the
  model's order; duplicates are refused, and no ordering is assumed.
- Pinned sets: `fixed4` = `train_final:I8wfZ`, `UtkbT`, `9unYH`, `O3ItP`
  (role `tune`, 36 GT instances); `fit16` = first 16 role-`fit` samples,
  disjoint from fixed4 — a capacity/memorisation diagnostic, not a held-out set.

Prediction documents are validated strictly with
`cadvec.recognition_submission.validate_document`: 12 exact fields, int commands
0..3, semantics 0..35, Thing instances `>= 0` (negative is a **GT** affordance and
is rejected for predictions), Stuff/35 instance `-1`, rgb 0..255, layerIds >= 1,
widths >= 0, 8 finite args per command, `obj_cts`/`boxes` one entry per Thing
instance in first-appearance order. GT validation is separate and looser (Thing
`-1` legal).

## Masks (optional)

Prediction masks are never required. A model that emits JSON is scored from JSON.
When masks exist, `--masks`/`masks_dir` (PNG pairs `<slug>.semantic.png` +
`<slug>.instance.png`, or `.npz` with `semantic`/`instance`/`source_size`) are
hashed and the document is re-derived with the canonical
`vectorize_masks(..., emitter="runs")`:

```
scale, letterbox_x, letterbox_y = viewbox_transform(width, height, (0,0,100,100))
x_view = x_px / scale - letterbox_x / scale        # letterboxed, aspect preserved
instance -1 is rebuilt wherever semantic is not a Thing class (0..29)
```

`mask_role="authoritative"` (default) refuses any JSON↔mask drift with
`document_mismatch`. `mask_role="diagnostic"` treats the dense map as a
winner-take-all rendering: drift is recorded (`mask_drift`, `attachment_notes`,
caveats) and the JSON stays the scored object. Use `diagnostic` for independent
overlapping instance models; a per-pixel map cannot hold two instances on one
pixel.

## Metrics (canonical, not reimplemented)

Scoring is `cadvec.recognition_raster_data.score_documents(gt, pred,
raster_size=1000, canvas_size=(100.0, 100.0))`: 100x100 canvas, 1000 px raster,
1 px stroke, per-class Hungarian one-to-one, a match only when IoU **strictly**
> 0.5, Thing instance `-1` skipped, Stuff (30..34) merged per class, class 35
ignored.

| quantity | definition |
| --- | --- |
| recall (instance matching) | `TP / (TP + FN)` — the parent's >90% objective |
| precision | `TP / (TP + FP)` |
| F1 | `2*TP / (2*TP + FP + FN)` (equals the harmonic mean when both rates exist) |
| PQ | `SumIoU / (TP + 0.5*FP + 0.5*FN)` |
| SQ | `SumIoU / TP` |
| RQ | `TP / (TP + 0.5*FP + 0.5*FN)` |

Any zero denominator is `None`, never `0`, never `NaN`. The report carries the
totals, `thing` and `stuff` blocks, `per_class` (with `group`), `per_image`
(`gt_keys`, `gt_skipped`, `pred_keys`), `denominators`, and a `consistency` block
that re-derives `TP+FN == GT keys`, `TP+FP == pred keys`, the class sums, the PQ
formula and Thing+Stuff totals from the emitted documents; `score` refuses to
write a report when that block is not clean.

`success.passed` requires **all** of: `recall > 0.90`, no GT leak
(`gt_read` after the seal only, prediction hashes unchanged), and `PQ >=`
reference when a *comparable* reference is supplied. Without a comparable
reference the status is `recall_only_reference_unavailable` and `passed` stays
false. The fixed four are already-exposed TUNE samples: the report says
`exposed_samples: true`, `heldout: false`, and "diagnostic, not a pristine
held-out result". A FIT16 report says it measures capacity/memorisation, not
generalisation.

## Reference baseline: FloorPlanCAD / PanCADNet

Verified original PanCADNet numbers (metadata, **not** a gate):

- PQ **0.561**, SQ **0.838**, RQ **0.660**
- 30-class vector-entity **length-weighted IoU**, macro-averaged across classes
- source: arXiv:2105.07147 v2, Tables 5 and 6
- a later reported PQ 59.5 is an unverified protocol difference

This lane's PQ is a proxy pixel-IoU PQ (1000 px raster, 1 px stroke, same-class
strict Hungarian > 0.5). It is **not directly comparable** to the PanCADNet
length-weighted vector-entity PQ, and no cross-protocol comparison or gate is
allowed. `baselines/floorplancad_pancadnet_pq.json` is metadata-only
(`usable_as_gate: false`); `C.load_reference_baseline` returns it with status
`incomparable_normalization`, so passing it to `score` documents the reference
and cannot change `passed`. No 0.60 threshold exists anywhere in this lane; only
an externally verified, protocol-compatible number may become a gate.

## Guards

- Writes only inside this lane (`require_lane_owned`), atomically, refusing to
  overwrite; a second `prepare` or an existing report is refused.
- Any path containing `test_final`, `dev_final`, `eval_pool`, `reserve`,
  `.runtime` is refused (`sealed_path`), including via mask and checkpoint paths.
- `score` re-verifies the seal and every sealed artifact's size + sha256, and
  re-derives mask documents, **before** the first GT read; `allow_gt=True` is the
  explicit grant, `allow_gt` absent gives `gt_not_authorized`.
- `batch` writes predictions only: `seal_written: false`, `gt_read: false`,
  `svg_read: false`, `checkpoint_loads: 1`.
- Representative codes: `prediction_missing`, `prediction_invalid`,
  `document_mismatch`, `source_size_mismatch`, `artifact_changed`,
  `predictions_exist`, `prediction_exists`, `gt_not_authorized`, `sealed_path`,
  `inputs_duplicate`, `batch_png_required`, `checkpoint_missing`,
  `metric_inconsistency`, `mask_not_index`, `mask_shape_mismatch`.

## Batch producer

`instance_eval.batch.run_batch` mirrors the official inference path without
copying it: one `model.infer.load_checkpoint` for the batch (official config,
thresholds and `detections_per_img` untouched), then per PNG
`load_png` → `model.infer.predict_arrays` → masks at `mask_threshold` →
`model.emit.emit_instances(..., emitter="runs")`. Independent masks are
vectorised one by one, so overlaps survive and Stuff merges per class; the
canonical viewBox affine is applied inside the emitter. No connected components,
no ink intersection, no tuning, no SVG, no GT. All documents are built before the
first file is written and every target is checked first, so a failed batch leaves
no partial set.

Dependency injection (`loader`, `predict`, `emit`, `png_loader`) keeps the tests
CPU-only and torch-free: the batch tests use the real emitter with fake
detections, and assert one checkpoint load, PNG-only reads, all predictions
written before any seal, overlapping instances preserved, Stuff merged,
letterboxed pixels landing inside the viewBox, and refusal of existing outputs,
SVGs and sealed paths.

Mask content that falls outside the letterboxed viewBox window transforms to
coordinates beyond the 0..100 canvas. The emitter does not clamp it, so the batch
counts such primitives as `out_of_viewbox_primitives` in the envelope `source`
and in the run summary, next to `diagnostics` from the emitter. A nonzero count
marks pixels the viewBox never covered; it is reported, never silently dropped.

## Tests

```
python tests/test_instance_eval.py        # 53 tests, stdlib unittest (torch-free)
python tests/test_batch_integration.py    # real model adapters; needs torch + torchvision
```

The second file is the actual-API integration proof: it builds a small
`weights="none"` Mask R-CNN from the sibling `model` package, saves a checkpoint
with the lane's real payload schema (`cadvec.direct-instance-maskrcnn/1`) in a
temporary directory, writes two generated PNG fixtures (96×96 square and 96×128
non-square), and drives `batch.run_batch` through its **default** adapters —
`default_loader` (exactly one `load_checkpoint`), `default_png_loader`,
`default_predict`, `default_emit` — with explicit fixture inputs instead of a
manifest. It then seals the finished predictions with `evaluator.prepare` and
checks the native geometry, the 12-field document schema, `emitter="runs"`,
`no_ground_truth_read`, sealed artifacts, and an overwrite refusal.

Its thresholds are fixture-only (`score_threshold=0.999`, 96 px inference size,
`weights="none"`); they are declared here so nobody mistakes them for deployed
defaults. Nothing in this file scores against GT or reads a dataset.

Run the model tests under the approved CPU-hidden CUDA runtime (Python 3.12,
torchvision overlay), so the sibling Mask R-CNN really builds:

```bash
export PYTHONPATH="<torchvision-overlay>"   # pip install --target 的同版本 cu130 wheel 目录
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=-1
"<python-with-torch>" tests/test_batch_integration.py
```

Under the default interpreter (no torch) the three model tests skip with an
explicit reason, and the module still checks the torch-free pieces — the real
shared emitter's affine, overlap and Stuff-merge behaviour, plus an AST contract
check that `model.infer`/`model.emit` still keep the keywords `batch.default_*`
calls. If that sibling API drifts, the contract test fails before the model owner
wastes a GPU run.

A random `weights="none"` detector can fire inside the letterbox padding, where
the inverse affine legitimately yields coordinates outside 0..100 (e.g. x =
-12.5). The integration test therefore never assumes predictions sit on canvas:
it asserts the exact pixel→viewBox affine and that `out_of_viewbox_primitives`
matches an independent count, so the diagnostic stays honest instead of being
clamped away.

Coverage of `test_instance_eval.py`: IoU ≈ 0.49 vs ≈ 0.51, exact IoU == 0.5
counted as a strict miss, duplicate prediction → FP, missed GT → FN, wrong class
→ FP+FN, Stuff merged, empty denominators → None, F1 from counts (TP0 cases
included), strict prediction validation vs permissive GT validation, pre-GT seal
ordering and stale/tampered refusal, mask re-derivation and affine correctness,
`mask_role` diagnostic drift, FIT16 selection/explicit order/duplicates,
hand-computed report fractions checked against `fractions.Fraction`, baseline
statuses, and the batch producer suite above. No test opens a real PNG/SVG/GT, a
checkpoint, a GPU or the network.

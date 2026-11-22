# Canonical direct-instance target contract

Native-resolution targets for the renewed PNG → class/instance route: one mask
per canonical GT key, `labels = class + 1`, exclusive pixel boxes, dense ink and
an explicit loss domain. References the project rasterizer by source path, so
the GT rasters are bit-identical to the canonical scorer's raster.

No dataset, PNG or GT file is read here. The tests build synthetic documents and
images in memory.

## Files

| Path | Role |
| --- | --- |
| `targets/targets.py` | contract implementation (`build_targets`, `crop_targets`) |
| `targets/__init__.py` | package re-exports |
| `targets/test_targets.py` | 16 synthetic unittest cases |
| `targets/README.md` | this contract note |

## API

```python
build_targets(image, gt_document, *, raster_size=None, canvas_size=(100.0, 100.0)) -> dict
crop_targets(image, target, origin=(x, y), size=384) -> dict
```

`image` is CHW or HWC RGB, float in `[0, 1]` or integer in `0..255`
(CPU/GPU tensors exposing `detach().cpu().numpy()` are accepted). It supplies
**only** the native spatial shape and the ink support; geometry never comes from
pixels. `gt_document` needs `commands`, `args`, `lengths`, `semanticIds`,
`instanceIds`, `width`, `height` (the other official fields are ignored).
Everything malformed raises `TargetError(ValueError)` — fail closed.

### `build_targets` result

| Key | Type / shape | Meaning |
| --- | --- | --- |
| `masks` | `bool (N, H, W)` | one mask per canonical key, sorted by `(class, instance)` |
| `keys` | `tuple[tuple[int, int], ...]` | canonical `(class, instance)` keys, index-aligned with `masks`/`boxes` |
| `labels` | `int64 (H, W)` | canonical class `+ 1`; `0` = no target |
| `boxes` | `float32 (N, 4)` | `xyxy`, exclusive extents of each mask in native pixels |
| `ink` | `bool (H, W)` | dense PNG ink support |
| `loss_domain` | `bool (H, W)` | `union(masks) | blank_negative`; every pixel is either a target positive, a licensed negative, or excluded |
| `unknown_geometry` | `bool (H, W)` | raster of skipped GT primitives (`semantic 35`, Thing `instance < 0`); always UNKNOWN |
| `blank_negative` | `bool (H, W)` | `(~ink) & (~unknown_geometry) & (~union)`; disjoint from every target mask and from skipped geometry |
| `native_size` | `{"height": H, "width": W}` | array shape order |
| `raster_size` | `{"width": W, "height": H}` | rasterizer argument order |
| `canvas_size` | `{"width": 100.0, "height": 100.0}` | logical canvas |
| `image_layout` | `"chw"` / `"hwc"` | detected input layout |
| `label_offset`, `mask_policy`, `ink_policy`, `domain_policy` | str/int | policy provenance |
| `diagnostics` | dict of ints | counts listed below |

Empty documents return `masks (0, H, W)`, `boxes (0, 4)`, `keys ()`,
`labels == 0`, `blank_negative == ~ink` (no union, no unknown geometry) and
`loss_domain == ~ink`.

### Canonical keys

* Thing `semantic 0..29` with `instance >= 0` → `(class, instance)`.
* Stuff `semantic 30..34` → `(class, -1)`; primitives of one Stuff class merge
  into one mask regardless of the document instance.
* `semantic == 35` → skipped, never a target, never a negative.
* Thing `instance < 0` (semantic-only GT) → skipped; its ink is unknown.
* Duplicate keys merge with logical OR. Keys are sorted, so output order never
  depends on document order.
* Masks may overlap. `labels` resolves an overlapping pixel to the **lowest
  canonical key** (sorted first wins); `diagnostics["overlap_pixels"]` and
  `diagnostics["conflict_pixels"]` (cross-class overlap) report it.

### Raster parity

Every mask is produced by
`src/cadvec/recognition_raster_data.py::_draw_mask` with the same
`canvas_size=(100, 100)`, command semantics (line `0/1` via
`ImageDraw.line(width=1)`; circle `2` / ellipse `3` from the four-sample bounding
box via `ImageDraw.ellipse(outline=1, width=1)`) and key logic
(`_segment_key`). The module is imported from its explicit file path under a
private package alias (`_cadvec_direct_instance_targets`), so an installed
`cadvec` cannot shadow it and the project's relative imports still work.
`CADVEC_SRC_DIR` overrides the `src` directory.

`masks` are compared bit-for-bit against
`rasterize_document(document, raster_size=(W, H), canvas_size=(100, 100))` in the
tests.

### Ink, labels and loss domain

`ink` follows the project PNG policy with no SVG colour, width or layer input:

```
ink = min(R, G, B) < 254 / 255        # after compositing over white
```

* `loss_domain = union(all target masks) | blank_negative`, where
  `blank_negative = (~ink) & (~unknown_geometry) & (~union)`.
* Negatives are disjoint from every target mask:
  `blank_negative & union == 0`. A known target raster can include pixels the
  PNG never inked (e.g. a diagonal/chord); those pixels remain positives via
  `union` and can never be negative. This is enforced by
  `model/model.py prepare_detector_targets`, which rejects
  `negative & union`; here the mask is excluded before the domain is formed.
* Ink pixels explained by no canonical key are **excluded**: `semantic 35`,
  Thing without an instance (`unknown_ink_from_skipped_geometry_pixels`), and
  residual ink the GT geometry never drew (`unknown_ink_residual_pixels`).
* Skipped GT geometry is UNKNOWN even where it falls outside the PNG ink: the
  four-point sample args of an arc/circle can chord across blank pixels, and
  `unknown_geometry` removes exactly those pixels from the negative license.
  This is why the domain is not `union | ~ink`.
* Known target pixels keep precedence: `union` is applied first and subtracted
  from the blank mask, so a positive that also lies on skipped geometry stays a
  positive and is never also a negative.
* Ink-free background that no skipped geometry crosses is a negative; known
  target pixels are positives. Presence in `unknown_geometry` alone never
  creates a negative.
* `masks` are **not** clipped to `ink` (`mask_policy`): ink only defines the loss
  domain, so the geometry target stays image-independent and exactly matches the
  canonical raster. If a trainer wants ink-clipped positives, intersect at use
  time (`masks & ink`); the diagnostics on `known_ink_pixels` /
  `target_pixels` quantify the difference.

### Native resolution

The contract never rescales. `raster_size=None` means "the image spatial
shape"; passing a `raster_size` is an assertion that must equal `(W, H)`, else
`TargetError`. This keeps `masks`, `ink` and `loss_domain` on one pixel grid.
Canvas scaling still follows the canonical formula (`sx = W / canvas_w`,
`sy = H / canvas_h`), so a non-square raster maps anisotropically exactly like
the scorer.

### `crop_targets`

Plain window utility — not a sampling policy.

* Window is `[x, x + size) × [y, y + size)` in native pixels; `origin` may be
  negative or beyond the image.
* No rescaling. `masks`, `labels`, `ink`, `loss_domain`, `unknown_geometry` and
  `blank_negative` are sliced and placed at the corresponding offset; `image` is
  sliced and padded with white `1.0`.
* Placed outside the native image: no ink, no mask, no label, no loss domain,
  no unknown geometry, no blank negative (those pixels do not exist), `image` =
  `1.0`.
* `boxes` are translated by `-origin` then clipped to `[0, size]`; a target with
  no visible pixel gets `[0, 0, 0, 0]` and `visibility=False`.
* Returns `image`, `masks`, `labels`, `boxes`, `keys`, `ink`, `loss_domain`,
  `unknown_geometry`, `blank_negative`, `visibility (N,) bool`, `visible_count`,
  `target_count`, `origin`, `size`, `window (x0, y0, x1, y1)` in native
  coordinates, `native_size`, `raster_size`, `image_layout`. `image=None` is
  allowed and skips the image crop.
* Zero-target crops (empty document, only `semantic 35` or Thing
  `instance < 0` primitives) return `masks (0, size, size)`,
  `boxes (0, 4)`, `keys ()`, `visibility (0,)`; they do not raise.
* Cropping preserves the invariants: cropped `blank_negative` is still disjoint
  from cropped `masks` and `unknown_geometry`, and
  `loss_domain == union(masks) | blank_negative` on the window.

## Diagnostics

`primitive_count`, `target_count`, `empty_target_count`,
`ignored_semantic35_primitives`,
`skipped_thing_missing_instance_primitives`, `skipped_primitive_count`,
`ink_pixels`, `ink_free_pixels`, `target_pixels`, `known_ink_pixels`,
`unknown_ink_pixels`, `unknown_ink_from_skipped_geometry_pixels`,
`unknown_ink_residual_pixels`, `loss_domain_pixels`, `blank_pixels`,
`unknown_geometry_pixels`, `unknown_geometry_ink_pixels`,
`unknown_geometry_blank_pixels`, `overlap_pixels`, `conflict_pixels`,
`ink_support_threshold_8bit`.

## Integration notes

* Build the image with the project loader so the ink policy matches:
  `recognition_raster_data._load_image(png)` (float CHW in `[0, 1]`, composited
  over white).
* Build `gt_document` with `document_from_parsed` / `document_from_svg` (or the
  evaluator's GT document); it already carries canonical `semanticIds 0..35`
  and `instanceIds >= -1`. Thing `-1` is legal GT and is skipped here, matching
  the evaluation contract's `validate_gt_document`.
* Consumption pattern: `target["masks"]` for PQ-style instance heads,
  `target["labels"]` for the semantic head, `target["boxes"]` for detection and
  `target["loss_domain"]` as the only negative/positive sampling mask.
* Keep `keys` with each sample; predictions must reuse the same canonical key
  space (`class 0..34`, instance `-1` for Stuff) to score under strict
  same-class IoU > 0.5.
* No metric, recall or baseline claim is made here. Evaluation lives in
  `../evaluation/`.

## Tests

```bash
cd outputs/recognition-direct-instance-20261114-01/targets
python -m unittest test_targets -v
```

Validated with the project Python (`Python 3.14.10`, numpy 2.5.4, Pillow 13.0.0):
**20 tests, 20 passed** (`Ran 20 tests ... OK`), also green via
`python -m unittest targets.test_targets` from the lane root.

Covered: exact line and circle/ellipse raster parity; Thing `-1` skipped and
unknown; `semantic 35` ignored; Stuff merged per class; unknown ink excluded
from the domain; skipped-geometry chord pixels never negative; known off-ink
diagonal positive with no unknown crossing and `blank_negative & union == 0`;
known-positive precedence over unknown geometry; canonical `+1` labels and
disjoint masks; exclusive boxes; empty document; zero-target build → crop
(including padding); non-square native shape and HWC/CHW equivalence; masks not
clipped to ink; crop translation, clipping, padding and visibility; duplicate
and order determinism; overlap precedence; malformed inputs fail closed.

# cadvec — stage 1 data foundation

Small, offline, standard-library-only package for the CAD floor-plan data.
No network access, no model, no platform-specific label conversion.

## Commands

From the repository root (no install needed if `PYTHONPATH=src` is set):

```bash
export PYTHONPATH=src
python -m cadvec inventory --datasets datasets --out data/processed
python -m cadvec parse-train --datasets datasets --split train_final --out data/processed --sample 5
python -m cadvec parse-svg tests/fixtures/train_like.svg --out data/processed/sample_one.json
```

Editable install (`python -m pip install -e .`) also works when setuptools is
available; in a fully offline environment use `PYTHONPATH=src` as above.

`test_final` is refused by `parse-train`/`parse-svg` (exit code 3) unless
`--allow-test-annotations` is passed, and that flag is only for future use
after the organiser authorises annotation access. The path's own split is
authoritative: declaring `--split train_final` for a file under `test_final`
is a conflict and fails closed, with or without the opt-in flag.

Exit codes: `0` success, `1` parse completed but a file failed or produced
`error` diagnostics, `2` usage/IO/serialisation error (including `OSError`
such as `PermissionError`/`IsADirectoryError` and any refused write), `3`
annotation-policy violation. `parse-train` reports file failures and files
with error diagnostics separately.

Write targets are checked before any mkdir/open/write: a `--out`,
`--sample-out`, `inventory.json` or `files.csv` that resolves inside the
dataset tree (through `.`, `..` or a symlink), onto the input file itself, or
to the same final path as another output is refused.

`parse-train --limit N --sample M` applies `--limit` once and uses that same
selected set for the summary and the samples; samples are never re-read from
the directory and `M` is capped by the selection. A selected file that fails
to parse is counted but not sampled. `--sample 0` or `--limit 0` writes no
sample file.

`inventory` returns 2 when the dataset root is missing/not a directory or
when none of the requested split directories exist. A partially missing set
of optional splits still succeeds and prints a warning; the report keeps
`present=false` entries.

## Raw facts vs. derived data vs. platform adaptation

**Raw facts (read from files, preserved verbatim)**

* file names, sizes, SHA-256 digests, PNG/SVG pairing per split
* `viewBox` string and its four parsed numbers; root `width` / `height` strings
* `<g>` layer document order (1-based `index`), nesting `path`, `id`,
  `inkscape:label`, `inkscape:groupmode`, and *all* raw attributes
* geometry primitive document order (`local_index`) and *all* raw attributes,
  including original `semantic-id` / `instance-id` strings
* raw `d` path data and its command list, raw `cx/cy/r/rx/ry` values, and the
  complete outer-to-inner `transform` chain (`root` → `group` → `primitive`,
  recorded, **not** applied)
* per primitive: `coordinate_frame: local-pre-transform`,
  `transform_applied: false`, `transform_resolved: false`
* non-finite numbers (`nan`, `inf`, `1e999`) are rejected rather than stored;
  every number in a written JSON document is strictly serialisable

**Derived (computed here, traceable, never a substitute for raw values)**

* four arc-length sample points per primitive, polyline length, bounding box;
  samples follow drawn segments only, so a moveto jump between subpaths is
  never interpolated as a connection (also `subpath_lengths` and
  `drawable_subpath_count`)
* circle circumference and Ramanujan-II ellipse perimeter approximation
* diagnostics (missing / duplicate / variant alias / non-integer / out-of-domain)
* optional fallback values and offset mapping, **only** when a config file
  explicitly supplies them; such values live under `label.derived` and carry a
  provenance string

**Platform adaptation (deliberately not implemented here)**

* raw `semantic-id` → platform semantic index (`0–34`)
* placeholder semantics (`35` for missing semantic, `-1` for missing instance)
* Thing/Stuff classification and instance-ID policy
* `commands` / `args` / `lengths` / `rgb` / `layerIds` / `widths` JSON assembly
* PNG → vector inference of any kind, and local PQ on `dev_final`

Every one of those requires an explicit mapping from the official schema or a
later confirmed platform contract. `config.example.json` shows the shape of an
explicit config; its maps are empty on purpose.

## Data policy enforced by code

* `._*` resource forks and `.DS_Store` are never counted as samples
  (`policy.is_ignored_name`, used by both inventory and parser listing).
* `train_final`: SVG annotations may be parsed for training and in-split
  validation (`policy.ensure_annotation_access`).
* `test_final`: file-level inventory only; parsing annotations raises
  `AnnotationAccessError` before any file is opened.
* `dev_final`: PNG-only, no local ground truth and **no local PQ**.
* Outputs are written only to `data/processed` (or an explicit `--out`);
  nothing writes into `datasets/`.

## Output files

* `data/processed/inventory.json` — per-split counts, pairing, digest summary,
  policy and exclusion rules (no per-file record array)
* `data/processed/files.csv` — one row per non-ignored file: split, name, role,
  extension, stem, size, digest
* `data/processed/train_parse_summary.json` — aggregated parse of a train split:
  primitive counts, raw semantic/instance value histograms, layer counts,
  diagnostics by code, problem files
* `data/processed/train_parse_samples.jsonl` — a few full documents with raw
  attributes, layer order, labels, geometry commands and samples

## Module map

| module | responsibility |
| --- | --- |
| `policy.py` | resource-fork exclusion, split policies, annotation-access guard |
| `inventory.py` | split scan, pairing, digests, CSV/JSON writers |
| `geometry.py` | path tokeniser, absolute polylines, arc/curve sampling, lengths |
| `attributes.py` | alias config, label reading, boundary diagnostics, opt-in mapping |
| `svg_parse.py` | document walk, layer/primitive records, split aggregation |
| `diagnostics.py` | structured diagnostic records and counters |
| `cli.py` | `inventory`, `parse-train`, `parse-svg` commands |

## Known limitations

* Curves and arcs are resampled (default 8 segments) for length and the four
  sample points; this is an approximation of exact SVG path length.
* `transform` is recorded as a chain but not applied; primitives keep local
  coordinates and a warning diagnostic names the frames. No SVG transform
  engine (matrix composition, nested viewports) is implemented in this stage,
  so these coordinates must not be treated as rendered absolute geometry.
  A transform that only affects `<text>` raises no geometry diagnostic.
* Text, dimensions and other non-geometry elements are counted but not
  converted into primitives.
* Ellipse length uses Ramanujan II; the platform reference format also treats
  it as an approximation.
* Extreme but finite coordinates whose sums, differences or arc parameters
  overflow float64 are rejected with an `error` diagnostic; raw `d` is kept
  and derived length/bbox/samples are withheld. Arbitrary-magnitude
  coordinates are out of scope.
* No platform JSON is produced, so no PQ/SQ/RQ is computed locally.

"""Stroke thickness of the emitted segments, measured on the canonical raster.

Per emitted class/instance segment the mask is rasterised exactly as the scorer
does (raster 1000x1000, viewBox 100) and its thickness is taken as
``2 * area / perimeter``, which is the width of an elongated stroke (a 1 px wide
line of length L gives area L and perimeter 2L, i.e. thickness 1.0; a 2 px wide
line gives 2.0).  This answers the question a lower binarisation threshold raises:
did the strokes get fatter, or did hidden strokes appear?

  python tools/emit_stats.py <results_dir> [--out stats.json] [--per-image]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# delivery/tools/x.py -> repository root
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LANE = REPO + "/outputs/recognition-direct-instance-20261114-01"
sys.path.insert(0, os.path.join(LANE, "evaluation"))
sys.path.insert(0, os.path.join(REPO, "src"))

from instance_eval import contract as C  # noqa: E402
import cadvec.recognition_raster_data as RD  # noqa: E402

RASTER = (1000, 1000, (100.0, 100.0))


def thickness(mask):
    """2 * area / perimeter of a boolean mask, with the frame counting as outside."""

    area = int(mask.sum())
    if area == 0:
        return 0.0, 0
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    core = padded[1:-1, 1:-1]
    inner = (
        padded[0:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, 0:-2] & padded[1:-1, 2:]
    )
    perimeter = int(np.count_nonzero(core & ~inner))
    if perimeter == 0:
        return 0.0, area
    return 2.0 * area / perimeter, area


def stats_for(document):
    segments = RD._segment_masks(document, *RASTER)
    rows = []
    for seg in segments:
        value, area = thickness(seg["mask"])
        if area == 0:
            continue
        rows.append(
            {"class_id": int(seg["class_id"]), "instance_id": int(seg["instance_id"]),
             "thickness": round(value, 6), "area": area}
        )
    return rows


def summarise(values):
    if not values:
        return {"segments": 0, "median_thickness": None, "mean_thickness": None,
                "p10_thickness": None, "p90_thickness": None, "median_area": None}
    array = np.asarray(values, dtype=float)
    return {
        "segments": int(array.size),
        "median_thickness": round(float(np.median(array)), 6),
        "mean_thickness": round(float(array.mean()), 6),
        "p10_thickness": round(float(np.percentile(array, 10)), 6),
        "p90_thickness": round(float(np.percentile(array, 90)), 6),
    }


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--out", default=None)
    parser.add_argument("--inputs", default=None)
    parser.add_argument("--per-image", action="store_true")
    args = parser.parse_args(argv)

    names = sorted(
        name for name in os.listdir(args.results_dir) if name.endswith(".json")
    )
    if args.inputs:
        with open(args.inputs, encoding="utf-8") as handle:
            keep = {
                os.path.splitext(os.path.basename(row["png_path"]))[0]
                for row in json.load(handle)["samples"]
            }
        names = [name for name in names if os.path.splitext(name)[0] in keep]

    thickness_values, area_values, per_image = [], [], []
    empty = 0
    for name in names:
        with open(os.path.join(args.results_dir, name), encoding="utf-8") as handle:
            envelope = json.load(handle)
        document = envelope
        if "document" in envelope:
            document, _ = C.unwrap_prediction(envelope, envelope.get("sample_id", ""))
        rows = stats_for(document)
        if not rows:
            empty += 1
        thickness_values.extend(row["thickness"] for row in rows)
        area_values.extend(row["area"] for row in rows)
        if args.per_image:
            per_image.append({"document": name, "segments": len(rows),
                              "median_thickness": summarise([r["thickness"] for r in rows])["median_thickness"]})

    summary = {
        "results_dir": args.results_dir,
        "documents": len(names),
        "documents_emitting_nothing": empty,
        **summarise(thickness_values),
        "median_area": round(float(np.median(area_values)), 3) if area_values else None,
        "mean_area": round(float(np.mean(area_values)), 3) if area_values else None,
    }
    if args.per_image:
        summary["per_image"] = per_image
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_image"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

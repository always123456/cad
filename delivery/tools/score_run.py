"""Score a delivery results dir with the canonical held-out scorer.

Writes the summary (without per_image) plus per-image rows into this lane, never into
the read-only reference trees.

  python tools/score_run.py <results_dir> [--out <json>] [--per-image]
"""

from __future__ import annotations

import json
import os
import sys
import time

# Directory holding the canonical scorer; defaults to the delivery package root.
SCORER_ROOT = os.environ.get("CAD_SCORER_ROOT_DIR", "") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    results = argv[1]
    out_path = None
    keep_per_image = False
    rest = argv[2:]
    while rest:
        key = rest.pop(0)
        if key == "--out":
            out_path = rest.pop(0)
        elif key == "--per-image":
            keep_per_image = True
        else:
            raise SystemExit("unknown argument %r" % key)

    sys.path.insert(0, SCORER_ROOT)
    import score_ship  # noqa: E402  (path set above)

    started = time.time()
    result = score_ship.score(results, None)
    elapsed = time.time() - started
    per_image = result.pop("per_image", None)
    summary = dict(result, results_dir=results, wall_s=round(elapsed, 1))
    print(json.dumps(summary, indent=1))
    if out_path:
        payload = dict(summary)
        if keep_per_image and per_image is not None:
            payload["per_image"] = per_image
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

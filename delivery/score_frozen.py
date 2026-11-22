"""Score the frozen run with the unmodified canonical scorer.

Thin wrapper: imports score() from score_ship.py (byte-identical copy of the parent
lane's scorer, which is the one every number in docs/FROZEN.md was produced with) and
points it at <FREEZE>/run/results. No metric arithmetic lives here.

  <PYTHON> package/score_frozen.py [results_dir] [out.json]

Default results_dir is <PACKAGE_DIR>/../run/results; default out is stdout only.
Requires the competition repository (instance_eval, cadvec) the lane's scorer imports.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from score_ship import score  # noqa: E402


def main(argv):
    frozen = os.path.dirname(HERE)
    results = argv[0] if len(argv) > 0 else os.path.join(frozen, "run", "results")
    out = argv[1] if len(argv) > 1 else None
    res = score(results, None)
    brief = {k: v for k, v in res.items() if k != "per_image"}
    print(json.dumps(brief, indent=2))
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

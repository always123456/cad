"""Bootstrap for scripts that need the v2 emit path: pre-import model.emit, then run the
script given as argv[1] (see cad-pool-expand01/fl_boot.py for why).
"""
import os
import runpy
import sys

from instance_eval import contract as C

LANE = os.path.dirname(str(C.LANE_ROOT))
if LANE not in sys.path:
    sys.path.insert(0, LANE)
import model.emit  # noqa: E402,F401  (pins sys.modules["model"] to the package)

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

if __name__ == "__main__":
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")

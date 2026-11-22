"""Bootstrap for fl_pool.py.

`cad-inkrefine-sweep01/fastemit.py` (imported by cad-verifier01/extract.py) puts
``<LANE>/model`` on sys.path, which makes the top-level module ``model/model.py``
shadow the ``model`` *package*.  The delivery's lazy ``batch._emit_module()`` then
fails with "attempted relative import with no known parent package".  Importing
``model.emit`` once as a package, before fastemit is touched, pins the right
module in sys.modules; nothing else changes.
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
sys.path.insert(0, HERE)

if __name__ == "__main__":
    sys.argv = [os.path.join(HERE, "fl_pool.py")] + sys.argv[1:]
    runpy.run_path(os.path.join(HERE, "fl_pool.py"), run_name="__main__")

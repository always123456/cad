"""Explicit, collision-free imports of the sibling target contract and project API.

Nothing here copies target logic: the real sibling module at
``outputs/recognition-direct-instance-20261114-01/targets/targets.py`` is loaded
under a private module name, and the canonical rasterizer/vectorizer/submission
modules are reached through that contract's private package alias.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

_TARGETS_MODULE_NAME = "_direct_instance_targets_contract_model_lane"


def targets_path() -> Path:
    override = os.environ.get("CADVEC_DIRECT_TARGETS_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "targets" / "targets.py").resolve()


def load_targets(path: Optional[Path] = None) -> ModuleType:
    """Load the sibling targets module by explicit path, cached in sys.modules."""

    module = sys.modules.get(_TARGETS_MODULE_NAME)
    if module is not None:
        return module
    resolved = Path(path) if path is not None else targets_path()
    if not resolved.is_file():
        raise FileNotFoundError(
            "targets contract not found at %r (set CADVEC_DIRECT_TARGETS_PATH)" % str(resolved)
        )
    spec = importlib.util.spec_from_file_location(_TARGETS_MODULE_NAME, str(resolved))
    if spec is None or spec.loader is None:
        raise ImportError("cannot build import spec for %r" % str(resolved))
    module = importlib.util.module_from_spec(spec)
    sys.modules[_TARGETS_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_TARGETS_MODULE_NAME, None)
        raise
    return module


def project_modules() -> Dict[str, Any]:
    """Return ``raster_data``, ``vectorize`` and ``submission`` project modules."""

    targets = load_targets()
    raster = targets.project_raster()
    package = targets.PROJECT_PACKAGE
    vectorize = importlib.import_module(package + ".png_raster_vectorize")
    submission = importlib.import_module(package + ".recognition_submission")
    for name, module in (
        ("recognition_raster_data", raster),
        ("png_raster_vectorize", vectorize),
        ("recognition_submission", submission),
    ):
        if module is None:
            raise ImportError("project module %r unavailable" % name)
    return {
        "targets": targets,
        "raster_data": raster,
        "vectorize": vectorize,
        "submission": submission,
        "package": package,
    }

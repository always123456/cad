"""Offline CAD floor-plan data utilities (algorithm stage 1).

The package deliberately keeps three concerns apart:

* :mod:`cadvec.inventory`  - file-level inventory (names, sizes, digests, pairing)
* :mod:`cadvec.svg_parse`  - reading SVG structure into traceable Python data
* :mod:`cadvec.attributes` - label-attribute aliases and validation rules

Nothing in this package converts a raw SVG ``semantic-id`` into a platform
prediction label.  Any such mapping must be supplied explicitly through a
configuration file (see ``attributes.LabelConfig``).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

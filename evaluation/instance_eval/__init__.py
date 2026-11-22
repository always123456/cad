"""Model-independent instance evaluation for prediction documents or masks.

``prepare``  validates and hashes the supplied predictions -- a directory of
             12-field documents, or prediction masks turned into those documents
             by :mod:`instance_eval.provider` -- and, when the masks are supplied
             too, re-derives each document from them.  Then it writes a pre-GT
             seal.  It never reads GT, GT PNG pixels or an SVG.
``score``    verifies that seal byte-for-byte, then reads GT once (parent grant
             required), scores with the existing canonical document scorer and
             reports PQ/recall/precision/F1.

The lane is deliberately independent of any model: its input is a directory of
12-field result documents (or prediction masks) for a named sample set -- the
fixed four exposed TUNE samples by default, a future capacity-16 FIT diagnostic
set or a future TUNE set through the same API.

:mod:`instance_eval.batch` is the optional PNG-only producer in front of that
API: one checkpoint load, ``predict_arrays`` per PNG, and the shared
``emit_instances`` document (independent masks, ``emitter="runs"``).  It writes
prediction envelopes only; sealing and scoring stay with ``prepare``/``score``.
"""

from . import batch, contract, evaluator, metrics, provider

__all__ = ["batch", "contract", "evaluator", "metrics", "provider"]

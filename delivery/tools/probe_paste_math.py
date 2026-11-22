"""Micro-benchmark: which space keeps a thin sub-0.5-margin peak under bilinear paste.

A 112x112 ROI mask with a 1-cell or 2-cell stroke at probability ``peak`` over a
uniform background is pasted onto a 1000x1000 canvas with the *real* torchvision
bilinear path, once in probability space and once in logit space (with the -12
padding).  Reports the maximum after pasting and whether any pixel still passes
the 0.5 decision.

  python tools/probe_paste_math.py
"""

from __future__ import annotations

import json
import math
import os
import sys

import torch
from torchvision.models.detection import roi_heads

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mask_paste  # noqa: E402


def probe(peak, background, cells, box):
    torso = math.log(peak / (1 - peak))
    base = math.log(background / (1 - background))
    logits = torch.full((1, 1, 112, 112), base)
    logits[0, 0, 50:50 + cells, 10:102] = torso
    boxes = torch.tensor([box], dtype=torch.float32)
    shape = (1000, 1000)

    probabilities = roi_heads.paste_masks_in_image(logits.sigmoid(), boxes, shape)
    with mask_paste.logit_paste(-12.0):
        pasted = roi_heads.paste_masks_in_image(logits, boxes, shape)
    return {
        "peak": peak,
        "background": background,
        "cells": cells,
        "prob_max": round(float(probabilities.max()), 6),
        "prob_pixels_over_0.5": int((probabilities >= 0.5).sum()),
        "logit_max_probability": round(float(pasted.sigmoid().max()), 6),
        "logit_pixels_over_0": int((pasted >= 0.0).sum()),
    }


def main():
    rows = []
    for peak in (0.55, 0.65, 0.95):
        for cells in (1, 2):
            rows.append(probe(peak, 0.02, cells, [100.0, 100.0, 900.0, 900.0]))
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

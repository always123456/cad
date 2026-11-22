"""Logit-space mask paste: keep the 0.5 decision boundary, measure the stroke.

The frozen decode chain pastes *probabilities*.  ``torchvision.models.detection``
takes the mask logits at ROI resolution, puts them through ``sigmoid``
(``roi_heads.maskrcnn_inference``), zero-pads one mask cell
(``roi_heads.expand_masks``), bilinearly resizes that probability field onto the
native canvas (``roi_heads.paste_mask_in_image``) and the lane then binarises at
0.5.  Bilinear averaging in probability space is what destroys this dataset's
1px strokes: a candidate whose ROI peak is 0.550220 pastes to 0.454369 and is
gone at the 0.5 test.

``--mask-paste logits`` defers the sigmoid to *after* the paste.  Sigmoid is
strictly monotone, so ``p >= 0.5`` is exactly ``logit >= 0``: the decision
boundary is unchanged while the interpolation averages logits instead of
probabilities.  Whether that keeps more of a thin stroke is an empirical
question, not an assumption -- averaging logits weights a far-below-boundary
neighbourhood much more heavily than averaging probabilities does (the sigmoid
is steep near 0), so the variant is measured against the probability path on a
fixed subset rather than presumed better.

The mechanism is a scoped monkeypatch of three module-level functions of the
*installed* torchvision (never edited on disk, never imported at module load):

  ``roi_heads.maskrcnn_inference``  class-select the logits, no ``sigmoid``
  ``roi_heads.expand_masks``        pad with a very negative logit, not 0
                                    (logit 0 is p = 0.5, i.e. a half-on border)
  ``roi_heads.paste_mask_in_image`` canvas initialised with that same logit
                                    (torchvision fills it with 0.0, which in
                                    logits space is p = 0.5 everywhere outside
                                    the box -- every pixel would read as
                                    foreground)

All three are resolved as module globals at call time -- ``RoIHeads.forward``
calls ``maskrcnn_inference(...)`` directly, and ``paste_masks_in_image`` calls
``expand_masks(...)`` and ``paste_mask_in_image(...)`` -- so the patch reaches
them without touching the model, the loader, ``fine_head``'s chunked mask
runner or the lane's inference module.  The rest is torchvision's own code: the
predicted-class channel selection, ``expand_boxes(boxes, scale).to(torch.int64)``
(the box integerisation rule), the canvas clipping and therefore the instance
count.

``--mask-paste prob`` is the default and installs nothing: the run calls the
lane's ``predict_arrays`` untouched and its documents stay byte-identical.
"""

from __future__ import annotations

MASK_PASTE_PROB = "prob"
MASK_PASTE_LOGITS = "logits"
MASK_PASTE_MODES = (MASK_PASTE_PROB, MASK_PASTE_LOGITS)
DEFAULT_MASK_PASTE = MASK_PASTE_PROB

#: Padding value for ``expand_masks`` in logits space.  The probability path
#: pads with 0.0 (= certain background); the nearest finite logit is -inf, so
#: use a value whose sigmoid (6.14e-6 at -12) is far below any 0.5 test while
#: staying finite for bilinear interpolation.
DEFAULT_LOGIT_PAD = -12.0

_ROI_HEADS_PATH = "torchvision.models.detection.roi_heads"


# --------------------------------------------------------------------------- #
# torchvision interception
# --------------------------------------------------------------------------- #
def _roi_heads_module():
    """The installed ``torchvision.models.detection.roi_heads`` module."""

    import importlib

    return importlib.import_module(_ROI_HEADS_PATH)


def _maskrcnn_inference_logits(x, labels):
    """``roi_heads.maskrcnn_inference`` without the sigmoid.

    Same predicted-class channel selection and the same per-image split as
    torchvision's function; the values stay logits.
    """

    import torch

    num_masks = x.shape[0]
    boxes_per_image = [label.shape[0] for label in labels]
    labels = torch.cat(labels)
    index = torch.arange(num_masks, device=labels.device)
    mask_logits = x[index, labels][:, None]
    return mask_logits.split(boxes_per_image, dim=0)


def _expand_masks_padded(pad):
    """``roi_heads.expand_masks`` that pads with ``pad`` instead of 0.

    The box scale stays torchvision's ``(M + 2 * padding) / M``, so the box
    expansion (and its int64 cast in ``paste_masks_in_image``) is unchanged.
    """

    import torch
    from torch.nn import functional as F

    value = float(pad)

    def expand_masks(mask, padding):
        size = mask.shape[-1]
        scale = float(size + 2 * padding) / size
        padded_mask = F.pad(mask, (padding,) * 4, value=value)
        return padded_mask, scale

    expand_masks.__name__ = "expand_masks"
    expand_masks._cad_logit_pad = value
    return expand_masks


def _paste_mask_in_image_filled(fill):
    """``roi_heads.paste_mask_in_image`` with a non-zero canvas.

    torchvision allocates the canvas with ``torch.zeros`` and writes only the
    box interior, which is right for probabilities (0.0 = certain background)
    and wrong for logits (0.0 = p 0.5 = foreground).  Everything else -- the
    ``box`` width/height rule, the ``align_corners=False`` bilinear resize, the
    clip against the canvas -- is copied verbatim from torchvision so the
    geometry stays bit-for-bit the same.
    """

    import torch
    from torch.nn import functional as F

    value = float(fill)

    def paste_mask_in_image(mask, box, im_h, im_w):
        width = int(box[2] - box[0] + 1)
        height = int(box[3] - box[1] + 1)
        width = max(width, 1)
        height = max(height, 1)

        mask = mask.expand((1, 1, -1, -1))
        mask = F.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False)
        mask = mask[0][0]

        im_mask = torch.full((im_h, im_w), value, dtype=mask.dtype, device=mask.device)
        x_0 = max(int(box[0]), 0)
        x_1 = min(int(box[2]) + 1, im_w)
        y_0 = max(int(box[1]), 0)
        y_1 = min(int(box[3]) + 1, im_h)

        im_mask[y_0:y_1, x_0:x_1] = mask[
            (y_0 - int(box[1])) : (y_1 - int(box[1])),
            (x_0 - int(box[0])) : (x_1 - int(box[0])),
        ]
        return im_mask

    paste_mask_in_image.__name__ = "paste_mask_in_image"
    paste_mask_in_image._cad_logit_pad = value
    return paste_mask_in_image


class logit_paste:  # noqa: N801 - used as a context manager
    """Patch the installed torchvision mask paste to run in logits space.

    ``with logit_paste(pad): model([image])`` returns ``outputs["masks"]`` as
    native-canvas *logits*.  All three patch targets are restored on exit, also
    on error, so the process is left exactly as it was found.
    """

    def __init__(self, pad=DEFAULT_LOGIT_PAD):
        self.pad = float(pad)
        self._module = None
        self._saved = {}

    def __enter__(self):
        module = _roi_heads_module()
        self._module = module
        self._saved = {
            "maskrcnn_inference": module.maskrcnn_inference,
            "expand_masks": module.expand_masks,
            "paste_mask_in_image": module.paste_mask_in_image,
        }
        module.maskrcnn_inference = _maskrcnn_inference_logits
        module.expand_masks = _expand_masks_padded(self.pad)
        module.paste_mask_in_image = _paste_mask_in_image_filled(self.pad)
        return self

    def __exit__(self, exc_type, exc, tb):
        module = self._module
        if module is not None:
            for name, function in self._saved.items():
                setattr(module, name, function)
        self._module = None
        self._saved = {}
        return False


# --------------------------------------------------------------------------- #
# probabilities from pasted logits
# --------------------------------------------------------------------------- #
def sigmoid(arrays):
    """``sigmoid`` of a float array, float32, C-contiguous (torch when present)."""

    import numpy as np

    values = np.ascontiguousarray(np.asarray(arrays, dtype=np.float32))
    try:
        import torch
    except ImportError:  # pragma: no cover - torch ships with the inference env
        return (1.0 / (1.0 + np.exp(-values.astype(np.float64)))).astype(np.float32)
    with torch.no_grad():
        return torch.sigmoid(torch.from_numpy(values)).numpy()


def wrap_predict(base, pad=DEFAULT_LOGIT_PAD):
    """Wrap a ``predict(context, image)`` in the logits paste.

    ``base`` is the untouched predict -- the lane's ``default_predict`` or an
    injected stand-in -- so its boxes, labels, scores, thresholds and every
    shape/range check are exactly the probability path's.  Only the pasted
    values change: they come back as logits and are mapped through ``sigmoid``
    here, which puts the 0.5 test exactly on the logit-0 crossing.
    """

    value = float(pad)

    def predict(context, image):
        with logit_paste(value):
            result = base(context, image)
        out = dict(result)
        out["mask_probabilities"] = sigmoid(result["mask_probabilities"])
        out["mask_paste"] = MASK_PASTE_LOGITS
        out["mask_logit_pad"] = value
        return out

    predict.__name__ = "predict_logit_paste"
    predict.__wrapped__ = base
    return predict

"""Pretrained torchvision Mask R-CNN v2 route for direct instance output.

This module owns the model/architecture seam only.  It builds a COCO-initialised
``maskrcnn_resnet50_fpn_v2``, replaces the two class-specific predictor heads
with 36-output heads (background + canonical classes 0..34) while reusing the
learned, class-agnostic mask deconvolution, and exposes a partial-label training
forward that never supervises unknown ink as negative.

Hard constraints implemented here:

* every parameter is trainable (``trainable_backbone_layers=5``);
* BatchNorm running statistics are frozen by calling ``freeze_bn_stats`` before
  each forward; affine weight/bias remain trainable;
* training crops use native resolution with an explicit 384/384 transform;
  inference uses an explicit 1000/1000 transform and maps masks back to the PNG's
  own H/W;
* labels are model classes 1..35 for canonical classes 0..34, background 0;
* the official target contract supplies per-instance masks, ink and
  ``loss_domain``; negative supervision is restricted to
  ``loss_domain & ~union(target masks)`` (the target owner's
  ``blank_negative``), so skipped Thing(-1)/semantic-35 geometry is never a
  negative even when the PNG is blank there.

No framework monkeypatch is installed.  The custom forward calls the public
torchvision building blocks directly and records effective positive/negative/
domain counts for every loss.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:  # torchvision is an explicit runtime dependency
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.models.detection.rpn import concat_box_prediction_layers
except Exception as exc:  # pragma: no cover - runtime is installed separately
    raise RuntimeError(
        "torchvision with the detection models is required (validated runtime: "
        "torch 2.14.1+cu130 / torchvision 0.29.1+cu130 or 0.29.2+cpu): %s" % exc
    ) from exc

from . import partial


NUM_CANONICAL_CLASSES = 35
NUM_MODEL_CLASSES = NUM_CANONICAL_CLASSES + 1  # background 0
CANONICAL_TO_MODEL_OFFSET = 1
TRAIN_CROP_SIZE = 384
TRAIN_JITTER = 24
DEFAULT_SEED = 20261108


@dataclass(frozen=True)
class DetectorConfig:
    """Exact, serialisable configuration of the detector seam."""

    weights: str = "none"  # "none" or "coco"
    num_classes: int = NUM_MODEL_CLASSES
    train_min_size: int = TRAIN_CROP_SIZE
    train_max_size: int = TRAIN_CROP_SIZE
    inference_min_size: int = 1000
    inference_max_size: int = 1000
    mask_roi_pool_size: int = 14
    mask_roi_output_size: int = 28  # MaskRCNNPredictor deconvolves 14 -> 28
    rpn_pre_nms_train: int = 2000
    rpn_post_nms_train: int = 2000
    rpn_pre_nms_test: int = 1000
    rpn_post_nms_test: int = 1000
    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.5
    box_detections_per_img: int = 300
    mask_threshold: float = 0.5
    mask_target_mode: str = "soft"
    # Coordinate convention for the *target/domain* ROIAlign only.  The mask
    # ROI pool and inference transform are deliberately untouched.
    mask_target_aligned: bool = True
    mask_target_sampling_ratio: int = 2
    mask_target_projection: str = "legacy"
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    seed: int = DEFAULT_SEED
    # Additive escape hatch for non-native training sizes (e.g. 512px crops).
    # Default False keeps the historical native-384/384 contract exactly.
    allow_variable_train_size: bool = False

    def __post_init__(self) -> None:
        if self.num_classes != NUM_MODEL_CLASSES:
            raise ValueError("num_classes must be 36 (background + 35 canonical)")
        if self.allow_variable_train_size:
            if int(self.train_min_size) != int(self.train_max_size):
                raise ValueError(
                    "allow_variable_train_size requires train_min_size == train_max_size"
                )
            size = int(self.train_min_size)
            if size <= 0 or size % 32 != 0:
                raise ValueError(
                    "allow_variable_train_size requires a positive multiple of 32, got %r" % (size,)
                )
        elif self.train_min_size != TRAIN_CROP_SIZE or self.train_max_size != TRAIN_CROP_SIZE:
            raise ValueError("training transform is fixed to native 384/384")
        if self.mask_roi_pool_size * 2 != self.mask_roi_output_size:
            raise ValueError("mask output is the 2x deconvolution of the ROI pool")
        if self.mask_target_mode not in ("soft", "hard"):
            raise ValueError("mask_target_mode must be 'soft' or 'hard'")
        if not isinstance(self.mask_target_aligned, bool):
            raise ValueError("mask_target_aligned must be a bool")
        if self.mask_target_sampling_ratio not in (-1, 2):
            raise ValueError("mask_target_sampling_ratio must be -1 or 2")
        if self.mask_target_projection not in ("legacy", "paste_consistent"):
            raise ValueError(
                "mask_target_projection must be 'legacy' or 'paste_consistent'"
            )

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["image_mean"] = list(self.image_mean)
        value["image_std"] = list(self.image_std)
        return value

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "DetectorConfig":
        data = dict(value)
        if "image_mean" in data:
            data["image_mean"] = tuple(float(v) for v in data["image_mean"])
        if "image_std" in data:
            data["image_std"] = tuple(float(v) for v in data["image_std"])
        return cls(**data)


def freeze_bn_stats(model: nn.Module) -> int:
    """Put every BatchNorm (including torchvision frozen variants) in eval mode.

    Running buffers are never written; ``weight``/``bias`` stay trainable.  Call
    this immediately before every training forward because ``model.train()``
    would otherwise re-enable running-stat updates.
    """

    frozen = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            module.eval()
            frozen += 1
            continue
        if module.__class__.__name__ == "FrozenBatchNorm2d":
            module.eval()
            frozen += 1
    return frozen


def set_trainable(model: nn.Module, trainable: bool = True) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(trainable)


def _replace_predictors(model: nn.Module, num_classes: int) -> Dict[str, Any]:
    """Replace only the class-specific heads; reuse the mask deconvolution."""

    roi_heads = model.roi_heads
    old_box = roi_heads.box_predictor
    old_mask = roi_heads.mask_predictor
    box_in = int(old_box.cls_score.in_features)
    roi_heads.box_predictor = FastRCNNPredictor(box_in, num_classes)

    mask_in = int(old_mask.conv5_mask.in_channels)
    mask_dim = int(old_mask.conv5_mask.out_channels)
    new_mask = MaskRCNNPredictor(mask_in, mask_dim, num_classes)
    reused = False
    try:
        new_mask.conv5_mask.load_state_dict(old_mask.conv5_mask.state_dict(), strict=True)
        reused = True
    except Exception:
        reused = False
    roi_heads.mask_predictor = new_mask
    return {
        "box_predictor_in_features": box_in,
        "mask_deconv_in_channels": mask_in,
        "mask_deconv_out_channels": mask_dim,
        "mask_deconv_reused_from_pretrained": reused,
        "num_classes": num_classes,
    }


def build_model(
    config: Optional[DetectorConfig] = None,
    *,
    weights_file: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Build the detector.

    ``weights_file`` is a local COCO checkpoint (91-class torchvision state
    dict); this lane never downloads assets itself.  ``config.weights='coco'``
    asks torchvision for its built-in COCO weights and is only appropriate when
    the asset owner has already cached them.
    """

    config = config or DetectorConfig()
    torch.manual_seed(config.seed)

    weights = None
    weights_source = "none"
    if weights_file is not None:
        if not os.path.isfile(weights_file):
            raise FileNotFoundError("weights_file does not exist: %r" % weights_file)
        weights_source = "local_state_dict:%s" % os.path.abspath(weights_file)
        # Build the 91-class COCO topology, load it strictly, then replace heads.
        model = maskrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            num_classes=91,
            trainable_backbone_layers=5,
        )
        state = torch.load(weights_file, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
    elif config.weights == "coco":
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        weights_source = "torchvision:%s" % str(weights)
        model = maskrcnn_resnet50_fpn_v2(
            weights=weights,
            weights_backbone=None,
            trainable_backbone_layers=5,
        )
    elif config.weights == "none":
        model = maskrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            num_classes=config.num_classes,
            trainable_backbone_layers=5,
        )
    else:
        raise ValueError("weights must be 'none' or 'coco'")

    if int(model.roi_heads.box_predictor.cls_score.out_features) != config.num_classes:
        head_info = _replace_predictors(model, config.num_classes)
    else:
        head_info = {
            "box_predictor_in_features": int(model.roi_heads.box_predictor.cls_score.in_features),
            "mask_deconv_in_channels": int(model.roi_heads.mask_predictor.conv5_mask.in_channels),
            "mask_deconv_out_channels": int(model.roi_heads.mask_predictor.conv5_mask.out_channels),
            "mask_deconv_reused_from_pretrained": False,
            "num_classes": config.num_classes,
        }

    set_trainable(model, True)
    set_trainable(model.backbone, True)
    model.direct_instance_config = config
    model.mask_target_mode = config.mask_target_mode
    model.mask_target_aligned = config.mask_target_aligned
    model.mask_target_sampling_ratio = config.mask_target_sampling_ratio
    model.mask_target_projection = config.mask_target_projection

    transform = model.transform
    transform.min_size = (config.train_min_size,)
    transform.max_size = config.train_max_size
    transform.image_mean = list(config.image_mean)
    transform.image_std = list(config.image_std)
    model.training_transform = ((int(config.train_min_size),), int(config.train_max_size))

    model.roi_heads.score_thresh = float(config.box_score_thresh)
    model.roi_heads.nms_thresh = float(config.box_nms_thresh)
    model.roi_heads.detections_per_img = int(config.box_detections_per_img)
    model.roi_heads.batch_size_per_image = 512
    model.roi_heads.positive_fraction = 0.25
    model.rpn._pre_nms_top_n = {
        "training": int(config.rpn_pre_nms_train),
        "testing": int(config.rpn_pre_nms_test),
    }
    model.rpn._post_nms_top_n = {
        "training": int(config.rpn_post_nms_train),
        "testing": int(config.rpn_post_nms_test),
    }

    bn_count = freeze_bn_stats(model)
    mask_output = int(model.roi_heads.mask_predictor.mask_fcn_logits.out_channels)
    info = {
        "config": config.to_dict(),
        "weights_source": weights_source,
        "head": head_info,
        "batch_norm_modules_frozen": int(bn_count),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "mask_roi_pool_output_size": int(model.roi_heads.mask_roi_pool.output_size[0]),
        "mask_head_spatial_output": int(config.mask_roi_output_size),
        "mask_target_projection": str(config.mask_target_projection),
        "mask_target_aligned": bool(config.mask_target_aligned),
        "mask_target_sampling_ratio": int(config.mask_target_sampling_ratio),
        "max_detections_per_image": int(config.box_detections_per_img),
    }
    return model, info


def set_inference_transform(model: nn.Module, config: DetectorConfig) -> None:
    model.transform.min_size = (int(config.inference_min_size),)
    model.transform.max_size = int(config.inference_max_size)


def assert_training_transform(model: nn.Module, config: Optional[DetectorConfig] = None) -> None:
    """Re-assert the frozen native-384 training transform before every forward.

    ``set_inference_transform`` mutates the same model instance used for
    synthetic inference tests; training must never silently run at 1000px or
    resize crops.  The expected transform is captured at build time so synthetic
    tests can shrink it; a real model has no override path.
    """

    if config is None:
        config = getattr(model, "direct_instance_config", None)
    expected = getattr(model, "training_transform", None)
    if expected is None and config is not None:
        expected = ((int(config.train_min_size),), int(config.train_max_size))
    if expected is None:
        raise RuntimeError("model has neither direct_instance_config nor training_transform")
    if (
        config is not None
        and not getattr(config, "allow_variable_train_size", False)
        and (
            int(config.train_min_size) != TRAIN_CROP_SIZE
            or int(config.train_max_size) != TRAIN_CROP_SIZE
        )
    ):
        raise RuntimeError("training transform must be native 384/384")
    expected_min, expected_max = expected
    expected_min = tuple(int(v) for v in expected_min)
    expected_max = int(expected_max)
    actual_min = tuple(int(v) for v in model.transform.min_size)
    actual_max = int(model.transform.max_size)
    if actual_min != expected_min or actual_max != expected_max:
        raise RuntimeError(
            "training transform drift: expected min_size=%r max_size=%r, got min_size=%r max_size=%r"
            % (expected_min, expected_max, actual_min, actual_max)
        )


# --------------------------------------------------------------------------- #
# targets / images
# --------------------------------------------------------------------------- #


def _as_float_tensor(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError("%s contains non-finite values" % name)
    return tensor


def prepare_detector_targets(
    target: Dict[str, Any],
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Convert the target owner's per-instance API into detector tensors.

    ``target['boxes']`` / ``keys`` are the N-vector source; the dense
    ``target['labels']`` is diagnostic only and is deliberately ignored.
    """

    keys = list(target["keys"])
    count = len(keys)
    boxes = _as_float_tensor(target["boxes"], "boxes").reshape(-1, 4)
    if boxes.shape[0] != count:
        raise ValueError("boxes count %d != keys %d" % (boxes.shape[0], count))
    labels = torch.tensor(
        [int(key[0]) + CANONICAL_TO_MODEL_OFFSET for key in keys], dtype=torch.int64
    )
    masks = torch.as_tensor(target["masks"], dtype=torch.bool)
    if masks.ndim != 3 or masks.shape[0] != count:
        raise ValueError("masks must be NHW with N == number of keys")

    domain = torch.as_tensor(target["loss_domain"], dtype=torch.bool)
    if domain.ndim != 2 or domain.shape != masks.shape[1:]:
        raise ValueError("loss_domain must be HW matching masks")
    union = masks.any(dim=0) if count else torch.zeros_like(domain)

    if "blank_negative" in target and target["blank_negative"] is not None:
        negative = torch.as_tensor(target["blank_negative"], dtype=torch.bool)
        if negative.shape != domain.shape:
            raise ValueError("blank_negative must be HW matching loss_domain")
    else:
        # Exact algebra of the accepted target contract:
        # loss_domain = union | blank_negative  =>  blank = loss_domain & ~union.
        negative = domain & ~union
    if bool((negative & union).any()):
        raise ValueError("blank_negative overlaps a positive target mask")
    if "unknown_geometry" in target and target["unknown_geometry"] is not None:
        unknown = torch.as_tensor(target["unknown_geometry"], dtype=torch.bool)
        if unknown.shape != domain.shape:
            raise ValueError("unknown_geometry must be HW matching loss_domain")
        if bool((negative & unknown).any()):
            raise ValueError("blank_negative licenses unknown_geometry pixels")

    return {
        "boxes": boxes.to(device),
        "labels": labels.to(device),
        "masks": masks.to(device),
        "loss_domain": domain.to(device),
        "blank_negative": negative.to(device),
        "image_id": torch.tensor([int(target.get("image_id", 0))], dtype=torch.int64).to(device),
    }


def _resize_bool_map(value: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if tuple(value.shape[-2:]) == tuple(size):
        return value
    return (
        F.interpolate(
            value[None, None].float(), size=tuple(size), mode="nearest"
        )[0, 0]
        > 0.5
    )


def _resize_masks(masks: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if masks.numel() == 0:
        return masks
    if tuple(masks.shape[-2:]) == tuple(size):
        return masks
    return (
        F.interpolate(masks[:, None].float(), size=tuple(size), mode="nearest")[:, 0]
        > 0.5
    )


def _zero_loss(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    total = None
    for parameter in parameters:
        term = parameter.sum() * 0.0
        total = term if total is None else total + term
    if total is None:
        return torch.zeros((), requires_grad=True)
    return total


# --------------------------------------------------------------------------- #
# custom partial-label training forward
# --------------------------------------------------------------------------- #


def training_losses(
    model: nn.Module,
    images: Sequence[torch.Tensor],
    targets: Sequence[Dict[str, torch.Tensor]],
) -> Dict[str, Any]:
    """One training forward/backward-ready loss computation.

    Standard torchvision sampling is replaced at the two seams where it could
    turn unknown ink into a negative example:

    * RPN anchors below the low IoU threshold become 0 only if their entire
      pixel rectangle is in ``blank_negative``; otherwise -1 (ignored before
      the sampler);
    * ROI proposals below the low threshold are treated the same way.

    All other torchvision losses remain untouched in form.  Mask BCE uses the
    target's ``loss_domain`` cells only; unknown ink is outside the denominator.
    """

    if model.training is False:
        model.train()
    assert_training_transform(model)
    freeze_bn_stats(model)

    images = [image.float() for image in images]
    images_t, targets_t = model.transform(list(images), list(targets))
    features = model.backbone(images_t.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    feature_list = list(features.values())
    image_sizes = [tuple(int(v) for v in size) for size in images_t.image_sizes]

    rpn = model.rpn
    roi_heads = model.roi_heads

    objectness_list, deltas_list = rpn.head(feature_list)
    anchors = rpn.anchor_generator(images_t, feature_list)
    num_anchors_per_level = [int(o[0].numel()) for o in objectness_list]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(
        objectness_list, deltas_list
    )
    proposals_decoded = rpn.box_coder.decode(
        pred_bbox_deltas.detach(), anchors
    ).view(len(images), -1, 4)
    proposals, _proposal_scores = rpn.filter_proposals(
        proposals_decoded,
        objectness,
        image_sizes,
        num_anchors_per_level,
    )

    # Resize licensed maps/masks to the actual transformed image size.  The
    # accepted contract is native-resolution; this branch is inert for 384
    # crops but keeps arbitrary synthetic H/W exact.
    negative_maps: List[torch.Tensor] = []
    domain_maps: List[torch.Tensor] = []
    gt_masks: List[torch.Tensor] = []
    for target, size in zip(targets, image_sizes):
        device = target["boxes"].device
        negative_maps.append(
            _resize_bool_map(target["blank_negative"].to(device), size)
        )
        domain_maps.append(_resize_bool_map(target["loss_domain"].to(device), size))
        gt_masks.append(_resize_masks(target["masks"].to(device), size))

    gt_boxes = [target["boxes"] for target in targets_t]
    gt_labels = [target["labels"] for target in targets_t]

    # ---- RPN losses ------------------------------------------------------- #
    rpn_labels, rpn_matched_boxes = partial.assign_partial_rpn_labels(
        rpn, anchors, gt_boxes, negative_maps
    )
    rpn_regression_targets = rpn.box_coder.encode(rpn_matched_boxes, anchors)
    rpn_losses, rpn_counts = partial.rpn_losses(
        objectness,
        pred_bbox_deltas,
        rpn_labels,
        rpn_regression_targets,
        batch_size_per_image=256,
        positive_fraction=0.5,
    )

    # ---- ROI box head ----------------------------------------------------- #
    proposals_with_gt = [
        torch.cat((proposal, boxes), dim=0) if boxes.numel() else proposal
        for proposal, boxes in zip(proposals, gt_boxes)
    ]
    matched_idxs, roi_labels = roi_heads.assign_targets_to_proposals(
        proposals_with_gt, gt_boxes, gt_labels
    )
    roi_labels = partial.filter_roi_negatives(
        roi_labels, proposals_with_gt, negative_maps
    )
    sampled_inds = roi_heads.subsample(roi_labels)

    proposals_sampled: List[torch.Tensor] = []
    labels_sampled: List[torch.Tensor] = []
    matched_sampled: List[torch.Tensor] = []
    for image_index, indices in enumerate(sampled_inds):
        proposals_sampled.append(proposals_with_gt[image_index][indices])
        labels_sampled.append(roi_labels[image_index][indices])
        matched_sampled.append(matched_idxs[image_index][indices])

    total_sampled = int(sum(value.numel() for value in labels_sampled))
    counts: Dict[str, Any] = {
        "rpn": rpn_counts,
        "roi_sampled": total_sampled,
        "roi_positive": int(
            sum(int((value >= 1).sum().item()) for value in labels_sampled)
        ),
        "roi_negative": int(
            sum(int((value == 0).sum().item()) for value in labels_sampled)
        ),
        "roi_ignored_total": int(
            sum(int((value < 0).sum().item()) for value in roi_labels)
        ),
        "target_count": int(sum(len(value) for value in gt_labels)),
    }

    zero_box = _zero_loss(roi_heads.box_predictor.parameters())
    zero_mask = _zero_loss(roi_heads.mask_predictor.parameters())
    mask_projection = str(getattr(model, "mask_target_projection", "legacy"))
    mask_aligned_requested = bool(getattr(model, "mask_target_aligned", True))
    mask_sampling_requested = int(
        getattr(model, "mask_target_sampling_ratio", 2)
    )
    if mask_projection == "paste_consistent":
        mask_aligned_effective = False
        mask_sampling_effective = -1
    else:
        mask_aligned_effective = mask_aligned_requested
        mask_sampling_effective = mask_sampling_requested

    if total_sampled == 0:
        losses = {
            "loss_classifier": zero_box,
            "loss_box_reg": zero_box,
            "loss_mask": zero_mask,
            "loss_objectness": rpn_losses["loss_objectness"],
            "loss_rpn_box_reg": rpn_losses["loss_rpn_box_reg"],
        }
        counts["mask_licensed_cells"] = 0
        counts["mask_positive_cells"] = 0
        counts["mask_negative_cells"] = 0
        counts["mask_soft_positive_mass"] = 0.0
        counts["mask_hard_positive_cells"] = 0
        counts["mask_target_mode"] = str(getattr(model, "mask_target_mode", "soft"))
        counts["mask_target_projection"] = mask_projection
        counts["mask_target_aligned"] = mask_aligned_effective
        counts["mask_target_sampling_ratio"] = mask_sampling_effective
        counts["mask_target_aligned_requested"] = mask_aligned_requested
        counts["mask_target_sampling_ratio_requested"] = mask_sampling_requested
        counts["mask_original_positive_proposals"] = 0
        counts["mask_excluded_narrow_proposals"] = 0
        counts["mask_effective_supervised_proposals"] = 0
        return {"losses": losses, "counts": counts, "proposals": proposals}

    box_features = roi_heads.box_roi_pool(features, proposals_sampled, image_sizes)
    box_features = roi_heads.box_head(box_features)
    class_logits, box_regression = roi_heads.box_predictor(box_features)

    labels_cat = torch.cat(labels_sampled, dim=0)
    matched_gt_boxes = []
    for image_index, matched in enumerate(matched_sampled):
        boxes = gt_boxes[image_index]
        if boxes.numel() == 0:
            # BoxCoder.encode splits reference boxes by len(proposals_i); the
            # empty-GT branch must therefore provide one zero row per proposal,
            # not a single row.
            matched_gt_boxes.append(
                torch.zeros((int(matched.numel()), 4), device=proposals_sampled[image_index].device)
            )
        else:
            matched_gt_boxes.append(boxes[matched.clamp(min=0)])
    regression_targets = torch.cat(
        roi_heads.box_coder.encode(matched_gt_boxes, proposals_sampled), dim=0
    )

    loss_classifier = F.cross_entropy(class_logits, labels_cat)
    positive_mask = labels_cat >= 1
    if bool(positive_mask.any()):
        num_boxes = class_logits.shape[0]
        box_regression_flat = box_regression.reshape(
            num_boxes, box_regression.size(-1) // 4, 4
        )
        loss_box_reg = (
            F.smooth_l1_loss(
                box_regression_flat[positive_mask, labels_cat[positive_mask]],
                regression_targets[positive_mask],
                beta=1 / 9,
                reduction="sum",
            )
            / labels_cat.numel()
        )
    else:
        loss_box_reg = box_regression.sum() * 0.0

    # ---- mask head, licensed-domain BCE ---------------------------------- #
    positive_proposals = [
        proposals_sampled[i][labels_sampled[i] >= 1]
        for i in range(len(images))
    ]
    positive_matched = [
        matched_sampled[i][labels_sampled[i] >= 1]
        for i in range(len(images))
    ]
    positive_labels = labels_cat[positive_mask]
    total_positive = int(positive_labels.numel())
    counts["mask_positive_proposals"] = total_positive

    if total_positive == 0:
        loss_mask = zero_mask
        counts["mask_licensed_cells"] = 0
        counts["mask_positive_cells"] = 0
        counts["mask_negative_cells"] = 0
        counts["mask_soft_positive_mass"] = 0.0
        counts["mask_hard_positive_cells"] = 0
        counts["mask_target_mode"] = str(getattr(model, "mask_target_mode", "soft"))
        counts["mask_target_projection"] = mask_projection
        counts["mask_target_aligned"] = mask_aligned_effective
        counts["mask_target_sampling_ratio"] = mask_sampling_effective
        counts["mask_target_aligned_requested"] = mask_aligned_requested
        counts["mask_target_sampling_ratio_requested"] = mask_sampling_requested
        counts["mask_original_positive_proposals"] = 0
        counts["mask_excluded_narrow_proposals"] = 0
        counts["mask_effective_supervised_proposals"] = 0
    else:
        mask_features = roi_heads.mask_roi_pool(
            features, positive_proposals, image_sizes
        )
        mask_features = roi_heads.mask_head(mask_features)
        mask_logits = roi_heads.mask_predictor(mask_features)
        loss_mask, mask_counts = partial.mask_domain_bce(
            mask_logits,
            gt_masks,
            domain_maps,
            positive_proposals,
            positive_matched,
            positive_labels,
            output_size=int(config_mask_output(mask_logits)),
            target_mode=str(getattr(model, "mask_target_mode", "soft")),
            aligned=mask_aligned_requested,
            sampling_ratio=mask_sampling_requested,
            projection=mask_projection,
        )
        counts["mask_licensed_cells"] = mask_counts["licensed_cells"]
        counts["mask_positive_cells"] = mask_counts["positive_cells"]
        counts["mask_negative_cells"] = mask_counts["negative_cells"]
        counts["mask_soft_positive_mass"] = mask_counts["soft_positive_mass"]
        counts["mask_hard_positive_cells"] = mask_counts["hard_positive_cells"]
        counts["mask_target_mode"] = mask_counts["target_mode"]
        counts["mask_target_projection"] = mask_counts["target_projection"]
        counts["mask_target_aligned"] = mask_counts["target_aligned"]
        counts["mask_target_sampling_ratio"] = mask_counts[
            "target_sampling_ratio"
        ]
        counts["mask_target_aligned_requested"] = mask_counts[
            "target_aligned_requested"
        ]
        counts["mask_target_sampling_ratio_requested"] = mask_counts[
            "target_sampling_ratio_requested"
        ]
        counts["mask_original_positive_proposals"] = mask_counts[
            "original_positive_proposals"
        ]
        counts["mask_excluded_narrow_proposals"] = mask_counts[
            "excluded_narrow_proposals"
        ]
        counts["mask_effective_supervised_proposals"] = mask_counts[
            "effective_supervised_proposals"
        ]

    losses = {
        "loss_classifier": loss_classifier,
        "loss_box_reg": loss_box_reg,
        "loss_mask": loss_mask,
        "loss_objectness": rpn_losses["loss_objectness"],
        "loss_rpn_box_reg": rpn_losses["loss_rpn_box_reg"],
    }
    return {"losses": losses, "counts": counts, "proposals": proposals}


def config_mask_output(mask_logits: torch.Tensor) -> int:
    return int(mask_logits.shape[-1])


def total_loss(losses: Dict[str, torch.Tensor]) -> torch.Tensor:
    total = None
    for value in losses.values():
        total = value if total is None else total + value
    if total is None:
        raise ValueError("no losses")
    return total

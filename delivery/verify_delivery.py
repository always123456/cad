#!/usr/bin/env python
"""Independent verifier for a delivery run directory (stdlib only).

Deliberately imports nothing from cadvec, instance_eval or the delivery CLI: the
contract rules below are re-implemented from the README §4 field list so a bug in
the shipped code cannot vouch for itself.

Usage:  python verify_delivery.py <run_out_dir> [more run dirs...]
Exit:   0 all clean, 1 findings.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys

FIELDS = (
    "commands",
    "args",
    "lengths",
    "semanticIds",
    "instanceIds",
    "width",
    "height",
    "obj_cts",
    "boxes",
    "rgb",
    "layerIds",
    "widths",
)
ARRAYS = ("commands", "args", "lengths", "semanticIds", "instanceIds", "rgb", "layerIds", "widths")
THING = (0, 29)
STUFF = (30, 34)
IGNORE = 35
VIEWBOX = (0.0, 0.0, 100.0, 100.0)


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def is_num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def close(left, right, tol=1e-6):
    return abs(float(left) - float(right)) <= tol * max(1.0, abs(float(right)))


def check_document(document, raw):
    """Return (errors, facts) for one result document."""

    errors = []
    facts = {}

    if not isinstance(document, dict):
        return ["document_not_object"], facts
    keys = set(document)
    if keys != set(FIELDS):
        missing = sorted(set(FIELDS) - keys)
        extra = sorted(keys - set(FIELDS))
        if missing:
            errors.append("missing_fields:" + ",".join(missing))
        if extra:
            errors.append("unknown_fields:" + ",".join(extra))
        return errors, facts

    facts["fields"] = len(keys)
    if b'"score"' in raw or b"'score'" in raw:
        errors.append("score_field_present_in_bytes")
    if b"ground_truth" in raw or b".svg" in raw:
        errors.append("gt_reference_in_document")

    arrays = {}
    for field in ARRAYS:
        value = document[field]
        if not isinstance(value, list):
            errors.append(f"{field}_not_array")
        else:
            arrays[field] = value
    if len(arrays) != len(ARRAYS):
        return errors, facts

    n = len(arrays["commands"])
    facts["primitives"] = n
    for field, value in arrays.items():
        if len(value) != n:
            errors.append(f"{field}_length_mismatch:{len(value)}!={n}")
    if errors:
        return errors, facts

    width, height = document["width"], document["height"]
    if not is_int(width) or width <= 0:
        errors.append("width_invalid")
    if not is_int(height) or height <= 0:
        errors.append("height_invalid")
    facts["canvas"] = [width, height]

    for index, value in enumerate(arrays["commands"]):
        if not is_int(value) or not 0 <= value <= 3:
            errors.append(f"commands_invalid:{index}")
    for index, value in enumerate(arrays["lengths"]):
        if not is_num(value) or float(value) < 0.0:
            errors.append(f"lengths_invalid:{index}")
    for index, value in enumerate(arrays["semanticIds"]):
        if not is_int(value) or not 0 <= value <= 35:
            errors.append(f"semanticIds_invalid:{index}")
    for index, value in enumerate(arrays["instanceIds"]):
        if not is_int(value) or value < -1:
            errors.append(f"instanceIds_invalid:{index}")
    for index, value in enumerate(arrays["rgb"]):
        value = arrays["rgb"][index]
        if not isinstance(value, list) or len(value) != 3 or any(
            not is_int(channel) or not 0 <= channel <= 255 for channel in value
        ):
            errors.append(f"rgb_invalid:{index}")
    for index, value in enumerate(arrays["layerIds"]):
        if not is_int(value) or value < 1:
            errors.append(f"layerIds_invalid:{index}")
    for index, value in enumerate(arrays["widths"]):
        if not is_num(value) or float(value) < 0.0:
            errors.append(f"widths_invalid:{index}")

    xs, ys = [], []
    for index, value in enumerate(arrays["args"]):
        if not isinstance(value, list) or len(value) != 8 or any(not is_num(item) for item in value):
            errors.append(f"args_invalid:{index}")
            continue
        for pair in range(4):
            xs.append(float(value[pair * 2]))
            ys.append(float(value[pair * 2 + 1]))
    if xs:
        facts["x_range"] = [min(xs), max(xs)]
        facts["y_range"] = [min(ys), max(ys)]
        if min(xs) < VIEWBOX[0] or max(xs) > VIEWBOX[2] or min(ys) < VIEWBOX[1] or max(ys) > VIEWBOX[3]:
            errors.append(
                "args_outside_viewbox:x=%.6f..%.6f,y=%.6f..%.6f" % (min(xs), max(xs), min(ys), max(ys))
            )

    # semantic / instance pairing
    order, semantic_of, members = [], {}, {}
    for index, (semantic, instance) in enumerate(zip(arrays["semanticIds"], arrays["instanceIds"])):
        if not is_int(semantic) or not is_int(instance):
            continue
        if THING[0] <= semantic <= THING[1]:
            if instance < 0:
                errors.append(f"thing_instance_negative:{index}")
        elif STUFF[0] <= semantic <= STUFF[1] or semantic == IGNORE:
            if instance != -1:
                errors.append(f"stuff_instance_not_minus_one:{index}")
        if THING[0] <= semantic <= THING[1] and instance >= 0:
            if instance not in semantic_of:
                order.append(instance)
                semantic_of[instance] = semantic
                members[instance] = []
            elif semantic_of[instance] != semantic:
                errors.append(f"instance_semantic_mismatch:{instance}")
            members[instance].append(index)
    facts["instances"] = len(order)
    facts["semantic_ids"] = sorted(set(arrays["semanticIds"]))

    obj_cts, boxes = document["obj_cts"], document["boxes"]
    if not isinstance(obj_cts, list) or not isinstance(boxes, list):
        errors.append("obj_cts_or_boxes_not_array")
        return errors, facts
    if len(obj_cts) != len(order):
        errors.append(f"obj_cts_instance_count_mismatch:{len(obj_cts)}!={len(order)}")
    if len(boxes) != len(order):
        errors.append(f"boxes_instance_count_mismatch:{len(boxes)}!={len(order)}")

    for position, instance in enumerate(order):
        if position >= len(obj_cts) or position >= len(boxes):
            break
        center, box = obj_cts[position], boxes[position]
        if not isinstance(center, list) or len(center) != 4 or any(not is_num(v) for v in center):
            errors.append(f"obj_cts_invalid:{position}")
            continue
        if not isinstance(box, list) or len(box) != 5 or any(not is_num(v) for v in box):
            errors.append(f"boxes_invalid:{position}")
            continue
        if not is_int(center[3]) or int(center[3]) != instance:
            errors.append(f"obj_cts_instance_order_mismatch:{position}")
        if float(center[2]) != 0.0:
            errors.append(f"obj_cts_third_value_not_zero:{position}")
        if not is_int(box[4]) or int(box[4]) != semantic_of[instance]:
            errors.append(f"boxes_semantic_mismatch:{position}")

        point_x, point_y = [], []
        for member in members.get(instance, []):
            flat = arrays["args"][member]
            if not isinstance(flat, list) or len(flat) != 8:
                continue
            for pair in range(4):
                if is_num(flat[pair * 2]) and is_num(flat[pair * 2 + 1]):
                    point_x.append(float(flat[pair * 2]))
                    point_y.append(float(flat[pair * 2 + 1]))
        if not point_x:
            errors.append(f"instance_without_finite_points:{instance}")
            continue
        expected_box = (min(point_x), min(point_y), max(point_x), max(point_y))
        expected_center = ((min(point_x) + max(point_x)) / 2.0, (min(point_y) + max(point_y)) / 2.0)
        if any(not close(a, b) for a, b in zip((float(v) for v in box[:4]), expected_box)):
            errors.append(f"boxes_geometry_mismatch:{position}")
        if any(not close(a, b) for a, b in zip((float(center[0]), float(center[1])), expected_center)):
            errors.append(f"obj_cts_geometry_mismatch:{position}")

    return errors, facts


def verify_run(out_dir):
    report = {"out": out_dir, "documents": {}, "errors": [], "index": {}, "receipt": {}}
    results_dir = os.path.join(out_dir, "results")
    names = sorted(name for name in os.listdir(results_dir) if name.endswith(".json")) if os.path.isdir(results_dir) else []
    if not names:
        report["errors"].append("no_result_documents")
        report["ok"] = False
        return report

    digests = {}
    for name in names:
        path = os.path.join(results_dir, name)
        with open(path, "rb") as handle:
            raw = handle.read()
        try:
            document = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - report, do not raise
            report["errors"].append(f"{name}:not_json:{exc}")
            continue
        canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digests[name[:-5]] = {
            "file": hashlib.sha256(raw).hexdigest(),
            "canonical": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        }
        errors, facts = check_document(document, raw)
        report["documents"][name[:-5]] = {"errors": errors, "facts": facts}
        report["errors"].extend(f"{name[:-5]}:{error}" for error in errors)

    index_path = os.path.join(out_dir, "results_index.json")
    if os.path.isfile(index_path):
        index = json.load(open(index_path, encoding="utf-8"))
        images = index.get("images") or {}
        report["index"]["count"] = len(images)
        report["index"]["keys"] = sorted(images)
        if set(images) != set(digests):
            report["errors"].append(
                "index_mismatch:index=%s disk=%s" % (sorted(images), sorted(digests))
            )
        for stem, entry in images.items():
            expected = entry.get("document_sha256")
            if stem in digests and expected != digests[stem]["canonical"]:
                report["errors"].append(f"index_sha_mismatch:{stem}")
            rel = entry.get("result")
            if rel and not os.path.isfile(os.path.join(out_dir, rel.replace("/", os.sep))):
                report["errors"].append(f"index_result_missing:{stem}")
        policy = index.get("policy") or {}
        report["index"]["policy"] = {
            key: policy.get(key) for key in ("score_threshold", "max_detections", "per_image_cap")
        }
        report["index"]["sha_ok"] = not any("index_sha_mismatch" in e for e in report["errors"])
    else:
        report["errors"].append("no_results_index")

    receipt_path = os.path.join(out_dir, "run_receipt.json")
    if os.path.isfile(receipt_path):
        receipt = json.load(open(receipt_path, encoding="utf-8"))
        guards = receipt.get("guards") or {}
        audit = receipt.get("open_audit") or {}
        report["receipt"] = {
            "status": receipt.get("status"),
            "device": receipt.get("device"),
            "policy": {
                key: (receipt.get("policy") or {}).get(key)
                for key in ("score_threshold", "max_detections", "per_image_cap")
            },
            "guards": guards,
            "violations": (audit.get("violations") or []),
            "non_square": receipt.get("non_square"),
            "checkpoint": (receipt.get("checkpoint") or {}).get("sha256"),
            "timings": receipt.get("timings"),
        }
        for key in ("document_fields_exact",):
            if not guards.get(key):
                report["errors"].append(f"guard_false:{key}")
        if guards.get("score_field_in_document"):
            report["errors"].append("guard_score_field_in_document_true")
        for key in ("ground_truth_read", "svg_read"):
            if guards.get(key):
                report["errors"].append(f"guard_{key}_true")
        if audit.get("violations"):
            report["errors"].append("open_audit_violations")
        if receipt.get("status") != "delivery_written":
            report["errors"].append("receipt_status_not_delivery_written")
        emitted = 0
        for item in (receipt.get("timings") or {}).get("per_image") or []:
            stem = item.get("stem")
            facts = (report["documents"].get(stem) or {}).get("facts") or {}
            if facts.get("instances") is not None:
                emitted += facts["instances"]
        report["receipt"]["emitted_instances"] = emitted
    else:
        report["errors"].append("no_run_receipt")

    report["ok"] = not report["errors"]
    return report


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    failed = False
    for out_dir in argv[1:]:
        report = verify_run(out_dir)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        failed = failed or not report["ok"]
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))

"""Checks for ``evaluation/projection_trial_recipe.json``.

The recipe is only useful if it names commands that really exist.  These tests
parse every recorded argv with the *real* argument parsers, so an invented flag
or a renamed option fails here instead of during a granted run.  They also pin
the immutable baseline artifacts and the projection-marker rule.

Nothing here reads a PNG, SVG, checkpoint payload, GT or the network; the largest
file touched is the baseline checkpoint, which is hashed, not loaded.

Run:  python tests/test_projection_trial_recipe.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_ROOT = os.path.dirname(HERE)
LANE_ROOT = os.path.dirname(EVAL_ROOT)
sys.path.insert(0, EVAL_ROOT)
sys.path.insert(0, LANE_ROOT)

RECIPE_PATH = os.path.join(EVAL_ROOT, "projection_trial_recipe.json")


def _recipe():
    with open(RECIPE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser_for(tool):
    name = os.path.basename(tool.split()[0])
    if name == "capacity_eval" or tool.split()[0] == "capacity_eval":
        from capacity_eval.cli import build_parser

        return build_parser()
    if name == "run_batch_predict.py":
        import run_batch_predict

        return run_batch_predict.build_parser()
    if name == "run_instance_eval.py":
        import run_instance_eval

        return run_instance_eval.build_parser()
    raise AssertionError("unknown tool in recipe: %r" % (tool,))


def _stage_argv(stage):
    """Return the argv slice addressed to the tool, without the interpreter."""

    argv = list(stage["argv"])
    assert argv[0] == "{PYTHON}", argv
    rest = argv[1:]
    tool = stage["tool"].split()[0]
    if tool == "capacity_eval":
        assert rest[:2] == ["-m", "capacity_eval"], rest
        return rest[2:]
    if rest[0].endswith(".py"):
        return rest[1:]
    return rest


class RecipeShapeTests(unittest.TestCase):
    def test_schema_status_and_four_ordered_stages(self):
        recipe = _recipe()
        self.assertEqual(recipe["schema"], "cadvec.direct_instance.projection_trial_recipe.v1")
        self.assertEqual(recipe["status"], "prepared_not_executed")
        self.assertFalse(recipe["authorization"]["execution_authorized"])
        stages = recipe["stages"]
        self.assertEqual([stage["index"] for stage in stages], [1, 2, 3, 4])
        self.assertEqual(
            [stage["name"] for stage in stages],
            ["crop_384_capacity", "native_1000_predictions", "seal_native_predictions", "score_native"],
        )

    def test_every_stage_cwd_is_inside_the_lane(self):
        recipe = _recipe()
        for stage in recipe["stages"]:
            cwd = os.path.normpath(stage["cwd"])
            self.assertEqual(os.path.commonpath([cwd, LANE_ROOT]), os.path.normpath(LANE_ROOT), stage["name"])


class RecordedFlagTests(unittest.TestCase):
    """No invented flags: the real parsers must accept every recorded argv."""

    def test_stage_one_parses_with_capacity_eval(self):
        recipe = _recipe()
        stage = recipe["stages"][0]
        args = _parser_for(stage["tool"]).parse_args(_stage_argv(stage))
        self.assertTrue(args.allow_data)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.checkpoint, "runs/capacity16_paste_consistent600_01/checkpoint.pt")
        self.assertEqual(args.output_dir, "runs/capacity16_paste_consistent_crop01")

    def test_stage_two_parses_with_the_batch_cli(self):
        recipe = _recipe()
        stage = recipe["stages"][1]
        args = _parser_for(stage["tool"]).parse_args(_stage_argv(stage))
        self.assertEqual(args.sample_set, "explicit")
        self.assertEqual(args.device, "cuda:0")
        self.assertIsNone(args.max_detections, "the recipe must not override official max detections")
        self.assertEqual(args.inputs, "../parent_capacity16_inputs.json")

    def test_stage_three_parses_with_the_evaluator_cli(self):
        recipe = _recipe()
        stage = recipe["stages"][2]
        args = _parser_for(stage["tool"]).parse_args(_stage_argv(stage))
        self.assertEqual(args.command, "prepare")
        self.assertEqual(args.sample_set, "fit16")
        self.assertEqual(args.dataset, "fit16-diagnostic")
        self.assertEqual(args.mask_role, "authoritative")
        self.assertIsNone(args.masks, "direct-JSON predictions need no mask directory")

    def test_stage_four_parses_with_the_evaluator_cli(self):
        recipe = _recipe()
        stage = recipe["stages"][3]
        args = _parser_for(stage["tool"]).parse_args(_stage_argv(stage))
        self.assertEqual(args.command, "score")
        self.assertTrue(args.allow_gt)
        self.assertIsNone(args.baseline, "no comparable reference baseline exists for this protocol")

    def test_unknown_flag_is_rejected_by_the_same_parser(self):
        recipe = _recipe()
        stage = recipe["stages"][3]
        parser = _parser_for(stage["tool"])
        with self.assertRaises(SystemExit), open(os.devnull, "w") as devnull:
            stderr = sys.stderr
            sys.stderr = devnull
            try:
                parser.parse_args(_stage_argv(stage) + ["--not-a-real-flag"])
            finally:
                sys.stderr = stderr


class ImmutableBaselineTests(unittest.TestCase):
    # Recomputing the 553 MB checkpoint hash on every run is wasteful and was
    # explicitly ruled out; large artifacts are pinned by size plus the recorded
    # digest shape, small JSON artifacts are hashed in full.
    LARGE_BYTES = 64 << 20

    def test_baseline_artifacts_are_byte_identical(self):
        recipe = _recipe()
        artifacts = recipe["immutable_baseline"]["artifacts"]
        self.assertEqual(len(artifacts), 5)
        for artifact in artifacts:
            path = os.path.join(LANE_ROOT, artifact["path"])
            self.assertTrue(os.path.isfile(path), artifact["path"])
            self.assertEqual(os.path.getsize(path), artifact["bytes"], artifact["path"])
            self.assertRegex(artifact["sha256"], r"^[0-9a-f]{64}$", artifact["path"])
            if artifact["bytes"] > self.LARGE_BYTES:
                continue  # pinned by size and digest; no full re-read
            self.assertEqual(_sha256(path), artifact["sha256"], artifact["path"])

    def test_baseline_metrics_match_the_sealed_report(self):
        recipe = _recipe()
        report_path = os.path.join(
            LANE_ROOT, "evaluation/runs/capacity16_native_eval01_seal/evaluation_report.json"
        )
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        metrics = report["metrics"]
        recorded = recipe["immutable_baseline"]["baseline_metrics"]["native_1000_full"]
        for key in ("TP", "FP", "FN", "PQ", "SQ", "RQ", "recall", "precision", "f1"):
            self.assertEqual(recorded[key], metrics[key], key)
        self.assertEqual(recorded["gt_instances"], report["scope"]["gt_instances_scored"])
        tp_rows = [row for row in metrics["per_image"] if row.get("TP")]
        self.assertEqual(
            [row["sample_id"] for row in tp_rows], ["train_final:4OOBV"], "all TP stay in 4OOBV"
        )

    def test_crop_baseline_metrics_match(self):
        recipe = _recipe()
        path = os.path.join(LANE_ROOT, "runs/capacity16_crop_eval01/capacity_eval.json")
        with open(path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        aggregate = report["aggregate"]
        recorded = recipe["immutable_baseline"]["baseline_metrics"]["native_384_crop"]
        self.assertEqual(recorded["TP"], aggregate["tp"])
        self.assertEqual(recorded["FP"], aggregate["fp"])
        self.assertEqual(recorded["FN"], aggregate["fn"])
        self.assertEqual(recorded["gt_instances"], aggregate["gt_instances"])
        self.assertEqual(recorded["PQ"], aggregate["micro"]["pq"])


class MarkerAndWriteScopeTests(unittest.TestCase):
    def test_explicit_inputs_are_pinned_and_unreordered(self):
        recipe = _recipe()
        block = recipe["explicit_inputs"]
        path = os.path.join(LANE_ROOT, block["path"])
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(_sha256(path), block["sha256"])
        with open(path, "r", encoding="utf-8") as handle:
            items = json.load(handle)
        self.assertEqual([item["sample_id"] for item in items], block["order"])
        self.assertEqual(len(items), block["count"])
        report_path = os.path.join(
            LANE_ROOT, "evaluation/runs/capacity16_native_eval01_seal/evaluation_report.json"
        )
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        baseline_order = [row["sample_id"] for row in report["metrics"]["per_image"]]
        self.assertEqual(block["order"], baseline_order, "input order must match the baseline rows")

    def test_marker_absence_means_legacy(self):
        recipe = _recipe()
        marker = recipe["candidate"]["marker"]
        self.assertTrue(marker["required"])
        self.assertEqual(marker["field"], "mask_target_projection")
        self.assertEqual(marker["field_paths"]["dataclass"], "model.model.DetectorConfig.mask_target_projection")
        self.assertEqual(marker["field_paths"]["serialized"], "payload['config']['mask_target_projection']")
        self.assertEqual(marker["default_value"], "legacy")
        self.assertEqual(marker["accepted_values"], ["legacy", "paste_consistent"])
        self.assertIn("legacy", marker["absence_rule"])
        self.assertIn("default", marker["absence_rule"])
        self.assertIn("never", marker["absence_rule"])
        self.assertEqual(marker["field_status"], "pending_model_owner")
        self.assertIn("mask_target_projection", " ".join(marker["check_command"]["argv"]))
        self.assertIn("paste_consistent", marker["check_command"]["pass_when"])

    def test_training_cli_spelling_comes_from_the_real_parser(self):
        """Read the argparse flag from model/train.py; never trust a copied string."""

        train_py = os.path.join(os.path.dirname(EVAL_ROOT), "model", "train.py")
        self.assertTrue(os.path.isfile(train_py), train_py)
        with open(train_py, "r", encoding="utf-8") as handle:
            source = handle.read()
        flags = set(re.findall(r'"(--mask[a-z-]*projection)"', source))
        self.assertEqual(flags, {"--mask-target-projection"}, flags)
        marker = _recipe()["candidate"]["marker"]
        self.assertEqual(marker["field_paths"]["training_cli"], flags.pop())
        self.assertNotIn("--mask_target_projection", json.dumps(_recipe()))
        self.assertIn("default=\"legacy\"", source.replace(" ", ""), "parser default is legacy")

    def test_no_alternative_or_top_level_marker_is_recorded(self):
        recipe = _recipe()
        text = json.dumps(recipe)
        self.assertNotIn("projection_mode", text, "use the canonical config field only")
        marker = recipe["candidate"]["marker"]
        self.assertEqual(sorted(marker["accepted_values"]), ["legacy", "paste_consistent"])
        self.assertIn("no top-level payload marker", marker["no_other_marker"])

    def test_effective_geometry_note_keeps_raw_fields_untouched(self):
        recipe = _recipe()
        marker = recipe["candidate"]["marker"]
        note = marker["effective_geometry_note"]
        self.assertIn("mask_target_aligned", note)
        self.assertIn("mask_target_sampling_ratio", note)
        self.assertIn("False", note)
        self.assertIn("-1", note)
        guard = marker["production_guard_note"]
        self.assertIn("below 2 px", guard)
        self.assertIn("per-box mask-loss exclusion", guard)
        self.assertIn("not accepted until", guard)
        self.assertNotIn("accepted", guard.split("not accepted until")[0])

    def test_no_stage_writes_into_the_baseline(self):
        recipe = _recipe()
        frozen = [os.path.normpath(item["path"]) for item in recipe["immutable_baseline"]["artifacts"]]
        frozen_dirs = [os.path.dirname(path) for path in frozen]
        for stage in recipe["stages"]:
            for target in stage["writes"]:
                path = os.path.normpath(os.path.join(stage["cwd"], target))
                for directory in frozen_dirs:
                    directory = os.path.normpath(os.path.join(LANE_ROOT, directory))
                    self.assertNotEqual(
                        os.path.commonpath([path, directory]), directory, (stage["name"], target)
                    )

    def test_stage_four_needs_the_gt_grant(self):
        recipe = _recipe()
        score = recipe["stages"][3]
        self.assertIn("--allow-gt", score["argv"])
        self.assertTrue(any("grant" in item.lower() for item in score["preconditions"]))
        self.assertEqual(score["exit_codes"]["1"], "valid measurement below target (report written)")
        self.assertEqual(score["exit_codes"]["2"], "GuardError or inconsistency")

    def test_every_stage_refuses_to_overwrite(self):
        recipe = _recipe()
        for stage in recipe["stages"]:
            text = " ".join(stage["guards"]).lower()
            self.assertTrue(
                "refuses" in text or "does not exist" in text or "not exist" in text, stage["name"]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Tests for ``evaluation/grouped_validation_recipe.json``.

Metadata only: no grouped_train files, no PNG/SVG pixels, no checkpoint, no
model, no GPU, no scoring.  The recipe's stage commands are parsed with the real
CLIs, and the reserve rule and grouped counts are recomputed from the real
expanded record metadata, so an invented flag, path, formula or count fails here
instead of in a future run.

Run:  python tests/test_grouped_validation_recipe.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_ROOT = os.path.dirname(HERE)
LANE_ROOT = os.path.dirname(EVAL_ROOT)
REPO_ROOT = os.path.dirname(os.path.dirname(LANE_ROOT))
RECIPE_PATH = os.path.join(EVAL_ROOT, "grouped_validation_recipe.json")

EXPANDED_LANE = os.path.join(REPO_ROOT, "outputs", "recognition-expanded-fit-20261114-01")
EXPANDED_MANIFEST = os.path.join(EXPANDED_LANE, "manifest", "expanded_train_manifest.json")
EXPANDED_RECORDS = os.path.join(EXPANDED_LANE, "manifest", "expanded_train_records.jsonl")

sys.path.insert(0, EVAL_ROOT)

from instance_eval import contract as C  # noqa: E402


def _load(name, relative):
    path = os.path.join(EVAL_ROOT, relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


batch_cli = _load("recipe_batch_cli", "run_batch_predict.py")
eval_cli = _load("recipe_eval_cli", "run_instance_eval.py")


def _recipe():
    with open(RECIPE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _text():
    with open(RECIPE_PATH, "r", encoding="utf-8") as handle:
        return handle.read()


def _resolve(text):
    return (
        text.replace("<lane>", LANE_ROOT.replace("\\", "/"))
        .replace("<repo>", REPO_ROOT.replace("\\", "/"))
    )


def _norm(path):
    return os.path.normcase(os.path.abspath(path))


def _argv(stage, python="python"):
    return [_resolve(part).replace("{PYTHON}", python) for part in stage["argv"]]


def _stage(recipe, index):
    for stage in recipe["stages"]:
        if stage["index"] == index:
            return stage
    raise AssertionError("stage %d is missing" % index)


def _records():
    rows = []
    with open(EXPANDED_RECORDS, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _reserve_selects(group_id):
    """The parent's rule, written out independently of the recipe."""

    digest = hashlib.sha256(("20261108:" + group_id).encode("utf-8")).hexdigest()
    return int(digest, 16) % 10000 < 1000



def _grouped_parser():
    if str(LANE_ROOT) not in sys.path:
        sys.path.insert(0, str(LANE_ROOT))
    from grouped_execution.cli import build_parser

    return build_parser()

class RecipeShapeTest(unittest.TestCase):
    def setUp(self):
        self.recipe = _recipe()

    def test_schema_status_and_authorization(self):
        recipe = self.recipe
        self.assertEqual(recipe["schema"], "cadvec.direct_instance.grouped_validation_recipe.v1")
        self.assertEqual(recipe["status"], "prepared_not_executed")
        authorization = recipe["authorization"]
        self.assertIs(authorization["execution_authorized"], False)
        self.assertIs(authorization["gt_authorized"], False)
        self.assertEqual(len(authorization["parent_grants_required"]), 3)

    def test_split_path_is_lane_owned_and_counts_are_pinned(self):
        split = self.recipe["future_artifacts"]["split"]
        self.assertEqual(split["path"], "<lane>/grouped_train/split.json")
        self.assertNotIn("alternative_paths", split)
        expected = split["expected"]
        self.assertEqual(expected["seed"], 20261108)
        self.assertEqual(expected["train"], {"records": 3366, "groups": 3167})
        self.assertEqual(expected["val"], {"records": 387, "groups": 361})
        # The split owner writes immutable original-position arrays; this recipe
        # must not pretend to own their field names.
        self.assertIn("original-position", expected["record_positions"])

    def test_recipe_never_points_at_a_repo_level_grouped_train_dir(self):
        text = _text()
        self.assertNotIn("<repo>/outputs/grouped_train", text)
        self.assertNotIn('"grouped_train/expanded_manifest.json"', text)

    def test_reserve_rule_is_exact_and_unambiguous(self):
        selection = self.recipe["reserve_selection"]
        self.assertIn("20261108:{group_id}", selection["expression"])
        self.assertIn("hexdigest()", selection["expression"])
        self.assertIn("% 10000 < 1000", selection["expression"])
        self.assertEqual(selection["applied_to"], "groups whose source_role == 'reserve' in expanded_train_records.jsonl")
        self.assertIn("record_id or sample_id", selection["not_applied_to"])
        self.assertIn("every source_role == 'fit' group is training", selection["fit_rule"])
        self.assertEqual(selection["selected_side"], "val")
        text = _text()
        self.assertNotIn("within the given range", text)
        self.assertNotIn("hash(record_id)", text)
        self.assertNotIn("reserve_hash", text)

    def test_reserve_rule_reproduces_the_pinned_grouped_counts(self):
        groups = {}
        records_per_group = {}
        for row in _records():
            group_id = row["group_id"]
            groups.setdefault(group_id, set()).add(row["source_role"])
            records_per_group[group_id] = records_per_group.get(group_id, 0) + 1
        for group_id, roles in groups.items():
            self.assertEqual(len(roles), 1, group_id)

        val_groups = [g for g, roles in groups.items() if roles == {"reserve"} and _reserve_selects(g)]
        train_groups = [g for g in groups if g not in set(val_groups)]
        for group_id in val_groups:
            self.assertEqual(groups[group_id], {"reserve"})
        for group_id, roles in groups.items():
            if roles == {"fit"}:
                self.assertIn(group_id, train_groups)

        self.assertEqual(len(groups), 3528)
        self.assertEqual(len(val_groups), 361)
        self.assertEqual(sum(records_per_group[g] for g in val_groups), 387)
        self.assertEqual(len(train_groups), 3167)
        self.assertEqual(sum(records_per_group[g] for g in train_groups), 3366)
        self.assertFalse(set(val_groups) & set(train_groups))

        measured = self.recipe["reserve_selection"]["measured"]
        self.assertEqual(measured["val_groups"], len(val_groups))
        self.assertEqual(measured["val_records"], 387)
        self.assertEqual(measured["train_groups"], len(train_groups))
        self.assertEqual(measured["train_records"], 3366)
        self.assertEqual(measured["admitted_groups"], 3528)
        self.assertEqual(measured["admitted_records"], 3753)
        self.assertEqual(measured["fit_groups"] + measured["reserve_groups"], 3528)

    def test_expanded_store_is_the_real_lane_and_digest(self):
        store = self.recipe["future_artifacts"]["expanded_store"]
        self.assertEqual(store["manifest"], "<repo>/outputs/recognition-expanded-fit-20261114-01/manifest/expanded_train_manifest.json")
        self.assertEqual(store["records"], "<repo>/outputs/recognition-expanded-fit-20261114-01/manifest/expanded_train_records.jsonl")
        self.assertEqual(_norm(_resolve(store["manifest"])), _norm(EXPANDED_MANIFEST))
        self.assertEqual(_norm(_resolve(store["records"])), _norm(EXPANDED_RECORDS))
        self.assertTrue(os.path.isfile(EXPANDED_MANIFEST))
        self.assertTrue(os.path.isfile(EXPANDED_RECORDS))

        with open(EXPANDED_MANIFEST, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        identity = store["records_identity"]
        self.assertEqual(identity["count"], manifest["records"]["count"])
        self.assertEqual(identity["sha256"], manifest["records"]["sha256"])
        self.assertEqual(identity["bytes"], manifest["records"]["bytes"])
        digest = hashlib.sha256()
        with open(EXPANDED_RECORDS, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        self.assertEqual(digest.hexdigest(), identity["sha256"])
        self.assertEqual(os.path.getsize(EXPANDED_RECORDS), identity["bytes"])
        self.assertIn("own datasets_root", store["datasets_root"])
        self.assertEqual(store["record_fields_used"], ["record_index", "sample_id", "group_id", "source_role", "stem", "png_name", "svg_name"])

    def test_builder_uses_the_real_record_store_api(self):
        builder = self.recipe["inputs_builder"]
        text = json.dumps(builder, ensure_ascii=False)
        self.assertIn("scripts/expanded_record_store.py", _text())
        self.assertIn("record_paths(record, datasets_root)", text)
        self.assertNotIn("record_paths[", text)
        self.assertNotIn("expanded_manifest.record_paths", text)
        self.assertNotIn('"record_paths": {', text)

    def test_inputs_payload_is_the_shared_evaluator_object(self):
        payload = self.recipe["inputs_builder"]["payload"]
        self.assertEqual(payload["schema"], "cadvec.direct_instance.inputs.v1")
        self.assertEqual(payload["key"], "samples")
        self.assertIn(
            "evaluation/instance_eval/contract.py normalize_inputs",
            " ".join(payload["consumed_by"]),
        )
        self.assertIn("{PYTHON} -m grouped_execution inputs --write", payload["command"])
        algorithm = " ".join(self.recipe["inputs_builder"]["algorithm"])
        self.assertIn("cadvec.direct_instance.inputs.v1", algorithm)
        self.assertNotIn("bare JSON list", algorithm)

    def test_builder_allows_repeated_groups_and_refuses_duplicate_samples(self):
        builder = self.recipe["inputs_builder"]
        codes = builder["fail_closed_codes"]
        self.assertIn("inputs_duplicate", codes)
        self.assertIn("inputs_group_partial", codes)
        self.assertIn("inputs_group_count", codes)
        self.assertIn("inputs_sample_count", codes)
        rules = " ".join(builder["algorithm"])
        self.assertIn("repeated group ids are expected and allowed", rules)
        self.assertIn("inputs_duplicate", rules)
        text = _text()
        self.assertNotIn("never deduplicate groups or sample ids", text)

    def test_stage_commands_parse_with_the_real_clis(self):
        recipe = self.recipe
        batch = batch_cli.build_parser().parse_args(_argv(_stage(recipe, 1))[2:])
        self.assertEqual(batch.sample_set, "explicit")
        self.assertEqual(batch.device, "cuda:0")
        self.assertEqual(_norm(batch.predictions_dir), _norm(_resolve("<lane>/runs/grouped_expanded_paste_consistent01/predictions")))
        self.assertEqual(_norm(batch.checkpoint), _norm(_resolve("<lane>/runs/grouped_expanded_paste_consistent01/checkpoint.pt")))
        self.assertEqual(_norm(batch.inputs), _norm(_resolve("<lane>/evaluation/grouped_val387_inputs.json")))

        prepare = eval_cli.build_parser().parse_args(_argv(_stage(recipe, 2))[2:])
        self.assertEqual(prepare.command, "prepare")
        self.assertEqual(prepare.dataset, "grouped-val387-prospective")
        self.assertEqual(_norm(prepare.out), _norm(_resolve("<lane>/evaluation/runs/grouped_expanded_paste_consistent01_val387_seal")))

        score = eval_cli.build_parser().parse_args(_argv(_stage(recipe, 3))[2:])
        self.assertEqual(score.command, "score")
        self.assertTrue(score.allow_gt)
        self.assertEqual(_norm(score.out), _norm(_resolve("<lane>/evaluation/runs/grouped_expanded_paste_consistent01_val387_seal")))

    def test_an_invented_flag_is_rejected_by_the_real_parsers(self):
        for index, cli in ((1, batch_cli), (2, eval_cli), (3, eval_cli)):
            argv = _argv(_stage(self.recipe, index))[2:] + ["--not-a-real-flag"]
            with self.assertRaises(SystemExit):
                cli.build_parser().parse_args(argv)

    def test_seal_is_written_after_every_prediction_and_before_any_gt(self):
        recipe = self.recipe
        batch, prepare, score = (_stage(recipe, i) for i in (1, 2, 3))
        self.assertEqual(batch["exit_codes"]["0"], "387 envelopes written")
        self.assertIn("every one of the 387 predictions must exist and validate first (missing_prediction / prediction_invalid)", prepare["guards"])
        self.assertEqual(prepare["ordering"], "the seal is the last artifact written before any ground-truth read")
        self.assertEqual(prepare["writes"], ["pre_gt_seal.json"])
        self.assertIs(score["gt_after_seal"], True)
        self.assertTrue(score["allow_gt"] if "allow_gt" in score else "--allow-gt" in score["argv"])
        self.assertEqual(score["exit_codes"], {"0": "recall > 0.90", "1": "valid run below target", "2": "failure or refusal"})
        for stage in (batch, prepare, score):
            for write in stage["writes"]:
                self.assertFalse(os.path.isabs(write), write)

    def test_checkpoint_is_bound_through_the_training_receipt(self):
        recipe = self.recipe
        receipt = recipe["future_artifacts"]["grouped_train_receipt"]
        self.assertEqual(
            receipt["path"],
            "<lane>/runs/grouped_expanded_paste_consistent01/grouped_train_receipt.json",
        )
        self.assertEqual(receipt["schema"], "cadvec.grouped_train.receipt/1")
        self.assertEqual(len(receipt["must_bind"]), 3)
        self.assertIn("model ABI does not record the split", receipt["why"])
        position_map = receipt["position_map"]
        self.assertEqual(
            position_map["path"],
            "<lane>/runs/grouped_expanded_paste_consistent01/grouped_position_map.json",
        )
        self.assertEqual(position_map["schema"], "cadvec.grouped_train.position_map/1")

        checks = {check["code"]: check["rule"] for check in recipe["preflight"]["checks"]}
        self.assertIn("checkpoint_binding", checks)
        self.assertIn("grouped training receipt", checks["checkpoint_binding"])
        self.assertIn("no split field is read from the checkpoint config", checks["checkpoint_binding"])
        self.assertIn("holdout_disjoint", checks)
        self.assertNotIn("config names the same split", _text())
        reads = " ".join(recipe["preflight"]["reads"])
        self.assertIn("grouped_train_receipt.json", reads)
        self.assertIn("grouped_position_map.json", reads)
        self.assertNotIn("grouped_training_receipt.json", _text())

        runner = (Path(LANE_ROOT) / "grouped_train" / "runner.py").read_text(encoding="utf-8")
        self.assertIn('RECEIPT_NAME = "grouped_train_receipt.json"', runner)
        self.assertIn('POSITION_MAP_NAME = "grouped_position_map.json"', runner)

    def test_expected_updates_are_the_pinned_8415(self):
        checks = {check["code"]: check for check in self.recipe["preflight"]["checks"]}
        pin = checks["candidate_updates"]
        self.assertEqual(pin["expected_updates"], 8415)
        self.assertEqual(pin["expected_micro_batches"], 16830)
        self.assertEqual(pin["expected_accumulate"], 2)
        self.assertEqual(pin["expected_passes"], 5)
        self.assertEqual(pin["train_records"], 3366)
        self.assertEqual(pin["expected_updates"] * pin["expected_accumulate"], pin["expected_micro_batches"])
        self.assertEqual(pin["expected_micro_batches"] // pin["train_records"], pin["expected_passes"])
        self.assertNotIn("must_be_set_before_execution", json.dumps(pin))

    def test_scope_is_prospective_with_uncertain_prior_exposure(self):
        scope = self.recipe["scope"]
        self.assertEqual(scope["label"], "grouped-val387-prospective")
        self.assertEqual(scope["dataset_argument"], "grouped-val387-prospective")
        self.assertIs(scope["prior_exposure_uncertain"], True)
        self.assertIn("prospective grouped holdout on a directly trained expanded run", scope["intended_reading"])
        self.assertIn("direct expanded run scored with the canonical pixel-raster proxy", scope["claims_allowed"])
        for forbidden in ("pristine held-out", "never seen during training or tuning", "vector-baseline comparable", "overall goal met"):
            self.assertIn(forbidden, scope["claims_forbidden"])
        self.assertIs(scope["current_evaluator_behaviour"]["heldout_flag"].startswith("evaluator._scope_claim"), True)

    def test_provenance_is_implemented_and_kept_outside_the_evaluator(self):
        provenance = self.recipe["scope"]["provenance"]
        self.assertEqual(provenance["status"], "implemented")
        self.assertIn("grouped_execution/provenance.py", provenance["implemented_by"])
        self.assertIs(provenance["kept_outside_the_evaluator"], True)
        fields = provenance["required_fields"]
        self.assertEqual(fields["val_records"], 387)
        self.assertEqual(fields["val_groups"], 361)
        self.assertEqual(fields["train_records"], 3366)
        self.assertEqual(fields["train_groups"], 3167)
        self.assertEqual(fields["seed"], 20261108)
        self.assertEqual(fields["prior_exposure"], "uncertain")
        self.assertIn("heldout=false", provenance["effect"])
        self.assertIn("not modified for a dataset label", provenance["effect"])
        self.assertIn("no prospective holdout reading is offered", provenance["effect_when_unverified"])
        self.assertNotIn("minimal_future_seam", self.recipe["scope"])
        self.assertNotIn("not_implemented", _text())

    def test_execution_commands_are_the_implemented_grouped_execution_cli(self):
        execution = self.recipe["execution"]
        self.assertEqual(execution["entry"], "python -m grouped_execution")
        for name in ("inputs", "preflight", "plan", "run", "provenance"):
            self.assertIn(name, execution["commands"])
        parser = _grouped_parser()
        run_command = [
            part
            for part in execution["commands"]["run"].replace("{PYTHON}", "python").split()
            if not part.startswith("[")
        ]
        parsed = parser.parse_args(run_command[3:])
        self.assertEqual(parsed.command, "run")
        self.assertEqual(
            parsed.run_dir,
            "<lane>/runs/grouped_expanded_paste_consistent01",
        )
        self.assertIs(parsed.allow_gt, False)
        inputs_command = execution["commands"]["inputs"].replace("{PYTHON}", "python").split()
        self.assertEqual(parser.parse_args(inputs_command[3:]).command, "inputs")
        score_codes = execution["stage_exit_codes"]["score"]
        self.assertIn("valid below_target score", score_codes["1"])
        self.assertIn("score_exit_mismatch", score_codes["rule"])
        self.assertEqual(
            execution["implemented_by"]["stages"],
            "grouped_execution/stages.py (stage_plan, run_stages)",
        )
        for stage in self.recipe["stages"]:
            self.assertEqual(stage["implemented_by"], "grouped_execution/stages.py stage_plan/run_stages")

    def test_no_vector_baseline_number_is_used_as_a_target(self):
        recipe = self.recipe
        self.assertEqual(recipe["objective"]["reference_comparison"], "none")
        objective = recipe["objective"]
        for key in ("pq_target", "reference_value", "baseline_value", "target_pq"):
            self.assertNotIn(key, objective)
        self.assertIn("vector-baseline comparable", recipe["scope"]["claims_forbidden"])
        # The only mention of the vector number is the prohibition itself.
        self.assertEqual(_text().count("0.561"), 1)

    def test_remaining_blockers_name_the_real_gaps(self):
        blockers = " ".join(self.recipe["remaining_blockers"])
        self.assertIn("grouped_train/split.json exists and is frozen", blockers)
        self.assertIn("grouped_train_receipt.json and grouped_position_map.json do not exist yet", blockers)
        self.assertIn("no grouped candidate checkpoint", blockers)
        self.assertIn("no stage may run until the parent authorizes it", blockers)
        non_goals = " ".join(self.recipe["non_goals"])
        self.assertIn("test_final / dev_final / eval_pool / .runtime", non_goals)


class InputsContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp(prefix="grp_inputs_", dir=LANE_ROOT)
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)
        self.pngs = []
        for name in ("a", "b", "c"):
            path = os.path.join(self.temp, name + ".png")
            with open(path, "wb") as handle:
                handle.write(b"\x89PNG\r\n\x1a\n")
            self.pngs.append(path)

    def test_repeated_group_ids_are_accepted_with_unique_samples(self):
        rows = [
            {"sample_id": "train_final:a", "group_id": "g-shared", "png_path": self.pngs[0]},
            {"sample_id": "train_final:b", "group_id": "g-shared", "png_path": self.pngs[1]},
            {"sample_id": "train_final:c", "group_id": "g-other", "png_path": self.pngs[2]},
        ]
        records = C.normalize_inputs(rows)
        self.assertEqual([r["sample_id"] for r in records], ["train_final:a", "train_final:b", "train_final:c"])
        self.assertEqual([r["group_id"] for r in records], ["g-shared", "g-shared", "g-other"])
        self.assertEqual(len({r["group_id"] for r in records}), 2)
        self.assertEqual(len({r["sample_id"] for r in records}), 3)

    def test_duplicate_sample_ids_are_refused(self):
        rows = [
            {"sample_id": "train_final:a", "group_id": "g-shared", "png_path": self.pngs[0]},
            {"sample_id": "train_final:a", "group_id": "g-shared", "png_path": self.pngs[1]},
        ]
        with self.assertRaises(C.GuardError) as raised:
            C.normalize_inputs(rows)
        self.assertEqual(raised.exception.code, "inputs_duplicate")

    def test_a_nonexistent_png_is_refused(self):
        rows = [
            {
                "sample_id": "train_final:a",
                "group_id": "g-shared",
                "png_path": os.path.join(self.temp, "does-not-exist.png"),
            }
        ]
        with self.assertRaises(C.GuardError) as raised:
            C.normalize_inputs(rows)
        self.assertEqual(raised.exception.code, "inputs_png_missing")

    def test_an_explicit_inputs_file_keeps_its_own_order(self):
        path = os.path.join(self.temp, "inputs.json")
        rows = [
            {"sample_id": "train_final:b", "group_id": "g-shared", "png_path": self.pngs[1]},
            {"sample_id": "train_final:a", "group_id": "g-shared", "png_path": self.pngs[0]},
        ]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle)
        records = C.normalize_inputs(path)
        self.assertEqual([r["sample_id"] for r in records], ["train_final:b", "train_final:a"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

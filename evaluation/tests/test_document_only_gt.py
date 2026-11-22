"""Synthetic tests for the document-only default GT builder.

Run:  python tests/test_document_only_gt.py -v     (stdlib unittest)

Scoring consumes ``gt_document`` only, so the default builder must call
``raster_data.document_from_svg(svg, curve_steps=128, split=None)`` and never
decode the PNG, never build dense training targets, never call ``build_sample``.
These tests pin the exact call, the guard order (seal before GT), the missing /
sealed path refusals, and document parity with the old ``build_sample`` path on
a tiny synthetic PNG + SVG that covers lines, an arc, a circle, an unknown
instance and an ignored semantic id.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_ROOT = os.path.dirname(HERE)
sys.path.insert(0, EVAL_ROOT)

from instance_eval import contract as C  # noqa: E402
from instance_eval import evaluator as E  # noqa: E402

PNG_SIZE = (100, 100)

SYNTHETIC_SVG = """<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     width="100" height="100" viewBox="0 0 100 100">
  <g id="layer1" inkscape:label="synthetic" inkscape:groupmode="layer">
    <path d="M 10 10 L 40 10" semantic-id="1" instance-id="1" stroke="rgb(200,30,40)" stroke-width="1"/>
    <path d="M 10 20 A 10 10 0 0 1 30 20" semantic-id="2" instance-id="2" stroke="rgb(10,220,30)"/>
    <circle cx="60" cy="60" r="8" semantic-id="3" instance-id="3" stroke="rgb(30,40,200)"/>
    <path d="M 50 80 L 80 80" semantic-id="4" stroke="rgb(90,90,90)" stroke-width="2"/>
    <path d="M 60 10 L 90 10" semantic-id="35" stroke="rgb(0,0,0)"/>
  </g>
</svg>
"""

# Only labelled, instance-carrying primitives: legal as a prediction document.
CLEAN_SVG = """<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     width="100" height="100" viewBox="0 0 100 100">
  <g id="layer1" inkscape:label="clean" inkscape:groupmode="layer">
    <path d="M 10 10 L 40 10" semantic-id="1" instance-id="1" stroke="rgb(200,30,40)"/>
    <path d="M 10 20 A 10 10 0 0 1 30 20" semantic-id="2" instance-id="2" stroke="rgb(10,220,30)"/>
    <circle cx="60" cy="60" r="8" semantic-id="3" instance-id="3" stroke="rgb(30,40,200)"/>
  </g>
</svg>
"""

DENSE_FUNCTIONS = (
    "build_sample",
    "_load_image",
    "_image_ink_support",
    "_dense_primitive_records",
    "_dense_label_entries",
    "_dense_segment_masks",
)


def write_png(path: str) -> str:
    """A tiny real PNG; still never opened by the document-only GT path."""

    numpy = C.require_numpy()
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    array = numpy.full((PNG_SIZE[1], PNG_SIZE[0], 3), 255, dtype=numpy.uint8)
    array[10:21, 10:41] = 0
    Image.fromarray(array, mode="RGB").save(path)
    return path


def write_text(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path


def write_json(path: str, payload) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def raising_data():
    """A stand-in raster_data whose every entry point is a tripwire."""

    class Tripwire:
        def __getattr__(self, name):
            def trip(*_args, **_kwargs):
                raise AssertionError("raster_data.%s must not be called" % name)

            return trip

    return Tripwire()


class TempCase(unittest.TestCase):
    def setUp(self) -> None:
        self.work = tempfile.mkdtemp(prefix="doc_only_gt_", dir=C.LANE_ROOT)
        self.addCleanup(shutil.rmtree, self.work, True)

    def path(self, *parts: str) -> str:
        return os.path.join(self.work, *parts)


# --------------------------------------------------------------------------- #
# the default builder calls exactly the document converter
# --------------------------------------------------------------------------- #
class DefaultBuilderCallTests(TempCase):
    def test_default_builder_calls_document_converter_with_exact_args(self):
        calls: list[tuple[str, dict]] = []
        sentinel = {"canonical": "gt"}

        class Data:
            def document_from_svg(self, path, **kwargs):
                calls.append((path, kwargs))
                return sentinel

            def __getattr__(self, name):
                raise AssertionError("unexpected raster_data.%s" % name)

        svg = self.path("synthetic.svg")
        png = self.path("synthetic.png")
        record = {
            "sample_id": "train_final:SYN1",
            "group_id": "g-syn",
            "png_path": png,
            "svg_path": svg,
        }
        with mock.patch.object(C, "raster_data", lambda: Data()):
            builder = E._default_gt_builder()
            result = builder(record)

        self.assertEqual(calls, [(str(svg), {"curve_steps": 128, "split": None})])
        self.assertEqual(result, {"gt_document": sentinel})
        self.assertEqual(sorted(result), ["gt_document"])

    def test_default_builder_never_loads_the_png_or_dense_targets(self):
        class Data:
            document_calls = 0

            def document_from_svg(self, path, **_kwargs):
                Data.document_calls += 1
                return {"commands": []}

            def __getattr__(self, name):
                raise AssertionError("unexpected raster_data.%s" % name)

        trips = []

        def trip(name):
            def fail(*_args, **_kwargs):
                trips.append(name)
                raise AssertionError("raster_data.%s must not be called" % name)

            return fail

        record = {
            "sample_id": "train_final:SYN1",
            "png_path": self.path("synthetic.png"),
            "svg_path": self.path("synthetic.svg"),
        }
        with mock.patch.object(C, "raster_data", lambda: Data()):
            builder = E._default_gt_builder()
            for name in DENSE_FUNCTIONS:
                patcher = mock.patch.object(Data, name, trip(name), create=True)
                patcher.start()
                self.addCleanup(patcher.stop)
            document = builder(record)["gt_document"]

        self.assertEqual(trips, [])
        self.assertEqual(Data.document_calls, 1)
        self.assertEqual(document, {"commands": []})


# --------------------------------------------------------------------------- #
# guard order and path refusals stay exactly as before
# --------------------------------------------------------------------------- #
class GuardTests(TempCase):
    def test_missing_svg_path_is_gt_source_missing(self):
        with mock.patch.object(C, "raster_data", raising_data):
            builder = E._default_gt_builder()
            with self.assertRaises(C.GuardError) as caught:
                builder(
                    {
                        "sample_id": "train_final:SYN1",
                        "png_path": self.path("synthetic.png"),
                        "svg_path": None,
                    }
                )
        self.assertEqual(caught.exception.code, "gt_source_missing")
        self.assertIn("train_final:SYN1", caught.exception.detail)

    def test_sealed_svg_path_is_refused_before_any_read(self):
        with mock.patch.object(C, "raster_data", raising_data):
            builder = E._default_gt_builder()
            with self.assertRaises(C.GuardError) as caught:
                builder(
                    {
                        "sample_id": "train_final:SYN1",
                        "png_path": self.path("synthetic.png"),
                        "svg_path": self.path("test_final", "SYN1.svg"),
                    }
                )
        self.assertEqual(caught.exception.code, "sealed_path")
        self.assertIn("svg_path", caught.exception.detail)

    def test_sealed_png_path_is_refused_before_any_read(self):
        with mock.patch.object(C, "raster_data", raising_data):
            builder = E._default_gt_builder()
            with self.assertRaises(C.GuardError) as caught:
                builder(
                    {
                        "sample_id": "train_final:SYN1",
                        "png_path": self.path("eval_pool", "SYN1.png"),
                        "svg_path": self.path("SYN1.svg"),
                    }
                )
        self.assertEqual(caught.exception.code, "sealed_path")
        self.assertIn("png_path", caught.exception.detail)


# --------------------------------------------------------------------------- #
# synthetic document parity with the old build_sample path
# --------------------------------------------------------------------------- #
class SyntheticSvgParityTests(TempCase):
    def test_document_only_gt_equals_build_sample_gt_document(self):
        data = C.raster_data()
        png = write_png(self.path("fixtures", "synthetic.png"))
        svg = write_text(self.path("fixtures", "synthetic.svg"), SYNTHETIC_SVG)

        record = {
            "sample_id": "train_final:SYN1",
            "png_path": png,
            "svg_path": svg,
        }
        document_only = E._default_gt_builder()(record)["gt_document"]
        legacy = data.build_sample(png, svg, record["sample_id"])["gt_document"]

        self.assertEqual(set(document_only), set(C.DOCUMENT_FIELDS))
        for field in C.DOCUMENT_FIELDS:
            self.assertEqual(
                C.canonical_json(document_only[field]),
                C.canonical_json(legacy[field]),
                "field %r differs between document_from_svg and build_sample" % field,
            )
        self.assertEqual(C.canonical_json(document_only), C.canonical_json(legacy))
        C.validate_gt_document(document_only, record["sample_id"])

        # The fixture really exercises the interesting cases.
        self.assertEqual(len(document_only["commands"]), 5)
        self.assertIn(2, document_only["commands"])  # circle
        self.assertIn(1, document_only["commands"])  # arc path
        self.assertIn(-1, document_only["instanceIds"])  # unlabelled instance
        self.assertIn(34, document_only["semanticIds"])  # semantic-id 35 -> ignore


# --------------------------------------------------------------------------- #
# end to end: prepare then score with the real default builder
# --------------------------------------------------------------------------- #
class DocumentOnlyEndToEndTests(TempCase):
    def inputs(self, svg: str, png: str) -> list[dict]:
        return [
            {
                "sample_id": "train_final:SYN1",
                "group_id": "g-syn",
                "png_path": png,
                "svg_path": svg,
            }
        ]

    def test_score_uses_svg_only_and_still_seals_before_gt(self):
        data = C.raster_data()
        png = write_png(self.path("fixtures", "SYN1.png"))
        svg = write_text(self.path("fixtures", "SYN1.svg"), CLEAN_SVG)
        document = data.document_from_svg(svg, curve_steps=128, split=None)

        predictions = self.path("predictions")
        write_json(os.path.join(predictions, C.slug("train_final:SYN1") + ".json"), document)
        out = self.path("seal")
        E.prepare(predictions, out, inputs=self.inputs(svg, png))

        trace: list[tuple[str, str]] = []
        with mock.patch.object(data, "build_sample", side_effect=AssertionError("build_sample")), \
                mock.patch.object(data, "_load_image", side_effect=AssertionError("_load_image")), \
                mock.patch.object(
                    data, "_dense_primitive_records", side_effect=AssertionError("dense")
                ), \
                mock.patch.object(
                    data, "_dense_label_entries", side_effect=AssertionError("dense")
                ), \
                mock.patch.object(
                    data, "_dense_segment_masks", side_effect=AssertionError("dense")
                ):
            report = E.score(out, allow_gt=True, trace=trace)

        self.assertAlmostEqual(report["metrics"]["PQ"], 1.0, places=9)
        self.assertAlmostEqual(report["metrics"]["recall"], 1.0, places=9)
        names = [name for name, _ in trace]
        self.assertIn("seal_verified", names)
        self.assertIn("gt_built", names)
        self.assertLess(names.index("seal_verified"), names.index("gt_built"))
        self.assertEqual(names.count("gt_built"), 1)

    def test_score_refuses_a_missing_seal_before_building_gt(self):
        out = self.path("empty")
        os.makedirs(out, exist_ok=True)
        with mock.patch.object(C, "raster_data", raising_data):
            with self.assertRaises(C.GuardError) as caught:
                E.score(out, allow_gt=True)
        self.assertEqual(caught.exception.code, "seal_missing")

    def test_score_without_grant_never_builds_gt(self):
        with mock.patch.object(C, "raster_data", raising_data):
            with self.assertRaises(C.GuardError) as caught:
                E.score(self.path("seal"), allow_gt=False)
        self.assertEqual(caught.exception.code, "gt_not_authorized")


if __name__ == "__main__":
    unittest.main(verbosity=2)

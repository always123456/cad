"""Focused contract tests for ``targets.py`` on synthetic images/documents.

No dataset, PNG or GT file is opened: every image is an in-memory array and
every document is built here.  The only external dependency is the project
rasterizer itself, used as the parity oracle.
"""

from __future__ import annotations

import unittest

import numpy as np

try:  # ``python -m unittest targets.test_targets`` style
    from . import targets as contract
except ImportError:  # ``python -m unittest test_targets`` inside this folder
    import targets as contract

RASTER = contract.project_raster()
CANVAS = (100.0, 100.0)


# --------------------------------------------------------------------------- #
# synthetic helpers
# --------------------------------------------------------------------------- #


def _doc(primitives, width=100, height=100):
    commands, args, lengths, semantic, instance = [], [], [], [], []
    for command, points, semantic_id, instance_id in primitives:
        commands.append(command)
        flat = []
        for x, y in points:
            flat.extend((float(x), float(y)))
        args.append(flat)
        lengths.append(0.0)
        semantic.append(semantic_id)
        instance.append(instance_id)
    return {
        "commands": commands,
        "args": args,
        "lengths": lengths,
        "semanticIds": semantic,
        "instanceIds": instance,
        "width": width,
        "height": height,
    }


def _line_points(x0, y0, x1, y1):
    return [
        (
            x0 + (x1 - x0) * t,
            y0 + (y1 - y0) * t,
        )
        for t in (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
    ]


def _circle_points(cx, cy, rx, ry=None):
    ry = rx if ry is None else ry
    return [(cx - rx, cy), (cx, cy - ry), (cx + rx, cy), (cx, cy + ry)]


def _blank(height=100, width=100):
    return np.ones((3, height, width), dtype=np.float32)


def _paint_line(image, x0, y0, x1, y1, value=0.0):
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for t in np.linspace(0.0, 1.0, steps + 1):
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        image[:, y, x] = value
    return image


def _project_masks(document, width, height):
    segments = RASTER.rasterize_document(
        document, raster_size=(width, height), canvas_size=CANVAS
    )
    return {
        (int(segment["class_id"]), int(segment["instance_id"])): np.asarray(
            segment["mask"], dtype=bool
        )
        for segment in segments
    }


def _target_masks(target):
    return {
        key: target["masks"][index]
        for index, key in enumerate(target["keys"])
    }


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #


class ContractTest(unittest.TestCase):
    def test_line_targets_match_project_rasterizer(self):
        document = _doc(
            [
                (0, _line_points(10, 10, 20, 10), 0, 0),
                (0, _line_points(40, 40, 55, 55), 7, 3),
                (0, _line_points(80, 20, 80, 60), 29, 0),
            ]
        )
        target = contract.build_targets(_blank(), document)
        expected = _project_masks(document, 100, 100)

        self.assertEqual(target["keys"], tuple(sorted(expected)))
        self.assertEqual(target["masks"].dtype, np.bool_)
        self.assertEqual(target["masks"].shape, (3, 100, 100))
        for key, mask in _target_masks(target).items():
            np.testing.assert_array_equal(mask, expected[key])

        horizontal = _target_masks(target)[(0, 0)]
        self.assertTrue(horizontal[10, 10])
        self.assertTrue(horizontal[10, 20])
        self.assertTrue(horizontal[10, 15])
        self.assertFalse(horizontal[11, 15])
        self.assertFalse(horizontal[10, 25])

    def test_circle_and_ellipse_match_project_rasterizer(self):
        document = _doc(
            [
                (2, _circle_points(50, 50, 10), 2, 0),
                (3, _circle_points(30, 30, 12, 6), 30, 4),
            ]
        )
        target = contract.build_targets(_blank(), document)
        expected = _project_masks(document, 100, 100)

        self.assertEqual(target["keys"], ((2, 0), (30, -1)))
        for key, mask in _target_masks(target).items():
            np.testing.assert_array_equal(mask, expected[key])

        circle = _target_masks(target)[(2, 0)]
        self.assertTrue(circle[50, 40])
        self.assertTrue(circle[40, 50])
        self.assertTrue(circle[50, 60])
        self.assertTrue(circle[60, 50])
        self.assertFalse(circle[50, 50])
        self.assertFalse(circle[50, 29])

    def test_thing_missing_instance_is_skipped_and_unknown(self):
        image = _blank()
        _paint_line(image, 10, 10, 20, 10)
        document = _doc([(0, _line_points(10, 10, 20, 10), 5, -1)])
        target = contract.build_targets(image, document)

        self.assertEqual(target["keys"], ())
        self.assertEqual(target["masks"].shape, (0, 100, 100))
        self.assertEqual(target["boxes"].shape, (0, 4))
        self.assertFalse(target["loss_domain"][10, 15])
        self.assertTrue(target["ink"][10, 15])
        diagnostics = target["diagnostics"]
        self.assertEqual(diagnostics["skipped_thing_missing_instance_primitives"], 1)
        self.assertEqual(diagnostics["target_count"], 0)
        self.assertGreaterEqual(
            diagnostics["unknown_ink_from_skipped_geometry_pixels"], 9
        )
        self.assertEqual(
            diagnostics["unknown_ink_pixels"],
            diagnostics["unknown_ink_from_skipped_geometry_pixels"]
            + diagnostics["unknown_ink_residual_pixels"],
        )

    def test_semantic_35_is_ignored(self):
        image = _blank()
        _paint_line(image, 60, 60, 70, 60)
        document = _doc(
            [
                (0, _line_points(60, 60, 70, 60), 35, 2),
                (0, _line_points(10, 40, 20, 40), 4, 0),
            ]
        )
        target = contract.build_targets(image, document)

        self.assertEqual(target["keys"], ((4, 0),))
        self.assertEqual(target["diagnostics"]["ignored_semantic35_primitives"], 1)
        self.assertFalse(target["loss_domain"][60, 65])
        self.assertTrue(target["loss_domain"][40, 15])
        self.assertEqual(target["labels"][40, 15], 5)
        self.assertEqual(target["labels"][60, 65], 0)

    def test_stuff_merged_per_class(self):
        document = _doc(
            [
                (0, _line_points(10, 20, 30, 20), 30, 3),
                (0, _line_points(10, 40, 30, 40), 30, 7),
                (0, _line_points(10, 60, 30, 60), 31, 1),
            ]
        )
        target = contract.build_targets(_blank(), document)
        masks = _target_masks(target)

        self.assertEqual(target["keys"], ((30, -1), (31, -1)))
        self.assertEqual(len(masks), 2)
        merged = masks[(30, -1)]
        self.assertTrue(merged[20, 20])
        self.assertTrue(merged[40, 20])
        self.assertFalse(merged[60, 20])
        expected = _project_masks(document, 100, 100)
        np.testing.assert_array_equal(merged, expected[(30, -1)])

    def test_loss_domain_excludes_unknown_ink_only(self):
        image = _blank()
        _paint_line(image, 10, 30, 30, 30)   # known target ink
        _paint_line(image, 10, 60, 30, 60)   # semantic-35 ink
        document = _doc(
            [
                (0, _line_points(10, 30, 30, 30), 6, 1),
                (0, _line_points(10, 60, 30, 60), 35, 0),
            ]
        )
        target = contract.build_targets(image, document)
        domain = target["loss_domain"]
        diagnostics = target["diagnostics"]

        self.assertTrue(domain[30, 20])
        self.assertFalse(domain[60, 20])
        self.assertTrue(domain[90, 90])
        union = target["masks"].any(axis=0)
        expected_blank = ~target["ink"] & ~target["unknown_geometry"] & ~union
        expected = union | expected_blank
        self.assertTrue(np.array_equal(domain, expected))
        self.assertTrue(np.array_equal(target["blank_negative"], expected_blank))
        self.assertFalse(bool((target["blank_negative"] & union).any()))
        self.assertFalse(bool((target["blank_negative"] & target["unknown_geometry"]).any()))
        self.assertFalse(bool(target["blank_negative"][60, 20]))
        self.assertTrue(bool(target["unknown_geometry"][60, 20]))
        self.assertEqual(int(domain.sum()), int(expected.sum()))
        self.assertEqual(diagnostics["known_ink_pixels"], target["masks"][0].sum())

    def test_labels_are_canonical_plus_one_and_disjoint(self):
        document = _doc(
            [
                (0, _line_points(10, 10, 30, 10), 0, 0),
                (0, _line_points(10, 70, 30, 70), 34, 5),
            ]
        )
        target = contract.build_targets(_blank(), document)
        labels = target["labels"]
        first, second = target["masks"]

        self.assertEqual(labels.dtype, np.int64)
        self.assertEqual(target["label_offset"], 1)
        self.assertTrue(np.all(labels[first] == 1))
        self.assertTrue(np.all(labels[second] == 35))
        self.assertTrue(np.all(labels[~(first | second)] == 0))
        self.assertFalse(np.any(first & second))
        self.assertEqual(target["diagnostics"]["overlap_pixels"], 0)

    def test_boxes_are_exclusive_mask_extents(self):
        document = _doc([(0, _line_points(10, 20, 19, 29), 3, 2)])
        target = contract.build_targets(_blank(), document)
        mask = target["masks"][0]
        box = target["boxes"][0]

        self.assertEqual(target["boxes"].dtype, np.float32)
        self.assertEqual(box.tolist(), [10.0, 20.0, 20.0, 30.0])
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        self.assertEqual(box.tolist(), [
            float(cols[0]),
            float(rows[0]),
            float(cols[-1] + 1),
            float(rows[-1] + 1),
        ])

    def test_empty_document_is_deterministic(self):
        image = _blank()
        target = contract.build_targets(image, _doc([]))

        self.assertEqual(target["keys"], ())
        self.assertEqual(target["masks"].shape, (0, 100, 100))
        self.assertEqual(target["boxes"].shape, (0, 4))
        self.assertEqual(target["labels"].sum(), 0)
        self.assertTrue(target["loss_domain"].all())
        diagnostics = target["diagnostics"]
        self.assertEqual(diagnostics["primitive_count"], 0)
        self.assertEqual(diagnostics["target_count"], 0)
        self.assertEqual(diagnostics["ink_pixels"], 0)
        self.assertEqual(diagnostics["unknown_ink_pixels"], 0)
        self.assertEqual(diagnostics["loss_domain_pixels"], 100 * 100)
        self.assertEqual(int(target["unknown_geometry"].sum()), 0)
        self.assertTrue(target["blank_negative"].all())
        self.assertEqual(target["native_size"], {"height": 100, "width": 100})

    def test_non_square_native_shape_and_hwc_layout(self):
        height, width = 50, 100
        document = _doc([(0, _line_points(5, 5, 15, 5), 8, 1)], width=100, height=100)
        chw = _blank(height, width)
        target_chw = contract.build_targets(chw, document)
        target_hwc = contract.build_targets(
            np.transpose(chw, (1, 2, 0)).copy(), document
        )

        self.assertEqual(target_chw["native_size"], {"height": 50, "width": 100})
        self.assertEqual(target_chw["raster_size"], {"width": 100, "height": 50})
        self.assertEqual(target_chw["masks"].shape, (1, 50, 100))
        self.assertEqual(target_chw["ink"].shape, (50, 100))
        # canvas is 100x100 but the raster is 100x50: y scales by 0.5, x by 1.0.
        self.assertEqual(target_chw["boxes"][0].tolist(), [5.0, 2.0, 16.0, 3.0])
        expected = _project_masks(document, 100, 50)
        np.testing.assert_array_equal(target_chw["masks"][0], expected[(8, 1)])

        np.testing.assert_array_equal(target_hwc["masks"], target_chw["masks"])
        np.testing.assert_array_equal(target_hwc["ink"], target_chw["ink"])
        np.testing.assert_array_equal(target_hwc["labels"], target_chw["labels"])
        np.testing.assert_array_equal(target_hwc["boxes"], target_chw["boxes"])

    def test_masks_are_not_clipped_to_ink(self):
        document = _doc([(0, _line_points(10, 10, 20, 10), 1, 0)])
        target = contract.build_targets(_blank(), document)

        self.assertGreater(int(target["masks"][0].sum()), 0)
        self.assertEqual(int(target["ink"].sum()), 0)
        self.assertTrue(target["loss_domain"].all())
        self.assertEqual(target["diagnostics"]["known_ink_pixels"], 0)
        self.assertIn("never clipped to ink", target["mask_policy"])

    def test_crop_translation_and_clipping(self):
        image = _blank()
        _paint_line(image, 20, 50, 40, 50)
        document = _doc([(0, _line_points(20, 50, 40, 50), 3, 0)])
        target = contract.build_targets(image, document)

        crop = contract.crop_targets(image, target, origin=(10, 40), size=20)
        self.assertEqual(crop["window"], (10, 40, 30, 60))
        self.assertEqual(crop["visible_count"], 1)
        self.assertEqual(crop["target_count"], 1)
        self.assertTrue(bool(crop["visibility"][0]))
        self.assertEqual(crop["boxes"][0].tolist(), [10.0, 10.0, 20.0, 11.0])
        np.testing.assert_array_equal(
            crop["masks"][0], target["masks"][0][40:60, 10:30]
        )
        np.testing.assert_array_equal(crop["labels"], target["labels"][40:60, 10:30])
        np.testing.assert_array_equal(crop["ink"], target["ink"][40:60, 10:30])
        np.testing.assert_array_equal(
            crop["loss_domain"], target["loss_domain"][40:60, 10:30]
        )
        np.testing.assert_array_equal(
            crop["unknown_geometry"], target["unknown_geometry"][40:60, 10:30]
        )
        np.testing.assert_array_equal(
            crop["blank_negative"], target["blank_negative"][40:60, 10:30]
        )
        self.assertEqual(crop["image"].shape, (3, 20, 20))
        np.testing.assert_array_equal(
            crop["image"], image[:, 40:60, 10:30]
        )

    def test_crop_outside_window_visibility_and_padding(self):
        image = _blank()
        _paint_line(image, 10, 10, 20, 10)
        document = _doc([(0, _line_points(10, 10, 20, 10), 4, 0)])
        target = contract.build_targets(image, document)

        far = contract.crop_targets(image, target, origin=(90, 90), size=20)
        self.assertFalse(bool(far["visibility"][0]))
        self.assertEqual(far["visible_count"], 0)
        self.assertEqual(far["boxes"][0].tolist(), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(int(far["masks"].sum()), 0)
        self.assertEqual(int(far["ink"].sum()), 0)
        self.assertTrue(far["loss_domain"][:10, :10].all())
        self.assertEqual(int(far["loss_domain"][10:, :].sum()), 0)
        self.assertTrue(np.all(far["image"][:, 10:, :] == 1.0))

        negative = contract.crop_targets(image, target, origin=(-5, -5), size=20)
        self.assertEqual(negative["window"], (0, 0, 15, 15))
        self.assertTrue(bool(negative["visibility"][0]))
        self.assertEqual(negative["boxes"][0].tolist(), [15.0, 15.0, 20.0, 16.0])
        self.assertTrue(negative["loss_domain"][5:, 5:].all())
        self.assertEqual(int(negative["loss_domain"][:5, :].sum()), 0)
        self.assertTrue(np.all(negative["image"][:, :5, :] == 1.0))

    def test_zero_target_build_and_crop(self):
        cases = {
            "empty": _doc([]),
            "thing_minus1": _doc([(0, _line_points(10, 10, 20, 10), 5, -1)]),
            "semantic35": _doc([(2, _circle_points(50, 50, 10), 35, 0)]),
        }
        for name, document in cases.items():
            with self.subTest(document=name):
                image = _blank()
                target = contract.build_targets(image, document)
                self.assertEqual(target["keys"], ())
                self.assertEqual(target["masks"].shape, (0, 100, 100))
                self.assertEqual(target["boxes"].shape, (0, 4))
                for origin, size in (((0, 0), 384), ((95, 95), 20), ((-5, -7), 24)):
                    crop = contract.crop_targets(image, target, origin=origin, size=size)
                    self.assertEqual(crop["masks"].shape, (0, size, size))
                    self.assertEqual(crop["boxes"].shape, (0, 4))
                    self.assertEqual(crop["keys"], ())
                    self.assertEqual(crop["visibility"].shape, (0,))
                    self.assertEqual(int(crop["visibility"].sum()), 0)
                    self.assertEqual(crop["visible_count"], 0)
                    self.assertEqual(crop["target_count"], 0)
                    self.assertEqual(int(crop["masks"].sum()), 0)
                    self.assertEqual(crop["labels"].shape, (size, size))
                    self.assertEqual(int(crop["labels"].sum()), 0)
                    self.assertEqual(crop["image"].shape, (3, size, size))
                    window_x0, window_y0, window_x1, window_y1 = crop["window"]
                    self.assertEqual(
                        int(crop["loss_domain"].sum()),
                        int(
                            target["loss_domain"][
                                window_y0:window_y1, window_x0:window_x1
                            ].sum()
                        ),
                    )
                    self.assertEqual(
                        int(crop["unknown_geometry"].sum()),
                        int(
                            target["unknown_geometry"][
                                window_y0:window_y1, window_x0:window_x1
                            ].sum()
                        ),
                    )
                    self.assertEqual(
                        int(crop["blank_negative"].sum()),
                        int(
                            target["blank_negative"][
                                window_y0:window_y1, window_x0:window_x1
                            ].sum()
                        ),
                    )

    def test_skipped_geometry_blank_pixels_are_not_negatives(self):
        image = _blank()
        image[:, 10, 10] = 0.0
        image[:, 90, 90] = 0.0
        document = _doc([(0, _line_points(10, 10, 90, 90), 35, 0)])
        target = contract.build_targets(image, document)

        self.assertFalse(bool(target["ink"][50, 50]))
        self.assertTrue(bool(target["unknown_geometry"][50, 50]))
        self.assertFalse(bool(target["loss_domain"][50, 50]))
        self.assertFalse(bool(target["blank_negative"][50, 50]))
        self.assertTrue(bool(target["loss_domain"][80, 20]))
        self.assertTrue(bool(target["blank_negative"][80, 20]))
        self.assertGreater(
            target["diagnostics"]["unknown_geometry_blank_pixels"], 10
        )
        np.testing.assert_array_equal(
            target["blank_negative"], ~target["ink"] & ~target["unknown_geometry"]
        )
        np.testing.assert_array_equal(target["loss_domain"], target["blank_negative"])

    def test_known_off_ink_diagonal_is_positive_not_negative(self):
        image = _blank()
        document = _doc([(0, _line_points(10, 10, 90, 90), 7, 0)])
        target = contract.build_targets(image, document)
        union = target["masks"].any(axis=0)

        self.assertTrue(bool(union[50, 50]))
        self.assertFalse(bool(target["ink"][50, 50]))
        self.assertFalse(bool(target["blank_negative"][50, 50]))
        self.assertTrue(bool(target["loss_domain"][50, 50]))
        self.assertFalse(bool((target["blank_negative"] & union).any()))
        self.assertFalse(
            bool((target["blank_negative"] & target["unknown_geometry"]).any())
        )
        self.assertTrue(bool(target["blank_negative"][80, 20]))
        self.assertTrue(bool(target["loss_domain"][80, 20]))
        np.testing.assert_array_equal(
            target["blank_negative"],
            ~target["ink"] & ~target["unknown_geometry"] & ~union,
        )

    def test_known_positive_precedence_over_unknown_geometry(self):
        image = _blank()
        _paint_line(image, 10, 50, 90, 50)
        document = _doc(
            [
                (0, _line_points(10, 10, 90, 90), 35, 0),
                (0, _line_points(10, 50, 90, 50), 4, 0),
            ]
        )
        target = contract.build_targets(image, document)

        self.assertTrue(bool(target["unknown_geometry"][50, 50]))
        self.assertTrue(bool(target["masks"][0][50, 50]))
        self.assertTrue(bool(target["loss_domain"][50, 50]))
        self.assertFalse(bool(target["blank_negative"][50, 50]))
        np.testing.assert_array_equal(
            target["loss_domain"],
            target["masks"].any(axis=0) | target["blank_negative"],
        )

    def test_duplicate_primitives_and_order_are_deterministic(self):
        first = (0, _line_points(10, 10, 40, 10), 2, 5)
        second = (0, _line_points(10, 50, 40, 50), 30, 9)
        forward = contract.build_targets(_blank(), _doc([first, second, first]))
        reverse = contract.build_targets(_blank(), _doc([second, first, first]))

        self.assertEqual(forward["keys"], reverse["keys"])
        np.testing.assert_array_equal(forward["masks"], reverse["masks"])
        np.testing.assert_array_equal(forward["labels"], reverse["labels"])
        np.testing.assert_array_equal(forward["boxes"], reverse["boxes"])

    def test_overlap_resolves_to_lowest_canonical_key(self):
        document = _doc(
            [
                (0, _line_points(10, 50, 90, 50), 5, 0),
                (0, _line_points(50, 10, 50, 90), 3, 0),
            ]
        )
        target = contract.build_targets(_blank(), document)

        self.assertEqual(target["keys"], ((3, 0), (5, 0)))
        self.assertEqual(target["labels"][50, 50], 4)
        self.assertGreaterEqual(target["diagnostics"]["overlap_pixels"], 1)
        self.assertGreaterEqual(target["diagnostics"]["conflict_pixels"], 1)

    def test_malformed_inputs_fail_closed(self):
        good_document = _doc([(0, _line_points(10, 10, 20, 10), 1, 0)])
        bad_image = [
            np.ones((100, 100), dtype=np.float32),
            np.ones((4, 100, 100), dtype=np.float32),
            np.ones((3, 100, 100), dtype=np.float32) * np.nan,
            np.ones((3, 100, 100), dtype=np.float32) * 2.0,
            np.ones((3, 100, 100), dtype=np.complex64),
            np.ones((3, 100, 100), dtype=np.bool_),
        ]
        for index, image in enumerate(bad_image):
            with self.subTest(image=index):
                with self.assertRaises(contract.TargetError):
                    contract.build_targets(image, good_document)

        documents = []
        missing = _doc([(0, _line_points(10, 10, 20, 10), 1, 0)])
        missing.pop("lengths")
        documents.append(("missing_lengths", missing))
        short_args = _doc([(0, _line_points(10, 10, 20, 10), 1, 0)])
        short_args["args"][0] = short_args["args"][0][:7]
        documents.append(("short_args", short_args))
        ragged = _doc([(0, _line_points(10, 10, 20, 10), 1, 0)])
        ragged["instanceIds"].append(0)
        documents.append(("ragged", ragged))
        bad_semantic = _doc([(0, _line_points(10, 10, 20, 10), 36, 0)])
        documents.append(("semantic_36", bad_semantic))
        bad_instance = _doc([(0, _line_points(10, 10, 20, 10), 1, -2)])
        documents.append(("instance_minus2", bad_instance))
        bad_command = _doc([(4, _line_points(10, 10, 20, 10), 1, 0)])
        documents.append(("command_4", bad_command))
        non_finite = _doc([(0, _line_points(10, 10, 20, 10), 1, 0)])
        non_finite["args"][0][0] = float("inf")
        documents.append(("non_finite", non_finite))
        for name, document in documents:
            with self.subTest(document=name):
                with self.assertRaises(contract.TargetError):
                    contract.build_targets(_blank(), document)

        with self.assertRaises(contract.TargetError):
            contract.build_targets(_blank(), good_document, raster_size=50)
        with self.assertRaises(contract.TargetError):
            contract.build_targets(_blank(), good_document, canvas_size=(0.0, 100.0))
        with self.assertRaises(contract.TargetError):
            contract.build_targets(None, good_document)

        target = contract.build_targets(_blank(), good_document)
        for name, kwargs in (
            ("bad_origin", {"origin": (1.5, 0)}),
            ("bad_size", {"origin": (0, 0), "size": 0}),
        ):
            with self.subTest(crop=name):
                with self.assertRaises(contract.TargetError):
                    contract.crop_targets(_blank(), target, **kwargs)
        broken = {key: value for key, value in target.items() if key != "ink"}
        with self.assertRaises(contract.TargetError):
            contract.crop_targets(_blank(), broken)
        with self.assertRaises(contract.TargetError):
            contract.crop_targets(_blank(50, 50), target)


if __name__ == "__main__":
    unittest.main(verbosity=2)

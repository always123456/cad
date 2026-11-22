"""--allow-official-inputs: the one documented relaxation of the sealed-path guard.

Off (the default, and the platform run) every sealed token refuses, as before.
On, exactly one case is admitted: an input PNG whose path holds a ``datasets``
directory (the official corpora, e.g. ``datasets/dev_final``).  --out, --checkpoint,
--inputs and non-PNG paths keep the full refusal, and the switch is traced in the
run summary / receipt (``guards.allow_official_inputs``).
"""

from __future__ import annotations

import os
import tempfile
import unittest

from tests.support import args, delivery, touch

REFUSE = delivery.DeliveryRefused


class AllowOfficialInputsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="delivery-official-")
        self.checkpoint = touch(os.path.join(self.tmp, "checkpoint.pt"))
        self.out = os.path.join(self.tmp, "out")

    def check(self, **overrides):
        fields = {"checkpoint": self.checkpoint, "out": self.out}
        fields.update(overrides)
        namespace = args(**fields)
        return delivery.validate_delivery_paths(
            out=namespace.out,
            checkpoint=namespace.checkpoint,
            png=namespace.png,
            inputs=namespace.inputs,
            device=namespace.device,
            allow_official_inputs=getattr(namespace, "allow_official_inputs", False),
        )

    def test_only_input_pngs_under_datasets_are_admitted(self):
        official = touch(os.path.join(self.tmp, "datasets", "dev_final", "x.png"))

        # off (the default): the official corpora stay sealed
        with self.assertRaises(REFUSE) as ctx:
            self.check(png=official)
        self.assertEqual(ctx.exception.code, "sealed_path")

        # on: an input PNG under datasets/ is admitted, nothing else
        plan = self.check(png=official, allow_official_inputs=True)
        self.assertEqual(plan["png"], os.path.abspath(official))

        # on: a sealed token outside datasets/ is still refused
        runtime = touch(os.path.join(self.tmp, ".runtime", "y.png"))
        with self.assertRaises(REFUSE) as ctx:
            self.check(png=runtime, allow_official_inputs=True)
        self.assertEqual(ctx.exception.code, "sealed_path")

        # on: --out keeps the full refusal, even under datasets/
        with self.assertRaises(REFUSE) as ctx:
            self.check(
                png=official,
                out=os.path.join(self.tmp, "datasets", "test_final", "out"),
                allow_official_inputs=True,
            )
        self.assertEqual(ctx.exception.code, "sealed_path")

        # on: --inputs (the manifest path) keeps the full refusal too
        manifest = touch(os.path.join(self.tmp, "datasets", "dev_final", "inputs.json"))
        with self.assertRaises(REFUSE) as ctx:
            self.check(inputs=manifest, allow_official_inputs=True)
        self.assertEqual(ctx.exception.code, "sealed_path")

        # and the flag is traced where an operator looks for it
        traced = delivery.run_delivery(
            args(png=official, out=self.out, checkpoint=self.checkpoint, device="cpu",
                 allow_official_inputs=True, dry_run=True)
        )
        self.assertTrue(traced["guards"]["allow_official_inputs"])

        # without the flag the same run is refused before anything is planned
        with self.assertRaises(REFUSE) as ctx:
            delivery.run_delivery(
                args(png=official, out=os.path.join(self.tmp, "out2"),
                     checkpoint=self.checkpoint, device="cpu", dry_run=True)
            )
        self.assertEqual(ctx.exception.code, "sealed_path")

    def test_default_run_carries_no_guards_block(self):
        fixture = touch(os.path.join(self.tmp, "box.png"))
        summary = delivery.run_delivery(
            args(png=fixture, out=self.out, checkpoint=self.checkpoint, device="cpu", dry_run=True)
        )
        self.assertNotIn("guards", summary)


if __name__ == "__main__":
    unittest.main()

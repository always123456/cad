"""Contract check over the frozen result documents.

Reuses package/verify_delivery.check_document (the delivery's own validator) and adds the
frozen-freeze specifics: exact 12 top-level fields, args in 4-point viewBox coordinates,
no out-of-viewBox rows, no Stuff classes, valid instanceIds.

Usage: python contract_check.py [<results_dir>] [<out.json>]
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import verify_delivery as VD  # noqa: E402

STUFF = set(range(VD.STUFF[0], VD.STUFF[1] + 1)) | {VD.IGNORE}


def main(argv):
    results = argv[1] if len(argv) > 1 else os.path.join(HERE, "..", "run", "results")
    out = argv[2] if len(argv) > 2 else os.path.join(HERE, "..", "evidence", "contract_check.json")
    results = os.path.abspath(results)

    names = sorted(n for n in os.listdir(results) if n.endswith(".json"))
    report = {
        "results_dir": "<FREEZE>/run/results",
        "documents": len(names),
        "fields_exact_12": 0,
        "primitives": 0,
        "rows": 0,
        "args_points_4": True,
        "args_in_viewbox": True,
        "no_stuff_classes": True,
        "instance_ids_valid": True,
        "class_ids": {},
        "errors": [],
    }
    for name in names:
        path = os.path.join(results, name)
        raw = open(path, "rb").read()
        doc = json.loads(raw.decode("utf-8"))
        errors, facts = VD.check_document(doc, raw)
        if errors:
            report["errors"].extend(f"{name}:{e}" for e in errors)
        if facts.get("fields") == 12:
            report["fields_exact_12"] += 1
        report["primitives"] += facts.get("primitives", 0)

        for row in doc["args"]:
            report["rows"] += 1
            if len(row) != 8:
                report["args_points_4"] = False
                report["errors"].append(f"{name}:args_not_4_points:{len(row)}")
                continue
            xs = row[0::2]
            ys = row[1::2]
            if min(xs) < 0.0 or max(xs) > 100.0 or min(ys) < 0.0 or max(ys) > 100.0:
                report["args_in_viewbox"] = False
                report["errors"].append(f"{name}:args_outside_viewbox")

        for semantic in doc["semanticIds"]:
            report["class_ids"][str(semantic)] = report["class_ids"].get(str(semantic), 0) + 1
            if semantic in STUFF:
                report["no_stuff_classes"] = False
                report["errors"].append(f"{name}:stuff_class:{semantic}")

    report["instance_ids_valid"] = not any("instanceIds_invalid" in e for e in report["errors"])
    report["ok"] = not report["errors"] and report["fields_exact_12"] == len(names)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in report.items() if k not in ("class_ids",)}, indent=1))
    print("class_ids:", report["class_ids"])


if __name__ == "__main__":
    main(sys.argv)

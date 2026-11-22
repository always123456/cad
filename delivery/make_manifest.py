"""Write MANIFEST.json for a freeze directory: per-file sha256, sizes, totals.

Usage: python make_manifest.py [<freeze_dir>]
"""

import hashlib
import json
import os
import sys
import time

SKIP = {"MANIFEST.json"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def results_digest(paths):
    """Order-independent digest of a set of result documents."""
    rows = sorted("%s  %s" % (os.path.basename(p), sha256(p)) for p in paths)
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def main(argv):
    freeze = os.path.abspath(argv[1] if len(argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    files = []
    for base, dirs, names in os.walk(freeze):
        dirs.sort()
        for name in sorted(names):
            path = os.path.join(base, name)
            rel = os.path.relpath(path, freeze).replace(os.sep, "/")
            if rel in SKIP:
                continue
            files.append((rel, path))
    files.sort()

    entries = []
    areas = {}
    total = 0
    for rel, path in files:
        size = os.path.getsize(path)
        total += size
        area = rel.split("/")[0] if "/" in rel else "(root)"
        a = areas.setdefault(area, {"files": 0, "bytes": 0})
        a["files"] += 1
        a["bytes"] += size
        entries.append({"path": rel, "bytes": size, "sha256": sha256(path)})

    results = [p for rel, p in files if rel.startswith("run/results/") and rel.endswith(".json")]
    pool = [p for rel, p in files if rel.startswith("run/pool/results/") and rel.endswith(".json")]

    manifest = {
        "freeze": os.path.basename(freeze),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_root": "<source project tree, kept in place>",
        "tooling": {"python": "<PYTHON> 3.14", "numpy": "2.5.x", "scikit-learn": "1.9.x"},
        "totals": {"files": len(entries), "bytes": total, "areas": areas},
        "results_digest": results_digest(results),
        "results_documents": len(results),
        "pool_results_digest": results_digest(pool),
        "pool_results_documents": len(pool),
        "weights": [
            {
                "path": e["path"],
                "bytes": e["bytes"],
                "sha256": e["sha256"],
            }
            for e in entries
            if e["path"].startswith("weights/")
        ],
        "files": entries,
    }
    out = os.path.join(freeze, "MANIFEST.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
    print(json.dumps({k: manifest[k] for k in ("freeze", "totals", "results_documents", "results_digest", "pool_results_digest", "weights")}, indent=1))


if __name__ == "__main__":
    main(sys.argv)

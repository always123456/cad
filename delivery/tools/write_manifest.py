"""Rewrite manifest.json from the package's own files.

  python tools/write_manifest.py [--root <package dir>]

The file hashes make the package auditable on its own; the measurement blocks
carried over from the copy source stay under ``inherited`` so no number is
silently dropped, while this package's own measurements live in MASK_PASTE.md
and DELIVERY.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

SCHEMA = "cad-delivery-manifest/1"
INCLUDE_SUFFIXES = (".py", ".md", ".json", ".png", ".ps1")
EXCLUDE_DIRS = {"runs", "__pycache__", ".git", "fixtures_tmp"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def walk(root):
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in EXCLUDE_DIRS)
        for name in sorted(files):
            if name == "manifest.json":
                continue
            if not name.endswith(INCLUDE_SUFFIXES):
                continue
            full = os.path.join(base, name)
            yield os.path.relpath(full, root).replace("\\", "/"), full


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = parser.parse_args(argv)
    root = os.path.abspath(args.root)
    path = os.path.join(root, "manifest.json")
    inherited = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            previous = json.load(handle)
        inherited = {key: value for key, value in previous.items() if key not in ("schema", "root", "files", "count")}
    files = []
    for rel, full in walk(root):
        files.append({"path": rel, "bytes": os.path.getsize(full), "sha256": sha256_file(full)})
    manifest = {"schema": SCHEMA, "root": root, "files": files, "count": len(files)}
    if inherited:
        manifest["inherited"] = {
            "note": (
                "measurement blocks copied verbatim from the package this one was copied from "
                "(cad-png-delivery-mrcnn03); they describe a different policy (cap 12, nonzero ink) "
                "and are history, not this package's numbers"
            ),
            "blocks": inherited,
        }
    manifest["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"root": root, "count": len(files)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

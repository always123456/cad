"""Run one A/B cell and write its durable receipt.

Waits (in this process) while another python job holds the single GPU, then runs
the delivery CLI in-process so the peak VRAM and the post-run CUDA counters can be
read from the same allocator, and finally writes ``cell_receipt.json`` next to the
run: argv, checkpoint sha256, peak VRAM, wall, CUDA allocated/reserved after exit
and the hash of the outputs index.

  python tools/run_cell.py --out runs/ab-prob50 [--mask-paste logits]
                           [--mask-threshold 0.45] [--cells 0.5]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

INPUTS = os.path.join(ROOT, "runs", "subset96-inputs.json")
CHECKPOINT = os.environ.get(
    "CAD_FINE_MASK_HEAD_CHECKPOINT",
    "",
) or os.path.join("runs", "fine_mask112_upt250_snapshot.pt")
POLL_S = 20.0
MAX_WAIT_S = 3600.0


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_apps():
    """(pid, used MiB) of every process the driver reports as holding the GPU.

    The driver also lists desktop processes that merely keep a context open
    (explorer, dwm, vendor tray apps); only a python process is a competing job.
    """

    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    apps = []
    for line in out.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        apps.append((int(parts[0]), parts[1]))
    return apps


def process_name(pid):
    out = subprocess.run(
        ["tasklist", "/FI", "PID eq %d" % pid, "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
    )
    line = out.stdout.strip().splitlines()
    if not line:
        return ""
    fields = [field.strip('"') for field in line[0].split('","')]
    return fields[0].strip('"').lower()


def wait_for_gpu():
    """Block while a foreign *python* process holds the GPU; never kill it."""

    started = time.time()
    waited = 0.0
    while True:
        foreign = [
            (pid, mib)
            for pid, mib in compute_apps()
            if pid != os.getpid() and "python" in process_name(pid)
        ]
        if not foreign:
            return waited
        if time.time() - started > MAX_WAIT_S:
            raise SystemExit(
                "GPU still held by %r after %.0fs; not waiting further" % (foreign, waited)
            )
        print("gpu busy %r, sleeping %.0fs" % (foreign, POLL_S), flush=True)
        time.sleep(POLL_S)
        waited = time.time() - started


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--mask-paste", default="prob")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--inputs", default=INPUTS)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    command = [
        "cad_mrcnn_delivery.py",
        "--inputs", args.inputs,
        "--out", out,
        "--checkpoint", args.checkpoint,
        "--mask-head", "112", "--mask-chunk", "24", "--drop-stuff",
        "--refine-mask", "a6", "--refine-ink", "composite128", "--refine-ratio", "0.0",
        "--score-threshold", "0.99", "--max-detections", "300", "--per-image-cap", "20",
        "--mask-paste", args.mask_paste,
        "--device", "cuda:0",
    ]
    if args.mask_threshold != 0.5:
        command += ["--mask-threshold", repr(args.mask_threshold)]
    if args.dry_run:
        command.append("--dry-run")

    import torch

    import cad_mrcnn_delivery as delivery

    waited = wait_for_gpu()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    code = delivery.main(command[1:])
    wall = time.time() - started
    peak = int(torch.cuda.max_memory_allocated())
    reserved_peak = int(torch.cuda.max_memory_reserved())
    torch.cuda.empty_cache()
    allocated = int(torch.cuda.memory_allocated())
    reserved = int(torch.cuda.memory_reserved())

    receipt = {
        "schema": "cad-mrcnn07.cell_receipt/1",
        "argv": command,
        "cwd": os.getcwd(),
        "exit": code,
        "checkpoint": {"path": args.checkpoint, "sha256": sha256(args.checkpoint)},
        "device": "cuda:0",
        "gpu_wait_s": round(waited, 1),
        "wall_s": round(wall, 2),
        "vram": {
            "peak_allocated_bytes": peak,
            "peak_reserved_bytes": reserved_peak,
            "allocated_after_exit": allocated,
            "reserved_after_exit": reserved,
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results_index_sha256": (
            sha256(os.path.join(out, "results_index.json"))
            if os.path.exists(os.path.join(out, "results_index.json"))
            else None
        ),
        "run_receipt_sha256": (
            sha256(os.path.join(out, "run_receipt.json"))
            if os.path.exists(os.path.join(out, "run_receipt.json"))
            else None
        ),
    }
    target = os.path.join(out, "cell_receipt.json")
    os.makedirs(out, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=1, ensure_ascii=False)
    print(json.dumps({k: receipt[k] for k in ("exit", "wall_s", "vram", "gpu_wait_s")}, indent=1))
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

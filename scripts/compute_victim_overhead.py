#!/usr/bin/env python3
"""Measure Def+ runtime overhead for artifact victims (off vs on).

Wall-clock is process start through last sample (same quantity as the
private system-cost runs). Inputs are decoded before the clock starts.
Repeats are independent process launches; the TSV reports the median.

Sample caps (override with --n):
  mnist / cifar          2000
  imagenet / celea / chest  1000

Defense knobs match compute_victim_acc.py / compile env.
Requires --backend.

Usage:
  python3 scripts/compute_victim_overhead.py --backend mc
  python3 scripts/compute_victim_overhead.py --backend ic --case IC01
  python3 scripts/compute_victim_overhead.py --backend tv --repeats 5
  python3 scripts/compute_victim_overhead.py --backend ta --case TA01 --n 200
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import compute_victim_acc as acc  # noqa: E402

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TVM_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
}
N_LIGHT = 2000
N_HEAVY = 1000


def default_n(spec: acc.DataSpec) -> int:
    return N_LIGHT if spec.subset == "full" else N_HEAVY


def pin_env(extra: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(THREAD_ENV)
    env.update(extra)
    env.setdefault("GLOW_INPUT_ZERO_DITHER_SILENT", "1")
    env.setdefault("TVM_INPUT_ZERO_DITHER_SILENT", "1")
    return env


def preload(
    adapter: acc.Adapter, spec: acc.DataSpec, indices: list[int],
) -> list[np.ndarray]:
    tensors: list[np.ndarray] = []
    n = len(indices)
    for i, idx in enumerate(indices, 1):
        image, _target = adapter.get(idx)
        tensors.append(acc.image_to_nchw(image, spec))
        if i % 200 == 0 or i == n:
            print(f"  preload {i}/{n}", flush=True)
    return tensors


def time_stream(
    cmd: list[str],
    extra_env: dict[str, str],
    n_out: int,
    tensors: list[np.ndarray],
    stderr_path: Path,
) -> float:
    env = pin_env(extra_env)
    t0 = time.perf_counter()
    with acc.StreamSession(cmd, env, n_out, stderr_path) as session:
        for tensor in tensors:
            session.infer(tensor)
    return time.perf_counter() - t0


def mc_cmd(art: Path, case_id: str, mode: str) -> list[str]:
    bundle = art / "victims" / "glow" / "libs" / case_id / mode
    runner = bundle / f"{case_id}_runner"
    weights = bundle / f"{case_id}.weights.bin"
    if not runner.is_file():
        raise FileNotFoundError(f"missing MC runner: {runner}")
    if not weights.is_file():
        raise FileNotFoundError(f"missing MC weights: {weights}")
    return [str(runner), str(weights), "--stream"]


def tvm_cmd(
    art: Path, case_id: str, mode: str, backend: str, tvm_build: Path,
) -> tuple[list[str], dict[str, str]]:
    tvm_dir = art / "victims" / "tvm"
    libs = tvm_dir / "libs" / case_id
    extra_ld = f"{tvm_build}:{tvm_build / 'lib'}"
    ld = extra_ld + ((":" + os.environ["LD_LIBRARY_PATH"]) if os.environ.get("LD_LIBRARY_PATH") else "")
    if backend == "tv":
        runner = tvm_dir / "bin" / "generic_tvm_native_vm_runner"
        library = libs / mode / f"{case_id}_tvm.so"
        shape_path = libs / mode / "input_shape.txt"
        for path in (runner, library, shape_path):
            if not path.exists():
                raise FileNotFoundError(path)
        cmd = [
            str(runner), "--library", str(library),
            "--input-shape", shape_path.read_text().strip(), "--stream",
        ]
    else:
        runner = libs / f"{case_id}_tvm_aot_runner"
        library = libs / mode / f"{case_id}_tvm_aot_kernels.so"
        constants = libs / mode / "constants"
        for path in (runner, library, constants):
            if not path.exists():
                raise FileNotFoundError(path)
        cmd = [
            str(runner), "--library", str(library),
            "--constants-dir", str(constants), "--stream",
        ]
    return cmd, {"LD_LIBRARY_PATH": ld}


def prepare_ic_batches(
    art: Path,
    row: dict[str, str],
    spec: acc.DataSpec,
    tensors: list[np.ndarray],
    work: Path,
    ic_bin: Path,
) -> tuple[list[str], Path, Path]:
    onnx = art / "victims" / "models" / row["onnx_rel"]
    if not ic_bin.is_file():
        raise FileNotFoundError(f"missing image-classifier: {ic_bin}")
    if not onnx.is_file():
        raise FileNotFoundError(f"missing ONNX: {onnx}")
    per_file = 1 if row["family"] in acc.FORCE_SINGLE_SAMPLE_FILE else 100
    batch_dir = work / "inputs"
    batch_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for start in range(0, len(tensors), per_file):
        chunk = tensors[start:start + per_file]
        batch = np.concatenate(chunk, axis=0).astype(np.float32, copy=False)
        path = batch_dir / f"batch_{start // per_file:04d}.npy"
        np.save(path, batch)
        paths.append(str(path))
    list_path = work / "input_batches.list.txt"
    list_path.write_text("\n".join(paths) + "\n", encoding="utf-8")
    raw_out = work / "raw_outputs.bin"
    cmd = [
        str(ic_bin),
        f"-model={onnx}",
        "-backend=Interpreter",
        "-model-input=data",
        f"-input-image-list-file={list_path}",
        "-output-name=output",
        "-image-layout=NCHW",
        "-input-layout=NCHW",
        "-minibatch=1",
        "-minibatch-threads=1",
        "-topk=0",
        f"-dump-output-binary-file={raw_out}",
    ]
    return cmd, raw_out, work


def time_ic(cmd: list[str], extra_env: dict[str, str], raw_out: Path, work: Path) -> float:
    if raw_out.exists():
        raw_out.unlink()
    env = pin_env(extra_env)
    env["LD_LIBRARY_PATH"] = acc.IC_LD_LIBRARY_PATH
    t0 = time.perf_counter()
    completed = subprocess.run(cmd, env=env, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.perf_counter() - t0
    (work / "stdout.txt").write_bytes(completed.stdout)
    (work / "stderr.txt").write_bytes(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"image-classifier failed ({completed.returncode})\n"
            f"{completed.stderr.decode('utf-8', 'replace')}"
        )
    return elapsed


def fmt(value: float) -> str:
    return "nan" if value != value else f"{value:.6f}"


def main() -> int:
    art = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend", choices=("mc", "ic", "tv", "ta"), required=True)
    parser.add_argument(
        "--case", action="append", default=[], dest="cases",
        help="MC01 / IC01 / TV01 / TA01 (repeatable). Default: all from manifest.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--n", type=int, default=None, help="samples per run. Default: 2000 or 1000.")
    parser.add_argument("--manifest", type=Path, default=art / "victims/models/MANIFEST.txt")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--imagenet50-96-root", type=Path, default=None)
    parser.add_argument("--subset-manifest", type=Path, default=None)
    parser.add_argument("--out-tsv", type=Path, default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--ic-bin", type=Path, default=None)
    parser.add_argument("--tvm-build", type=Path, default=None)
    parser.add_argument("--relu-patch-bits", type=acc.parse_int_auto, default=acc.RELU_PATCH_BITS)
    parser.add_argument("--relu-patch-fixed-bits", type=acc.parse_int_auto, default=acc.RELU_PATCH_FIXED_BITS)
    parser.add_argument("--relu-patch-fixed-value", type=acc.parse_int_auto, default=acc.RELU_PATCH_FIXED_VALUE)
    parser.add_argument("--relu-patch-inc", type=acc.parse_int_auto, default=acc.RELU_PATCH_INC)
    parser.add_argument("--relu-patch-positive", type=acc.parse_int_auto, default=acc.RELU_PATCH_POSITIVE)
    parser.add_argument("--dither-mode", default=acc.DITHER_MODE)
    parser.add_argument("--dither-thresh", type=float, default=acc.DITHER_THRESH)
    parser.add_argument("--dither-eps-min", type=float, default=acc.DITHER_EPS_MIN)
    parser.add_argument("--dither-eps-max", type=float, default=acc.DITHER_EPS_MAX)
    parser.add_argument("--dither-seed", type=acc.parse_int_auto, default=acc.DITHER_SEED)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be > 0")
    if args.n is not None and args.n <= 0:
        raise SystemExit("--n must be > 0")

    data_root = acc.resolve_dataset_root(args.dataset_root)
    imagenet_root = acc.resolve_imagenet50_root(args.imagenet50_96_root, data_root)
    work_root = args.work_root or (art / "overhead_out" / args.backend)
    out_tsv = args.out_tsv or (work_root / "overhead_results.tsv")
    meta = acc.load_manifest(args.manifest, args.backend)
    case_ids = acc.normalize_cases(args.cases, args.backend, sorted(meta))
    unknown = [case_id for case_id in case_ids if case_id not in meta]
    if unknown:
        raise SystemExit(f"unknown case(s) for --backend {args.backend}: {unknown}")
    subset_indices = acc.load_subset_manifest(args.subset_manifest) if args.subset_manifest else {}
    ic_bin = args.ic_bin or Path(
        os.environ.get("IC_BIN", art / "writeback-protect/glow/build_ic/bin/image-classifier")
    )
    tvm_build = args.tvm_build or (art / "writeback-protect" / "tvm" / "build-llvm14-glow")

    print(f"backend={args.backend}")
    print(f"repeats={args.repeats}")
    print(f"{args.backend}\ttask\tdataset\tn\trepeats\tt_off\tt_on\toverhead_pct")

    rows: list[dict[str, Any]] = []
    missing = 0
    for case_id in case_ids:
        info = meta[case_id]
        spec = acc.spec_for(info)
        n_want = args.n if args.n is not None else default_n(spec)
        print(f"=== {case_id} {info['family']}/{info['dataset']} n={n_want} ===", flush=True)
        try:
            adapter = acc.build_adapter(spec, data_root, imagenet_root)
            indices = acc.select_indices(spec, adapter, n_want, subset_indices)
            tensors = preload(adapter, spec, indices)
        except (FileNotFoundError, ModuleNotFoundError, RuntimeError, OSError) as exc:
            print(f"{case_id}\tmissing\t{exc}")
            rows.append({
                args.backend: case_id,
                "task": info["task"],
                "model": info["model"],
                "dataset": info["dataset"],
                "n": n_want,
                "repeats": 0,
                "status": "missing",
                "error": str(exc),
            })
            missing += 1
            continue

        ic_ready: tuple[list[str], Path, Path] | None = None
        if args.backend == "ic":
            try:
                ic_ready = prepare_ic_batches(
                    art, info, spec, tensors, work_root / case_id / "shared", ic_bin,
                )
            except (FileNotFoundError, RuntimeError, OSError) as exc:
                print(f"{case_id}\tmissing\t{exc}")
                rows.append({
                    args.backend: case_id,
                    "task": info["task"],
                    "model": info["model"],
                    "dataset": info["dataset"],
                    "n": len(tensors),
                    "repeats": 0,
                    "status": "missing",
                    "error": str(exc),
                })
                missing += 1
                continue

        off_secs: list[float] = []
        on_secs: list[float] = []
        status = "ok"
        error = ""
        for rep in range(1, args.repeats + 1):
            try:
                for mode, bucket in (("off", off_secs), ("on", on_secs)):
                    work = work_root / case_id / f"repeat_{rep:02d}" / mode
                    work.mkdir(parents=True, exist_ok=True)
                    extra = acc.glow_env(mode, args) if args.backend in {"mc", "ic"} else acc.tvm_env(mode, args)
                    if args.backend == "mc":
                        elapsed = time_stream(
                            mc_cmd(art, case_id, mode), extra, spec.n_out, tensors, work / "stream.stderr",
                        )
                    elif args.backend == "ic":
                        assert ic_ready is not None
                        cmd, raw_out, shared = ic_ready
                        elapsed = time_ic(cmd, extra, raw_out, work)
                        _ = shared
                    else:
                        cmd, ld = tvm_cmd(art, case_id, mode, args.backend, tvm_build)
                        extra = {**extra, **ld}
                        elapsed = time_stream(cmd, extra, spec.n_out, tensors, work / "stream.stderr")
                    bucket.append(elapsed)
                    print(f"  {case_id}/{mode} repeat {rep}/{args.repeats} {elapsed:.4f}s", flush=True)
            except (FileNotFoundError, RuntimeError, OSError) as exc:
                status = "partial" if off_secs or on_secs else "missing"
                error = str(exc)
                missing += 1
                print(f"{case_id}/repeat{rep}\t{status}\t{exc}")
                break

        paired = min(len(off_secs), len(on_secs))
        overheads = [
            (on_secs[i] - off_secs[i]) / off_secs[i] * 100.0
            for i in range(paired)
            if off_secs[i] > 0
        ]
        t_off = statistics.median(off_secs) if off_secs else float("nan")
        t_on = statistics.median(on_secs) if on_secs else float("nan")
        overhead = statistics.median(overheads) if overheads else float("nan")
        print(
            f"{case_id}\t{info['task']}\t{info['family']}/{info['dataset']}\t"
            f"{len(tensors)}\t{paired}\t{fmt(t_off)}\t{fmt(t_on)}\t{fmt(overhead)}"
        )
        rows.append({
            args.backend: case_id,
            "task": info["task"],
            "model": info["model"],
            "dataset": info["dataset"],
            "n": len(tensors),
            "repeats": paired,
            "t_off_sec": t_off,
            "t_on_sec": t_on,
            "overhead_pct": overhead,
            "status": status,
            "error": error,
        })

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        args.backend, "task", "model", "dataset", "n", "repeats",
        "t_off_sec", "t_on_sec", "overhead_pct", "status", "error",
    ]
    with out_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_tsv}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

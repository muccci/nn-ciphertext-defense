#!/usr/bin/env python3
"""Sweep ρ_trans and acc across perturbation strengths.

Nine ON points (ReLU 30 / 7 / inc 3,
fv in {106,108,110,112,114,116,120,124,126}) plus off. Default 14 MC
(no CelebA).

Uses the existing doors:
  collect_rho_trans.sh / compute_victim_rho_trans.py / compute_victim_acc.py

Does not write into acc_out/ (that is the single-config acc job).
This sweep has its own *_out, still split as acc and ρ_trans:

  perturbation_out/acc/
  perturbation_out/rho_trans/

  python3 scripts/run_perturbation.py
  python3 scripts/run_perturbation.py --stage pin --case MC02
  python3 scripts/run_perturbation.py --stage acc --limit 8
  python3 scripts/run_perturbation.py --stage plot
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ART = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import compute_victim_acc as acc  # noqa: E402
import compute_victim_rho_trans as rho_trans  # noqa: E402

COLLECT = SCRIPTS / "trace" / "collect_rho_trans.sh"
MANIFEST = ART / "victims" / "models" / "MANIFEST.txt"
DEFAULT_ROOT = ART / "perturbation_out"

DEFAULT_CASES = (
    "MC01", "MC02", "MC03", "MC04", "MC05", "MC06", "MC07", "MC08",
    "MC10", "MC11", "MC12", "MC14", "MC15", "MC17",
)
THRESH_4 = frozenset({"MC06", "MC11", "MC12", "MC14"})
FV_POINTS = (
    (106, "b30_fb7_fv0b1101010", "4.8e-7"),
    (108, "b30_fb7_fv0b1101100", "1.9e-6"),
    (110, "b30_fb7_fv0b1101110", "7.6e-6"),
    (112, "b30_fb7_fv0b1110000", "3.1e-5"),
    (114, "b30_fb7_fv0b1110010", "1.2e-4"),
    (116, "b30_fb7_fv0b1110100", "4.9e-4"),
    (120, "b30_fb7_fv0b1111000", "7.8e-3"),
    (124, "b30_fb7_fv0b1111100", "1.2e-1"),
    (126, "b30_fb7_fv0b1111110", "5.0e-1"),
)


def dither_thresh(case_id: str) -> float:
    return 4.0 if case_id in THRESH_4 else 1.0


def acc_ns(fv: int | None, thresh: float) -> argparse.Namespace:
    return argparse.Namespace(
        dither_mode="random",
        dither_thresh=thresh,
        dither_eps_min=1e-5,
        dither_eps_max=2e-5,
        dither_seed=1234,
        relu_patch_bits=30,
        relu_patch_fixed_bits=7,
        relu_patch_fixed_value=117 if fv is None else fv,
        relu_patch_inc=3,
        relu_patch_positive=0,
    )


def style_dataset(name: str) -> str:
    if name.startswith("mnist"):
        return "mnist"
    if name.startswith("cifar"):
        return "cifar"
    if name.startswith("chest"):
        return "chest"
    if name.startswith("imagenet"):
        return "imagenet50_96"
    return name


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}", flush=True)
    return path


def collect_pin(cases: list[str], pin_traces: Path, modes: str, fv: int | None) -> None:
    env = os.environ.copy()
    env["PIN_TRACES"] = str(pin_traces)
    env["MODES"] = modes
    if fv is not None:
        env["GLOW_RELU_PATCH_FIXED_VALUE"] = str(fv)
    cmd = ["bash", str(COLLECT), "mc", "--modes", modes]
    for case_id in cases:
        cmd.extend(["--case", case_id])
    print(f"==> collect {modes} fv={fv} PIN_TRACES={pin_traces}", flush=True)
    completed = subprocess.run(cmd, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"collect_rho_trans.sh failed modes={modes} fv={fv}")


def stage_pin(args: argparse.Namespace) -> None:
    collect_pin(args.cases, args.rho_trans_dir / "off", "off", None)
    for fv, _cid, _lab in FV_POINTS:
        collect_pin(args.cases, args.rho_trans_dir / f"fv_{fv}", "on", fv)


def case_jsons(root: Path, case_id: str, mode: str) -> list[Path]:
    return rho_trans.mode_jsons(root / "mc" / case_id, mode)


def stage_rho_trans(args: argparse.Namespace) -> Path:
    rows = []
    for case_id in args.cases:
        off_ps = case_jsons(args.rho_trans_dir / "off", case_id, "off")
        if not off_ps:
            rows.append({"mc": case_id, "config_id": "off", "status": "missing"})
            print(f"{case_id} off missing", flush=True)
            continue
        off = rho_trans.accumulate_paths(off_ps, rho_trans.accumulate_mc, case_id)
        rows.append({
            "mc": case_id, "config_id": "off", "x_index": 0, "x_label": "off",
            "rho_trans_off": off.ratio, "rho_trans_on": off.ratio, "R_pct": 0.0, "status": "ok",
        })
        for x_index, (fv, cid, lab) in enumerate(FV_POINTS, start=1):
            on_ps = case_jsons(args.rho_trans_dir / f"fv_{fv}", case_id, "on")
            if not on_ps:
                rows.append({"mc": case_id, "config_id": cid, "x_index": x_index, "x_label": lab, "status": "missing"})
                print(f"{case_id} {lab} missing", flush=True)
                continue
            on = rho_trans.accumulate_paths(on_ps, rho_trans.accumulate_mc, case_id)
            r_pct = rho_trans.r_adj(off, on) * 100.0
            rows.append({
                "mc": case_id, "config_id": cid, "x_index": x_index, "x_label": lab,
                "rho_trans_off": off.ratio, "rho_trans_on": on.ratio, "R_pct": r_pct, "status": "ok",
            })
            print(f"{case_id} {lab} rho_trans_on={on.ratio:.6f} R={r_pct:.2f}", flush=True)
    return write_tsv(
        args.rho_trans_dir / "rho_trans.tsv",
        rows,
        ["mc", "config_id", "x_index", "x_label", "rho_trans_off", "rho_trans_on", "R_pct", "status"],
    )


def stage_acc(args: argparse.Namespace) -> Path:
    data_root = acc.resolve_dataset_root(None)
    imagenet_root = acc.resolve_imagenet50_root(None, data_root)
    meta = acc.load_manifest(MANIFEST, "mc")
    rows = []
    off_metric: dict[str, float] = {}
    configs = [("off", None, "off", 0)] + [
        ("on", fv, cid, i) for i, (fv, cid, _lab) in enumerate(FV_POINTS, start=1)
    ]
    for case_id in args.cases:
        info = meta[case_id]
        spec = acc.spec_for(info)
        print(f"=== acc {case_id} {info['family']}/{info['dataset']} ===", flush=True)
        try:
            adapter = acc.build_adapter(spec, data_root, imagenet_root)
            indices = acc.select_indices(spec, adapter, args.limit, {})
        except (FileNotFoundError, ModuleNotFoundError, RuntimeError, OSError) as exc:
            print(f"[skip acc] {case_id} {exc}", flush=True)
            for _mode, _fv, cid, x_index in configs:
                rows.append({
                    "mc": case_id, "config_id": cid, "x_index": x_index,
                    "metric": spec.metric, "status": "missing", "error": str(exc),
                })
            continue
        for mode, fv, cid, x_index in configs:
            work = args.acc_dir / case_id / cid
            marker = work / "acc.json"
            if marker.is_file():
                value = float(marker.read_text().strip())
                print(f"[skip acc] {case_id}/{cid} {value:.6f}", flush=True)
            else:
                ns = acc_ns(fv, dither_thresh(case_id))
                print(f"==> acc {case_id}/{cid} n={len(indices)}", flush=True)
                result = acc.run_mc(ART, case_id, spec, adapter, indices, mode, ns, work)
                value = float(result["value"])
                work.mkdir(parents=True, exist_ok=True)
                marker.write_text(f"{value}\n", encoding="utf-8")
            if mode == "off":
                off_metric[case_id] = value
            delta = 0.0 if mode == "off" else value - off_metric.get(case_id, float("nan"))
            rows.append({
                "mc": case_id, "config_id": cid, "x_index": x_index,
                "metric": spec.metric, "value": value, "delta_vs_off": delta,
                "n": len(indices), "status": "ok",
            })
    return write_tsv(
        args.acc_dir / "acc.tsv",
        rows,
        ["mc", "config_id", "x_index", "metric", "value", "delta_vs_off", "n", "status"],
    )


def _series_style():
    colors = {
        "lenet": "#2f73bf", "vggnet": "#bf5b1f", "SqueezeNet": "#d98b20",
        "squeezenet": "#d98b20", "resnet": "#3b8c5a", "mobilenet": "#7e5ab8",
        "densenet": "#c44e52",
    }
    styles = {
        "mnist": {"linestyle": "-", "marker": "o"},
        "cifar": {"linestyle": "--", "marker": "s"},
        "imagenet50_96": {"linestyle": "-.", "marker": "^"},
        "chest": {"linestyle": ":", "marker": "D"},
    }
    return colors, styles


def _plot_one(series, ticks, ylabel, out: Path, ylim=None, hline=None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors, styles = _series_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.8))
    for (family, dataset), pts in series.items():
        pts = sorted(pts)
        st = styles.get(style_dataset(dataset), {"linestyle": "-", "marker": "o"})
        ax.plot(
            [p[0] for p in pts], [p[1] for p in pts],
            color=colors.get(family, "gray"), label=f"{family}/{dataset}", **st,
        )
    if hline is not None:
        ax.axhline(hline, color="0.5", ls="--", lw=0.8)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xticks(sorted(ticks))
    ax.set_xticklabels([ticks[i] for i in sorted(ticks)], rotation=30, ha="right")
    if series:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out}", flush=True)
    return out


def stage_plot(args: argparse.Namespace) -> None:
    meta = acc.load_manifest(MANIFEST, "mc")
    ticks = {0: "off"}
    series_rho_trans: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    series_acc: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    rho_trans_path = args.rho_trans_dir / "rho_trans.tsv"
    acc_path = args.acc_dir / "acc.tsv"
    if rho_trans_path.is_file():
        with rho_trans_path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row.get("status") != "ok":
                    continue
                info = meta[row["mc"]]
                xi = int(row["x_index"])
                ticks[xi] = row["x_label"]
                series_rho_trans[(info["family"], info["dataset"])].append((xi, float(row["rho_trans_on"])))
        _plot_one(series_rho_trans, ticks, r"$\rho_{\mathrm{trans}}$", args.rho_trans_dir / "rho_trans.png", ylim=(0, 0.6))
    if acc_path.is_file():
        with acc_path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row.get("status") != "ok":
                    continue
                info = meta[row["mc"]]
                xi = int(row["x_index"])
                if row.get("x_label"):
                    ticks[xi] = row["x_label"]
                elif xi and xi not in ticks:
                    ticks[xi] = str(xi)
                series_acc[(info["family"], info["dataset"])].append(
                    (xi, float(row["delta_vs_off"]) * 100.0)
                )
        _plot_one(series_acc, ticks, r"$\Delta$ accuracy (%)", args.acc_dir / "acc.png", hline=0.0)
    if not series_rho_trans and not series_acc:
        print("nothing to plot", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("all", "pin", "rho_trans", "acc", "plot"), default="all")
    p.add_argument("--case", action="append", default=[], dest="cases", help="MC id (repeatable). Default: 14-MC figure set.")
    p.add_argument("--limit", type=int, default=None, help="acc sample cap (debug)")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = p.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be > 0")
    args.cases = args.cases or list(DEFAULT_CASES)
    args.root = args.root.resolve()
    args.acc_dir = args.root / "acc"
    args.rho_trans_dir = args.root / "rho_trans"
    return args


def main() -> int:
    args = parse_args()
    args.acc_dir.mkdir(parents=True, exist_ok=True)
    args.rho_trans_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "pin"):
        stage_pin(args)
    if args.stage in ("all", "rho_trans"):
        stage_rho_trans(args)
    if args.stage in ("all", "acc"):
        stage_acc(args)
    if args.stage in ("all", "plot"):
        stage_plot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

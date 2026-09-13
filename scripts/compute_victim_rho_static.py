#!/usr/bin/env python3
"""Compute ρ_static on protected activation writes. Artifact-local.

Plaintext ONNX ReLU/Clip outputs, T test images. Off = raw writes.
On = Cfg-SR75 zero rewrite (30 / 7 / fv=117 / inc=3), simulated per
(layer index, site). Static set is {0} only.

Does not read Pin traces, runners, or bundles.
Default 11 MC: no CelebA, no MobileNet (MC11/12/13/14).

Usage:
  python3 scripts/compute_victim_rho_static.py
  python3 scripts/compute_victim_rho_static.py --case MC01
  python3 scripts/compute_victim_rho_static.py --num-images 512
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
ART = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import compute_victim_acc as acc  # noqa: E402

DEFAULT_CASES = (
    "MC01", "MC02", "MC03", "MC04", "MC05", "MC06", "MC07", "MC08",
    "MC10", "MC15", "MC17",
)
SKIP_FAMILIES = frozenset({"mobilenet"})
NUM_IMAGES = 512
LAYERS_PER_GROUP = 4
SEED = 1234
ZERO_U32 = np.uint32(0)
NEG_ZERO_U32 = np.uint32(0x80000000)


@dataclass
class CfgSR75:
    bit_count: int = 30
    fixed_bit_count: int = 7
    fixed_value: int = 0b1110101
    increment: int = 3

    def __post_init__(self) -> None:
        payload_bits = self.bit_count - self.fixed_bit_count
        self.patch_payload_mask = (1 << payload_bits) - 1 if payload_bits else 0
        fv_mask = (1 << self.fixed_bit_count) - 1
        self.fixed_window_bits = (self.fixed_value & fv_mask) << payload_bits


def splitmix32(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.uint32)
    z = z + np.uint32(0x9E3779B9)
    z = (z ^ (z >> np.uint32(16))) * np.uint32(0x85EBCA6B)
    z = (z ^ (z >> np.uint32(13))) * np.uint32(0xC2B2AE35)
    return (z ^ (z >> np.uint32(16))).astype(np.uint32, copy=False)


def derive_payloads(exec_seed: int, layer_idx: int, n_sites: int, mask: int) -> np.ndarray:
    sites = np.arange(n_sites, dtype=np.uint32)
    layer_part = (np.uint32(layer_idx) * np.uint32(0x85EBCA77)).astype(np.uint32)
    mixed = np.bitwise_xor(np.uint32(exec_seed), layer_part)
    mixed = np.bitwise_xor(mixed, sites * np.uint32(0xC2B2AE3D))
    return splitmix32(mixed) & np.uint32(mask)


def apply_defense_layer(y_off: np.ndarray, layer_idx: int, exec_seeds: np.ndarray, cfg: CfgSR75) -> np.ndarray:
    y_on = np.array(y_off, copy=True, order="C")
    t_len, n_sites = y_off.shape
    delta = np.uint32(cfg.increment)
    mask = np.uint32(cfg.patch_payload_mask)
    fixed = np.uint32(cfg.fixed_window_bits)
    for t in range(t_len):
        is_zero = y_off[t] == 0.0
        if not np.any(is_zero):
            continue
        payloads = derive_payloads(int(exec_seeds[t]), layer_idx, n_sites, cfg.patch_payload_mask)
        bits = fixed | (((payloads.astype(np.uint32) + delta) & mask).astype(np.uint32))
        y_on[t, is_zero] = bits.view(np.float32)[is_zero]
    return y_on


def float32_to_u32(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    out = bits.copy()
    out[out == NEG_ZERO_U32] = ZERO_U32
    return out


def layer_static_collisions(y: np.ndarray) -> tuple[int, int]:
    bits = float32_to_u32(y)
    is_zero = bits == ZERO_U32
    t_len = bits.shape[0]
    L = int(bits.size)
    if t_len == 0 or L == 0:
        return 0, L
    cs = np.cumsum(is_zero, axis=0, dtype=np.int64)
    prior = np.zeros_like(is_zero, dtype=bool)
    if t_len > 1:
        prior[1:] = cs[:-1] > 0
    return int((is_zero & prior).sum()), L


def list_activation_outs(onnx_path: Path) -> list[str]:
    import onnx

    model = onnx.load(str(onnx_path))
    return [n.output[0] for n in model.graph.node if n.op_type in {"Relu", "Clip"}]


def onnx_input_name(onnx_path: Path) -> str:
    import onnx

    return onnx.load(str(onnx_path)).graph.input[0].name


def build_session(onnx_path: Path, output_names: list[str]):
    import onnx
    import onnxruntime as ort
    from onnx import helper

    model = onnx.load(str(onnx_path))
    existing = {vi.name for vi in model.graph.output}
    for name in output_names:
        if name not in existing:
            model.graph.output.append(helper.ValueInfoProto(name=name))
            existing.add(name)
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])


def wrap_chest(adapter: acc.Adapter, need: int) -> acc.Adapter:
    if not hasattr(adapter, "names"):
        return adapter
    keep: list[int] = []
    image_dir = adapter.image_dir
    for i, name in enumerate(adapter.names):
        if (image_dir / name).is_file():
            keep.append(i)
            if len(keep) >= need:
                break
    if len(keep) < need:
        raise RuntimeError(f"chest adapter found only {len(keep)} images under {image_dir}")

    class Existing(acc.Adapter):
        def __len__(self) -> int:
            return len(keep)

        def get(self, index: int):
            return adapter.get(keep[index])

    return Existing()


def already_done(path: Path, num_images: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if payload.get("num_images") == num_images and "rho_static_off" in payload:
        return payload
    return None


def run_case(
    case_id: str,
    info: dict[str, str],
    onnx_path: Path,
    spec: acc.DataSpec,
    adapter: acc.Adapter,
    num_images: int,
    layers_per_group: int,
    out_dir: Path,
) -> dict[str, Any]:
    verdict_path = out_dir / case_id / "verdict.json"
    cached = already_done(verdict_path, num_images)
    if cached is not None:
        print(f"[skip] {case_id} T={num_images}", flush=True)
        return cached

    n_use = min(num_images, len(adapter))
    relu_outs = list_activation_outs(onnx_path)
    if not relu_outs:
        raise RuntimeError(f"{case_id}: no Relu/Clip in {onnx_path}")
    input_name = onnx_input_name(onnx_path)
    cfg = CfgSR75()
    case_dir = out_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    seeds_path = case_dir / "seeds.npz"
    if seeds_path.is_file():
        exec_seeds = np.load(seeds_path)["seeds"][:n_use].astype(np.uint32)
        if exec_seeds.shape[0] < n_use:
            exec_seeds = np.random.default_rng(SEED).integers(0, 2**32, size=n_use, dtype=np.uint32)
            np.savez_compressed(seeds_path, seeds=exec_seeds)
    else:
        exec_seeds = np.random.default_rng(SEED).integers(0, 2**32, size=n_use, dtype=np.uint32)
        np.savez_compressed(seeds_path, seeds=exec_seeds)

    off_s = off_L = on_s = on_L = 0
    layer_rows: list[dict[str, Any]] = []
    t0 = time.time()
    n_layers = len(relu_outs)
    print(
        f"[run] {case_id} {info['family']}/{info['dataset']} "
        f"T={n_use} layers={n_layers} {spec.height}x{spec.width}x{spec.channels}",
        flush=True,
    )

    for g_start in range(0, n_layers, layers_per_group):
        g_end = min(g_start + layers_per_group, n_layers)
        g_outs = relu_outs[g_start:g_end]
        sess = build_session(onnx_path, g_outs)
        buffers: list[list[np.ndarray]] = [[] for _ in g_outs]
        for ii in range(n_use):
            image, _label = adapter.get(ii)
            x = acc.image_to_nchw(image, spec)
            outs = sess.run(g_outs, {input_name: x})
            for buf, y in zip(buffers, outs):
                buf.append(np.ascontiguousarray(y[0], dtype=np.float32).reshape(-1))
            if (ii + 1) % 64 == 0 or ii + 1 == n_use:
                print(f"  {case_id} L{g_start:02d}-L{g_end - 1:02d} {ii + 1}/{n_use}", flush=True)
        for li, buf in enumerate(buffers):
            layer_idx = g_start + li
            y_off = np.stack(buf, axis=0)
            y_on = apply_defense_layer(y_off, layer_idx, exec_seeds, cfg)
            n_off, L_off = layer_static_collisions(y_off)
            n_on, L_on = layer_static_collisions(y_on)
            layer_rows.append({
                "layer": f"L{layer_idx:02d}",
                "n_elements": int(y_off.shape[1]),
                "n_static_off": n_off,
                "L_off": L_off,
                "rho_static_off": n_off / L_off if L_off else float("nan"),
                "n_static_on": n_on,
                "L_on": L_on,
                "rho_static_on": n_on / L_on if L_on else float("nan"),
            })
            off_s += n_off
            off_L += L_off
            on_s += n_on
            on_L += L_on

    rho_off = off_s / off_L if off_L else float("nan")
    rho_on = on_s / on_L if on_L else float("nan")
    r_static = 1.0 - (rho_on / rho_off) if rho_off > 0 else float("nan")
    verdict = {
        "mc": case_id,
        "task": info["task"],
        "model": info["model"],
        "dataset": info["dataset"],
        "onnx": str(onnx_path),
        "num_images": n_use,
        "n_layers": n_layers,
        "S_stat": "{0}",
        "n_static_off": off_s,
        "L_off": off_L,
        "rho_static_off": rho_off,
        "n_static_on": on_s,
        "L_on": on_L,
        "rho_static_on": rho_on,
        "R_static": r_static,
        "elapsed_s": round(time.time() - t0, 2),
        "layers": layer_rows,
    }
    verdict_path.write_text(json.dumps(verdict, indent=2))
    print(f"[done] {case_id} rho_static_off={rho_off:.6f} rho_static_on={rho_on:.6f}", flush=True)
    return verdict


def write_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "mc", "task", "model", "dataset", "T", "n_layers",
        "n_static_off", "L_off", "rho_static_off",
        "n_static_on", "L_on", "rho_static_on", "R_static", "status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--case", action="append", default=[], dest="cases")
    parser.add_argument("--manifest", type=Path, default=ART / "victims/models/MANIFEST.txt")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--imagenet50-96-root", type=Path, default=None)
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES)
    parser.add_argument("--layers-per-group", type=int, default=LAYERS_PER_GROUP)
    parser.add_argument("--out-root", type=Path, default=ART / "rho_static_out")
    args = parser.parse_args()
    if args.num_images <= 0:
        raise SystemExit("--num-images must be > 0")

    meta = acc.load_manifest(args.manifest, "mc")
    case_ids = acc.normalize_cases(args.cases, "mc", list(DEFAULT_CASES))
    unknown = [c for c in case_ids if c not in meta]
    if unknown:
        raise SystemExit(f"unknown case(s): {unknown}")
    skipped = [c for c in case_ids if meta[c]["family"] in SKIP_FAMILIES]
    if skipped:
        raise SystemExit(f"rho_static does not include MobileNet: {skipped}")

    data_root = acc.resolve_dataset_root(args.dataset_root)
    imagenet_root = acc.resolve_imagenet50_root(args.imagenet50_96_root, data_root)
    models = ART / "victims/models"
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"dataset_root={data_root}", flush=True)
    print("mc\trho_static_off\trho_static_on\tR_static", flush=True)

    rows: list[dict[str, Any]] = []
    missing = 0
    for case_id in case_ids:
        info = meta[case_id]
        spec = acc.spec_for(info)
        onnx_path = models / info["onnx_rel"]
        try:
            if not onnx_path.is_file():
                raise FileNotFoundError(onnx_path)
            adapter = acc.build_adapter(spec, data_root, imagenet_root)
            if spec.source == "chestxray14":
                adapter = wrap_chest(adapter, args.num_images)
            verdict = run_case(
                case_id, info, onnx_path, spec, adapter,
                args.num_images, args.layers_per_group, out_root,
            )
        except (FileNotFoundError, ModuleNotFoundError, RuntimeError, OSError) as exc:
            print(f"{case_id}\tmissing\t{exc}", flush=True)
            rows.append({
                "mc": case_id,
                "task": info["task"],
                "model": info["model"],
                "dataset": info["dataset"],
                "status": "missing",
            })
            missing += 1
            continue
        rows.append({
            "mc": verdict["mc"],
            "task": verdict["task"],
            "model": verdict["model"],
            "dataset": verdict["dataset"],
            "T": verdict["num_images"],
            "n_layers": verdict["n_layers"],
            "n_static_off": verdict["n_static_off"],
            "L_off": verdict["L_off"],
            "rho_static_off": f"{verdict['rho_static_off']:.9f}",
            "n_static_on": verdict["n_static_on"],
            "L_on": verdict["L_on"],
            "rho_static_on": f"{verdict['rho_static_on']:.9f}",
            "R_static": f"{verdict['R_static']:.9f}",
            "status": "ok",
        })
        print(
            f"{case_id}\t{verdict['rho_static_off']:.6f}\t"
            f"{verdict['rho_static_on']:.6f}\t{verdict['R_static']:.4f}",
            flush=True,
        )

    write_tsv(rows, out_root / "rho_static.tsv")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
